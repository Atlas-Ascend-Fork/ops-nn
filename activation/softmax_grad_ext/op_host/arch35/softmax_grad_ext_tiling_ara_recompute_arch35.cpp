/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_ext_tiling_ara_recompute.cc
 * \brief
 */
#include "softmax_grad_ext_tiling.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling {

constexpr int64_t DEFAULT_BIN_ADD_R_FACTOR = 128;
// 段长折叠容量耦合约束：默认段长须与 kernel 侧 scratch 总承载行数一致（人工同步易漏，编译期钉死）
static_assert(DEFAULT_BIN_ADD_R_FACTOR == 128,
              "must equal SCRATCH_ROW_NUM * ROWS_PER_SCRATCH(128) in softmax_grad_ext_ara_recompute.h");
// xSum 1 + temp 1 + reduceSumTemp 8；相对前向(11)去掉xMax——grad无max
constexpr int64_t UB_COMPUTE_FACTOR_GRAD = 10;
constexpr int64_t UB_RESERVED_BYTE = 1024 * 8;
constexpr int64_t UB_QUEUE_COUNT = 4; // grad/x1/x2/y 四个 T 队列

bool SoftmaxGradExtTilingARARecompute::IsCapable()
{
    // a0_为1的场景走AR模板
    OP_TILING_CHECK(
        a0_ == DIM_NUM_ONE,
        OP_LOGI(context_->GetNodeName(), "ARA recompute template is not capable, a0 should be greater than 1."),
        return false);
    return true;
}

int64_t SoftmaxGradExtTilingARARecompute::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

ge::graphStatus SoftmaxGradExtTilingARARecompute::ComputeBinaryAddParams()
{
    if (binAddRFactor_ == 0) { // 首次调用用默认段长；mc 复用调用时保留已设的 mc 段因子
        binAddRFactor_ = DEFAULT_BIN_ADD_R_FACTOR;
    }
    binAddRLoop_ = r_ / binAddRFactor_;
    binAddRTotalLoop_ = CeilDiv(r_, binAddRFactor_);
    binAddRTail_ = r_ - (binAddRTotalLoop_ - 1) * binAddRFactor_;
    // 计算最接近binAddRTotalLoop_的2^k
    binAddBasicBlockLoop_ = binAddRTotalLoop_ > 1 ?
                                (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(binAddRTotalLoop_ - 1))) :
                                0;
    mainFoldCount_ = binAddRLoop_ - binAddBasicBlockLoop_;
    if (binAddBasicBlockLoop_ == 0) {
        binAddCacheBufferCount_ = 1;
        binAddResultCacheID_ = 0;
    } else {
        binAddCacheBufferCount_ = ULONG_BIT_LEN - __builtin_clzl(binAddBasicBlockLoop_);
        binAddResultCacheID_ = GetCacheID(binAddBasicBlockLoop_ - 1);
    }

    OP_LOGI(context_->GetNodeName(),
            "Binary add rFactor: %ld, rLoop: %ld, binAddRTotalLoop_: %ld, rTail: %ld, basicBlockLoop: %ld, "
            "mainFoldCount:%ld,cacheBufferCount: "
            "%ld, resultCacheId:%ld ",
            binAddRFactor_, binAddRLoop_, binAddRTotalLoop_, binAddRTail_, binAddBasicBlockLoop_, mainFoldCount_,
            binAddCacheBufferCount_, binAddResultCacheID_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxGradExtTilingARARecompute::DoOpTiling()
{
    OP_CHECK_IF(
        ComputeBinaryAddParams() != ge::GRAPH_SUCCESS,
        OP_LOGI(context_->GetNodeName(), "ARA recompute template is not capable, compute binary add tiling not ok."),
        return ge::GRAPH_PARAM_INVALID);

    a0TileBase_ = xDtype_ == ge::DT_FLOAT ? blockSize_ / FLOAT32_BYTES : blockSize_ / FLOAT16_BYTES;

    // UB容量 = 4个T队列双缓冲（R方向按段长）+ fp32计算缓冲（xSum/temp/8行reduceSumTemp/cache）
    int64_t factorMax = (aicoreParams_.ubSize - UB_RESERVED_BYTE) / a0TileBase_ /
                        (binAddRFactor_ * DOUBLE_BUFFER * xDtypeSize_ * UB_QUEUE_COUNT +
                         FLOAT32_BYTES * (UB_COMPUTE_FACTOR_GRAD + binAddCacheBufferCount_));

    OP_CHECK_IF(factorMax <= 0,
                OP_LOGI(context_->GetNodeName(),
                        "ARA recompute template is not capable. merged shape is (%ld, %ld, %ld), ub size: %ldB, "
                        "tileBase: %ld, ub factor: %ld.",
                        a1_, r_, a0_, aicoreParams_.ubSize, a0TileBase_, factorMax),
                return ge::GRAPH_PARAM_INVALID);

    // 尽量占多核：flat 候选宽度从大到小 {64, 32, 16, 8}（需 ≤a0、为 a0TileBase 倍数、整除 vlFp32、
    // 且 blockLen 至少 2 block——小块搬运 MTE2 吞吐骤降，实测 bf16 32B/行 满核反而不如 64B/行 少核、
    // 宽度需不超 factorMax（UB 容量）），取首个使 a1*CeilDiv(a0, w) >= blockDim 的宽度；
    // 都不满足则回退 a0InnerMax 最小化切法
    int64_t blockDim = static_cast<int64_t>(aicoreParams_.blockDim);
    int64_t a0FactorMax = CeilDiv(a0_, a0TileBase_);
    int64_t totalTilesMax = a1_ * a0FactorMax;
    constexpr int64_t FLAT_WIDTHS[4] = {64, 32, 16, 8};
    int64_t a0Inner = 0;
    for (int64_t w : FLAT_WIDTHS) {
        if (w > a0_ || w % a0TileBase_ != 0 || vlFp32_ % w != 0) {
            continue;
        }
        if (w * xDtypeSize_ < 2 * blockSize_) {
            continue;
        }
        if (w / a0TileBase_ > factorMax) {
            continue; // 宽度超出 UB 容量（factorMax 为 a0TileBase_ 单位上限），降级更窄 flat 宽度
        }
        if (a1_ * CeilDiv(a0_, w) >= blockDim) {
            a0Inner = w / a0TileBase_;
            break;
        }
    }
    if (a0Inner == 0) {
        int64_t a0InnerMax = CeilDiv(totalTilesMax, blockDim);
        a0Inner = a0InnerMax < factorMax ? a0InnerMax : factorMax;
        a0Inner = a0Inner < a0FactorMax ? a0Inner : a0FactorMax;
    }

    tileA0Len_ = a0Inner * a0TileBase_;
    a0Outer_ = CeilDiv(a0_, tileA0Len_);
    tileA0Tail_ = a0_ - tileA0Len_ * (a0Outer_ - 1);

    // 不切R轴
    totalTiles_ = a1_ * a0Outer_;
    usedCoreNums_ = totalTiles_ < blockDim ? totalTiles_ : blockDim;
    tilesPerCore_ = CeilDiv(totalTiles_, usedCoreNums_);

    // mc（R 切分多核）：旧切法 tile 数占不满核且 a0 整除 64 lane（flat 满车道可行）时，
    // 改按 (a1, R 分块) 满核两阶段协同（P1 部分和 → SyncAll 汇总 S → SyncAll → P2 全宽输出），
    // 部分和复用输出张量头部 overlay，规避 TTK 下 workspace 仅 32B 的问题
    constexpr int64_t MC_EXTRA_UB_BYTES = 4096; // mc 附加队列预留
    bool mcCapable = blockDim >= 2 && a1_ <= blockDim && a0_ <= vlFp32_ && vlFp32_ % a0_ == 0 &&
                     totalTilesMax < blockDim && a1_ * (blockDim / a1_) > totalTilesMax;
    // mc 段因子：mc 不使用二分累加缓存结构，按四队列双缓冲段长核算（原非 mc 公式过度保守）
    int64_t denom = UB_QUEUE_COUNT * DOUBLE_BUFFER * a0_ * xDtypeSize_;
    int64_t computeBytes = 11 * a0_ * FLOAT32_BYTES; // xSum1 + reduceSumTemp8 + temp1 + cache1
    int64_t avail = aicoreParams_.ubSize - UB_RESERVED_BYTE * 2 - MC_EXTRA_UB_BYTES - computeBytes;
    int64_t mcSegFactor = avail / denom;
    mcSegFactor = mcSegFactor < DEFAULT_BIN_ADD_R_FACTOR ? mcSegFactor : DEFAULT_BIN_ADD_R_FACTOR;
    while (mcSegFactor > 0 && (mcSegFactor * a0_) % vlFp32_ != 0) { // 段内整 quad
        --mcSegFactor;
    }
    int64_t mcEnabled = 0;
    int64_t mcChunksPerA1 = 0;
    if (mcCapable && mcSegFactor >= 1) {
        mcEnabled = 1;
        mcEnabled_ = true;
        binAddRFactor_ = mcSegFactor;
        ComputeBinaryAddParams();
        mcChunksPerA1 = blockDim / a1_;
        usedCoreNums_ = a1_ * mcChunksPerA1;
        tileA0Len_ = a0_;
        a0Outer_ = 1;
        tileA0Tail_ = a0_;
        totalTiles_ = a1_;
        tilesPerCore_ = 1;
    }

    tilingData_.set_totalTiles(totalTiles_);
    tilingData_.set_tilesPerCore(tilesPerCore_);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    tilingData_.set_totalRLen(r_);
    tilingData_.set_totalA0Len(a0_);
    tilingData_.set_tileA0Outer(a0Outer_);
    tilingData_.set_tileA0Len(tileA0Len_);
    tilingData_.set_tileA0Tail(tileA0Tail_);
    tilingData_.set_x2IsScalar(isX2Scalar_ ? 1L : 0L);

    // binary add
    tilingData_.set_binAddRFactor(binAddRFactor_);
    tilingData_.set_binAddRLoop(binAddRLoop_);
    tilingData_.set_binAddRTotalLoop(binAddRTotalLoop_);
    tilingData_.set_binAddRTail(binAddRTail_);
    tilingData_.set_binAddBasicBlockLoop(binAddBasicBlockLoop_);
    tilingData_.set_binAddMainFoldCount(mainFoldCount_);
    tilingData_.set_binAddCacheBufferCount(binAddCacheBufferCount_);
    tilingData_.set_binAddResultCacheID(binAddResultCacheID_);
    tilingData_.set_mcEnabled(mcEnabled);
    tilingData_.set_mcA1Len(a1_);
    tilingData_.set_mcChunksPerA1(mcChunksPerA1);

    OP_LOGI(context_->GetNodeName(),
            "Do tiling success, tileA0Len: %ld, a0Outer_: %ld, tileA0Tail_: %ld, totalTiles_: %ld, tilesPerCore_: "
            "%ld, usedCoreNums_:%ld, mcEnabled: %ld, mcChunksPerA1: %ld",
            tileA0Len_, a0Outer_, tileA0Tail_, totalTiles_, tilesPerCore_, usedCoreNums_, mcEnabled, mcChunksPerA1);

    return ge::GRAPH_SUCCESS;
}

uint64_t SoftmaxGradExtTilingARARecompute::GetTilingKey() const { return TILINGKEY_ARA_RECOMPUTE; }

ge::graphStatus SoftmaxGradExtTilingARARecompute::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    if (mcEnabled_) {
        // kernel 使用 SyncAll 核间同步，需设置为 batch mode，避免 resident launch 下 SyncAll 永等
        constexpr uint32_t BATCH_MODE = 1;
        auto scheduleRet = context_->SetScheduleMode(BATCH_MODE);
        OP_CHECK_IF(scheduleRet != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "SetScheduleMode error"),
                    return ge::GRAPH_FAILED);
    }
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SoftmaxGradExt, SoftmaxGradExtTilingARARecompute, TEMPLATE_ARA_RECOMPUTE_PRIORITY);
} // namespace optiling
