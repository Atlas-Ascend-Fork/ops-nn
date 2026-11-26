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
 * \file batch_norm_grad_ext2_tiling_infer.cpp
 * \brief
 */

#include <cstdint>
#include <string>
#include "batch_norm_grad_ext2_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "batch_norm_grad_ext2_tiling_infer_base.h"

using namespace ge;

namespace optiling {
static constexpr int64_t TWO = 2;
static constexpr int64_t FOUR = 4;

static constexpr uint64_t BNG_RAR_RECOMPUTE_SPLIT_R1_INFER = 1;
static constexpr uint64_t BNG_RAR_RECOMPUTE_SPLIT_R0_INFER = 2;
static constexpr int ULONG_BIT_LEN = 64;
static constexpr int64_t BNG_PER_CORE_PROCESS_MIN_UB_SIZE = 1024;
// 与 kernel batch_norm_grad_ext2_infer.h 的 RS_MAX_A_DIM / RS_MIN_R1 保持一致
static constexpr int64_t RS_MAX_A_DIM = 8;
static constexpr int64_t RS_MIN_R1 = 16;

class BatchNormGradExt2InferTiling : public BatchNormGradExt2InferBase {
public:
    explicit BatchNormGradExt2InferTiling(gert::TilingContext* context) : BatchNormGradExt2InferBase(context) {}
    ~BatchNormGradExt2InferTiling() override = default;

protected:
    bool IsCapable() override
    {
        // R1AR0, R0 == 1时走RA模板
        OP_CHECK_IF(
            r0Dim == 1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "r0Dim", std::to_string(r0Dim).c_str(),
                                                  "RAR template is not capable when r0Dim is 1"),
            return false);

        CalcBasicInfo();

        OP_LOGD(context_, "BatchNormGradExt2InferTiling RAR template is capable, fused shape: (%ld, %ld, %ld)", r1Dim,
                aDim, r0Dim);
        return true;
    }

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus DoOpTilingForStage0();
    ge::graphStatus DoOpTilingForStage1();
    void DoRecomputeTilingSplitR1();
    void DoRecomputeTilingSplitR0();
    void DoBinaryAddTiling(BatchNormGradExt2BinaryAddTilingData& tilingData, int64_t quotient);

    int64_t onceProcUbNeed{0}; // A轴上执行一次，需要占用的ub大小
    bool rSplitMode{false};    // dgamma/dbeta 按 R1 跨核切分（kernel: useRSplit_）
    uint64_t binaryAddBufSize{0};
    uint64_t subTilingKey{0};
    int64_t r1Factor{0};
    int64_t r0Factor{0};
    int64_t ubRDimLoopNum{0};
    int64_t ubRDimFactor{0};
    int64_t ubRDimFactorAlign{0};
    int64_t ubRDimTailFactor{0};
    int64_t ubRDimTailFactorAlign{0};
    int64_t ubRDimTail{0};
    int64_t ubRDimTailLoopNum{0};
    int64_t ubRDimTailTail{0};
    int64_t ubRDimTailTailFactor{0};
    int64_t ubRDimTailTailFactorAlign{0};
    int64_t ubRDimTailTailLoopNum{0};
    BatchNormGradExt2InferTilingData tilingData_;
};

ge::graphStatus BatchNormGradExt2InferTiling::DoOpTilingForStage0()
{
    // 切分A、B基本块， （B0,A,B1） -- >(r1Outer*aOuter*r0Outer, B0inner*Ainner*B1innerA(TileBase))
    int64_t aInner = 1;
    int64_t ubBufferSize = (aicoreParams_.ubSize / DOUBLE_BUFFER -
                            (bytesPerWeight_ + bytesPerRunningVar_) * aInner * aTileBase_) /
                           bytesPerDy_ / INPUT_OUTPUT_NUM;

    // 先按照B切分，再切A
    // UB可载入最大tile块数
    int64_t factorMax = ubBufferSize / aTileBase_;

    // 默认策略: 先按照B0, B1把UB切满
    int64_t b1FactorMax = Ops::Base::CeilDiv(r0Dim, aTileBase_);          // 处理 b1 需要多少 vf
    int64_t b1Inner = factorMax <= b1FactorMax ? factorMax : b1FactorMax; // b1 vf 内循环处理多少个次
    int64_t r0Outer = Ops::Base::CeilDiv(r0Dim, b1Inner * aTileBase_);    // b1 外循环

    factorMax = factorMax / b1Inner;
    int64_t b0FactorMax = r1Dim;
    int64_t r1Inner = factorMax <= b0FactorMax ? factorMax : b0FactorMax;
    int64_t r1Outer = Ops::Base::CeilDiv(r1Dim, r1Inner);

    factorMax = factorMax / r1Inner;
    int64_t aFactorMax = aDim;
    aInner = factorMax <= aFactorMax ? factorMax : aFactorMax;
    int64_t aOuter = Ops::Base::CeilDiv(aDim, aInner);

    int64_t totalTiles = r1Outer * aOuter * r0Outer;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    int64_t tileBlockR1Tail = r1Dim - r1Inner * (r1Outer - 1);
    int64_t tileBlockATail = aDim - aInner * (aOuter - 1);
    int64_t tileBlockR0Tail = r0Dim - b1Inner * aTileBase_ * (r0Outer - 1);

    tilingData_.baseTilingData.totalTiles = totalTiles;
    tilingData_.baseTilingData.tilesPerCore = tilesPerCore;
    tilingData_.baseTilingData.usedCoreNums = usedCoreNums_;
    tilingData_.baseTilingData.r1Dim = r1Dim;
    tilingData_.baseTilingData.aDim = aDim;
    tilingData_.baseTilingData.r0Dim = r0Dim;
    tilingData_.baseTilingData.r1Outer = r1Outer;
    tilingData_.baseTilingData.aOuter = aOuter;
    tilingData_.baseTilingData.r0Outer = r0Outer;
    tilingData_.baseTilingData.tileBlockR1Len = r1Inner;
    tilingData_.baseTilingData.tileBlockR1Tail = tileBlockR1Tail;
    tilingData_.baseTilingData.tileBlockALen = aInner;
    tilingData_.baseTilingData.tileBlockATail = tileBlockATail;
    tilingData_.baseTilingData.tileBlockR0Len = b1Inner * aTileBase_;
    tilingData_.baseTilingData.tileBlockR0Tail = tileBlockR0Tail;
    tilingData_.baseTilingData.tileBlockAPaddingNum = 0;
    tilingData_.baseTilingData.epsilon = epsilon_;
    return ge::GRAPH_SUCCESS;
}

void BatchNormGradExt2InferTiling::DoRecomputeTilingSplitR1()
{
    if (Ops::Base::CeilDiv(static_cast<int64_t>(r1Dim * r0Dim * sizeof(float)), blockSize_) * blockSize_ <=
        static_cast<int64_t>(binaryAddBufSize)) {
        ubRDimFactor = r1Dim * r0Dim;
        ubRDimFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimFactor * sizeof(float)), blockSize_) *
                            blockSize_ / sizeof(float);
        tilingData_.ubRDimFactor = ubRDimFactor;
        tilingData_.ubRDimFactorAlign = ubRDimFactorAlign;
        tilingData_.ubRDimLoopNum = 1;
        tilingData_.ubRDimTail = 0;
        tilingData_.ubRDimTailFactor = 0;
        tilingData_.ubRDimTailFactorAlign = 0;
        tilingData_.ubRDimTailLoopNum = 0;
        tilingData_.ubRDimTailTail = 0;
        tilingData_.ubRDimTailTailFactor = 0;
        tilingData_.ubRDimTailTailFactorAlign = 0;
        tilingData_.ubRDimTailTailLoopNum = 0;
        return;
    }
    r1Factor = (binaryAddBufSize / (r0Dim * sizeof(float))) &
               (~1L); // 计算 ub 最大能装多少 r1          (r1, A, r0) -> (r1Dim / r1Factor, A, r1Factor * r0)
    ubRDimLoopNum = (1L << (ULONG_BIT_LEN - 1 -
                            __builtin_clzl(r1Dim / r1Factor))); // 小于等于r1Dim / r1Factor 的最大2次幂
    ubRDimFactor = r1Factor * r0Dim;
    ubRDimFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimFactor * sizeof(float)), blockSize_) *
                        blockSize_ / sizeof(float);
    ubRDimTail = r1Dim * r0Dim - ubRDimFactor * ubRDimLoopNum;
    ubRDimTailFactor = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimFactor / TWO : ubRDimFactor;
    ubRDimTailFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimTailFactor * sizeof(float)), blockSize_) *
                            blockSize_ / sizeof(float);
    ubRDimTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimTail / ubRDimFactor * TWO :
                                                                                       ubRDimTail / ubRDimFactor;
    ubRDimTailTail = ubRDimTail - ubRDimTailLoopNum * ubRDimTailFactor;
    ubRDimTailTailFactor = ubRDimTailFactor;
    ubRDimTailTailFactorAlign = ubRDimTailFactorAlign;
    ubRDimTailTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? TWO : 1;

    tilingData_.ubRDimFactor = ubRDimFactor;
    tilingData_.ubRDimFactorAlign = ubRDimFactorAlign;
    tilingData_.ubRDimLoopNum = ubRDimLoopNum;
    tilingData_.ubRDimTail = ubRDimTail;
    tilingData_.ubRDimTailFactor = ubRDimTailFactor;
    tilingData_.ubRDimTailFactorAlign = ubRDimTailFactorAlign;
    tilingData_.ubRDimTailLoopNum = ubRDimTailLoopNum;
    tilingData_.ubRDimTailTail = ubRDimTailTail;
    tilingData_.ubRDimTailTailFactor = ubRDimTailTailFactor;
    tilingData_.ubRDimTailTailFactorAlign = ubRDimTailTailFactorAlign;
    tilingData_.ubRDimTailTailLoopNum = ubRDimTailTailLoopNum;
}

void BatchNormGradExt2InferTiling::DoRecomputeTilingSplitR0()
{
    r0Factor = r0Dim * sizeof(float) / binaryAddBufSize;
    ubRDimLoopNum = r0Factor <= 0 ? 1L : std::max(1L, (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(r0Factor))));
    ubRDimFactor = std::min(static_cast<uint64_t>(r0Dim), binaryAddBufSize / sizeof(float));
    ubRDimFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimFactor * sizeof(float)), blockSize_) *
                        blockSize_ / sizeof(float);
    ubRDimTail = r0Dim - ubRDimFactor * ubRDimLoopNum;
    // ubRDimFactor / 2 * 2 may be less than ubRDimFactor
    ubRDimTailFactor = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? (ubRDimFactor + TWO - 1) / TWO :
                                                                                      ubRDimFactor;
    ubRDimTailFactorAlign = Ops::Base::CeilDiv(static_cast<int64_t>(ubRDimTailFactor * sizeof(float)), blockSize_) *
                            blockSize_ / sizeof(float);
    ubRDimTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? ubRDimTail / ubRDimFactor * TWO :
                                                                                       ubRDimTail / ubRDimFactor;
    ubRDimTailTail = ubRDimTail - ubRDimTailLoopNum * ubRDimTailFactor;
    ubRDimTailTailFactor = ubRDimTailFactor;
    ubRDimTailTailFactorAlign = ubRDimTailFactorAlign;
    ubRDimTailTailLoopNum = ubRDimFactorAlign * sizeof(float) > (binaryAddBufSize / TWO) ? TWO : 1;

    tilingData_.ubRDimFactor = ubRDimFactor;
    tilingData_.ubRDimFactorAlign = ubRDimFactorAlign;
    tilingData_.ubRDimLoopNum = ubRDimLoopNum;
    tilingData_.ubRDimTail = ubRDimTail;
    tilingData_.ubRDimTailFactor = ubRDimTailFactor;
    tilingData_.ubRDimTailFactorAlign = ubRDimTailFactorAlign;
    tilingData_.ubRDimTailLoopNum = ubRDimTailLoopNum;
    tilingData_.ubRDimTailTail = ubRDimTailTail;
    tilingData_.ubRDimTailTailFactor = ubRDimTailTailFactor;
    tilingData_.ubRDimTailTailFactorAlign = ubRDimTailTailFactorAlign;
    tilingData_.ubRDimTailTailLoopNum = ubRDimTailTailLoopNum;
}

void BatchNormGradExt2InferTiling::DoBinaryAddTiling(BatchNormGradExt2BinaryAddTilingData& tilingData, int64_t quotient)
{
    tilingData.binaryAddQuotient = quotient;
    int64_t vcaddNum = quotient / vlFp32_;
    if (vcaddNum <= vlFp32_) {
        tilingData.binaryAddk = 0;
        tilingData.binaryAddLastNum = vcaddNum;
    } else {
        int64_t binaryAddNum = vcaddNum / vlFp32_;
        tilingData.binaryAddk = __builtin_ctzl(binaryAddNum);
        tilingData.binaryAddLastNum = vlFp32_;
    }
}

ge::graphStatus BatchNormGradExt2InferTiling::DoOpTilingForStage1()
{
    // UB内二分累加buffer按32KB的float数算，占512字节
    // dgamma和dbeta的UB间二分累加buffer各占3KB
    // mean rstd gamma各占128字节，共512字节
    // 总共7KB
    constexpr uint64_t extraUbSize = 7 * 1024 + 512;
    binaryAddBufSize = (aicoreParams_.ubSize - extraUbSize) / TWO / DOUBLE_BUFFER;

    // R0不可全载，核内切R0
    subTilingKey = BNG_RAR_RECOMPUTE_SPLIT_R0_INFER;
    if (Ops::Base::CeilDiv(static_cast<int64_t>(r0Dim * sizeof(float) * TWO), blockSize_) * blockSize_ <=
        static_cast<int64_t>(binaryAddBufSize)) {
        // R0可全载，核内切R1
        subTilingKey = BNG_RAR_RECOMPUTE_SPLIT_R1_INFER;
    }
    // R（= r1Dim*r0Dim）很短、A 比 R 长时（如 C=1035140, R=16），dgamma/dbeta 的旧路径是逐通道串行
    // （每通道一次 4B 搬入/搬出 + 累加器初始化/finalize），实测被延迟打死（MBU 2.3%）。这种形状统一
    // 交给 SplitR0 模板里的"lane = 通道"实现（kernel：BatchNormGradExt2InferDgammaDbetaSplitR0 的
    // useChannelLane_ 分支），按 64 通道一批处理。
    if (r1Dim * r0Dim <= static_cast<int64_t>(vlFp32_) && aDim >= r1Dim * r0Dim) {
        subTilingKey = BNG_RAR_RECOMPUTE_SPLIT_R0_INFER;
    }
    // A 很短而 R1 很长时（NCHW/NCDHW 小通道形状，如 L1_191 NCHW 128x2x8709x2：r1=128, a=2, r0=17418），
    // dgamma/dbeta 的旧路径按 A 切核，只有 aDim 个核有活（2/72），其余核空闲，2.2M 元素逐 lane 累加
    // ≈1ms。改为按 r1 切核：blockNum 提到满核数，每核处理 r1 的一段、全部通道，把每核每通道 4 个 dd
    // 字（s0hi/s0lo/s1hi/s1lo，workspace 已按该口径预留）落盘，SyncAll 后 0 号核补偿合并 + finalize
    // （kernel：同文件的 useRSplit_ 分支，条件必须与 kernel 侧完全一致）。
    rSplitMode = enableDgamma && enableDbeta && (aDim <= RS_MAX_A_DIM) &&
                 (aDim < static_cast<int64_t>(aicoreParams_.numBlocks)) && (r1Dim >= RS_MIN_R1) &&
                 (aDim * FOUR <= static_cast<int64_t>(aicoreParams_.numBlocks));
    if (rSplitMode) {
        subTilingKey = BNG_RAR_RECOMPUTE_SPLIT_R0_INFER;
    }

    onceProcUbNeed = INT64_MAX;
    int64_t minADimPerCore = std::max(1L, Ops::Base::CeilDiv(BNG_PER_CORE_PROCESS_MIN_UB_SIZE, onceProcUbNeed));
    int64_t blockNum = rSplitMode ? static_cast<int64_t>(aicoreParams_.numBlocks) : aicoreParams_.numBlocks;
    if (!rSplitMode && static_cast<int64_t>(minADimPerCore * aicoreParams_.numBlocks) > aDim) {
        // 每个核最少要处理 BNG_PER_CORE_PROCESS_MIN_UB_SIZE, minADimPerCore向上取整不可能是0
        blockNum = aDim / minADimPerCore;
        if (blockNum == 0) {
            blockNum = 1;
        }
    }
    OP_CHECK_IF(blockNum == 0, OP_LOGE_WITHOUT_REPORT(context_, "block num is 0, failed."), return ge::GRAPH_FAILED);

    int64_t tailBlockNum = aDim % blockNum;
    int64_t formerBlockDim = aDim / blockNum;
    int64_t tailBlockDim = formerBlockDim + 1;

    tilingData_.blockNum = blockNum;
    tilingData_.tailBlockNum = tailBlockNum;
    tilingData_.formerBlockDim = formerBlockDim;
    tilingData_.tailBlockDim = tailBlockDim;

    if (subTilingKey == BNG_RAR_RECOMPUTE_SPLIT_R1_INFER) {
        DoRecomputeTilingSplitR1();
    } else {
        DoRecomputeTilingSplitR0();
    }
    int64_t tailBinAddQuotient = ubRDimTailFactor == 0 ? 1 :
                                                         (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubRDimTailFactor)));
    int64_t generalBinAddQuotient = ubRDimFactor == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(ubRDimFactor)));

    DoBinaryAddTiling(tilingData_.generalBinAddTilingData, generalBinAddQuotient);
    DoBinaryAddTiling(tilingData_.tailBinAddTilingData, tailBinAddQuotient);

    tilingData_.enableDx = static_cast<int32_t>(enableDx);
    tilingData_.enableDgamma = static_cast<int32_t>(enableDgamma);
    tilingData_.enableDbeta = static_cast<int32_t>(enableDbeta);

    // ---- UB 账本：按 infer.h 中 SplitR0 / SplitR1 的真实 InitBuffer 重算并断言 ----
    // 上面的 binaryAddBufSize / factorMax 保留历史口径（决定 r0/r1 切分，不改），本账本只核算
    // dgamma/dbeta 阶段实际占用的 buffer（BUFFER_NUM = 2；ONE_BLK_SIZE = 32B；UB_ADD_BUF = 3KB；
    // compAccBuf_ = 4 * vlFp32 * 4B，是 kernel 新增后公式里一直没计的部分）。
    //   SplitR0: 2*dyBuf + 2*xBuf + 2*32(mean) + 2*32(var) + 3KB(dbetaOut,1 buffer) + 3KB(dgammaOut)
    //   SplitR1: 2*dyBuf + 2*xBuf + 2*32(mean) + 2*32(var) + 2*32(dbetaOut) + 2*32(dgammaOut)
    constexpr int64_t kInferBufferNum = 2;
    constexpr int64_t kInferOneBlkSize = 32;
    constexpr int64_t kInferAddOutBuf = 3 * 1024;
    const int64_t compAccUb = 4 * vlFp32_ * FLOAT32_BYTES;
    int64_t dbgUbNeed = 0;
    if (subTilingKey == BNG_RAR_RECOMPUTE_SPLIT_R1_INFER) {
        const int64_t rAlign = static_cast<int64_t>(ubRDimFactor);
        const int64_t dyBuf = Ops::Base::CeilAlign(rAlign * bytesPerDy_, TWO * blockSize_);
        dbgUbNeed = kInferBufferNum * (dyBuf + dyBuf) + kInferBufferNum * 4 * kInferOneBlkSize + compAccUb;
    } else {
        const int64_t tailFactorAlign = Ops::Base::CeilAlign(static_cast<int64_t>(ubRDimTailFactor) * bytesPerDy_,
                                                             blockSize_) /
                                        bytesPerDy_;
        const int64_t dyBuf = tailFactorAlign * TWO * bytesPerDy_;
        const int64_t xBuf = static_cast<int64_t>(ubRDimTailFactorAlign) * TWO * FLOAT32_BYTES;
        dbgUbNeed = kInferBufferNum * (dyBuf + xBuf + TWO * kInferOneBlkSize) + TWO * kInferAddOutBuf + compAccUb;
    }

    OP_LOGI(context_->GetNodeName(), "Infer dgamma/dbeta UB ledger: need %ldB, ub %luB, subTilingKey %lu.", dbgUbNeed,
            aicoreParams_.ubSize, subTilingKey);
    OP_CHECK_IF(dbgUbNeed > static_cast<int64_t>(aicoreParams_.ubSize),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "infer dgamma/dbeta UB footprint",
                                                      std::to_string(dbgUbNeed).c_str(),
                                                      "exceeds the unified buffer size of the infer template"),
                return ge::GRAPH_PARAM_INVALID);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradExt2InferTiling::DoOpTiling()
{
    DoOpTilingForStage0();
    DoOpTilingForStage1();
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradExt2InferTiling::GetTilingKey() const { return TILINGKEY_INFER_BASE + subTilingKey; }

ge::graphStatus BatchNormGradExt2InferTiling::PostTiling()
{
    context_->SetBlockDim(std::max(usedCoreNums_, tilingData_.blockNum));
    // rSplit 路径用 SyncAll 做跨核 partial 合并，必须 batch mode，否则 barrier 形同虚设
    // （结果是随运行漂移的脏值）。
    if (rSplitMode) {
        auto ret = context_->SetScheduleMode(1);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_, "SetScheduleMode(batch) failed for R1-split infer tiling."),
                    return ret);
    }
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto* tilingDataOut = context_->GetTilingData<std::decay_t<decltype(tilingData_)>>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataOut);
    *tilingDataOut = tilingData_;
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNormGradExt2, BatchNormGradExt2InferTiling, 91000);
} // namespace optiling
