/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_tiling_infer_channel_last.cpp
 * \brief
 */

#include "batch_norm_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "batch_norm_grad_tiling_infer_base.h"

using namespace ge;

namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "BatchNormGrad", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

static constexpr int ULONG_BIT_LEN = 64;
constexpr int64_t DEFAULT_BIN_ADD_R_FACTOR = 128;
constexpr int64_t FP32_BLOCK_ALIGN_NUM = 8;
constexpr int64_t FP16_BLOCK_ALIGN_NUM = 16;
constexpr int64_t CONST_ONE = 1;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t CONST_TEN = 10;

class BatchNormGradInferChannelLastTiling : public BatchNormGradInferBase {
public:
    explicit BatchNormGradInferChannelLastTiling(gert::TilingContext* context) : BatchNormGradInferBase(context) {}
    ~BatchNormGradInferChannelLastTiling() override = default;
    int64_t GetCacheID(const int64_t idx);

protected:
    bool IsCapable() override
    {
        OP_CHECK_IF(
            r0Dim != 1,
            OP_LOGD(context_,
                    "BatchNormGradInferChannelLastTiling BA template is not capable, fused shape: (%ld, %ld, %ld)",
                    r1Dim, aDim, r0Dim),
            return false);

        CalcBasicInfo();

        OP_LOGD(context_, "BatchNormGradInferChannelLastTiling BA template is capable, fused shape: (%ld, %ld, %ld)",
                r1Dim, aDim, r0Dim);
        return true;
    }

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus DoOpTilingForStage0();
    ge::graphStatus DoOpTilingForStage1();

    BatchNormGradInferChannelLastTilingData tilingData_;
};

ge::graphStatus BatchNormGradInferChannelLastTiling::DoOpTiling()
{
    DoOpTilingForStage0();
    DoOpTilingForStage1();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferChannelLastTiling::DoOpTilingForStage0()
{
    // 切分A、B基本块， （B,A） -- >(Bouter, Aouter, Binner*Ainner*aTileBase_)
    int64_t aInner = 1;
    int64_t ubBufferSize = (aicoreParams_.ubSize / DOUBLE_BUFFER -
                            (bytesPerWeight_ + bytesPerRunningVar_) * aInner * aTileBase_) /
                           bytesPerDy_ / INPUT_OUTPUT_NUM;

    // 先按照B切分，再切A
    int64_t bFactorMax = ubBufferSize / aTileBase_;
    int64_t bInner = r1Dim <= bFactorMax ? r1Dim : bFactorMax;
    int64_t bOuter = Ops::Base::CeilDiv(r1Dim, bInner);
    int64_t bTail = r1Dim % bInner;
    int64_t tileBlockBTail = bTail == 0 ? bInner : bTail;

    int64_t aFactorMax = aicoreParams_.ubSize / DOUBLE_BUFFER / aTileBase_ /
                         (bInner * INPUT_OUTPUT_NUM * bytesPerDy_ + bytesPerWeight_ + bytesPerRunningVar_);
    int64_t aInnerMax = aDim / aTileBase_;
    aInner = aInnerMax <= aFactorMax ? aInnerMax : aFactorMax;

    int64_t tileBlockALen = aInner == 0 ? aTileBase_ : aInner * aTileBase_;
    int64_t aOuter = Ops::Base::CeilDiv(aDim, tileBlockALen);
    int64_t aTail = aDim % tileBlockALen;
    int64_t tileBlockATail = aTail == 0 ? tileBlockALen : aTail;
    int64_t tileBlockAPaddingNum = tileBlockALen - tileBlockATail;

    // 切核 （Bouter, Binner, Aouter, Ainner*aTileBase_） -- > (Bouter*Aouter, Binner, Ainner*aTileBase_)
    int64_t totalTiles = aOuter * bOuter;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.numBlocks));
    usedCoreNums_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    tilingData_.dxTilingData.set_totalTiles(totalTiles);
    tilingData_.dxTilingData.set_tilesPerCore(tilesPerCore);
    tilingData_.dxTilingData.set_usedCoreNums(usedCoreNums_);
    tilingData_.dxTilingData.set_aDim(aDim);
    tilingData_.dxTilingData.set_aOuter(aOuter);
    tilingData_.dxTilingData.set_bOuter(bOuter);
    tilingData_.dxTilingData.set_tileBlockALen(tileBlockALen);
    tilingData_.dxTilingData.set_tileBlockATail(tileBlockATail);
    tilingData_.dxTilingData.set_tileBlockAPaddingNum(tileBlockAPaddingNum);
    tilingData_.dxTilingData.set_tileBlockBLen(bInner);
    tilingData_.dxTilingData.set_tileBlockBTail(tileBlockBTail);
    tilingData_.dxTilingData.set_epsilon(epsilon_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferChannelLastTiling::DoOpTilingForStage1()
{
    // 二分
    int64_t binAddRFactor = DEFAULT_BIN_ADD_R_FACTOR;
    int64_t binAddRLoop = r1Dim / binAddRFactor;
    int64_t binAddRTotalLoop = Ops::Base::CeilDiv(r1Dim, binAddRFactor);
    int64_t binAddRTail = r1Dim - (binAddRTotalLoop - 1) * binAddRFactor;
    // 计算最接近binAddRTotalLoop_的2^k
    int64_t binAddBasicBlockLoop = binAddRTotalLoop > 1 ?
                                       (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(binAddRTotalLoop - 1))) :
                                       0;
    int64_t mainFoldCount = binAddRLoop - binAddBasicBlockLoop;
    int64_t binAddCacheBufferCount = 1;
    int64_t binAddResultCacheID = 0;
    if (binAddBasicBlockLoop != 0) {
        binAddCacheBufferCount = ULONG_BIT_LEN - __builtin_clzl(binAddBasicBlockLoop);
        binAddResultCacheID = GetCacheID(binAddBasicBlockLoop - 1);
    }

    OP_LOGI(context_->GetNodeName(),
            "Binary add rFactor: %ld, rLoop: %ld, binAddRTotalLoop: %ld, rTail: %ld, basicBlockLoop: %ld, "
            "mainFoldCount:%ld,cacheBufferCount: "
            "%ld, resultCacheId:%ld ",
            binAddRFactor, binAddRLoop, binAddRTotalLoop, binAddRTail, binAddBasicBlockLoop, mainFoldCount,
            binAddCacheBufferCount, binAddResultCacheID);

    tilingData_.set_binAddRFactorStg1(binAddRFactor);
    tilingData_.set_binAddRLoopStg1(binAddRLoop);
    tilingData_.set_binAddRTotalLoopStg1(binAddRTotalLoop);
    tilingData_.set_binAddRTailStg1(binAddRTail);
    tilingData_.set_binAddBasicBlockLoopStg1(binAddBasicBlockLoop);
    tilingData_.set_binAddMainFoldCountStg1(mainFoldCount);
    tilingData_.set_binAddCacheBufferCountStg1(binAddCacheBufferCount);
    tilingData_.set_binAddResultCacheIDStg1(binAddResultCacheID);

    // 分核
    int64_t aBase = dyDtype_ == ge::DT_FLOAT ? FP32_BLOCK_ALIGN_NUM : FP16_BLOCK_ALIGN_NUM;
    int64_t factorMax = 0;
    if (dyDtype_ == ge::DT_FLOAT) {
        factorMax = aicoreParams_.ubSize / aBase / FLOAT32_BYTES /
                    (binAddRFactor * CONST_FOUR + CONST_TEN + binAddCacheBufferCount * CONST_TWO);
    } else {
        factorMax = aicoreParams_.ubSize / aBase /
                    (binAddRFactor * (CONST_FOUR * FLOAT16_BYTES + FLOAT32_BYTES * CONST_TWO) +
                     FLOAT32_BYTES * CONST_TEN + binAddCacheBufferCount * FLOAT32_BYTES * CONST_TWO);
    }

    OP_CHECK_IF(factorMax <= 0,
                ReportTilingStageError(context_->GetNodeName(),
                                       "DoOpTilingForStage0: the RA recompute template cannot fit the merged shape (" +
                                           std::to_string(r1Dim) + ", " + std::to_string(aDim) + ") into UB size " +
                                           std::to_string(aicoreParams_.ubSize) + "B, factorMax=" +
                                           std::to_string(factorMax) + "; tiling strategy failed for a valid input"),
                return ge::GRAPH_PARAM_INVALID);

    // 尽量占多核
    int64_t aFactorMax = Ops::Base::CeilDiv(aDim, aBase);
    int64_t aFactorInnerMax = Ops::Base::CeilDiv(aFactorMax, static_cast<int64_t>(aicoreParams_.numBlocks));
    int64_t aFactorInner = aFactorInnerMax < factorMax ? aFactorInnerMax : factorMax;
    int64_t aInner = aFactorInner * aBase;

    int64_t aOuter = Ops::Base::CeilDiv(aDim, aInner);
    int64_t aTail = aDim - aInner * (aOuter - 1);

    // 不切R轴
    int64_t aOuterPerCore = Ops::Base::CeilDiv(aOuter, static_cast<int64_t>(aicoreParams_.numBlocks));
    int64_t usedCoreNums = Ops::Base::CeilDiv(aOuter, aOuterPerCore);

    tilingData_.set_aDimStg1(aDim);
    tilingData_.set_aOuterStg1(aOuter);
    tilingData_.set_aInnerStg1(aInner);
    tilingData_.set_aTailStg1(aTail);
    tilingData_.set_aOuterPerCoreStg1(aOuterPerCore);
    tilingData_.set_usedCoreNumsStg1(usedCoreNums);

    tilingData_.set_enableDx(static_cast<int32_t>(enableDx));
    tilingData_.set_enableDgamma(static_cast<int32_t>(enableDgamma));
    tilingData_.set_enableDbeta(static_cast<int32_t>(enableDbeta));

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradInferChannelLastTiling::GetTilingKey() const { return TILINGKEY_INFER_CHANNEL_LAST_BASE; }

int64_t BatchNormGradInferChannelLastTiling::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

ge::graphStatus BatchNormGradInferChannelLastTiling::PostTiling()
{
    context_->SetBlockDim(static_cast<uint32_t>(
        std::max(tilingData_.get_usedCoreNumsStg1(), tilingData_.dxTilingData.get_usedCoreNums())));
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr,
                ReportTilingStageError(context_->GetNodeName(),
                                       "PostTiling: GetWorkspaceSizes(1) returned null; required output-slot count is "
                                       "1. Check the TilingContext construction, not the user device allocation"),
                return ge::GRAPH_FAILED);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNormGrad, BatchNormGradInferChannelLastTiling, 90000);
} // namespace optiling
