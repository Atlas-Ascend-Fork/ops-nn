/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_v3_grad_nhwc_tiling.cpp
 * \brief MaxPoolV3Grad NHWC tiling for large-C, small-kernel and large-kernel paths.
 */

#include "error_util.h"
#include "max_pool_v3_grad_tiling.h"
#include "../../../pool_grad_common/op_host/arch35/pool_grad_tiling_split_helper.h"

namespace optiling {

// Shared scalar constants are defined in max_pool_v3_grad_tiling_base.h.
static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t BATCH_LIMIT = 256;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();
static constexpr int64_t UB_RESERVED_SIZE = 5120;
static constexpr int64_t CACHE_LINE_SIZE = 128;

void MaxPoolV3GradNHWCTilingImpl::InitializationVars(gert::TilingContext* context, int64_t ubSize, int64_t coreNum)
{
    baseData_.vRegSize = Ops::Base::GetVRegSize(context);
    baseData_.ubBlockSize = Ops::Base::GetUbBlockSize(context);
    baseData_.inputBytes = ge::GetSizeByDataType(inputData_->inputDtype);
    baseData_.indexBytes = INT32_SIZE;
    baseData_.availableUb = ubSize - UB_RESERVED_SIZE;
    baseData_.totalCoreNum = coreNum;
    baseData_.coreUsedForBestPerformance = baseData_.totalCoreNum;

    int64_t oneBlockNumT1 = baseData_.ubBlockSize / baseData_.inputBytes;
    int64_t oneBlockNumT2 = baseData_.ubBlockSize / baseData_.indexBytes;
    baseData_.maxDataNumInOneBlock = std::max(oneBlockNumT1, oneBlockNumT2);
    baseData_.proDataNumInOneBeat = baseData_.vRegSize / std::max(baseData_.inputBytes, baseData_.indexBytes);
    baseData_.moveDataNumCacheLine = CACHE_LINE_SIZE / baseData_.inputBytes;

    baseData_.isPad = 0;
    if (inputData_->padH != 0 || inputData_->padW != 0 || inputData_->padHBack != 0 || inputData_->padWBack != 0) {
        baseData_.isPad = 1;
    }

    baseData_.hProBatchSize = PoolGradTiling::CalcProBatchSize(inputData_->hKernel, inputData_->hStride);
    baseData_.wProBatchSize = PoolGradTiling::CalcProBatchSize(inputData_->wKernel, inputData_->wStride);
    baseData_.isOverlap = (baseData_.hProBatchSize != 1 || baseData_.wProBatchSize != 1) ? 1 : 0;
}

bool MaxPoolV3GradNHWCTilingImpl::IsMeetTargetCoreNum() const
{
    const PoolGradTiling::PoolGradNhwcDims dims{inputData_->nX, inputData_->cX,      inputData_->hX,
                                                inputData_->wX, inputData_->hStride, inputData_->wStride};
    return PoolGradTiling::IsMeetTargetCoreNumNhwc(splitData_, dims, baseData_.coreUsedForBestPerformance);
}

bool MaxPoolV3GradNHWCTilingImpl::TrySplitN()
{
    const PoolGradTiling::PoolGradNhwcDims dims{inputData_->nX, inputData_->cX,      inputData_->hX,
                                                inputData_->wX, inputData_->hStride, inputData_->wStride};
    return PoolGradTiling::TrySplitN(splitData_, dims, baseData_.coreUsedForBestPerformance, *this);
}

bool MaxPoolV3GradNHWCTilingImpl::TrySplitAlignH()
{
    const PoolGradTiling::PoolGradNhwcDims dims{inputData_->nX, inputData_->cX,      inputData_->hX,
                                                inputData_->wX, inputData_->hStride, inputData_->wStride};
    return PoolGradTiling::TrySplitAlignH(splitData_, dims, *this);
}

bool MaxPoolV3GradNHWCTilingImpl::TrySplitAlignW()
{
    splitData_.nOutputInner = 1;
    splitData_.hOutputInner = inputData_->hX;
    splitData_.cOutputInner = inputData_->cX;

    splitData_.wOutputInner = inputData_->wStride;
    int64_t halfInput = inputData_->wX / 2;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(halfInput, inputData_->wStride);
        int64_t bestSplit = 1;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData_.wOutputInner = mid * inputData_->wStride;
            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestSplit = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData_.wOutputInner = bestSplit * inputData_->wStride;
        return true;
    }
    return false;
}

bool MaxPoolV3GradNHWCTilingImpl::TrySplitAlignC()
{
    splitData_.nOutputInner = 1;
    splitData_.hOutputInner = inputData_->hStride;
    splitData_.wOutputInner = inputData_->wStride;

    int64_t tmpCAligned = (inputData_->cX < baseData_.moveDataNumCacheLine) ? inputData_->cX :
                                                                              baseData_.moveDataNumCacheLine;
    splitData_.cOutputInner = tmpCAligned;
    if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
        int64_t left = 1;
        int64_t right = Ops::Base::CeilDiv(inputData_->cX / 2, baseData_.moveDataNumCacheLine);
        // Keep the validated initial channel tile if the binary search finds no larger valid tile.
        // Otherwise a channel tile larger than cX could exceed the UB budget.
        int64_t bestCInner = tmpCAligned;
        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            splitData_.cOutputInner = mid * baseData_.moveDataNumCacheLine;
            if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
                bestCInner = splitData_.cOutputInner;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        splitData_.cOutputInner = bestCInner;
        if (splitData_.cOutputInner > baseData_.proDataNumInOneBeat) {
            splitData_.cOutputInner = baseData_.proDataNumInOneBeat;
        }
        return true;
    }
    return false;
}

void MaxPoolV3GradNHWCTilingImpl::DynamicAdjustmentHW()
{
    if (splitData_.hOutputInner != 1) {
        splitData_.hOutputOuter++;
        splitData_.hOutputInner = Ops::Base::CeilDiv(inputData_->hX, splitData_.hOutputOuter);
        return;
    }
    splitData_.wOutputOuter++;
    splitData_.wOutputInner = Ops::Base::CeilDiv(inputData_->wX, splitData_.wOutputOuter);
}

void MaxPoolV3GradNHWCTilingImpl::SplitUnalignHWC()
{
    splitData_.nOutputInner = 1;
    if (baseData_.isPad == 0 && baseData_.isOverlap == 0) {
        splitData_.hOutputInner = inputData_->hStride;
        splitData_.wOutputInner = inputData_->wStride;
        int64_t tmpCAligned = (inputData_->cX < baseData_.moveDataNumCacheLine) ? inputData_->cX :
                                                                                  baseData_.moveDataNumCacheLine;
        splitData_.cOutputInner = tmpCAligned;
    } else {
        splitData_.hOutputInner = inputData_->hX;
        splitData_.wOutputInner = inputData_->wX;
        splitData_.cOutputInner = inputData_->cX;
    }

    splitData_.wOutputOuter = Ops::Base::CeilDiv(inputData_->wX, splitData_.wOutputInner);
    splitData_.hOutputOuter = Ops::Base::CeilDiv(inputData_->hX, splitData_.hOutputInner);

    while (splitData_.hOutputInner != 1 || splitData_.wOutputInner > baseData_.proDataNumInOneBeat) {
        if (!IsMeetTargetCoreNum() || !IsMeetUBSize()) {
            DynamicAdjustmentHW();
        } else {
            break;
        }
    }

    // Validate the selected channel tile before shrinking W, which cannot increase the buffer size.
    // Search larger vector-aligned channel tiles only after the initial tile fits in UB.
    // If it does not fit, fall back to cache-line alignment.
    if (inputData_->cX <= baseData_.proDataNumInOneBeat) {
        splitData_.cOutputInner = inputData_->cX;
    } else {
        splitData_.cOutputInner = baseData_.proDataNumInOneBeat;
        if (IsMeetUBSize()) {
            int64_t left = 1;
            int64_t right = Ops::Base::CeilDiv(inputData_->cX / 2, baseData_.proDataNumInOneBeat);
            int64_t bestCInner = baseData_.proDataNumInOneBeat;
            while (left <= right) {
                int64_t mid = left + (right - left) / 2;
                splitData_.cOutputInner = mid * baseData_.proDataNumInOneBeat;
                if (IsMeetUBSize()) {
                    bestCInner = splitData_.cOutputInner;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            splitData_.cOutputInner = bestCInner;
        } else {
            splitData_.cOutputInner = std::min(inputData_->cX, baseData_.moveDataNumCacheLine);
            if (!IsMeetUBSize()) {
                // Restore vector alignment and let W contraction satisfy the UB limit.
                splitData_.cOutputInner = baseData_.proDataNumInOneBeat;
            }
        }
    }

    // Keep shrinking W until the buffers fit in UB or the minimum tile is reached.
    // Large overlapping kernels can exceed UB even when hInner is already one.
    while (!IsMeetUBSize() && splitData_.wOutputInner > 1) {
        DynamicAdjustmentHW();
    }
}

void MaxPoolV3GradNHWCTilingImpl::SearchBestTiling()
{
    splitData_.isCheckRange = 0;
    if (TrySplitN()) {
        return;
    }
    if (baseData_.isPad == 0 && baseData_.isOverlap == 0) {
        if (TrySplitAlignH()) {
            return;
        }
        if (TrySplitAlignW()) {
            return;
        }
        if (TrySplitAlignC()) {
            return;
        }
    }
    splitData_.isCheckRange = 1;
    SplitUnalignHWC();
}

void MaxPoolV3GradNHWCTilingImpl::DoUBTiling()
{
    SearchBestTiling();
    IsMeetUBSize();

    splitData_.wOutputOuter = Ops::Base::CeilDiv(inputData_->wX, splitData_.wOutputInner);
    int64_t wTail = inputData_->wX % splitData_.wOutputInner;
    splitData_.wOutputTail = (wTail == 0) ? splitData_.wOutputInner : wTail;

    splitData_.hOutputOuter = Ops::Base::CeilDiv(inputData_->hX, splitData_.hOutputInner);
    int64_t hTail = inputData_->hX % splitData_.hOutputInner;
    splitData_.hOutputTail = (hTail == 0) ? splitData_.hOutputInner : hTail;

    splitData_.nOutputOuter = Ops::Base::CeilDiv(inputData_->nX, splitData_.nOutputInner);
    int64_t nTail = inputData_->nX % splitData_.nOutputInner;
    splitData_.nOutputTail = (nTail == 0) ? splitData_.nOutputInner : nTail;

    splitData_.cOutputOuter = Ops::Base::CeilDiv(inputData_->cX, splitData_.cOutputInner);
    int64_t cTail = inputData_->cX % splitData_.cOutputInner;
    splitData_.cOutputTail = (cTail == 0) ? splitData_.cOutputInner : cTail;
}

void MaxPoolV3GradNHWCTilingImpl::DoBlockTiling()
{
    // Recalculate outer and tail sizes after changing inner dimensions.
    splitData_.wOutputOuter = Ops::Base::CeilDiv(inputData_->wX, splitData_.wOutputInner);
    int64_t wTail = inputData_->wX % splitData_.wOutputInner;
    splitData_.wOutputTail = (wTail == 0) ? splitData_.wOutputInner : wTail;

    splitData_.hOutputOuter = Ops::Base::CeilDiv(inputData_->hX, splitData_.hOutputInner);
    int64_t hTail = inputData_->hX % splitData_.hOutputInner;
    splitData_.hOutputTail = (hTail == 0) ? splitData_.hOutputInner : hTail;

    splitData_.nOutputOuter = Ops::Base::CeilDiv(inputData_->nX, splitData_.nOutputInner);
    int64_t nTail = inputData_->nX % splitData_.nOutputInner;
    splitData_.nOutputTail = (nTail == 0) ? splitData_.nOutputInner : nTail;

    splitData_.cOutputOuter = Ops::Base::CeilDiv(inputData_->cX, splitData_.cOutputInner);
    int64_t cTail = inputData_->cX % splitData_.cOutputInner;
    splitData_.cOutputTail = (cTail == 0) ? splitData_.cOutputInner : cTail;

    PoolGradTiling::DoBlockTilingNhwc(splitData_, baseData_.totalCoreNum);
}

void MaxPoolV3GradNHWCTilingImpl::SearchOuterSingle(int64_t& inner, int64_t step)
{
    PoolGradTiling::SearchOuterSingle(
        inner, step, [this]() { DoBlockTiling(); }, [this]() { return splitData_.normalCoreProcessNum; });
}

void MaxPoolV3GradNHWCTilingImpl::SearchOuterSingleAligned(int64_t& inner, int64_t stride, int64_t kernelSize)
{
    if (kernelSize == 1) {
        // Single-element windows permit arbitrary tile boundaries.
        SearchOuterSingle(inner, 1);
        return;
    }
    if (inner <= stride) {
        return;
    }
    if (inner % stride != 0) {
        // Align an unsplit axis to the stride grid first.
        // Restore the original tile if alignment increases the normal core workload.
        int64_t originInner = inner;
        int64_t lastNormCoreProcessNum = splitData_.normalCoreProcessNum;
        inner = inner / stride * stride;
        DoBlockTiling();
        if (splitData_.normalCoreProcessNum > lastNormCoreProcessNum) {
            inner = originInner;
            DoBlockTiling();
            return;
        }
    }
    SearchOuterSingle(inner, stride);
}

void MaxPoolV3GradNHWCTilingImpl::AdjustInnerSplitForMultiCore()
{
    DoBlockTiling();
    if (splitData_.usedCoreNum >= baseData_.totalCoreNum) {
        return;
    }

    // Batch planes are independent and permit arbitrary inner sizes.
    SearchOuterSingle(splitData_.nOutputInner, 1);
    if (splitData_.usedCoreNum >= baseData_.totalCoreNum) {
        DoBufferCalculate();
        return;
    }

    if (splitData_.isCheckRange == 1) {
        // The unaligned path filters out-of-tile lanes with IS_CHECK_RANGE=1.
        SearchOuterSingle(splitData_.hOutputInner, 1);
        if (splitData_.usedCoreNum >= baseData_.totalCoreNum) {
            DoBufferCalculate();
            return;
        }
        SearchOuterSingle(splitData_.wOutputInner, 1);
    } else if (baseData_.isPad == 0 && baseData_.isOverlap == 0) {
        // Without range filtering, H/W tile boundaries must remain on the stride grid.
        SearchOuterSingleAligned(splitData_.hOutputInner, inputData_->hStride, inputData_->hKernel);
        if (splitData_.usedCoreNum >= baseData_.totalCoreNum) {
            DoBufferCalculate();
            return;
        }
        SearchOuterSingleAligned(splitData_.wOutputInner, inputData_->wStride, inputData_->wKernel);
    }
    // With padding or overlap and no range filtering, keep H/W unsplit and adjust only N.
    DoBufferCalculate();
}

void MaxPoolV3GradNHWCTilingImpl::DoBufferCalculate()
{
    int64_t hInputInner = Ops::Base::CeilDiv(splitData_.hOutputInner + inputData_->hKernel - KERNEL_OFFSET,
                                             inputData_->hStride);
    int64_t wInputInner = Ops::Base::CeilDiv(splitData_.wOutputInner + inputData_->wKernel - KERNEL_OFFSET,
                                             inputData_->wStride);

    int64_t cAligned = Ops::Base::CeilAlign(splitData_.cOutputInner, baseData_.maxDataNumInOneBlock);

    splitData_.gradBufferSize = splitData_.nOutputInner * hInputInner * wInputInner * cAligned * baseData_.inputBytes;
    splitData_.outputBufferSize = splitData_.nOutputInner * splitData_.hOutputInner * splitData_.wOutputInner *
                                  cAligned * std::max(FLOAT32_SIZE, baseData_.inputBytes);
    splitData_.argmaxBufferSize = splitData_.nOutputInner * splitData_.cOutputInner * hInputInner * wInputInner *
                                  baseData_.indexBytes;

    if (splitData_.isBigKernel == 1) {
        // Large kernels stream window data and retain the running maximum in the merge buffer.
        const int64_t kv = inputData_->hKernel * inputData_->wKernel;
        const int64_t mergeSize = cAligned * baseData_.inputBytes;
        const int64_t fullWindowBytes = kv * cAligned * baseData_.inputBytes;
        const int64_t minInput = cAligned * baseData_.inputBytes;
        const int64_t forwardInputAvailableBytes = std::max<int64_t>(
            0, (baseData_.availableUb - splitData_.argmaxBufferSize - mergeSize) / DOUBLE_BUFFER);
        const int64_t maxLoadBytes = std::max<int64_t>(minInput, forwardInputAvailableBytes);
        splitData_.inputBufferSize = std::min(fullWindowBytes, maxLoadBytes);
        const int64_t forwardSize = splitData_.inputBufferSize * DOUBLE_BUFFER + mergeSize;
        const int64_t backwardSize = splitData_.gradBufferSize + splitData_.outputBufferSize;
        const int64_t sharedSize = (forwardSize > backwardSize) ? forwardSize : backwardSize;
        splitData_.totalBufferSize = splitData_.argmaxBufferSize + sharedSize;
        return;
    }

    // Small kernels, including the large-C path, load the complete tile and overlapping window halo.
    int64_t hInExpect = splitData_.hOutputInner + (inputData_->hKernel - KERNEL_OFFSET) * DOUBLE_BUFFER;
    int64_t wInExpect = splitData_.wOutputInner + (inputData_->wKernel - KERNEL_OFFSET) * DOUBLE_BUFFER;
    splitData_.inputBufferSize = splitData_.nOutputInner * hInExpect * wInExpect * cAligned * baseData_.inputBytes;

    int64_t calcSize = baseData_.isPad ? splitData_.inputBufferSize : 0;
    int64_t forwardSize = splitData_.inputBufferSize * DOUBLE_BUFFER + calcSize;
    int64_t backwardSize = splitData_.gradBufferSize * DOUBLE_BUFFER + splitData_.outputBufferSize * DOUBLE_BUFFER;
    int64_t sharedSize = (forwardSize > backwardSize) ? forwardSize : backwardSize;
    splitData_.totalBufferSize = splitData_.argmaxBufferSize + sharedSize;
}

bool MaxPoolV3GradNHWCTilingImpl::IsMeetUBSize()
{
    DoBufferCalculate();
    if (baseData_.inputBytes <= FLOAT16_SIZE) {
        return splitData_.totalBufferSize <= baseData_.availableUb &&
               splitData_.inputBufferSize <= MAX_INPUT_ELEMENTS * baseData_.inputBytes &&
               splitData_.gradBufferSize <= MAX_INPUT_ELEMENTS * baseData_.inputBytes;
    }
    return splitData_.totalBufferSize <= baseData_.availableUb;
}

ge::graphStatus MaxPoolV3GradNHWCTilingImpl::SetTilingData(gert::TilingContext* context)
{
    MaxPoolV3GradNHWCTilingData* tilingData = context->GetTilingData<MaxPoolV3GradNHWCTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    tilingData->base.hArgmax = inputData_->hGrad;
    tilingData->base.wArgmax = inputData_->wGrad;
    tilingData->base.hOutput = inputData_->hX;
    tilingData->base.wOutput = inputData_->wX;
    tilingData->base.hKernel = inputData_->hKernel;
    tilingData->base.wKernel = inputData_->wKernel;
    tilingData->base.hStride = inputData_->hStride;
    tilingData->base.wStride = inputData_->wStride;
    tilingData->base.padH = inputData_->padH;
    tilingData->base.padW = inputData_->padW;
    tilingData->base.padHBack = inputData_->padHBack;
    tilingData->base.padWBack = inputData_->padWBack;

    tilingData->base.highAxisInner = splitData_.nOutputInner;
    tilingData->base.highAxisTail = splitData_.nOutputTail;
    tilingData->base.highAxisOuter = splitData_.nOutputOuter;
    tilingData->base.hOutputInner = splitData_.hOutputInner;
    tilingData->base.hOutputTail = splitData_.hOutputTail;
    tilingData->base.hOutputOuter = splitData_.hOutputOuter;
    tilingData->base.wOutputInner = splitData_.wOutputInner;
    tilingData->base.wOutputTail = splitData_.wOutputTail;
    tilingData->base.wOutputOuter = splitData_.wOutputOuter;
    tilingData->base.normalCoreProcessNum = splitData_.normalCoreProcessNum;
    tilingData->base.tailCoreProcessNum = splitData_.tailCoreProcessNum;
    tilingData->base.usedCoreNum = splitData_.usedCoreNum;
    tilingData->base.inputBufferSize = splitData_.inputBufferSize;
    tilingData->base.outputBufferSize = splitData_.outputBufferSize;
    tilingData->base.gradBufferSize = splitData_.gradBufferSize;
    tilingData->base.argmaxBufferSize = splitData_.argmaxBufferSize;
    tilingData->base.hProBatchSize = baseData_.hProBatchSize;
    tilingData->base.wProBatchSize = baseData_.wProBatchSize;
    tilingData->base.isBigKernel = splitData_.isBigKernel;

    tilingData->cDim = inputData_->cX;
    tilingData->cOutputInner = splitData_.cOutputInner;
    tilingData->cOutputTail = splitData_.cOutputTail;
    tilingData->cOutputOuter = splitData_.cOutputOuter;
    tilingData->isBigKernel = splitData_.isBigKernel;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3GradNHWCTilingImpl::DoOpTiling(gert::TilingContext* context)
{
    DoUBTiling();
    AdjustInnerSplitForMultiCore();
    DoBlockTiling(); // Keep final axis tails and core workloads consistent.
    return SetTilingData(context);
}

ge::graphStatus MaxPoolV3GradNHWCTilingImpl::PostTiling(gert::TilingContext* context, uint64_t key)
{
    auto ret = context->SetTilingKey(key);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Failed to set tiling key: %lu.", key);
        return ret;
    }
    ret = context->SetBlockDim(static_cast<uint32_t>(splitData_.usedCoreNum));
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Failed to set block dimension: %ld.", splitData_.usedCoreNum);
    }
    return ret;
}

ge::graphStatus MaxPoolV3GradNHWCSmallKernelTiling::GetShapeAttrsInfo()
{
    auto ret = MaxPoolV3GradTilingBase::GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    nhwcInputData_.nX = inputData.nX;
    nhwcInputData_.cX = inputData.cX;
    nhwcInputData_.hX = inputData.hX;
    nhwcInputData_.wX = inputData.wX;
    nhwcInputData_.hGrad = inputData.hGrad;
    nhwcInputData_.wGrad = inputData.wGrad;
    nhwcInputData_.hKernel = inputData.hKernel;
    nhwcInputData_.wKernel = inputData.wKernel;
    nhwcInputData_.hStride = inputData.hStride;
    nhwcInputData_.wStride = inputData.wStride;
    nhwcInputData_.padH = inputData.hPad;
    nhwcInputData_.padW = inputData.wPad;
    nhwcInputData_.padHBack = padBottom_;
    nhwcInputData_.padWBack = padRight_;
    nhwcInputData_.inputDataCount = totalInputElements_;
    nhwcInputData_.inputDtype = inputData.inputDtype;
    nhwcInputData_.inputFormat = inputData.inputFormat;
    return ge::GRAPH_SUCCESS;
}

bool MaxPoolV3GradNHWCSmallKernelTiling::IsCapable()
{
    if (nhwcInputData_.inputFormat != ge::Format::FORMAT_NHWC) {
        OP_LOGW(context_->GetNodeName(), "inputFormat invalid, expected NHWC");
        return false;
    }

    if (nhwcInputData_.inputDtype != ge::DT_FLOAT && nhwcInputData_.inputDtype != ge::DT_FLOAT16 &&
        nhwcInputData_.inputDtype != ge::DT_BF16 && nhwcInputData_.inputDtype != ge::DT_INT8 &&
        nhwcInputData_.inputDtype != ge::DT_UINT8 && nhwcInputData_.inputDtype != ge::DT_INT16 &&
        nhwcInputData_.inputDtype != ge::DT_UINT16 && nhwcInputData_.inputDtype != ge::DT_INT32 &&
        nhwcInputData_.inputDtype != ge::DT_INT64) {
        OP_LOGI(context_->GetNodeName(), "dtype %d not supported by nhwc fast path",
                static_cast<int32_t>(nhwcInputData_.inputDtype));
        return false;
    }

    // Empty dimensions are handled by the SIMT fallback.
    if (nhwcInputData_.nX <= 0 || nhwcInputData_.cX <= 0 || nhwcInputData_.hX <= 0 || nhwcInputData_.wX <= 0 ||
        nhwcInputData_.hGrad <= 0 || nhwcInputData_.wGrad <= 0) {
        OP_LOGI(context_->GetNodeName(), "empty dimension, fallback to simt");
        return false;
    }

    if (nhwcInputData_.inputDataCount > MAX_INT32) {
        OP_LOGI(context_->GetNodeName(), "inputDataCount:%ld exceeds int32, fast div not supported",
                nhwcInputData_.inputDataCount);
        return false;
    }

    int64_t kv = nhwcInputData_.hKernel * nhwcInputData_.wKernel;

    nhwcBase_.InitializationVars(context_, ubSize_, coreNum_);

    int64_t batchSize = nhwcBase_.GetBaseData().hProBatchSize * nhwcBase_.GetBaseData().wProBatchSize;
    if (batchSize >= BATCH_LIMIT) {
        OP_LOGI(context_->GetNodeName(), "batch size too large: %ld", batchSize);
        return false;
    }

    auto& split = nhwcBase_.GetSplitData();
    split.isBigKernel = (kv >= KSIZE_THRESHOLD) ? 1 : 0;

    split.nOutputInner = 1;
    split.hOutputInner = 1;
    split.wOutputInner = (split.isBigKernel == 1) ?
                             1 :
                             std::min(nhwcInputData_.wX, nhwcBase_.GetBaseData().proDataNumInOneBeat);
    split.cOutputInner = std::min(nhwcInputData_.cX, nhwcBase_.GetBaseData().proDataNumInOneBeat);

    nhwcBase_.DoBufferCalculate();
    return split.totalBufferSize <= nhwcBase_.GetBaseData().availableUb;
}

ge::graphStatus MaxPoolV3GradNHWCSmallKernelTiling::DoOpTiling()
{
    auto ret = nhwcBase_.DoOpTiling(context_);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    isCheckRange_ = nhwcBase_.GetSplitData().isCheckRange;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3GradNHWCSmallKernelTiling::GetWorkspaceSize()
{
    size_t* workspaceSizes = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaceSizes);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3GradNHWCSmallKernelTiling::PostTiling()
{
    return nhwcBase_.PostTiling(context_, GetTilingKey());
}

uint64_t MaxPoolV3GradNHWCSmallKernelTiling::GetTilingKey() const
{
    uint32_t kernelMode = TPL_NHWC_KERNEL;
    uint32_t format = TPL_NHWC_FORMAT;
    uint32_t indicesDtype = TPL_INT32;
    uint32_t isCheckRange = static_cast<uint32_t>(isCheckRange_);
    return GET_TPL_TILING_KEY(kernelMode, format, indicesDtype, isCheckRange);
}

REGISTER_OPS_TILING_TEMPLATE(MaxPoolV3Grad, MaxPoolV3GradNHWCSmallKernelTiling, 2);

} // namespace optiling
