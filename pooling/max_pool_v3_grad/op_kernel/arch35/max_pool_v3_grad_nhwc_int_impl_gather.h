/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL_V3_GRAD_NHWC_INT_IMPL_GATHER_H_
#define MAX_POOL_V3_GRAD_NHWC_INT_IMPL_GATHER_H_

#include "max_pool_v3_grad_nhwc_int_small_kernel.h"
#include "pool_utils/arch35/index/max_pool_with_argmax_index.h"

namespace MaxPoolV3GradNHWCIntNameSpace {

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::Init(
    GM_ADDR origX, GM_ADDR origY, GM_ADDR grad, GM_ADDR y, const MaxPoolV3GradNHWCTilingData& tilingData, TPipe& pipe)
{
    (void)origY;
    Base::ParseTilingData(tilingData);
    vToMte2Event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    mte3ToMte2Event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    Base::blockIdx_ = GetBlockIdx();
    if (Base::blockIdx_ >= tilingData_.base.usedCoreNum) {
        return;
    }
    Base::curCoreProcessNum_ = (Base::blockIdx_ + 1 == tilingData_.base.usedCoreNum) ?
                                   tilingData_.base.tailCoreProcessNum :
                                   tilingData_.base.normalCoreProcessNum;

    xGm_.SetGlobalBuffer((__gm__ T*)origX);
    Base::gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    Base::yGm_.SetGlobalBuffer((__gm__ T*)y);

    // Gather needs both a value lane and a 32-bit argmax lane per channel.
    // In particular, int8 channels must not exceed the index vector capacity.
    isBigC_ = cDim_ > static_cast<int64_t>(INTEGER_LANES<T> / 2);

    pipe.InitBuffer(Base::argmaxBuff_, tilingData_.base.argmaxBufferSize);
    pipe.InitBuffer(Base::helpBuf_, HELP_BUFFER);

    int64_t calcSize = Base::isPad_ ? tilingData_.base.inputBufferSize : 0;
    int64_t forwardSize = tilingData_.base.inputBufferSize * BUFFER_NUM + calcSize;
    int64_t backwardSize = tilingData_.base.gradBufferSize * BUFFER_NUM +
                           tilingData_.base.outputBufferSize * BUFFER_NUM;
    totalStageBufferSize_ = (forwardSize > backwardSize) ? forwardSize : backwardSize;

    pipe.InitBufPool(forwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_));
    pipe.InitBufPool(backwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_), forwardBufPool_);
    forwardBufPool_.InitBuffer(inputQue_, BUFFER_NUM, static_cast<uint32_t>(tilingData_.base.inputBufferSize));
    if (Base::isPad_) {
        forwardBufPool_.InitBuffer(inputCalcBuff_, static_cast<uint32_t>(tilingData_.base.inputBufferSize));
    }
    backwardBufPool_.InitBuffer(Base::gradQue_, BUFFER_NUM, static_cast<uint32_t>(tilingData_.base.gradBufferSize));
    backwardBufPool_.InitBuffer(Base::outputQue_, BUFFER_NUM, static_cast<uint32_t>(tilingData_.base.outputBufferSize));
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::Process()
{
    if (Base::blockIdx_ >= tilingData_.base.usedCoreNum) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < Base::curCoreProcessNum_; ++loopNum) {
        Base::ScalarCompute(loopNum);

        if (hArgmaxActual_ == 0 || wArgmaxActual_ == 0) {
            Base::ProcessNoArgmaxBlock();
            Base::Mte3Drain();
            continue;
        }

        ForwardScalarCompute();
        ForwardCopyIn();
        Forward();

        Base::VDrainToMte2();

        Base::CopyInGrad();
        Base::Backward();
        Base::CopyOut();
        Base::Mte3Drain();
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ForwardComputeTile()
{
    __ubuf__ T* xAddr = xForwardAddr_;
    __ubuf__ INDEX_T* argmaxAddr = (__ubuf__ INDEX_T*)argmaxBuff_.template Get<INDEX_T>().GetPhyAddr();

    int64_t indexVectorLanes = V_REG_SIZE / (sizeof(T) > sizeof(INDEX_T) ? sizeof(T) : sizeof(INDEX_T));
    int64_t channelCount = cOutputActual_;
    int64_t wcElems = wArgmaxActual_ * channelCount;
    int64_t hwcElems = hArgmaxActual_ * wcElems;

    if (isBigC_) {
        ChunkCGather(xAddr, argmaxAddr);
    } else if (hwcElems <= indexVectorLanes) {
        MultiNcGather(xAddr, argmaxAddr);
    } else if (wcElems <= indexVectorLanes) {
        MultiRowGather(xAddr, argmaxAddr);
    } else {
        SingleRowGather(xAddr, argmaxAddr);
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessWNHWCKernel(
    __ubuf__ T* xAddr, uint32_t colStride, uint32_t rowStride, Reg::RegTensor<int32_t>& indexReg, uint16_t hKernel,
    uint16_t wKernel, uint16_t repeatElem, Reg::RegTensor<INDEX_T>& argmaxHStart, Reg::RegTensor<INDEX_T>& argmaxWStart,
    Reg::RegTensor<INDEX_T>& argmaxHRes, Reg::RegTensor<INDEX_T>& argmaxWRes)
{
    Reg::RegTensor<int32_t> indexWithOffset;
    Reg::RegTensor<T> calcReg;
    Reg::RegTensor<T> maxReg;
    Reg::RegTensor<INDEX_T> candReg;

    Reg::MaskReg allMaskU32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();
    uint32_t gatherCount = static_cast<uint32_t>(repeatElem);
    Reg::MaskReg gatherMask = Reg::UpdateMask<T>(gatherCount);
    Reg::MaskReg gtMask;

    DuplicateLowestValueVF<T>(maxReg);

    for (uint16_t h = 0; h < hKernel; h++) {
        for (uint16_t w = 0; w < wKernel; w++) {
            int32_t relIndex = h * rowStride + w * colStride;

            Reg::Adds(indexWithOffset, indexReg, relIndex, allMaskU32);

            if constexpr (sizeof(T) >= sizeof(uint32_t)) {
                Reg::DataCopyGather(calcReg, xAddr, (Reg::RegTensor<uint32_t>&)indexWithOffset, gatherMask);
            } else {
                Reg::RegTensor<uint16_t> indexConvert;
                Reg::Pack(indexConvert, indexWithOffset);
                if constexpr (sizeof(T) == 1) {
                    Reg::MaskReg gatherMaskB16;
                    Reg::UnPack(gatherMaskB16, gatherMask);
                    Reg::DataCopyGather((Reg::RegTensor<uint16_t>&)calcReg, xAddr, indexConvert, gatherMaskB16);
                    Reg::Pack((Reg::RegTensor<uint8_t>&)calcReg, (Reg::RegTensor<uint16_t>&)calcReg);
                } else {
                    Reg::DataCopyGather(calcReg, xAddr, indexConvert, gatherMask);
                }
            }

            Reg::Compare<T, CMPMODE::GT>(gtMask, calcReg, maxReg, gatherMask);

            if constexpr (sizeof(INDEX_T) / sizeof(T) == 1) {
                Reg::Adds(candReg, argmaxHStart, static_cast<INDEX_T>(h), allMaskIdx);
                Reg::Select(argmaxHRes, candReg, argmaxHRes, gtMask);
                Reg::Adds(candReg, argmaxWStart, static_cast<INDEX_T>(w), allMaskIdx);
                Reg::Select(argmaxWRes, candReg, argmaxWRes, gtMask);
            } else {
                Reg::MaskReg gtMaskUnpack;
                gtMaskUnpack = ValueMaskToIndexMask<T>(gtMask);
                Reg::Adds(candReg, argmaxHStart, static_cast<INDEX_T>(h), allMaskIdx);
                Reg::Select(argmaxHRes, candReg, argmaxHRes, gtMaskUnpack);
                Reg::Adds(candReg, argmaxWStart, static_cast<INDEX_T>(w), allMaskIdx);
                Reg::Select(argmaxWRes, candReg, argmaxWRes, gtMaskUnpack);
            }

            Reg::Max(maxReg, maxReg, calcReg, gatherMask);
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__simd_callee__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ComposeAndStoreArgmax(
    Reg::RegTensor<INDEX_T>& argmaxHRes, Reg::RegTensor<INDEX_T>& argmaxWRes, __ubuf__ INDEX_T* argmaxAddr,
    uint16_t repeatElem, int32_t hOutputPlane, int32_t wOutputPlane)
{
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();

    if constexpr (IS_PAD) {
        Reg::RegTensor<INDEX_T> zeroReg;
        Reg::MaskReg ltMask;
        Reg::Duplicate(zeroReg, static_cast<INDEX_T>(0));
        Reg::Compare<INDEX_T, CMPMODE::LT>(ltMask, argmaxHRes, zeroReg, allMaskIdx);
        Reg::Select(argmaxHRes, zeroReg, argmaxHRes, ltMask);
        Reg::Compare<INDEX_T, CMPMODE::LT>(ltMask, argmaxWRes, zeroReg, allMaskIdx);
        Reg::Select(argmaxWRes, zeroReg, argmaxWRes, ltMask);
    }

    Reg::RegTensor<INDEX_T> argmaxRes;
    Reg::RegTensor<INDEX_T> tmp;
    Reg::Muls(argmaxRes, argmaxHRes, static_cast<INDEX_T>(wOutputPlane), allMaskIdx);
    Reg::Add(argmaxRes, argmaxRes, argmaxWRes, allMaskIdx);

    Reg::UnalignReg u1;
    Reg::DataCopyUnAlign(argmaxAddr, argmaxRes, u1, repeatElem);
    Reg::DataCopyUnAlignPost(argmaxAddr, u1, 0);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__simd_vf__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessWNHWC(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem, uint16_t cChunk, uint16_t hKernel,
    uint16_t wKernel, uint32_t colStride, uint32_t rowStride, uint32_t wBaseStep, int32_t hOrigin, int32_t wOrigin,
    int32_t wStride, int32_t hOutputPlane, int32_t wOutputPlane)
{
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<INDEX_T> hStart;
    Reg::RegTensor<INDEX_T> wStart;
    Reg::RegTensor<INDEX_T> hRes;
    Reg::RegTensor<INDEX_T> wRes;
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();

    PoolUtils::Index::GenGatterIndex2DVF<int32_t>(indexReg, static_cast<int32_t>(wBaseStep),
                                                  static_cast<int32_t>(cChunk), 1);
    PoolUtils::Index::GenGatterIndex2DVF<INDEX_T>(wStart, static_cast<INDEX_T>(wStride), static_cast<INDEX_T>(cChunk),
                                                  0);
    Reg::Adds(wStart, wStart, static_cast<INDEX_T>(wOrigin), allMaskIdx);
    Reg::Duplicate(hStart, static_cast<INDEX_T>(hOrigin));
    Reg::Copy(hRes, hStart, allMaskIdx);
    Reg::Copy(wRes, wStart, allMaskIdx);

    ProcessWNHWCKernel(xAddr, colStride, rowStride, indexReg, hKernel, wKernel, repeatElem, hStart, wStart, hRes, wRes);

    ComposeAndStoreArgmax<IS_PAD>(hRes, wRes, argmaxAddr, repeatElem, hOutputPlane, wOutputPlane);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__simd_vf__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessWNHWC2D(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem, uint16_t cChunk, uint16_t wOut,
    uint16_t hKernel, uint16_t wKernel, uint32_t colStride, uint32_t rowStride, uint32_t wBaseStep, uint32_t hBaseStep,
    int32_t hOrigin, int32_t wOrigin, int32_t wStride, int32_t hStride, int32_t hOutputPlane, int32_t wOutputPlane)
{
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<INDEX_T> hStart;
    Reg::RegTensor<INDEX_T> wStart;
    Reg::RegTensor<INDEX_T> hRes;
    Reg::RegTensor<INDEX_T> wRes;
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();
    int32_t wcChunk = static_cast<int32_t>(wOut) * static_cast<int32_t>(cChunk);

    PoolUtils::Index::GenGatterIndex3DVF<int32_t>(indexReg, static_cast<int32_t>(hBaseStep), wcChunk,
                                                  static_cast<int32_t>(wBaseStep), static_cast<int32_t>(cChunk), 1);
    PoolUtils::Index::GenGatterIndex3DVF<INDEX_T>(hStart, static_cast<INDEX_T>(hStride), static_cast<INDEX_T>(wcChunk),
                                                  0, static_cast<INDEX_T>(cChunk), 0);
    PoolUtils::Index::GenGatterIndex3DVF<INDEX_T>(wStart, 0, static_cast<INDEX_T>(wcChunk),
                                                  static_cast<INDEX_T>(wStride), static_cast<INDEX_T>(cChunk), 0);
    Reg::Adds(hStart, hStart, static_cast<INDEX_T>(hOrigin), allMaskIdx);
    Reg::Adds(wStart, wStart, static_cast<INDEX_T>(wOrigin), allMaskIdx);
    Reg::Copy(hRes, hStart, allMaskIdx);
    Reg::Copy(wRes, wStart, allMaskIdx);

    ProcessWNHWCKernel(xAddr, colStride, rowStride, indexReg, hKernel, wKernel, repeatElem, hStart, wStart, hRes, wRes);

    ComposeAndStoreArgmax<IS_PAD>(hRes, wRes, argmaxAddr, repeatElem, hOutputPlane, wOutputPlane);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__simd_vf__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessWNHWC3D(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem, uint16_t cChunk, uint16_t wOut, uint16_t hOut,
    uint16_t hKernel, uint16_t wKernel, uint32_t colStride, uint32_t rowStride, uint32_t wBaseStep, uint32_t hBaseStep,
    uint32_t ncBaseStep, int32_t hOrigin, int32_t wOrigin, int32_t wStride, int32_t hStride, int32_t hOutputPlane,
    int32_t wOutputPlane)
{
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<INDEX_T> hStart;
    Reg::RegTensor<INDEX_T> wStart;
    Reg::RegTensor<INDEX_T> hRes;
    Reg::RegTensor<INDEX_T> wRes;
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();
    int32_t wcChunk = static_cast<int32_t>(wOut) * static_cast<int32_t>(cChunk);
    int32_t hwcChunk = static_cast<int32_t>(hOut) * wcChunk;

    PoolUtils::Index::GenGatterIndex4DVF<int32_t>(indexReg, static_cast<int32_t>(ncBaseStep), hwcChunk,
                                                  static_cast<int32_t>(hBaseStep), wcChunk,
                                                  static_cast<int32_t>(wBaseStep), static_cast<int32_t>(cChunk), 1);
    PoolUtils::Index::GenGatterIndex4DVF<INDEX_T>(hStart, 0, static_cast<INDEX_T>(hwcChunk),
                                                  static_cast<INDEX_T>(hStride), static_cast<INDEX_T>(wcChunk), 0,
                                                  static_cast<INDEX_T>(cChunk), 0);
    PoolUtils::Index::GenGatterIndex4DVF<INDEX_T>(wStart, 0, static_cast<INDEX_T>(hwcChunk), 0,
                                                  static_cast<INDEX_T>(wcChunk), static_cast<INDEX_T>(wStride),
                                                  static_cast<INDEX_T>(cChunk), 0);
    Reg::Adds(hStart, hStart, static_cast<INDEX_T>(hOrigin), allMaskIdx);
    Reg::Adds(wStart, wStart, static_cast<INDEX_T>(wOrigin), allMaskIdx);
    Reg::Copy(hRes, hStart, allMaskIdx);
    Reg::Copy(wRes, wStart, allMaskIdx);

    ProcessWNHWCKernel(xAddr, colStride, rowStride, indexReg, hKernel, wKernel, repeatElem, hStart, wStart, hRes, wRes);

    ComposeAndStoreArgmax<IS_PAD>(hRes, wRes, argmaxAddr, repeatElem, hOutputPlane, wOutputPlane);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::SingleRowGather(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    uint16_t channelCount = static_cast<uint16_t>(cOutputActual_);
    uint16_t hKernel = tilingData_.base.hKernel;
    uint16_t wKernel = tilingData_.base.wKernel;
    uint32_t colStride = static_cast<uint32_t>(cAligned_);
    uint32_t rowStride = static_cast<uint32_t>(wInputActualPad_ * cAligned_);
    uint32_t wBaseStep = static_cast<uint32_t>(tilingData_.base.wStride * cAligned_);
    int32_t hOrigin = static_cast<int32_t>(hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH);
    int32_t wOrigin = static_cast<int32_t>(wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW);
    int32_t hStride = static_cast<int32_t>(tilingData_.base.hStride);
    int32_t wStride = static_cast<int32_t>(tilingData_.base.wStride);
    int32_t hOutputPlane = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t wOutputPlane = static_cast<int32_t>(tilingData_.base.wOutput);

    uint16_t indexVectorLanes = static_cast<uint16_t>(V_REG_SIZE /
                                                      (sizeof(T) > sizeof(INDEX_T) ? sizeof(T) : sizeof(INDEX_T)));
    uint16_t wFactor = indexVectorLanes / channelCount;
    uint16_t loopW = static_cast<uint16_t>(wArgmaxActual_ / wFactor);
    uint16_t tailW = static_cast<uint16_t>(wArgmaxActual_ - loopW * wFactor);
    if (tailW == 0) {
        loopW = loopW - 1;
        tailW = wFactor;
    }
    uint32_t wChunkElems = static_cast<uint32_t>(wFactor) * channelCount;
    uint32_t tailElems = static_cast<uint32_t>(tailW) * channelCount;

    uint32_t argmaxRowElems = static_cast<uint32_t>(wArgmaxActual_) * channelCount;
    uint32_t argmaxBatchElems = static_cast<uint32_t>(hArgmaxActual_) * argmaxRowElems;

    for (int64_t n = 0; n < nOutputActual_; n++) {
        for (int64_t h = 0; h < hArgmaxActual_; h++) {
            int64_t xOffset = n * ncBaseStep_ + h * hBaseStep_;
            int64_t argmaxOffset = static_cast<int64_t>(n) * argmaxBatchElems + h * argmaxRowElems;
            int32_t hOriginEff = hOrigin + static_cast<int32_t>(h) * hStride;
            for (uint16_t wLoop = 0; wLoop < loopW; wLoop++) {
                int32_t wOriginEff = wOrigin + static_cast<int32_t>(wLoop * wFactor) * wStride;
                DispatchProcessWNHWC(xAddr + xOffset + static_cast<int64_t>(wLoop) * wFactor * wBaseStep,
                                     argmaxAddr + argmaxOffset + static_cast<int64_t>(wLoop) * wChunkElems,
                                     static_cast<uint16_t>(wChunkElems), channelCount, hKernel, wKernel, colStride,
                                     rowStride, wBaseStep, hOriginEff, wOriginEff, wStride, hOutputPlane, wOutputPlane);
            }
            int32_t wOriginTail = wOrigin + static_cast<int32_t>(loopW * wFactor) * wStride;
            DispatchProcessWNHWC(xAddr + xOffset + static_cast<int64_t>(loopW) * wFactor * wBaseStep,
                                 argmaxAddr + argmaxOffset + static_cast<int64_t>(loopW) * wChunkElems,
                                 static_cast<uint16_t>(tailElems), channelCount, hKernel, wKernel, colStride, rowStride,
                                 wBaseStep, hOriginEff, wOriginTail, wStride, hOutputPlane, wOutputPlane);
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::MultiRowGather(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    uint16_t channelCount = static_cast<uint16_t>(cOutputActual_);
    uint16_t hKernel = tilingData_.base.hKernel;
    uint16_t wKernel = tilingData_.base.wKernel;
    uint32_t colStride = static_cast<uint32_t>(cAligned_);
    uint32_t rowStride = static_cast<uint32_t>(wInputActualPad_ * cAligned_);
    uint32_t wBaseStep = static_cast<uint32_t>(tilingData_.base.wStride * cAligned_);
    uint32_t hBaseStep = static_cast<uint32_t>(tilingData_.base.hStride * rowStride);
    int32_t hOrigin = static_cast<int32_t>(hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH);
    int32_t wOrigin = static_cast<int32_t>(wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW);
    int32_t hStride = static_cast<int32_t>(tilingData_.base.hStride);
    int32_t wStride = static_cast<int32_t>(tilingData_.base.wStride);
    int32_t hOutputPlane = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t wOutputPlane = static_cast<int32_t>(tilingData_.base.wOutput);

    uint16_t indexVectorLanes = static_cast<uint16_t>(V_REG_SIZE /
                                                      (sizeof(T) > sizeof(INDEX_T) ? sizeof(T) : sizeof(INDEX_T)));
    uint16_t wOutA = static_cast<uint16_t>(wArgmaxActual_);
    uint16_t hBatch = indexVectorLanes / static_cast<uint16_t>(wOutA * channelCount);
    uint16_t hLoopTimes = static_cast<uint16_t>(hArgmaxActual_ / hBatch);
    uint16_t hTail = static_cast<uint16_t>(hArgmaxActual_ - hLoopTimes * hBatch);
    if (hTail == 0) {
        hLoopTimes = hLoopTimes - 1;
        hTail = hBatch;
    }
    uint32_t chunkElems = static_cast<uint32_t>(hBatch) * wOutA * channelCount;
    uint32_t tailElems = static_cast<uint32_t>(hTail) * wOutA * channelCount;

    uint32_t argmaxRowElems = static_cast<uint32_t>(wArgmaxActual_) * channelCount;
    uint32_t argmaxBatchElems = static_cast<uint32_t>(hArgmaxActual_) * argmaxRowElems;

    for (int64_t n = 0; n < nOutputActual_; n++) {
        int64_t xBase = n * ncBaseStep_;
        int64_t argBase = static_cast<int64_t>(n) * argmaxBatchElems;
        for (uint16_t hLoop = 0; hLoop < hLoopTimes; hLoop++) {
            int32_t hOriginEff = hOrigin + static_cast<int32_t>(hLoop * hBatch) * hStride;
            DispatchProcessWNHWC2D(xAddr + xBase + static_cast<int64_t>(hLoop) * hBatch * hBaseStep_,
                                   argmaxAddr + argBase + static_cast<int64_t>(hLoop) * chunkElems,
                                   static_cast<uint16_t>(chunkElems), channelCount, wOutA, hKernel, wKernel, colStride,
                                   rowStride, wBaseStep, hBaseStep, hOriginEff, wOrigin, wStride, hStride, hOutputPlane,
                                   wOutputPlane);
        }
        int32_t hOriginTail = hOrigin + static_cast<int32_t>(hLoopTimes * hBatch) * hStride;
        DispatchProcessWNHWC2D(xAddr + xBase + static_cast<int64_t>(hLoopTimes) * hBatch * hBaseStep_,
                               argmaxAddr + argBase + static_cast<int64_t>(hLoopTimes) * chunkElems,
                               static_cast<uint16_t>(tailElems), channelCount, wOutA, hKernel, wKernel, colStride,
                               rowStride, wBaseStep, hBaseStep, hOriginTail, wOrigin, wStride, hStride, hOutputPlane,
                               wOutputPlane);
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::MultiNcGather(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    uint16_t channelCount = static_cast<uint16_t>(cOutputActual_);
    uint16_t hKernel = tilingData_.base.hKernel;
    uint16_t wKernel = tilingData_.base.wKernel;
    uint32_t colStride = static_cast<uint32_t>(cAligned_);
    uint32_t rowStride = static_cast<uint32_t>(wInputActualPad_ * cAligned_);
    uint32_t wBaseStep = static_cast<uint32_t>(tilingData_.base.wStride * cAligned_);
    uint32_t hBaseStep = static_cast<uint32_t>(tilingData_.base.hStride * rowStride);
    uint32_t ncBaseStep = static_cast<uint32_t>(hInputActualPad_ * rowStride);
    int32_t hOrigin = static_cast<int32_t>(hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH);
    int32_t wOrigin = static_cast<int32_t>(wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW);
    int32_t hStride = static_cast<int32_t>(tilingData_.base.hStride);
    int32_t wStride = static_cast<int32_t>(tilingData_.base.wStride);
    int32_t hOutputPlane = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t wOutputPlane = static_cast<int32_t>(tilingData_.base.wOutput);

    uint16_t indexVectorLanes = static_cast<uint16_t>(V_REG_SIZE /
                                                      (sizeof(T) > sizeof(INDEX_T) ? sizeof(T) : sizeof(INDEX_T)));
    uint16_t wOutA = static_cast<uint16_t>(wArgmaxActual_);
    uint16_t hOutA = static_cast<uint16_t>(hArgmaxActual_);
    uint16_t hwcElems = hOutA * wOutA * channelCount;
    uint16_t ncBatch = indexVectorLanes / hwcElems;
    uint16_t ncLoopTimes = static_cast<uint16_t>(nOutputActual_ / ncBatch);
    uint16_t ncTail = static_cast<uint16_t>(nOutputActual_ - ncLoopTimes * ncBatch);
    if (ncTail == 0) {
        ncLoopTimes = ncLoopTimes - 1;
        ncTail = ncBatch;
    }
    uint32_t chunkElems = static_cast<uint32_t>(ncBatch) * hwcElems;
    uint32_t tailElems = static_cast<uint32_t>(ncTail) * hwcElems;

    uint32_t argmaxRowElems = static_cast<uint32_t>(wArgmaxActual_) * channelCount;
    uint32_t argmaxBatchElems = static_cast<uint32_t>(hArgmaxActual_) * argmaxRowElems;

    for (uint16_t ncLoop = 0; ncLoop < ncLoopTimes; ncLoop++) {
        DispatchProcessWNHWC3D(xAddr + static_cast<int64_t>(ncLoop) * ncBatch * ncBaseStep,
                               argmaxAddr + static_cast<int64_t>(ncLoop) * chunkElems,
                               static_cast<uint16_t>(chunkElems), channelCount, wOutA, hOutA, hKernel, wKernel,
                               colStride, rowStride, wBaseStep, hBaseStep, ncBaseStep, hOrigin, wOrigin, wStride,
                               hStride, hOutputPlane, wOutputPlane);
    }
    DispatchProcessWNHWC3D(xAddr + static_cast<int64_t>(ncLoopTimes) * ncBatch * ncBaseStep,
                           argmaxAddr + static_cast<int64_t>(ncLoopTimes) * chunkElems,
                           static_cast<uint16_t>(tailElems), channelCount, wOutA, hOutA, hKernel, wKernel, colStride,
                           rowStride, wBaseStep, hBaseStep, ncBaseStep, hOrigin, wOrigin, wStride, hStride,
                           hOutputPlane, wOutputPlane);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ForwardScalarCompute()
{
    inputStrideW_ = cDim_;
    inputStrideH_ = tilingData_.base.wOutput * cDim_;

    hInputActualPad_ = (hArgmaxActual_ - 1) * tilingData_.base.hStride + (tilingData_.base.hKernel - 1) + 1;
    wInputActualPad_ = (wArgmaxActual_ - 1) * tilingData_.base.wStride + (tilingData_.base.wKernel - 1) + 1;

    ncBaseStep_ = static_cast<uint32_t>(hInputActualPad_ * wInputActualPad_ * cAligned_);
    hBaseStep_ = static_cast<uint32_t>(tilingData_.base.hStride * wInputActualPad_ * cAligned_);

    if (isPad_) {
        int64_t lRelBound = wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW;
        int64_t rRelBound = lRelBound + wInputActualPad_ - tilingData_.base.wOutput;
        leftOffsetToInputLeft_ = lRelBound >= 0 ? 0 : -lRelBound;
        rightOffsetToInputRight_ = rRelBound > 0 ? rRelBound : 0;

        int64_t tRelBound = hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH;
        int64_t downRelBound = tRelBound + hInputActualPad_ - tilingData_.base.hOutput;
        topOffsetToInputTop_ = tRelBound >= 0 ? 0 : -tRelBound;
        downOffsetToInputDown_ = downRelBound > 0 ? downRelBound : 0;

        hInputActualNoPad_ = hInputActualPad_ - topOffsetToInputTop_ - downOffsetToInputDown_;
        wInputActualNoPad_ = wInputActualPad_ - leftOffsetToInputLeft_ - rightOffsetToInputRight_;
    } else {
        leftOffsetToInputLeft_ = 0;
        rightOffsetToInputRight_ = 0;
        topOffsetToInputTop_ = 0;
        downOffsetToInputDown_ = 0;
        hInputActualNoPad_ = hInputActualPad_;
        wInputActualNoPad_ = wInputActualPad_;
    }

    int64_t hValidStart = hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH;
    hValidStart = hValidStart > 0 ? hValidStart : 0;
    int64_t wValidStart = wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW;
    wValidStart = wValidStart > 0 ? wValidStart : 0;

    forwardInputOffset_ = nAxisIndex_ * tilingData_.base.highAxisInner * tilingData_.base.hOutput *
                              tilingData_.base.wOutput * cDim_ +
                          hValidStart * inputStrideH_ + wValidStart * inputStrideW_ + cAxisGradOffset_;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ForwardCopyIn()
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    int64_t channelCount = cOutputActual_;
    int64_t wLoad = isPad_ ? wInputActualNoPad_ : wInputActualPad_;
    int64_t hLoad = isPad_ ? hInputActualNoPad_ : hInputActualPad_;

    // Load GM [N][H][W][C] into UB [N][H][W][cAligned].
    // LoopMode advances UB blocks by their aligned byte lengths.
    LoopModeParams loopParams;
    loopParams.loop2Size = static_cast<uint32_t>(nOutputActual_);
    loopParams.loop2SrcStride = static_cast<uint64_t>(tilingData_.base.hOutput * inputStrideH_ * sizeof(T));
    loopParams.loop2DstStride = static_cast<uint64_t>(hLoad * wLoad * cAligned_ * sizeof(T));
    loopParams.loop1Size = static_cast<uint32_t>(hLoad);
    loopParams.loop1SrcStride = static_cast<uint64_t>(inputStrideH_ * sizeof(T));
    loopParams.loop1DstStride = static_cast<uint64_t>(wLoad * cAligned_ * sizeof(T));

    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint32_t>(wLoad);
    copyParams.blockLen = static_cast<uint32_t>(channelCount * sizeof(T));
    copyParams.srcStride = static_cast<uint32_t>((cDim_ - channelCount) * sizeof(T));
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[forwardInputOffset_], copyParams, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inputQue_.EnQue(xLocal);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::Forward()
{
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    xForwardAddr_ = (__ubuf__ T*)xLocal.GetPhyAddr();

    if (isPad_) {
        LocalTensor<T> calcLocal = inputCalcBuff_.Get<T>();
        __ubuf__ T* calcAddr = (__ubuf__ T*)calcLocal.GetPhyAddr();
        int64_t calcTotalElems = nOutputActual_ * hInputActualPad_ * wInputActualPad_ * cAligned_;
        uint32_t vl = static_cast<uint32_t>(V_REG_SIZE / sizeof(T));
        DupCalcLowestVf(calcAddr, calcTotalElems, vl);
        CopyValidToCalcVf(
            calcAddr, xForwardAddr_, static_cast<uint32_t>(nOutputActual_), static_cast<uint32_t>(hInputActualNoPad_),
            static_cast<uint32_t>(wInputActualNoPad_ * cAligned_), static_cast<uint32_t>(hInputActualPad_),
            static_cast<uint32_t>(wInputActualPad_ * cAligned_),
            static_cast<uint32_t>(leftOffsetToInputLeft_ * cAligned_), static_cast<uint32_t>(topOffsetToInputTop_), vl);
        xForwardAddr_ = calcAddr;
    }

    LocalTensor<INDEX_T> argmaxLocal = argmaxBuff_.template Get<INDEX_T>();
    Duplicate(argmaxLocal, INDEX_T(-1), argmaxBufferSize_ / sizeof(INDEX_T));
    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);

    ForwardComputeTile();

    inputQue_.FreeTensor(xLocal);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::DupCalcLowestVf(__ubuf__ T* calcAddr,
                                                                                                  uint64_t totalElems,
                                                                                                  uint32_t vl)
{
    Reg::RegTensor<T> v0;
    DuplicateLowestValueVF<T>(v0);
    uint32_t loop = static_cast<uint32_t>(totalElems / vl);
    uint32_t tail = static_cast<uint32_t>(totalElems - loop * vl);
    Reg::MaskReg allMask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    for (uint32_t i = 0; i < loop; i++) {
        Reg::StoreAlign(calcAddr, v0, allMask);
        calcAddr += vl;
    }
    Reg::MaskReg tailMask = Reg::UpdateMask<T>(tail);
    Reg::StoreAlign(calcAddr, v0, tailMask);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::CopyValidToCalcVf(
    __ubuf__ T* calcAddr, __ubuf__ T* srcAddr, uint32_t nActual, uint32_t hNoPad, uint32_t wNoPadRowElems,
    uint32_t hPad, uint32_t wPadRowElems, uint32_t leftElems, uint32_t top, uint32_t vl)
{
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();

    uint32_t srcN = hNoPad * wNoPadRowElems;
    uint32_t srcH = wNoPadRowElems;
    uint32_t dstN = hPad * wPadRowElems;
    uint32_t dstH = wPadRowElems;

    uint16_t loopCols = static_cast<uint16_t>(wNoPadRowElems / vl);
    uint16_t tailCols = static_cast<uint16_t>(wNoPadRowElems - loopCols * vl);

    Reg::RegTensor<T> v0;
    Reg::UnalignReg u0;

    for (uint32_t n = 0; n < nActual; n++) {
        for (uint32_t h = 0; h < hNoPad; h++) {
            __ubuf__ T* srcCur = srcAddr + n * srcN + h * srcH;
            __ubuf__ T* dstCur = calcAddr + n * dstN + (h + top) * dstH + leftElems;
            for (uint16_t k = 0; k < loopCols; k++) {
                Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, srcCur, vl);
                Reg::DataCopyUnAlign(dstCur, v0, u0, vl);
            }
            Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, srcCur, vl);
            Reg::DataCopyUnAlign(dstCur, v0, u0, tailCols);
            Reg::DataCopyUnAlignPost(dstCur, u0, 0);
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::MaxPoolSingleChannelWithArgmax(
    __ubuf__ INDEX_T* argmaxAddr, __ubuf__ T* srcAddr, uint16_t kH, uint16_t kW, uint32_t rowStride, uint32_t colStride,
    uint16_t repeatElms, int32_t curInH, int32_t curInW, int32_t initialIndex, int32_t wOutputPlane)
{
    Reg::RegTensor<T> res;
    Reg::RegTensor<T> vd0;
    Reg::RegTensor<INDEX_T> argmaxRes;
    Reg::RegTensor<INDEX_T> candReg;

    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();
    uint32_t num = repeatElms;
    Reg::MaskReg p0 = Reg::UpdateMask<T>(num);
    Reg::MaskReg gtMask;

    DuplicateLowestValueVF<T>(res);
    Reg::Duplicate(argmaxRes, static_cast<INDEX_T>(initialIndex));

    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            Reg::AddrReg aReg = Reg::CreateAddrReg<T>(hIdx, rowStride, wIdx, colStride);
            Reg::LoadAlign(vd0, srcAddr, aReg);

            Reg::Compare<T, CMPMODE::GT>(gtMask, vd0, res, p0);

            int32_t kernelIndex = (curInH + static_cast<int32_t>(hIdx)) * wOutputPlane +
                                  (curInW + static_cast<int32_t>(wIdx));
            Reg::Duplicate(candReg, static_cast<INDEX_T>(kernelIndex));

            if constexpr (sizeof(INDEX_T) / sizeof(T) == 1) {
                Reg::Select(argmaxRes, candReg, argmaxRes, gtMask);
            } else {
                Reg::MaskReg gtMaskUnpack;
                gtMaskUnpack = ValueMaskToIndexMask<T>(gtMask);
                Reg::Select(argmaxRes, candReg, argmaxRes, gtMaskUnpack);
            }

            Reg::Max(res, vd0, res, p0);
        }
    }

    Reg::UnalignReg u1;
    Reg::DataCopyUnAlign(argmaxAddr, argmaxRes, u1, repeatElms);
    Reg::DataCopyUnAlignPost(argmaxAddr, u1, 0);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessBigCPosition(
    __ubuf__ INDEX_T* argmaxAddr, __ubuf__ T* srcAddr, uint16_t cLoop, uint16_t tailNum, uint16_t vl, uint16_t hKernel,
    uint16_t wKernel, uint32_t rowStride, uint32_t colStride, int32_t curInH, int32_t curInW, int32_t initialIndex,
    int32_t wOutputPlane)
{
    for (uint16_t m = 0; m < cLoop; m++) {
        MaxPoolSingleChannelWithArgmax(argmaxAddr + m * vl, srcAddr + m * vl, hKernel, wKernel, rowStride, colStride,
                                       vl, curInH, curInW, initialIndex, wOutputPlane);
    }
    MaxPoolSingleChannelWithArgmax(argmaxAddr + cLoop * vl, srcAddr + cLoop * vl, hKernel, wKernel, rowStride,
                                   colStride, tailNum, curInH, curInW, initialIndex, wOutputPlane);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ChunkCGather(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    uint16_t channelCount = static_cast<uint16_t>(cOutputActual_);
    uint16_t vl = static_cast<uint16_t>(V_REG_SIZE / (sizeof(T) > sizeof(INDEX_T) ? sizeof(T) : sizeof(INDEX_T)));
    uint16_t cLoop = channelCount / vl;
    uint16_t tailNum = static_cast<uint16_t>(channelCount - cLoop * vl);

    uint32_t colStride = static_cast<uint32_t>(cAligned_);
    uint32_t rowStride = static_cast<uint32_t>(wInputActualPad_ * cAligned_);
    uint16_t hKernel = static_cast<uint16_t>(tilingData_.base.hKernel);
    uint16_t wKernel = static_cast<uint16_t>(tilingData_.base.wKernel);

    int32_t wOutputPlane = static_cast<int32_t>(tilingData_.base.wOutput);

    uint32_t argmaxRowElems = static_cast<uint32_t>(wArgmaxActual_) * channelCount;
    uint32_t argmaxBatchElems = static_cast<uint32_t>(hArgmaxActual_) * argmaxRowElems;
    int64_t inputPlaneElems = static_cast<int64_t>(hInputActualPad_) * wInputActualPad_ * cAligned_;

    for (int64_t n = 0; n < nOutputActual_; n++) {
        for (int64_t h = 0; h < hArgmaxActual_; h++) {
            for (int64_t w = 0; w < wArgmaxActual_; w++) {
                int32_t curInH = static_cast<int32_t>((hArgmaxActualStart_ + h) * tilingData_.base.hStride -
                                                      tilingData_.base.padH);
                int32_t curInW = static_cast<int32_t>((wArgmaxActualStart_ + w) * tilingData_.base.wStride -
                                                      tilingData_.base.padW);

                const int32_t firstValidH = curInH < 0 ? 0 : curInH;
                const int32_t firstValidW = curInW < 0 ? 0 : curInW;
                const int32_t initialIndex = firstValidH * wOutputPlane + firstValidW;

                int64_t srcOffset = n * inputPlaneElems +
                                    h * tilingData_.base.hStride *
                                        (static_cast<int64_t>(wInputActualPad_) * cAligned_) +
                                    w * tilingData_.base.wStride * cAligned_;
                int64_t argmaxOffset = static_cast<int64_t>(n) * argmaxBatchElems + h * argmaxRowElems +
                                       static_cast<int64_t>(w) * channelCount;

                ProcessBigCPosition(argmaxAddr + argmaxOffset, xAddr + srcOffset, cLoop, tailNum, vl, hKernel, wKernel,
                                    rowStride, colStride, curInH, curInW, initialIndex, wOutputPlane);
            }
        }
    }
}

} // namespace MaxPoolV3GradNHWCIntNameSpace
#endif // MAX_POOL_V3_GRAD_NHWC_INT_IMPL_GATHER_H_
