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
 * \file max_pool_v3_grad_nhwc_impl_scatter.h
 * \brief Shared NHWC backward implementation for small and large kernels.
 * Argmax uses dense [N][H][W][C] layout between the forward and backward stages.
 */

#ifndef MAX_POOL_V3_GRAD_NHWC_IMPL_SCATTER_H_
#define MAX_POOL_V3_GRAD_NHWC_IMPL_SCATTER_H_

#include <limits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../pool_grad_common/arch35/max_pool_grad_struct.h"
#include "max_pool_v3_grad_scatter.h"

namespace MaxPoolV3GradNHWCNameSpace {

using namespace AscendC;
using ComputeType = float;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t DOUBLE_HEAD = 2;
constexpr int32_t HELP_BUFFER = 5120;

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
class MaxPoolV3GradNHWCBackwardBase {
public:
    __aicore__ inline MaxPoolV3GradNHWCBackwardBase() {}

protected:
    __aicore__ inline void ParseTilingData(const MaxPoolV3GradNHWCTilingData& tilingData);
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyInGrad();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void Backward();
    __aicore__ inline void VDrainToMte2();
    __aicore__ inline void Mte3Drain();
    __aicore__ inline void BackwardCompute(__ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void CompensatedGatherProcessVF(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                                      __ubuf__ INDEX_T* argmaxAddr);

    __aicore__ inline void SingleLineProcessVF(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                               __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultipleLineWProcessVF(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                                  __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultipleLineHwProcessVF(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                                   __ubuf__ INDEX_T* argmaxAddr);

    __aicore__ inline int64_t PStart(int64_t index, int64_t pad, int64_t kernel, int64_t stride);
    __aicore__ inline int64_t PEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize);

    MaxPoolV3GradNHWCTilingData tilingData_;

    GlobalTensor<T> gradGm_;
    GlobalTensor<T> yGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> argmaxBuff_;
    TBuf<QuePosition::VECCALC> helpBuf_;

    int64_t blockIdx_ = 0;
    int64_t curCoreProcessNum_ = 0;
    event_t vToMte2Event_;
    event_t mte3ToMte2Event_;

    int64_t nAxisIndex_ = 0;
    int64_t cAxisIndex_ = 0;
    int64_t hAxisIndex_ = 0;
    int64_t wAxisIndex_ = 0;

    int64_t nOutputActual_ = 0;
    int64_t cOutputActual_ = 0;
    int64_t hOutputActual_ = 0;
    int64_t wOutputActual_ = 0;

    int64_t cAxisGradOffset_ = 0;

    int64_t hArgmaxActualStart_ = 0;
    int64_t hArgmaxActualEnd_ = 0;
    int64_t wArgmaxActualStart_ = 0;
    int64_t wArgmaxActualEnd_ = 0;
    int64_t hArgmaxActual_ = 0;
    int64_t wArgmaxActual_ = 0;

    int64_t cAligned_ = 0;
    int64_t cDim_ = 0;
    int64_t argmaxBufferSize_ = 0;

    bool isPad_ = false;
    int64_t vlT2_ = 0;

    int64_t curHProBatchSize_ = 0;
    int64_t curWProBatchSize_ = 0;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();
};

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::ParseTilingData(
    const MaxPoolV3GradNHWCTilingData& tilingData)
{
    tilingData_ = tilingData;
    cDim_ = tilingData.cDim;
    argmaxBufferSize_ = tilingData.base.argmaxBufferSize;
    isPad_ = (tilingData.base.padH != 0 || tilingData.base.padW != 0 || tilingData.base.padHBack != 0 ||
              tilingData.base.padWBack != 0);
    vlT2_ = static_cast<int64_t>(V_REG_SIZE) / sizeof(INDEX_T);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline int64_t MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::PStart(int64_t index, int64_t pad,
                                                                                            int64_t kernel,
                                                                                            int64_t stride)
{
    if (stride == 0) {
        return 0;
    }
    return (index + pad < kernel) ? 0 : (index + pad - kernel) / stride + 1;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline int64_t MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::PEnd(int64_t index, int64_t pad,
                                                                                          int64_t stride,
                                                                                          int64_t pooledSize)
{
    if (stride == 0) {
        return 0;
    }
    int64_t end = (index + pad) / stride + 1;
    return end < pooledSize ? end : pooledSize;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::ScalarCompute(int64_t loopNum)
{
    int64_t baseBlockIdx = blockIdx_ * tilingData_.base.normalCoreProcessNum + loopNum;

    int64_t cOuter = tilingData_.cOutputOuter;
    int64_t hOuter = tilingData_.base.hOutputOuter;
    int64_t wOuter = tilingData_.base.wOutputOuter;

    nAxisIndex_ = baseBlockIdx / (cOuter * hOuter * wOuter);
    int64_t rem1 = baseBlockIdx % (cOuter * hOuter * wOuter);
    cAxisIndex_ = rem1 / (hOuter * wOuter);
    int64_t rem2 = rem1 % (hOuter * wOuter);
    hAxisIndex_ = rem2 / wOuter;
    wAxisIndex_ = rem2 % wOuter;

    nOutputActual_ = (nAxisIndex_ == tilingData_.base.highAxisOuter - 1) ? tilingData_.base.highAxisTail :
                                                                           tilingData_.base.highAxisInner;
    cOutputActual_ = (cAxisIndex_ == cOuter - 1) ? tilingData_.cOutputTail : tilingData_.cOutputInner;
    hOutputActual_ = (hAxisIndex_ == hOuter - 1) ? tilingData_.base.hOutputTail : tilingData_.base.hOutputInner;
    wOutputActual_ = (wAxisIndex_ == wOuter - 1) ? tilingData_.base.wOutputTail : tilingData_.base.wOutputInner;

    hArgmaxActualStart_ = PStart(hAxisIndex_ * tilingData_.base.hOutputInner, tilingData_.base.padH,
                                 tilingData_.base.hKernel, tilingData_.base.hStride);
    hArgmaxActualEnd_ = PEnd(hAxisIndex_ * tilingData_.base.hOutputInner + hOutputActual_ - 1, tilingData_.base.padH,
                             tilingData_.base.hStride, tilingData_.base.hArgmax);
    wArgmaxActualStart_ = PStart(wAxisIndex_ * tilingData_.base.wOutputInner, tilingData_.base.padW,
                                 tilingData_.base.wKernel, tilingData_.base.wStride);
    wArgmaxActualEnd_ = PEnd(wAxisIndex_ * tilingData_.base.wOutputInner + wOutputActual_ - 1, tilingData_.base.padW,
                             tilingData_.base.wStride, tilingData_.base.wArgmax);

    hArgmaxActual_ = hArgmaxActualEnd_ - hArgmaxActualStart_;
    wArgmaxActual_ = wArgmaxActualEnd_ - wArgmaxActualStart_;

    cAxisGradOffset_ = cAxisIndex_ * tilingData_.cOutputInner;
    int64_t blockElems = BLOCK_SIZE / sizeof(T);
    cAligned_ = (cOutputActual_ + blockElems - 1) / blockElems * blockElems;

    curHProBatchSize_ = tilingData_.base.hProBatchSize > hArgmaxActual_ ? hArgmaxActual_ :
                                                                          tilingData_.base.hProBatchSize;
    curWProBatchSize_ = tilingData_.base.wProBatchSize > wArgmaxActual_ ? wArgmaxActual_ :
                                                                          tilingData_.base.wProBatchSize;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::CopyInGrad()
{
    LocalTensor<T> gradLocal = gradQue_.AllocTensor<T>();
    const int64_t C = cOutputActual_;
    const int64_t rowElt = wArgmaxActual_ * cAligned_;
    const int64_t planeElt = hArgmaxActual_ * rowElt;

    const int64_t gradStrideW = cDim_;
    const int64_t gradStrideH = tilingData_.base.wArgmax * gradStrideW;
    const int64_t gradBase = nAxisIndex_ * tilingData_.base.highAxisInner * tilingData_.base.hArgmax * gradStrideH +
                             hArgmaxActualStart_ * gradStrideH + wArgmaxActualStart_ * gradStrideW + cAxisGradOffset_;

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint32_t>(wArgmaxActual_);
    copyParams.blockLen = static_cast<uint32_t>(C * sizeof(T));
    copyParams.srcStride = static_cast<uint32_t>((cDim_ - C) * sizeof(T));
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    for (int64_t n = 0; n < nOutputActual_; ++n) {
        for (int64_t h = 0; h < hArgmaxActual_; ++h) {
            int64_t dstOffset = n * planeElt + h * rowElt;
            int64_t srcOffset = gradBase + n * tilingData_.base.hArgmax * gradStrideH + h * gradStrideH;
            DataCopyPad(gradLocal[dstOffset], gradGm_[srcOffset], copyParams, padParams);
        }
    }
    gradQue_.EnQue(gradLocal);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::CopyOut()
{
    LocalTensor<T> yLocal = outputQue_.DeQue<T>();
    const int64_t C = cOutputActual_;
    const int64_t rowElt = wOutputActual_ * cAligned_;
    const int64_t planeElt = hOutputActual_ * rowElt;

    const int64_t yStrideW = cDim_;
    const int64_t yStrideH = tilingData_.base.wOutput * yStrideW;
    const int64_t yBase = nAxisIndex_ * tilingData_.base.highAxisInner * tilingData_.base.hOutput * yStrideH +
                          hAxisIndex_ * tilingData_.base.hOutputInner * yStrideH +
                          wAxisIndex_ * tilingData_.base.wOutputInner * yStrideW + cAxisGradOffset_;

    for (int64_t n = 0; n < nOutputActual_; n++) {
        int64_t gmOffset = yBase + n * tilingData_.base.hOutput * yStrideH;
        int64_t ubOffset = n * planeElt;

        DataCopyExtParams copyParam;
        copyParam.blockCount = static_cast<uint16_t>(wOutputActual_);
        copyParam.blockLen = static_cast<uint32_t>(C * sizeof(T));
        copyParam.srcStride = 0;
        copyParam.dstStride = static_cast<uint32_t>((cDim_ - C) * sizeof(T));
        copyParam.rsv = 0;

        LoopModeParams loopParam;
        loopParam.loop1Size = static_cast<uint16_t>(hOutputActual_);
        loopParam.loop2Size = 1;
        loopParam.loop1SrcStride = static_cast<uint64_t>(rowElt * sizeof(T));
        loopParam.loop2SrcStride = static_cast<uint64_t>(planeElt * sizeof(T));
        loopParam.loop1DstStride = static_cast<uint64_t>(yStrideH * sizeof(T));
        loopParam.loop2DstStride = static_cast<uint64_t>(yStrideH * sizeof(T));

        SetLoopModePara(loopParam, DataCopyMVType::UB_TO_OUT);
        DataCopyPad(yGm_[gmOffset], yLocal[static_cast<uint32_t>(ubOffset)], copyParam);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
    outputQue_.FreeTensor(yLocal);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(nOutputActual_ * hOutputActual_ * wOutputActual_ * cAligned_);
    LocalTensor<T> yLocal = outputQue_.AllocTensor<T>();
    Duplicate(yLocal, T(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::Backward()
{
    uint32_t calcCount = static_cast<uint32_t>(nOutputActual_ * hOutputActual_ * wOutputActual_ * cAligned_);
    LocalTensor<float> yLocal = outputQue_.AllocTensor<float>();
    Duplicate(yLocal, 0.f, calcCount);
    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
    LocalTensor<T> gradLocal = gradQue_.DeQue<T>();
    LocalTensor<INDEX_T> argmaxLocal = argmaxBuff_.Get<INDEX_T>();

    BackwardCompute((__ubuf__ float*)yLocal.GetPhyAddr(), (__ubuf__ T*)gradLocal.GetPhyAddr(),
                    (__ubuf__ INDEX_T*)argmaxLocal.GetPhyAddr());
    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);

    if constexpr (std::negation<std::is_same<T, float>>::value) {
        Cast(yLocal.ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, calcCount);
    }
    outputQue_.EnQue(yLocal);
    gradQue_.FreeTensor(gradLocal);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::VDrainToMte2()
{
    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::Mte3Drain()
{
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
}

constexpr int32_t PATTERN_BASE_WORD = 64;
constexpr int32_t PATTERN_STRIDE_WORD = 64;

constexpr int32_t SLOT_PATTERN_ONE = 2;      // 3D (H,W batch, C) -> initialReg{Index,Argmax}One
constexpr int32_t SLOT_PATTERN_W_BATCH = 10; // 2D (W batch, C) -> initialReg{Index,Argmax}WBatch

__simd_callee__ inline void GenPattern2D(AscendC::Reg::RegTensor<int32_t>& indexReg, __ubuf__ int32_t* helpAddr,
                                         int32_t groupStride, int32_t cCnt, int32_t slot)
{
    // lane = g*cCnt + c → offset = g*groupStride + c
    __ubuf__ int32_t* dstAddr = helpAddr + PATTERN_BASE_WORD + slot * PATTERN_STRIDE_WORD;
    AscendC::Reg::RegTensor<int32_t> laneReg;
    AscendC::Reg::RegTensor<int32_t> gReg;
    AscendC::Reg::RegTensor<int32_t> cReg;
    AscendC::Reg::RegTensor<int32_t> cCntReg;
    AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Arange(laneReg, 0);
    AscendC::Reg::Duplicate(cCntReg, cCnt, p0);
    AscendC::Reg::Div(gReg, laneReg, cCntReg, p0);
    AscendC::Reg::Mul(cReg, gReg, cCntReg, p0);
    AscendC::Reg::Sub(cReg, laneReg, cReg, p0);
    AscendC::Reg::Muls(gReg, gReg, groupStride, p0);
    AscendC::Reg::Add(gReg, gReg, cReg, p0);
    AscendC::Reg::StoreAlign(dstAddr, gReg, p0);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign(indexReg, dstAddr);
}

__simd_callee__ inline void GenPattern3D(AscendC::Reg::RegTensor<int32_t>& indexReg, __ubuf__ int32_t* helpAddr,
                                         int32_t hiGroupStride, int32_t loGroupStride, int32_t loGroupCount,
                                         int32_t cCnt, int32_t slot)
{
    // lane = hi*(loCnt*cCnt) + lo*cCnt + c → offset = hi*hiStride + lo*loStride + c
    __ubuf__ int32_t* dstAddr = helpAddr + PATTERN_BASE_WORD + slot * PATTERN_STRIDE_WORD;
    AscendC::Reg::RegTensor<int32_t> laneReg;
    AscendC::Reg::RegTensor<int32_t> hiReg;
    AscendC::Reg::RegTensor<int32_t> loReg;
    AscendC::Reg::RegTensor<int32_t> remReg;
    AscendC::Reg::RegTensor<int32_t> cReg;
    AscendC::Reg::RegTensor<int32_t> cCntReg;
    AscendC::Reg::RegTensor<int32_t> loCntReg;
    AscendC::Reg::MaskReg p0 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Arange(laneReg, 0);
    AscendC::Reg::Duplicate(cCntReg, cCnt, p0);
    AscendC::Reg::Duplicate(loCntReg, loGroupCount, p0);
    AscendC::Reg::Mul(remReg, loCntReg, cCntReg, p0);
    AscendC::Reg::Div(hiReg, laneReg, remReg, p0);
    AscendC::Reg::Mul(remReg, hiReg, remReg, p0);
    AscendC::Reg::Sub(remReg, laneReg, remReg, p0);
    AscendC::Reg::Div(loReg, remReg, cCntReg, p0);
    AscendC::Reg::Mul(cReg, loReg, cCntReg, p0);
    AscendC::Reg::Sub(cReg, remReg, cReg, p0);
    AscendC::Reg::Muls(hiReg, hiReg, hiGroupStride, p0);
    AscendC::Reg::Muls(loReg, loReg, loGroupStride, p0);
    AscendC::Reg::Add(hiReg, hiReg, loReg, p0);
    AscendC::Reg::Add(hiReg, hiReg, cReg, p0);
    AscendC::Reg::StoreAlign(dstAddr, hiReg, p0);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    AscendC::Reg::LoadAlign(indexReg, dstAddr);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void SingleLineProcessVf(__ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr,
                                            int32_t wInput, int32_t wOutputActual, int32_t hOutputActual,
                                            int32_t curHIndex, int32_t curWIndex, int32_t cOutputAligned,
                                            uint16_t nOutputActual, uint16_t hArgmaxActual, uint16_t wArgmaxActual,
                                            int32_t cOutputActual, uint32_t magicW, uint32_t shiftW)
{
    constexpr uint16_t computeSizeFP32 = 64;
    int32_t cStride = cOutputAligned;
    int32_t baseOffsetConst = -(curHIndex * wOutputActual + curWIndex) * cStride;
    uint16_t cRepeatimes = static_cast<uint16_t>(cOutputActual / computeSizeFP32);
    uint16_t cRemain = static_cast<uint16_t>(cOutputActual - cRepeatimes * computeSizeFP32);
    AscendC::Reg::RegTensor<int32_t> hLowerReg;
    AscendC::Reg::RegTensor<int32_t> wLowerReg;
    AscendC::Reg::RegTensor<int32_t> hUpperReg;
    AscendC::Reg::RegTensor<int32_t> wUpperReg;
    if constexpr (IS_CHECK_RANGE == 1) {
        AscendC::Reg::Duplicate(hLowerReg, int32_t(curHIndex));
        AscendC::Reg::Duplicate(wLowerReg, int32_t(curWIndex));
        AscendC::Reg::Duplicate(hUpperReg, int32_t(hOutputActual + curHIndex));
        AscendC::Reg::Duplicate(wUpperReg, int32_t(wOutputActual + curWIndex));
    }
    AscendC::Reg::RegTensor<uint32_t> magicWReg;
    AscendC::Reg::Duplicate(magicWReg, magicW);

    AscendC::Reg::RegTensor<int32_t> parallelRegIndex;
    AscendC::Reg::RegTensor<int32_t> parallelRegArgmax;
    AscendC::Reg::RegTensor<int32_t> initialRegIndex;
    AscendC::Reg::Arange(initialRegIndex, 0);
    AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<int32_t> cIncReg;

    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nGradOffset = nIdx * hArgmaxActual * wArgmaxActual * cStride;
        uint32_t nArgmaxOffset = nIdx * hArgmaxActual * wArgmaxActual * cOutputActual;
        int32_t baseOffset = int32_t(nIdx * hOutputActual * wOutputActual * cStride) + baseOffsetConst;
        for (uint16_t hIdx = 0; hIdx < hArgmaxActual; ++hIdx) {
            uint32_t hGradBase = hIdx * wArgmaxActual * cStride + nGradOffset;
            uint32_t hArgmaxBase = hIdx * wArgmaxActual * cOutputActual + nArgmaxOffset;
            for (uint16_t wIdx = 0; wIdx < wArgmaxActual; ++wIdx) {
                uint32_t hwGradBase = wIdx * cStride + hGradBase;
                uint32_t hwArgmaxBase = wIdx * cOutputActual + hArgmaxBase;
                for (uint16_t cRepeatIdx = 0; cRepeatIdx < cRepeatimes; ++cRepeatIdx) {
                    uint32_t cOffset = cRepeatIdx * computeSizeFP32;
                    uint32_t gradOffset = hwGradBase + cOffset;
                    uint32_t argmaxOffset = hwArgmaxBase + cOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(gradOffset), allMaskU32);
                    AscendC::Reg::Adds(parallelRegArgmax, initialRegIndex, static_cast<int32_t>(argmaxOffset),
                                       allMaskU32);
                    AscendC::Reg::Arange(cIncReg, static_cast<int32_t>(cOffset));
                    DoSingleNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                        (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr,
                        (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, computeSizeFP32,
                        magicWReg, static_cast<int16_t>(shiftW), wOutputActual, wInput, baseOffset, cOutputAligned,
                        cIncReg, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
                }
                {
                    uint32_t cOffset = cRepeatimes * computeSizeFP32;
                    uint32_t gradOffset = hwGradBase + cOffset;
                    uint32_t argmaxOffset = hwArgmaxBase + cOffset;
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(gradOffset), allMaskU32);
                    AscendC::Reg::Adds(parallelRegArgmax, initialRegIndex, static_cast<int32_t>(argmaxOffset),
                                       allMaskU32);
                    AscendC::Reg::Arange(cIncReg, static_cast<int32_t>(cOffset));
                    DoSingleNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                        (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr,
                        (__local_mem__ INDEX_T*)argmaxAddr, parallelRegArgmax, parallelRegIndex, cRemain, magicWReg,
                        static_cast<int16_t>(shiftW), wOutputActual, wInput, baseOffset, cOutputAligned, cIncReg,
                        hLowerReg, wLowerReg, hUpperReg, wUpperReg);
                }
            }
        }
    }
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::SingleLineProcessVF(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_.base.hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_.base.wOutputInner);
    int32_t cOutputAligned = static_cast<int32_t>(cAligned_);
    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    uint16_t hArgmaxActual = static_cast<uint16_t>(hArgmaxActual_);
    uint16_t wArgmaxActual = static_cast<uint16_t>(wArgmaxActual_);
    int32_t cOutputActual = static_cast<int32_t>(cOutputActual_);

    uint32_t magicW = 0;
    uint32_t shiftW = 0;
    GetUintDivMagicAndShift<uint32_t>(magicW, shiftW, static_cast<uint32_t>(wInput));

    SingleLineProcessVf<T, INDEX_T, IS_CHECK_RANGE>(yAddr, gradAddr, argmaxAddr, wInput, wOutputActual, hOutputActual,
                                                    curHIndex, curWIndex, cOutputAligned, nOutputActual, hArgmaxActual,
                                                    wArgmaxActual, cOutputActual, magicW, shiftW);
}

template <typename INDEX_T>
__simd_vf__ inline void CompensatedGatherProcessVf(__ubuf__ float* yChunk, __ubuf__ float* gradCandidateChunk,
                                                   __ubuf__ INDEX_T* argmaxCandidateChunk, uint32_t gradRowStride,
                                                   uint32_t argmaxRowStride, uint32_t candidateRows,
                                                   uint32_t candidateCols, uint32_t gradColStride,
                                                   uint32_t argmaxColStride, int32_t targetIndex, uint32_t count)
{
    Reg::RegTensor<uint32_t> laneReg;
    Reg::Arange((Reg::RegTensor<int32_t>&)laneReg, 0);
    uint32_t gradCount = count;
    uint32_t argmaxCount = count;
    Reg::MaskReg channelMask = Reg::UpdateMask<float>(gradCount);
    Reg::MaskReg argmaxMask = GenT2Mask<INDEX_T>(argmaxCount);
    Reg::RegTensor<int32_t> targetReg;
    Reg::Duplicate(targetReg, targetIndex);
    Reg::RegTensor<float> zeroReg;
    Reg::RegTensor<float> sumReg;
    Reg::RegTensor<float> correctionReg;
    Reg::Duplicate(zeroReg, 0.f);
    Reg::Duplicate(sumReg, 0.f);
    Reg::Duplicate(correctionReg, 0.f);

    for (uint32_t h = 0; h < candidateRows; ++h) {
        __ubuf__ float* gradPtr = gradCandidateChunk + h * gradRowStride;
        __ubuf__ INDEX_T* argmaxPtr = argmaxCandidateChunk + h * argmaxRowStride;
        for (uint32_t w = 0; w < candidateCols; ++w) {
            Reg::RegTensor<int32_t> argmaxReg;
            Reg::RegTensor<float> valueReg;
            GetConCurrentInput<float, INDEX_T>(argmaxReg, valueReg, gradPtr, argmaxPtr, laneReg, laneReg, channelMask,
                                               argmaxMask);
            Reg::MaskReg matchMask;
            Reg::Compare<int32_t, CMPMODE::EQ>(matchMask, argmaxReg, targetReg, channelMask);
            Reg::Select(valueReg, valueReg, zeroReg, matchMask);

            // Neumaier retains the rounding residual even when the new term is larger than the sum.
            Reg::RegTensor<float> nextReg;
            Reg::RegTensor<float> absSumReg;
            Reg::RegTensor<float> absValueReg;
            Reg::RegTensor<float> largerReg;
            Reg::RegTensor<float> smallerReg;
            Reg::RegTensor<float> residualReg;
            Reg::Add(nextReg, sumReg, valueReg, channelMask);
            Reg::Abs(absSumReg, sumReg, channelMask);
            Reg::Abs(absValueReg, valueReg, channelMask);
            Reg::MaskReg sumLargerMask;
            Reg::Compare<float, CMPMODE::GE>(sumLargerMask, absSumReg, absValueReg, channelMask);
            Reg::Select(largerReg, sumReg, valueReg, sumLargerMask);
            Reg::Select(smallerReg, valueReg, sumReg, sumLargerMask);
            Reg::Sub(residualReg, largerReg, nextReg, channelMask);
            Reg::Add(residualReg, residualReg, smallerReg, channelMask);
            Reg::Add(correctionReg, correctionReg, residualReg, channelMask);
            // Non-finite sums must retain ordinary IEEE addition semantics.
            Reg::RegTensor<float> absNextReg;
            Reg::RegTensor<float> finiteLimitReg;
            Reg::Abs(absNextReg, nextReg, channelMask);
            Reg::Duplicate(finiteLimitReg, std::numeric_limits<float>::max());
            Reg::MaskReg finiteMask;
            Reg::Compare<float, CMPMODE::LE>(finiteMask, absNextReg, finiteLimitReg, channelMask);
            Reg::Select(correctionReg, correctionReg, zeroReg, finiteMask);
            sumReg = nextReg;
            gradPtr += gradColStride;
            argmaxPtr += argmaxColStride;
        }
    }
    Reg::Add(sumReg, sumReg, correctionReg, channelMask);
    Reg::UnalignRegForStore u0;
    Reg::StoreUnAlign(yChunk, sumReg, u0, count);
    Reg::StoreUnAlignPost(yChunk, u0, 0);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::CompensatedGatherProcessVF(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    constexpr int64_t channelChunk = 64;
    const int64_t curH = hAxisIndex_ * tilingData_.base.hOutputInner;
    const int64_t curW = wAxisIndex_ * tilingData_.base.wOutputInner;
    const int64_t wInput = tilingData_.base.wOutput;
    const int64_t gradRowStride = wArgmaxActual_ * cAligned_;
    const int64_t argmaxRowStride = wArgmaxActual_ * cOutputActual_;
    for (int64_t n = 0; n < nOutputActual_; ++n) {
        for (int64_t h = 0; h < hOutputActual_; ++h) {
            int64_t hStart = PStart(curH + h, tilingData_.base.padH, tilingData_.base.hKernel,
                                    tilingData_.base.hStride);
            int64_t hEnd = PEnd(curH + h, tilingData_.base.padH, tilingData_.base.hStride, tilingData_.base.hArgmax);
            hStart = hStart < hArgmaxActualStart_ ? hArgmaxActualStart_ : hStart;
            hEnd = hEnd > hArgmaxActualEnd_ ? hArgmaxActualEnd_ : hEnd;
            for (int64_t w = 0; w < wOutputActual_; ++w) {
                int64_t wStart = PStart(curW + w, tilingData_.base.padW, tilingData_.base.wKernel,
                                        tilingData_.base.wStride);
                int64_t wEnd = PEnd(curW + w, tilingData_.base.padW, tilingData_.base.wStride,
                                    tilingData_.base.wArgmax);
                wStart = wStart < wArgmaxActualStart_ ? wArgmaxActualStart_ : wStart;
                wEnd = wEnd > wArgmaxActualEnd_ ? wArgmaxActualEnd_ : wEnd;
                if (hStart >= hEnd || wStart >= wEnd) {
                    continue;
                }
                const int64_t candidateOffset = (n * hArgmaxActual_ + hStart - hArgmaxActualStart_) * wArgmaxActual_ +
                                                wStart - wArgmaxActualStart_;
                const int64_t yOffset = ((n * hOutputActual_ + h) * wOutputActual_ + w) * cAligned_;
                const int32_t targetIndex = static_cast<int32_t>((curH + h) * wInput + curW + w);
                for (int64_t c = 0; c < cOutputActual_; c += channelChunk) {
                    const uint32_t count = static_cast<uint32_t>(
                        cOutputActual_ - c < channelChunk ? cOutputActual_ - c : channelChunk);
                    CompensatedGatherProcessVf<INDEX_T>(
                        yAddr + yOffset + c, (__ubuf__ float*)gradAddr + candidateOffset * cAligned_ + c,
                        argmaxAddr + candidateOffset * cOutputActual_ + c, static_cast<uint32_t>(gradRowStride),
                        static_cast<uint32_t>(argmaxRowStride), static_cast<uint32_t>(hEnd - hStart),
                        static_cast<uint32_t>(wEnd - wStart), static_cast<uint32_t>(cAligned_),
                        static_cast<uint32_t>(cOutputActual_), targetIndex, count);
                }
            }
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::BackwardCompute(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    if constexpr (!std::is_same<T, float>::value) {
        SingleLineProcessVF(yAddr, gradAddr, argmaxAddr);
        return;
    }
    if constexpr (std::is_same<T, float>::value) {
        if (tilingData_.base.isBigKernel && tilingData_.base.hProBatchSize * tilingData_.base.wProBatchSize >= 16) {
            CompensatedGatherProcessVF(yAddr, gradAddr, argmaxAddr);
            return;
        }
    }
    uint32_t wConcurrentCount = curWProBatchSize_ > 0 ? static_cast<uint32_t>(wArgmaxActual_ / curWProBatchSize_) : 0;
    uint32_t hConcurrentCount = curHProBatchSize_ > 0 ? static_cast<uint32_t>(hArgmaxActual_ / curHProBatchSize_) : 0;

    const uint32_t computeSizeFP32 = static_cast<uint32_t>(V_REG_SIZE) / sizeof(float);
    const uint32_t cActualU32 = static_cast<uint32_t>(cOutputActual_);
    const bool fitWC = cActualU32 > 0 && cActualU32 <= computeSizeFP32;
    const bool fitHWC = fitWC && wConcurrentCount * cActualU32 <= computeSizeFP32;

    const uint32_t concW = computeSizeFP32 / cActualU32;
    const bool fillW = fitWC && wConcurrentCount >= concW && wConcurrentCount > 0;
    const uint32_t concH = fillW ? concW / wConcurrentCount : 0;
    const bool fillH = fillW && hConcurrentCount >= concH && hConcurrentCount > 0;

    if (concW <= 1 || wConcurrentCount * DOUBLE_HEAD * sizeof(INDEX_T) > V_REG_SIZE || !fitWC || !fillW) {
        SingleLineProcessVF(yAddr, gradAddr, argmaxAddr);
    } else if (wConcurrentCount * hConcurrentCount * DOUBLE_HEAD * sizeof(INDEX_T) > V_REG_SIZE || !fitHWC || !fillH) {
        MultipleLineWProcessVF(yAddr, gradAddr, argmaxAddr);
    } else {
        MultipleLineHwProcessVF(yAddr, gradAddr, argmaxAddr);
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MultipleLineWProcessVf(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                               __ubuf__ INDEX_T* argmaxAddr, __ubuf__ int32_t* helpAddr, int32_t wInput,
                                               int32_t wOutputActual, int32_t hOutputActual, int32_t curHIndex,
                                               int32_t curWIndex, int32_t cOutputAligned, int32_t cOutputActual,
                                               uint32_t wBatchConcurrency, uint16_t repeatimes, uint32_t wRemainBatch,
                                               uint16_t wRemainTail, uint32_t magicW, uint32_t shiftW,
                                               uint32_t divCMagic, int16_t divCShift);

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::MultipleLineWProcessVF(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    constexpr int32_t V_REG_SIZE = platform::GetVRegSize();

    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_.base.hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_.base.wOutputInner);
    int32_t cOutputAligned = static_cast<int32_t>(cAligned_);
    int32_t cOutputActual = static_cast<int32_t>(cOutputActual_);
    int32_t cStride = cOutputAligned;
    int32_t cActual = cOutputActual;

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    uint16_t hArgmaxActual = static_cast<uint16_t>(hArgmaxActual_);
    int32_t wArgmaxActual = static_cast<int32_t>(wArgmaxActual_);
    uint16_t wProBatchSize = static_cast<uint16_t>(tilingData_.base.wProBatchSize);
    wProBatchSize = (wProBatchSize > wArgmaxActual) ? static_cast<uint16_t>(wArgmaxActual) : wProBatchSize;

    uint32_t wFullBatchCount = static_cast<uint32_t>(wArgmaxActual) / wProBatchSize;
    uint16_t computeSizeFP32 = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSizeFP32 / static_cast<uint16_t>(cOutputActual);
    uint16_t repeatimes = static_cast<uint16_t>(wFullBatchCount / concurrencyCount);
    uint32_t wRemain = static_cast<uint32_t>(wArgmaxActual) - repeatimes * wProBatchSize * concurrencyCount;
    uint32_t wRemainBatch = wRemain / wProBatchSize;
    uint16_t wRemainTail = static_cast<uint16_t>(wRemain - wRemainBatch * wProBatchSize);
    uint32_t wBatchConcurrency = static_cast<uint32_t>(wProBatchSize) | (static_cast<uint32_t>(concurrencyCount) << 16);

    uint32_t magicW = 0;
    uint32_t shiftW = 0;
    GetUintDivMagicAndShift<uint32_t>(magicW, shiftW, static_cast<uint32_t>(wInput));
    DivMagic divC = PrecomputeDiv(static_cast<uint32_t>(cOutputActual));
    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        uint32_t nGradOffset = nIdx * hArgmaxActual * wArgmaxActual * cStride;
        uint32_t nArgmaxOffset = nIdx * hArgmaxActual * wArgmaxActual * cActual;
        uint32_t nOutputOffset = nIdx * hOutputActual * wOutputActual * cStride;
        for (uint16_t hIdx = 0; hIdx < hArgmaxActual; ++hIdx) {
            uint32_t hGradBase = hIdx * wArgmaxActual * cStride + nGradOffset;
            uint32_t hArgmaxBase = hIdx * wArgmaxActual * cActual + nArgmaxOffset;
            MultipleLineWProcessVf<T, INDEX_T, IS_CHECK_RANGE>(
                yAddr + nOutputOffset, gradAddr + hGradBase, argmaxAddr + hArgmaxBase,
                (__ubuf__ int32_t*)helpBuf_.template Get<int32_t>().GetPhyAddr(), wInput, wOutputActual, hOutputActual,
                curHIndex, curWIndex, cOutputAligned, cOutputActual, wBatchConcurrency, repeatimes, wRemainBatch,
                wRemainTail, magicW, shiftW, divC.magic, divC.shift);
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MultipleLineWProcessVf(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                               __ubuf__ INDEX_T* argmaxAddr, __ubuf__ int32_t* helpAddr, int32_t wInput,
                                               int32_t wOutputActual, int32_t hOutputActual, int32_t curHIndex,
                                               int32_t curWIndex, int32_t cOutputAligned, int32_t cOutputActual,
                                               uint32_t wBatchConcurrency, uint16_t repeatimes, uint32_t wRemainBatch,
                                               uint16_t wRemainTail, uint32_t magicW, uint32_t shiftW,
                                               uint32_t divCMagic, int16_t divCShift)
{
    uint16_t wProBatchSize = static_cast<uint16_t>(wBatchConcurrency & 0xffffU);
    uint16_t concurrencyCount = static_cast<uint16_t>(wBatchConcurrency >> 16);
    int32_t cStride = cOutputAligned;
    int32_t cActual = cOutputActual;
    uint32_t mask0 = concurrencyCount * cOutputActual;
    uint32_t mask1 = wRemainBatch * cOutputActual;
    uint32_t mask2 = cOutputActual;
    int32_t wBatchStride = wProBatchSize * cStride;
    int32_t wBatchStrideA = wProBatchSize * cActual;
    int32_t baseOffset = -(curHIndex * wOutputActual + curWIndex) * cStride;
    // The wrapper passes pointers already offset to the current row.
    constexpr uint32_t hGradBase = 0;
    constexpr uint32_t hArgmaxBase = 0;
    DivMagic divC = {divCMagic, divCShift};
    AscendC::Reg::RegTensor<int32_t> hLowerReg;
    AscendC::Reg::RegTensor<int32_t> wLowerReg;
    AscendC::Reg::RegTensor<int32_t> hUpperReg;
    AscendC::Reg::RegTensor<int32_t> wUpperReg;
    if constexpr (IS_CHECK_RANGE == 1) {
        AscendC::Reg::Duplicate(hLowerReg, int32_t(curHIndex));
        AscendC::Reg::Duplicate(wLowerReg, int32_t(curWIndex));
        AscendC::Reg::Duplicate(hUpperReg, int32_t(hOutputActual + curHIndex));
        AscendC::Reg::Duplicate(wUpperReg, int32_t(wOutputActual + curWIndex));
    }
    AscendC::Reg::RegTensor<uint32_t> magicWReg;
    AscendC::Reg::Duplicate(magicWReg, magicW);
    AscendC::Reg::RegTensor<int32_t> initialRegIndex;
    GenPattern2D(initialRegIndex, helpAddr, wBatchStride, cOutputActual, 0);
    AscendC::Reg::RegTensor<int32_t> initialRegArgmax;
    GenPattern2D(initialRegArgmax, helpAddr, wBatchStrideA, cOutputActual, 1);
    AscendC::Reg::RegTensor<int32_t> initialRegIndexOne;
    PoolGradCommon::Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initialRegIndexOne, 1, 1);
    AscendC::Reg::RegTensor<int32_t> parallelRegIndex;
    AscendC::Reg::RegTensor<int32_t> parallelRegArgmax;
    AscendC::Reg::RegTensor<int32_t> cIncReg;
    AscendC::Reg::Arange(cIncReg, 0);
    AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; ++wRepeatIdx) {
        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
            uint32_t offset = (wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize) * cStride + hGradBase;
            uint32_t argmaxOff = (wBatchIdx + wRepeatIdx * concurrencyCount * wProBatchSize) * cActual + hArgmaxBase;
            AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset), allMaskU32);
            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff), allMaskU32);
            DoMulNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr, (__local_mem__ INDEX_T*)argmaxAddr,
                parallelRegArgmax, parallelRegIndex, mask0, magicWReg, static_cast<int16_t>(shiftW), wOutputActual,
                wInput, baseOffset, cOutputAligned, cOutputActual, divC, cIncReg, hLowerReg, wLowerReg, hUpperReg,
                wUpperReg);
        }
    }
    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
        uint32_t offset = (wBatchIdx + repeatimes * concurrencyCount * wProBatchSize) * cStride + hGradBase;
        uint32_t argmaxOff = (wBatchIdx + repeatimes * concurrencyCount * wProBatchSize) * cActual + hArgmaxBase;
        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset), allMaskU32);
        AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff), allMaskU32);
        DoMulNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
            (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr, (__local_mem__ INDEX_T*)argmaxAddr,
            parallelRegArgmax, parallelRegIndex, mask1, magicWReg, static_cast<int16_t>(shiftW), wOutputActual, wInput,
            baseOffset, cOutputAligned, cOutputActual, divC, cIncReg, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
    }
    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
        uint32_t offset = (wBatchIdx + wRemainBatch * wProBatchSize + repeatimes * concurrencyCount * wProBatchSize) *
                              cStride +
                          hGradBase;
        uint32_t argmaxOff = (wBatchIdx + wRemainBatch * wProBatchSize +
                              repeatimes * concurrencyCount * wProBatchSize) *
                                 cActual +
                             hArgmaxBase;
        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, static_cast<int32_t>(offset), allMaskU32);
        AscendC::Reg::Adds(parallelRegArgmax, initialRegIndexOne, static_cast<int32_t>(argmaxOff), allMaskU32);
        DoSingleNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
            (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr, (__local_mem__ INDEX_T*)argmaxAddr,
            parallelRegArgmax, parallelRegIndex, mask2, magicWReg, static_cast<int16_t>(shiftW), wOutputActual, wInput,
            baseOffset, cOutputAligned, cIncReg, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
    }
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MultipleLineHwProcessVf(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                                __ubuf__ INDEX_T* argmaxAddr, __ubuf__ int32_t* helpAddr,
                                                int32_t wInput, int32_t wOutputActual, int32_t hOutputActual,
                                                int32_t curHIndex, int32_t curWIndex, int32_t cOutputAligned,
                                                int32_t cOutputActual, int32_t wArgmaxActual, uint32_t hwProBatchSize,
                                                uint32_t wFullBatchCount, uint32_t hBlockConcurrency,
                                                uint32_t hRemainBatchTail, uint32_t magicW, uint32_t shiftW,
                                                uint32_t divCMagic, int16_t divCShift, uint16_t nIdx);

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>::MultipleLineHwProcessVF(
    __ubuf__ float* yAddr, __ubuf__ T* gradAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    constexpr int32_t V_REG_SIZE = platform::GetVRegSize();

    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    int32_t wOutputActual = static_cast<int32_t>(wOutputActual_);
    int32_t hOutputActual = static_cast<int32_t>(hOutputActual_);
    int32_t curHIndex = static_cast<int32_t>(hAxisIndex_ * tilingData_.base.hOutputInner);
    int32_t curWIndex = static_cast<int32_t>(wAxisIndex_ * tilingData_.base.wOutputInner);
    int32_t cOutputAligned = static_cast<int32_t>(cAligned_);
    int32_t cOutputActual = static_cast<int32_t>(cOutputActual_);

    uint16_t nOutputActual = static_cast<uint16_t>(nOutputActual_);
    uint16_t hArgmaxActual = static_cast<uint16_t>(hArgmaxActual_);
    int32_t wArgmaxActual = static_cast<int32_t>(wArgmaxActual_);
    uint16_t hProBatchSize = static_cast<uint16_t>(tilingData_.base.hProBatchSize);
    hProBatchSize = (hProBatchSize > hArgmaxActual) ? hArgmaxActual : hProBatchSize;
    uint16_t wProBatchSize = static_cast<uint16_t>(tilingData_.base.wProBatchSize);
    wProBatchSize = (wProBatchSize > wArgmaxActual) ? static_cast<uint16_t>(wArgmaxActual) : wProBatchSize;

    uint32_t wFullBatchCount = static_cast<uint32_t>(wArgmaxActual) / wProBatchSize;
    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t computeSizeFP32 = V_REG_SIZE / sizeof(float);
    uint16_t concurrencyCount = computeSizeFP32 / static_cast<uint16_t>(cOutputActual);
    uint16_t hConcurrentCount = concurrencyCount / static_cast<uint16_t>(wFullBatchCount);
    uint16_t blockConcurrentCount = hFullBatchCount / hConcurrentCount;
    uint16_t hRemain = hArgmaxActual - blockConcurrentCount * hConcurrentCount * hProBatchSize;
    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;
    uint32_t hwProBatchSize = static_cast<uint32_t>(hProBatchSize) | (static_cast<uint32_t>(wProBatchSize) << 16);
    uint32_t hBlockConcurrency = static_cast<uint32_t>(hConcurrentCount) |
                                 (static_cast<uint32_t>(blockConcurrentCount) << 16);
    uint32_t hRemainBatchTail = static_cast<uint32_t>(hRemainBatchCount) | (static_cast<uint32_t>(hRemainTail) << 16);

    uint32_t magicW = 0;
    uint32_t shiftW = 0;
    GetUintDivMagicAndShift<uint32_t>(magicW, shiftW, static_cast<uint32_t>(wInput));
    DivMagic divC = PrecomputeDiv(static_cast<uint32_t>(cOutputActual));
    for (uint16_t nIdx = 0; nIdx < nOutputActual; ++nIdx) {
        MultipleLineHwProcessVf<T, INDEX_T, IS_CHECK_RANGE>(
            yAddr, gradAddr, argmaxAddr, (__ubuf__ int32_t*)helpBuf_.template Get<int32_t>().GetPhyAddr(), wInput,
            wOutputActual, hOutputActual, curHIndex, curWIndex, cOutputAligned, cOutputActual, wArgmaxActual,
            hwProBatchSize, wFullBatchCount, hBlockConcurrency, hRemainBatchTail, magicW, shiftW, divC.magic,
            divC.shift, nIdx);
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MultipleLineHwProcessVf(__ubuf__ float* yAddr, __ubuf__ T* gradAddr,
                                                __ubuf__ INDEX_T* argmaxAddr, __ubuf__ int32_t* helpAddr,
                                                int32_t wInput, int32_t wOutputActual, int32_t hOutputActual,
                                                int32_t curHIndex, int32_t curWIndex, int32_t cOutputAligned,
                                                int32_t cOutputActual, int32_t wArgmaxActual, uint32_t hwProBatchSize,
                                                uint32_t wFullBatchCount, uint32_t hBlockConcurrency,
                                                uint32_t hRemainBatchTail, uint32_t magicW, uint32_t shiftW,
                                                uint32_t divCMagic, int16_t divCShift, uint16_t nIdx)
{
    uint16_t hProBatchSize = static_cast<uint16_t>(hwProBatchSize & 0xffffU);
    uint16_t wProBatchSize = static_cast<uint16_t>(hwProBatchSize >> 16);
    uint16_t hConcurrentCount = static_cast<uint16_t>(hBlockConcurrency & 0xffffU);
    uint16_t blockConcurrentCount = static_cast<uint16_t>(hBlockConcurrency >> 16);
    uint16_t hRemainBatchCount = static_cast<uint16_t>(hRemainBatchTail & 0xffffU);
    uint16_t hRemainTail = static_cast<uint16_t>(hRemainBatchTail >> 16);
    uint16_t wRemainTail = static_cast<uint16_t>(wArgmaxActual - wProBatchSize * wFullBatchCount);
    // Derive products here to keep the VF scalar argument set small.
    int32_t cStride = cOutputAligned;
    int32_t cActual = cOutputActual;
    uint16_t hArgmaxActual = blockConcurrentCount * hConcurrentCount * hProBatchSize +
                             hRemainBatchCount * hProBatchSize + hRemainTail;
    uint32_t nGradOffset = nIdx * hArgmaxActual * wArgmaxActual * cStride;
    uint32_t nArgmaxOffset = nIdx * hArgmaxActual * wArgmaxActual * cActual;
    uint32_t maskBlock = wFullBatchCount * hConcurrentCount * cOutputActual;
    uint32_t maskRemainBatch = wFullBatchCount * hRemainBatchCount * cOutputActual;
    uint32_t maskRemainTail = wFullBatchCount * cOutputActual;
    uint32_t blockOne = hConcurrentCount * cOutputActual;
    uint32_t remainBatchOne = hRemainBatchCount * cOutputActual;
    uint32_t remainTailOne = cOutputActual;
    int32_t hBatchStride = hProBatchSize * wArgmaxActual * cStride;
    int32_t wBatchStride = wProBatchSize * cStride;
    int32_t hBatchStrideA = hProBatchSize * wArgmaxActual * cActual;
    int32_t wBatchStrideA = wProBatchSize * cActual;
    int32_t baseOffset = nIdx * hOutputActual * wOutputActual * cStride -
                         (curHIndex * wOutputActual + curWIndex) * cStride;
    DivMagic divC = {divCMagic, divCShift};
    AscendC::Reg::RegTensor<int32_t> hLowerReg;
    AscendC::Reg::RegTensor<int32_t> wLowerReg;
    AscendC::Reg::RegTensor<int32_t> hUpperReg;
    AscendC::Reg::RegTensor<int32_t> wUpperReg;
    if constexpr (IS_CHECK_RANGE == 1) {
        AscendC::Reg::Duplicate(hLowerReg, int32_t(curHIndex));
        AscendC::Reg::Duplicate(wLowerReg, int32_t(curWIndex));
        AscendC::Reg::Duplicate(hUpperReg, int32_t(hOutputActual + curHIndex));
        AscendC::Reg::Duplicate(wUpperReg, int32_t(wOutputActual + curWIndex));
    }
    AscendC::Reg::RegTensor<uint32_t> magicWReg;
    AscendC::Reg::Duplicate(magicWReg, magicW);
    AscendC::Reg::RegTensor<int32_t> initialRegIndex;
    GenPattern3D(initialRegIndex, helpAddr, hBatchStride, wBatchStride, static_cast<int32_t>(wFullBatchCount),
                 cOutputActual, 0);
    AscendC::Reg::RegTensor<int32_t> initialRegArgmax;
    GenPattern3D(initialRegArgmax, helpAddr, hBatchStrideA, wBatchStrideA, static_cast<int32_t>(wFullBatchCount),
                 cOutputActual, 1);
    AscendC::Reg::RegTensor<int32_t> initialRegIndexOne;
    GenPattern2D(initialRegIndexOne, helpAddr, hBatchStride, cOutputActual, SLOT_PATTERN_ONE);
    AscendC::Reg::RegTensor<int32_t> initialRegArgmaxOne;
    GenPattern2D(initialRegArgmaxOne, helpAddr, hBatchStrideA, cOutputActual, SLOT_PATTERN_ONE + 1);
    AscendC::Reg::RegTensor<int32_t> initialRegIndexWBatch;
    GenPattern2D(initialRegIndexWBatch, helpAddr, wBatchStride, cOutputActual, SLOT_PATTERN_W_BATCH);
    AscendC::Reg::RegTensor<int32_t> initialRegArgmaxWBatch;
    GenPattern2D(initialRegArgmaxWBatch, helpAddr, wBatchStrideA, cOutputActual, SLOT_PATTERN_W_BATCH + 1);
    AscendC::Reg::RegTensor<int32_t> parallelRegIndex;
    AscendC::Reg::RegTensor<int32_t> parallelRegArgmax;
    AscendC::Reg::RegTensor<int32_t> cIncReg;
    AscendC::Reg::Arange(cIncReg, 0);
    AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

    for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; ++hIdx) {
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; ++hProBatchIdx) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
                uint32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                   hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                      cStride +
                                  nGradOffset;
                uint32_t argmaxOff = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                      hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                         cActual +
                                     nArgmaxOffset;
                AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset), allMaskU32);
                AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff), allMaskU32);
                DoMulNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                    (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr, (__local_mem__ INDEX_T*)argmaxAddr,
                    parallelRegArgmax, parallelRegIndex, maskBlock, magicWReg, static_cast<int16_t>(shiftW),
                    wOutputActual, wInput, baseOffset, cOutputAligned, cOutputActual, divC, cIncReg, hLowerReg,
                    wLowerReg, hUpperReg, wUpperReg);
            }
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
                uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                   hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                      cStride +
                                  nGradOffset;
                uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                      hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount) *
                                         cActual +
                                     nArgmaxOffset;
                AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, static_cast<int32_t>(offset), allMaskU32);
                AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxOne, static_cast<int32_t>(argmaxOff), allMaskU32);
                DoMulNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                    (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr, (__local_mem__ INDEX_T*)argmaxAddr,
                    parallelRegArgmax, parallelRegIndex, blockOne, magicWReg, static_cast<int16_t>(shiftW),
                    wOutputActual, wInput, baseOffset, cOutputAligned, cOutputActual, divC, cIncReg, hLowerReg,
                    wLowerReg, hUpperReg, wUpperReg);
            }
        }
    }
    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainBatchCount; ++hProBatchIdx) {
        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
            uint32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                               blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                  cStride +
                              nGradOffset;
            uint32_t argmaxOff = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                  blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                     cActual +
                                 nArgmaxOffset;
            AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, static_cast<int32_t>(offset), allMaskU32);
            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmax, static_cast<int32_t>(argmaxOff), allMaskU32);
            DoMulNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr, (__local_mem__ INDEX_T*)argmaxAddr,
                parallelRegArgmax, parallelRegIndex, maskRemainBatch, magicWReg, static_cast<int16_t>(shiftW),
                wOutputActual, wInput, baseOffset, cOutputAligned, cOutputActual, divC, cIncReg, hLowerReg, wLowerReg,
                hUpperReg, wUpperReg);
        }
        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
            uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                               blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                  cStride +
                              nGradOffset;
            uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                  blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                     cActual +
                                 nArgmaxOffset;
            AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, static_cast<int32_t>(offset), allMaskU32);
            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxOne, static_cast<int32_t>(argmaxOff), allMaskU32);
            DoMulNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr, (__local_mem__ INDEX_T*)argmaxAddr,
                parallelRegArgmax, parallelRegIndex, remainBatchOne, magicWReg, static_cast<int16_t>(shiftW),
                wOutputActual, wInput, baseOffset, cOutputAligned, cOutputActual, divC, cIncReg, hLowerReg, wLowerReg,
                hUpperReg, wUpperReg);
        }
    }
    for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; ++hProBatchIdx) {
        for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; ++wBatchIdx) {
            uint32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                               hRemainBatchCount * hProBatchSize * wArgmaxActual +
                               blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                  cStride +
                              nGradOffset;
            uint32_t argmaxOff = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                  hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                  blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                     cActual +
                                 nArgmaxOffset;
            AscendC::Reg::Adds(parallelRegIndex, initialRegIndexWBatch, static_cast<int32_t>(offset), allMaskU32);
            AscendC::Reg::Adds(parallelRegArgmax, initialRegArgmaxWBatch, static_cast<int32_t>(argmaxOff), allMaskU32);
            DoMulNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr, (__local_mem__ INDEX_T*)argmaxAddr,
                parallelRegArgmax, parallelRegIndex, maskRemainTail, magicWReg, static_cast<int16_t>(shiftW),
                wOutputActual, wInput, baseOffset, cOutputAligned, cOutputActual, divC, cIncReg, hLowerReg, wLowerReg,
                hUpperReg, wUpperReg);
        }
        for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; ++wBatchIdx) {
            uint32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                               hRemainBatchCount * hProBatchSize * wArgmaxActual +
                               blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                  cStride +
                              nGradOffset;
            uint32_t argmaxOff = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                  hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                  blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual) *
                                     cActual +
                                 nArgmaxOffset;
            AscendC::Reg::RegTensor<int32_t> initialRegIndexTailOne;
            PoolGradCommon::Gen2DIndexOne((AscendC::Reg::RegTensor<int32_t>&)initialRegIndexTailOne, 1, 1);
            AscendC::Reg::Adds(parallelRegIndex, initialRegIndexTailOne, static_cast<int32_t>(offset), allMaskU32);
            AscendC::Reg::Adds(parallelRegArgmax, initialRegIndexTailOne, static_cast<int32_t>(argmaxOff), allMaskU32);
            DoSingleNCNhwcFastDiv<T, INDEX_T, IS_CHECK_RANGE>(
                (__local_mem__ ComputeType*)yAddr, (__local_mem__ T*)gradAddr, (__local_mem__ INDEX_T*)argmaxAddr,
                parallelRegArgmax, parallelRegIndex, remainTailOne, magicWReg, static_cast<int16_t>(shiftW),
                wOutputActual, wInput, baseOffset, cOutputAligned, cIncReg, hLowerReg, wLowerReg, hUpperReg, wUpperReg);
        }
    }
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

} // namespace MaxPoolV3GradNHWCNameSpace

#endif // MAX_POOL_V3_GRAD_NHWC_IMPL_SCATTER_H_
