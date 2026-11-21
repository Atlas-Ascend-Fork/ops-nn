/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL_GRAD_NCHW_BACKWARD_BASE_H_
#define MAX_POOL_GRAD_NCHW_BACKWARD_BASE_H_

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool_grad_struct.h"
#include "../inc/platform.h"
#include "max_pool_grad_with_argmax_base_common.h"
#include "max_pool_grad_nchw_fields_common.h"
#include "max_pool_grad_nchw_scatter_common.h"

namespace MaxPoolGradNCHWBackwardBaseNameSpace {
using namespace AscendC;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;

using ::PEnd;
using ::PStart;
using MaxPoolGradNCHWNameSpace::DoMulNCNchw;
using MaxPoolGradNCHWNameSpace::DoMulNCNchwFastDiv;
using MaxPoolGradNCHWNameSpace::DoSingleNCNchw;
using MaxPoolGradNCHWNameSpace::DoSingleNCNchwFastDiv;
using MaxPoolGradNCHWNameSpace::Gen3DIndexOneNchw;
using MaxPoolGradNCHWNameSpace::GenInitial3DIndicesNchw;

constexpr int32_t BK_BUFFER_NUM = 2;
constexpr int64_t RATIO = 2;
constexpr uint32_t DOUBLE = 2;
constexpr uint32_t NCHW_INT_INDEX_LANES = 64;

template <typename T1, const uint32_t IS_CHECK_RANGE>
class MaxPoolGradNCHWBackwardBase : protected MaxPoolGradNCHWCommonNameSpace::MaxPoolGradNchwCommonFields {
public:
    __aicore__ inline MaxPoolGradNCHWBackwardBase() {}

protected:
    __aicore__ inline void ParseTilingData(const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData);
    __aicore__ inline void ScalarCompute(int64_t loopNum);
    __aicore__ inline void CopyInGrad();
    __aicore__ inline void CopyOut();
    __aicore__ inline void ProcessNoArgmaxBlock();
    __aicore__ inline void BackwardCompute(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                           __local_mem__ int32_t* argmaxAddr);
    __aicore__ inline void BackwardComputeDispatch(__local_mem__ int32_t* argmaxAddr);
    __aicore__ inline void BackwardComputeInt(__local_mem__ T1* yAddr, __local_mem__ T1* gradAddr,
                                              __local_mem__ int32_t* argmaxAddr);
    template <const bool IS_OVERLAP>
    __simd_vf__ inline void BackwardComputeIntSimdVf(__ubuf__ T1* yAddr, __ubuf__ T1* gradAddr,
                                                     __ubuf__ int32_t* argmaxAddr, uint16_t hArgmax, uint16_t wArgmax,
                                                     int32_t wArgmaxAligned, uint16_t highAxis, uint16_t wPro,
                                                     uint16_t rowResidues, int32_t wFull, int32_t wOutAligned,
                                                     int32_t wOutActual, int32_t hOutActual, int32_t curH, int32_t curW,
                                                     int32_t ncStride, uint32_t wDivMagic, uint32_t wDivShift);
    template <const bool IS_OVERLAP>
    __aicore__ inline void singleLineProcessVF(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                               __local_mem__ int32_t* argmaxAddr, __local_mem__ uint32_t* helpAddr);
    template <const bool IS_OVERLAP>
    __aicore__ inline void multipleLineHwProcessVF(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                   __local_mem__ int32_t* argmaxAddr, __local_mem__ uint32_t* helpAddr);
    template <const bool IS_OVERLAP>
    __aicore__ inline void multipleLineProcessVF2(__local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr,
                                                  __local_mem__ int32_t* argmaxAddr, __local_mem__ uint32_t* helpAddr);

    TQue<QuePosition::VECIN, BK_BUFFER_NUM> gradQue_;
    TQue<QuePosition::VECOUT, BK_BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> argmaxBuf_;
    TBuf<QuePosition::VECCALC> helpBuf_;

    GlobalTensor<T1> gradGm_;
    GlobalTensor<T1> yGm_;

    int64_t inputBufferSize_ = 1;
    int64_t hArgmaxActualStart_ = 0;
    int64_t wArgmaxActualStart_ = 0;
    bool isOverlap_ = false;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();
    constexpr static int64_t MAX_DATA_NUM_IN_ONE_BLOCK =
        BLOCK_SIZE / sizeof(T1) > BLOCK_SIZE / sizeof(int32_t) ? BLOCK_SIZE / sizeof(T1) : BLOCK_SIZE / sizeof(int32_t);
    constexpr static uint16_t vlArgmax_ = platform::GetVRegSize() / sizeof(int32_t);
};

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::ParseTilingData(
    const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData)
{
    ParseCommonTilingData(tilingData);
    inputBufferSize_ = tilingData.inputBufferSize;
    argmaxPlaneSize_ = hArgmax_ * wArgmax_;
    isOverlap_ = ((kernelH_ - 1) * dilationH_ + 1 > strideH_) || ((kernelW_ - 1) * dilationW_ + 1 > strideW_);
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::ScalarCompute(int64_t loopNum)
{
    int64_t hArgmaxActualStart = 0;
    int64_t wArgmaxActualStart = 0;
    ScalarComputeCommon(loopNum, MAX_DATA_NUM_IN_ONE_BLOCK, hArgmaxActualStart, wArgmaxActualStart);
    hArgmaxActualStart_ = hArgmaxActualStart;
    wArgmaxActualStart_ = wArgmaxActualStart;
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::CopyInGrad()
{
    LocalTensor<T1> gradLocal = gradQue_.AllocTensor<T1>();
    int64_t argmaxGmOffset = highAxisArgmaxOffset_ + hAxisArgmaxOffset_ + wAxisArgmaxOffset_;
    DataCopyPadExtParams<T1> paramsT1 = {false, 0, 0, 0};
    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop1SrcStride = argmaxPlaneSize_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = hArgmaxActual_ * wArgmaxAligned_ * sizeof(T1);
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop2DstStride = 0;
    SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(hArgmaxActual_),
                                        static_cast<uint32_t>(wArgmaxActual_ * sizeof(T1)),
                                        static_cast<uint32_t>((wArgmax_ - wArgmaxActual_) * sizeof(T1)), 0, 0};
    DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyOutParamT1, paramsT1);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    gradQue_.EnQue(gradLocal);
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::CopyOut()
{
    LocalTensor<T1> yLocal = outputQue_.DeQue<T1>();
    int64_t outputPlaneSize = hOutput_ * wOutput_;
    int64_t highOutputAxisOffset = highAxisIndex_ * highAxisInner_ * outputPlaneSize;
    int64_t hOutputAxisOffset = hAxisIndex_ * hOutputInner_ * wOutput_;
    int64_t wOutputAxisOffset = wAxisIndex_ * wOutputInner_;
    int64_t outputGmOffset = highOutputAxisOffset + hOutputAxisOffset + wOutputAxisOffset;

    LoopModeParams loopModeParamsT1;
    loopModeParamsT1.loop1Size = highAxisActual_;
    loopModeParamsT1.loop1SrcStride = hOutputActual_ * wOutputAligned_ * sizeof(T1);
    loopModeParamsT1.loop1DstStride = outputPlaneSize * sizeof(T1);
    loopModeParamsT1.loop2Size = 1;
    loopModeParamsT1.loop2SrcStride = 0;
    loopModeParamsT1.loop2DstStride = 0;
    SetLoopModePara(loopModeParamsT1, DataCopyMVType::UB_TO_OUT);
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(hOutputActual_),
                                        static_cast<uint32_t>(wOutputActual_ * sizeof(T1)), 0,
                                        static_cast<uint32_t>((wOutput_ - wOutputActual_) * sizeof(T1)), 0};
    DataCopyPad(yGm_[outputGmOffset], yLocal, copyOutParamT1);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    outputQue_.FreeTensor(yLocal);
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::ProcessNoArgmaxBlock()
{
    uint32_t calcCount = static_cast<uint32_t>(outputBufferSize_) / sizeof(T1);
    LocalTensor<T1> yLocal = outputQue_.AllocTensor<T1>();
    Duplicate(yLocal, T1(0), calcCount);
    outputQue_.EnQue(yLocal);
    CopyOut();
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::BackwardCompute(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ int32_t* argmaxAddr)
{
    uint32_t wConcurrentCount = wArgmaxActual_ / curWProBatchSize_;
    uint32_t hConcurrentCount = hArgmaxActual_ / curHProBatchSize_;
    LocalTensor<uint32_t> helpTensor = helpBuf_.Get<uint32_t>();
    __local_mem__ uint32_t* helpAddr = (__local_mem__ uint32_t*)helpTensor.GetPhyAddr();
    if (isOverlap_) {
        if (wConcurrentCount * DOUBLE * sizeof(int32_t) > V_REG_SIZE) {
            singleLineProcessVF<true>(yAddr, gradAddr, argmaxAddr, helpAddr);
        } else if (wConcurrentCount * hConcurrentCount * DOUBLE * sizeof(int32_t) > V_REG_SIZE) {
            multipleLineHwProcessVF<true>(yAddr, gradAddr, argmaxAddr, helpAddr);
        } else {
            multipleLineProcessVF2<true>(yAddr, gradAddr, argmaxAddr, helpAddr);
        }
    } else {
        if (wConcurrentCount * DOUBLE * sizeof(int32_t) > V_REG_SIZE) {
            singleLineProcessVF<false>(yAddr, gradAddr, argmaxAddr, helpAddr);
        } else if (wConcurrentCount * hConcurrentCount * DOUBLE * sizeof(int32_t) > V_REG_SIZE) {
            multipleLineHwProcessVF<false>(yAddr, gradAddr, argmaxAddr, helpAddr);
        } else {
            multipleLineProcessVF2<false>(yAddr, gradAddr, argmaxAddr, helpAddr);
        }
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::BackwardComputeDispatch(
    __local_mem__ int32_t* argmaxAddr)
{
    uint32_t calCount = static_cast<uint32_t>(highAxisActual_ * hOutputActual_ * wOutputAligned_);

    if constexpr (std::is_integral<T1>::value) {
        LocalTensor<T1> yLocal = outputQue_.template AllocTensor<T1>();
        Duplicate(yLocal, T1(0), calCount);
        LocalTensor<T1> gradLocal = gradQue_.template DeQue<T1>();
        __local_mem__ T1* yAddr = (__local_mem__ T1*)yLocal.GetPhyAddr();
        __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
        BackwardComputeInt(yAddr, gradAddr, argmaxAddr);
        outputQue_.EnQue(yLocal);
        gradQue_.FreeTensor(gradLocal);
    } else {
        LocalTensor<computeType> yLocal = outputQue_.template AllocTensor<computeType>();
        Duplicate(yLocal, computeType(0), calCount);
        LocalTensor<T1> gradLocal = gradQue_.template DeQue<T1>();
        __local_mem__ computeType* yAddr = (__local_mem__ computeType*)yLocal.GetPhyAddr();
        __local_mem__ T1* gradAddr = (__local_mem__ T1*)gradLocal.GetPhyAddr();
        BackwardCompute(yAddr, gradAddr, argmaxAddr);
        if constexpr (!std::is_same<T1, float>::value) {
            Cast(yLocal.ReinterpretCast<T1>(), yLocal, RoundMode::CAST_RINT, calCount);
        }
        outputQue_.EnQue(yLocal);
        gradQue_.FreeTensor(gradLocal);
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
template <const bool IS_OVERLAP>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::singleLineProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ int32_t* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    int64_t wOutput = wOutput_;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;
    int64_t wArgmaxActual = wArgmaxActual_;
    int64_t wArgmaxAligned = wArgmaxAligned_;
    uint16_t hArgmaxActual = hArgmaxActual_;
    uint16_t wProBatchSize = curWProBatchSize_;
    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t computeSizeint32_t = V_REG_SIZE / sizeof(int32_t);
    uint16_t repeatimes = wFullBatchCount / computeSizeint32_t;
    uint16_t wRemain = wArgmaxActual - repeatimes * wProBatchSize * computeSizeint32_t;
    uint32_t wRemainBatchCount = wRemain / wProBatchSize;
    uint16_t wRemainTail = wRemain % wProBatchSize;
    uint32_t one = 1;
    uint32_t all = computeSizeint32_t;

    uint32_t magic = 0;
    uint32_t shift = 0;
    GetUintDivMagicAndShift<uint32_t>(magic, shift, static_cast<uint32_t>(wOutput));

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, int32_t(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<int32_t> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, int32_t(wOutput));

        AscendC::Reg::RegTensor<uint32_t> magicReg;
        AscendC::Reg::Duplicate(magicReg, magic);

        AscendC::Reg::RegTensor<uint32_t> initialRegGrad;
        AscendC::Reg::RegTensor<uint32_t> initialRegIndex;
        AscendC::Reg::RegTensor<uint32_t> parallelRegGrad;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

        PoolGradCommon::GenInitial1DIndices<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initialRegGrad, wProBatchSize);
        PoolGradCommon::GenInitial1DIndices<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t highIndexOffset = highIdx * hArgmaxActual * wArgmaxActual;
            uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
            for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
                for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatimes; wRepeatIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        uint32_t offset = (wBatchIdx + wRepeatIdx * computeSizeint32_t * wProBatchSize +
                                           hIdx * wArgmaxAligned + highArgmaxOffset);
                        uint32_t indexOffset = (wBatchIdx + wRepeatIdx * computeSizeint32_t * wProBatchSize +
                                                hIdx * wArgmaxActual + highIndexOffset);
                        AscendC::Reg::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                        DoSingleNCNchwFastDiv<T1, IS_CHECK_RANGE, IS_OVERLAP>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, all, magicReg,
                            static_cast<int16_t>(shift), curHIndex, curWIndex, wOutput, wOutputAligned,
                            highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    uint32_t offset = (wBatchIdx + repeatimes * computeSizeint32_t * wProBatchSize +
                                       hIdx * wArgmaxAligned + highArgmaxOffset);
                    uint32_t indexOffset = (wBatchIdx + repeatimes * computeSizeint32_t * wProBatchSize +
                                            hIdx * wArgmaxActual + highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                    DoSingleNCNchwFastDiv<T1, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, wRemainBatchCount, magicReg,
                        static_cast<int16_t>(shift), curHIndex, curWIndex, wOutput, wOutputAligned, highOutputOffset,
                        zeroConstReg, wMaxReg, hMaxReg);
                }
                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    uint32_t offset = (wBatchIdx + wRemainBatchCount * wProBatchSize +
                                       repeatimes * computeSizeint32_t * wProBatchSize + hIdx * wArgmaxAligned +
                                       highArgmaxOffset);
                    uint32_t indexOffset = (wBatchIdx + wRemainBatchCount * wProBatchSize +
                                            repeatimes * computeSizeint32_t * wProBatchSize + hIdx * wArgmaxActual +
                                            highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                    DoSingleNCNchwFastDiv<T1, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, one, magicReg,
                        static_cast<int16_t>(shift), curHIndex, curWIndex, wOutput, wOutputAligned, highOutputOffset,
                        zeroConstReg, wMaxReg, hMaxReg);
                }
            }
        }
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
template <const bool IS_OVERLAP>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::multipleLineHwProcessVF(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ int32_t* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    int64_t wOutput = wOutput_;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    uint16_t highAxisActual = static_cast<uint16_t>(highAxisActual_);
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;
    int64_t wArgmaxAligned = wArgmaxAligned_;
    int64_t wArgmaxActual = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;

    uint16_t hConcurrentCount = V_REG_SIZE / (wFullBatchCount * sizeof(int32_t));

    uint16_t blockConcurrentCount = hFullBatchCount / hConcurrentCount;
    uint16_t hRemain = hArgmaxActual - blockConcurrentCount * hConcurrentCount * hProBatchSize;

    uint16_t hRemainBatchCount = hRemain / hProBatchSize;
    uint16_t hRemainTail = hRemain - hRemainBatchCount * hProBatchSize;

    uint32_t blockOne = 1 * hConcurrentCount;
    uint32_t remainBatchOne = 1 * hRemainBatchCount;
    uint32_t remainTailOne = 1;
    uint32_t maskBlock = wFullBatchCount * hConcurrentCount;
    uint32_t maskRemainBatch = wFullBatchCount * hRemainBatchCount;
    uint32_t maskRemainTail = wFullBatchCount;

    uint32_t magic = 0;
    uint32_t shift = 0;
    GetUintDivMagicAndShift<uint32_t>(magic, shift, static_cast<uint32_t>(wOutput));

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, int32_t(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<int32_t> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, int32_t(wOutput));

        AscendC::Reg::RegTensor<uint32_t> magicReg;
        AscendC::Reg::Duplicate(magicReg, magic);

        AscendC::Reg::RegTensor<uint32_t> initialRegGrad;
        AscendC::Reg::RegTensor<uint32_t> initialRegGradOne;
        AscendC::Reg::RegTensor<uint32_t> initialRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initialRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegGrad;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        PoolGradCommon::GenInitial2DIndices<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initialRegGrad, wProBatchSize,
                                                     hProBatchSize, wArgmaxAligned, wFullBatchCount);
        PoolGradCommon::Gen2DIndexOne<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initialRegGradOne, hProBatchSize,
                                               wArgmaxAligned);
        PoolGradCommon::GenInitial2DIndices<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initialRegIndex, wProBatchSize,
                                                     hProBatchSize, wArgmaxActual, wFullBatchCount);
        PoolGradCommon::Gen2DIndexOne<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initialRegIndexOne, hProBatchSize,
                                               wArgmaxActual);

        for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
            uint32_t highArgmaxOffset = highIdx * hArgmaxActual * wArgmaxAligned;
            uint32_t highIndexOffset = highIdx * hArgmaxActual * wArgmaxActual;
            uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
            for (uint16_t hIdx = 0; hIdx < blockConcurrentCount; hIdx++) {
                for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                    for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                        int32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                          hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + highArgmaxOffset);
                        int32_t indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                               hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount +
                                               highIndexOffset);
                        AscendC::Reg::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                        DoSingleNCNchwFastDiv<T1, IS_CHECK_RANGE, IS_OVERLAP>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, maskBlock, magicReg,
                            static_cast<int16_t>(shift), curHIndex, curWIndex, wOutput, wOutputAligned,
                            highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }

                    for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                        int32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                          hIdx * wArgmaxAligned * hProBatchSize * hConcurrentCount + highArgmaxOffset);
                        int32_t indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                               hProBatchIdx * wArgmaxActual +
                                               hIdx * wArgmaxActual * hProBatchSize * hConcurrentCount +
                                               highIndexOffset);
                        AscendC::Reg::Adds(parallelRegGrad, initialRegGradOne, offset, allMaskU32);
                        AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, indexOffset, allMaskU32);
                        DoSingleNCNchwFastDiv<T1, IS_CHECK_RANGE, IS_OVERLAP>(
                            yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, blockOne, magicReg,
                            static_cast<int16_t>(shift), curHIndex, curWIndex, wOutput, wOutputAligned,
                            highOutputOffset, zeroConstReg, wMaxReg, hMaxReg);
                    }
                }
            }

            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    int32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                      blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                      highArgmaxOffset);
                    int32_t indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                           blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual +
                                           highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                    DoSingleNCNchwFastDiv<T1, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, maskRemainBatch, magicReg,
                        static_cast<int16_t>(shift), curHIndex, curWIndex, wOutput, wOutputAligned, highOutputOffset,
                        zeroConstReg, wMaxReg, hMaxReg);
                }

                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    int32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                      blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                      highArgmaxOffset);
                    int32_t indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                           blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual +
                                           highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initialRegGradOne, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, indexOffset, allMaskU32);
                    DoSingleNCNchwFastDiv<T1, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, remainBatchOne, magicReg,
                        static_cast<int16_t>(shift), curHIndex, curWIndex, wOutput, wOutputAligned, highOutputOffset,
                        zeroConstReg, wMaxReg, hMaxReg);
                }
            }

            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    int32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned +
                                      hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                                      blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                      highArgmaxOffset);
                    int32_t indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual +
                                           hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                           blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual +
                                           highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initialRegGrad, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, indexOffset, allMaskU32);
                    DoSingleNCNchwFastDiv<T1, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, maskRemainTail, magicReg,
                        static_cast<int16_t>(shift), curHIndex, curWIndex, wOutput, wOutputAligned, highOutputOffset,
                        zeroConstReg, wMaxReg, hMaxReg);
                }

                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    int32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                      hRemainBatchCount * hProBatchSize * wArgmaxAligned +
                                      blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxAligned +
                                      highArgmaxOffset);
                    int32_t indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                           hRemainBatchCount * hProBatchSize * wArgmaxActual +
                                           blockConcurrentCount * hConcurrentCount * hProBatchSize * wArgmaxActual +
                                           highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initialRegGradOne, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndexOne, indexOffset, allMaskU32);
                    DoSingleNCNchwFastDiv<T1, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, remainTailOne, magicReg,
                        static_cast<int16_t>(shift), curHIndex, curWIndex, wOutput, wOutputAligned, highOutputOffset,
                        zeroConstReg, wMaxReg, hMaxReg);
                }
            }
        }
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
template <const bool IS_OVERLAP>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::multipleLineProcessVF2(
    __local_mem__ computeType* yAddr, __local_mem__ T1* gradAddr, __local_mem__ int32_t* argmaxAddr,
    __local_mem__ uint32_t* helpAddr)
{
    int64_t wOutput = wOutput_;
    int64_t wOutputActual = wOutputActual_;
    int64_t wOutputAligned = wOutputAligned_;
    int64_t hOutputActual = hOutputActual_;
    int32_t highOutputPlaneActual = wOutputAligned * hOutputActual;
    int64_t highAxisActual = highAxisActual_;
    int64_t curHIndex = hAxisIndex_ * hOutputInner_;
    int64_t curWIndex = wAxisIndex_ * wOutputInner_;
    int64_t wArgmaxAligned = wArgmaxAligned_;
    int64_t wArgmaxActual = wArgmaxActual_;
    uint16_t hArgmaxActual = hArgmaxActual_;

    uint16_t hProBatchSize = curHProBatchSize_;
    uint16_t wProBatchSize = curWProBatchSize_;

    uint32_t wFullBatchCount = wArgmaxActual / wProBatchSize;
    uint16_t hFullBatchCount = hArgmaxActual / hProBatchSize;
    uint16_t wRemainTail = wArgmaxActual % wProBatchSize;
    uint32_t whFullBatchCount = wFullBatchCount * hFullBatchCount;

    uint16_t highConcurrentCount = V_REG_SIZE / (whFullBatchCount * sizeof(int32_t));

    uint16_t highBlockConcurrentCount = highAxisActual / highConcurrentCount;
    uint16_t highBlockRemainTail = highAxisActual - highBlockConcurrentCount * highConcurrentCount;

    uint16_t hRemainTail = hArgmaxActual - hFullBatchCount * hProBatchSize;

    uint32_t mask0 = highConcurrentCount * whFullBatchCount;
    uint32_t mask1 = highConcurrentCount * hFullBatchCount * 1;
    uint32_t mask2 = highConcurrentCount * 1 * wFullBatchCount;
    uint32_t mask3 = highConcurrentCount * 1 * 1;
    uint32_t mask4 = highBlockRemainTail * whFullBatchCount;
    uint32_t mask5 = highBlockRemainTail * hFullBatchCount * 1;
    uint32_t mask6 = highBlockRemainTail * 1 * wFullBatchCount;
    uint32_t mask7 = highBlockRemainTail * 1 * 1;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint32_t> initial3DRegGrad;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegGradOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegGrad;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegGradOne;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
        GenInitial3DIndicesNchw<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initial3DRegGrad, wProBatchSize,
                                         hProBatchSize, wArgmaxAligned, wFullBatchCount, hFullBatchCount,
                                         hArgmaxActual);
        Gen3DIndexOneNchw<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initial3DRegGradOne, hProBatchSize,
                                   wArgmaxAligned, hFullBatchCount, hArgmaxActual);
        PoolGradCommon::GenInitial2DIndices<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initial2DRegGrad, wProBatchSize,
                                                     hArgmaxActual, wArgmaxAligned, wFullBatchCount);
        PoolGradCommon::Gen2DIndexOne<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initial2DRegGradOne, hArgmaxActual,
                                               wArgmaxAligned);
        GenInitial3DIndicesNchw<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initial3DRegIndex, wProBatchSize,
                                         hProBatchSize, wArgmaxActual, wFullBatchCount, hFullBatchCount, hArgmaxActual);
        Gen3DIndexOneNchw<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initial3DRegIndexOne, hProBatchSize,
                                   wArgmaxActual, hFullBatchCount, hArgmaxActual);
        PoolGradCommon::GenInitial2DIndices<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initial2DRegIndex,
                                                     wProBatchSize, hArgmaxActual, wArgmaxActual, wFullBatchCount);
        PoolGradCommon::Gen2DIndexOne<int32_t>((AscendC::Reg::RegTensor<int32_t>&)initial2DRegIndexOne, hArgmaxActual,
                                               wArgmaxActual);
        AscendC::Reg::MaskReg allMaskStore = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::DataCopy(helpAddr, initial3DRegGrad, allMaskStore);
        AscendC::Reg::DataCopy(helpAddr + V_REG_SIZE / sizeof(uint32_t), initial3DRegGradOne, allMaskStore);
        AscendC::Reg::DataCopy(helpAddr + 2 * V_REG_SIZE / sizeof(uint32_t), initial2DRegGrad, allMaskStore);
        AscendC::Reg::DataCopy(helpAddr + 3 * V_REG_SIZE / sizeof(uint32_t), initial2DRegGradOne, allMaskStore);
        AscendC::Reg::DataCopy(helpAddr + 4 * V_REG_SIZE / sizeof(uint32_t), initial3DRegIndex, allMaskStore);
        AscendC::Reg::DataCopy(helpAddr + 5 * V_REG_SIZE / sizeof(uint32_t), initial3DRegIndexOne, allMaskStore);
        AscendC::Reg::DataCopy(helpAddr + 6 * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndex, allMaskStore);
        AscendC::Reg::DataCopy(helpAddr + 7 * V_REG_SIZE / sizeof(uint32_t), initial2DRegIndexOne, allMaskStore);
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, int32_t(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<int32_t> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, int32_t(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initial3DRegGrad;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegGradOne;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegGrad;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::DataCopy(initial3DRegGrad, helpAddr);
        AscendC::Reg::DataCopy(initial3DRegGradOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial3DRegIndex, helpAddr + 4 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial3DRegIndexOne, helpAddr + 5 * V_REG_SIZE / sizeof(uint32_t));

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
            uint32_t highIndexOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxActual;
            uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    int32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                    int32_t indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual + highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initial3DRegGrad, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, indexOffset, allMaskU32);
                    DoMulNCNchw<T1, int32_t, int32_t, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask0, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                        highOutputPlaneActual, whFullBatchCount);
                }

                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    int32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                      highArgmaxOffset);
                    int32_t indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                           highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initial3DRegGradOne, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, indexOffset, allMaskU32);
                    DoMulNCNchw<T1, int32_t, int32_t, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask1, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                        highOutputPlaneActual, hFullBatchCount);
                }
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, int32_t(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<int32_t> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, int32_t(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initial2DRegGrad;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegGradOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegGrad;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::DataCopy(initial2DRegGrad, helpAddr + 2 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegGradOne, helpAddr + 3 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndex, helpAddr + 6 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndexOne, helpAddr + 7 * V_REG_SIZE / sizeof(uint32_t));

        for (uint16_t highBlockIdx = 0; highBlockIdx < highBlockConcurrentCount; ++highBlockIdx) {
            uint32_t highArgmaxOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
            uint32_t highIndexOffset = highBlockIdx * highConcurrentCount * hArgmaxActual * wArgmaxActual;
            uint32_t highOutputOffset = highBlockIdx * highConcurrentCount * hOutputActual * wOutputAligned;
            for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    int32_t offset = (wBatchIdx + (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned +
                                      highArgmaxOffset);
                    int32_t indexOffset = (wBatchIdx +
                                           (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxActual +
                                           highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initial2DRegGrad, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, indexOffset, allMaskU32);
                    DoMulNCNchw<T1, int32_t, int32_t, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask2, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                        highOutputPlaneActual, wFullBatchCount);
                }

                for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                    int32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                      (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxAligned +
                                      highArgmaxOffset);
                    int32_t indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                           (hProBatchSize * hFullBatchCount + hProBatchIdx) * wArgmaxActual +
                                           highIndexOffset);
                    AscendC::Reg::Adds(parallelRegGrad, initial2DRegGradOne, offset, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, indexOffset, allMaskU32);
                    DoMulNCNchw<T1, int32_t, int32_t, IS_CHECK_RANGE, IS_OVERLAP>(
                        yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask3, wOutputConstReg,
                        curHIndex, curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg,
                        highOutputPlaneActual, 1);
                }
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, int32_t(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<int32_t> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, int32_t(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initial3DRegGrad;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegGradOne;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial3DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegGrad;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::DataCopy(initial3DRegGrad, helpAddr);
        AscendC::Reg::DataCopy(initial3DRegGradOne, helpAddr + V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial3DRegIndex, helpAddr + 4 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial3DRegIndexOne, helpAddr + 5 * V_REG_SIZE / sizeof(uint32_t));

        uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hProBatchSize; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                int32_t offset = (wBatchIdx + hProBatchIdx * wArgmaxAligned + highArgmaxOffset);
                int32_t indexOffset = (wBatchIdx + hProBatchIdx * wArgmaxActual + highIndexOffset);
                AscendC::Reg::Adds(parallelRegGrad, initial3DRegGrad, offset, allMaskU32);
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndex, indexOffset, allMaskU32);
                DoMulNCNchw<T1, int32_t, int32_t, IS_CHECK_RANGE, IS_OVERLAP>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask4, wOutputConstReg, curHIndex,
                    curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual,
                    whFullBatchCount);
            }

            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                int32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxAligned +
                                  highArgmaxOffset);
                int32_t indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount + hProBatchIdx * wArgmaxActual +
                                       highIndexOffset);
                AscendC::Reg::Adds(parallelRegGrad, initial3DRegGradOne, offset, allMaskU32);
                AscendC::Reg::Adds(parallelRegIndex, initial3DRegIndexOne, indexOffset, allMaskU32);
                DoMulNCNchw<T1, int32_t, int32_t, IS_CHECK_RANGE, IS_OVERLAP>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask5, wOutputConstReg, curHIndex,
                    curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual,
                    hFullBatchCount);
            }
        }
    }

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<int32_t> zeroConstReg;
        AscendC::Reg::RegTensor<int32_t> wMaxReg;
        AscendC::Reg::RegTensor<int32_t> hMaxReg;
        if constexpr (IS_CHECK_RANGE == 1) {
            AscendC::Reg::Duplicate(zeroConstReg, int32_t(0));
            AscendC::Reg::Duplicate(wMaxReg, int32_t(wOutputActual));
            AscendC::Reg::Duplicate(hMaxReg, int32_t(hOutputActual));
        }

        AscendC::Reg::RegTensor<int32_t> wOutputConstReg;
        AscendC::Reg::Duplicate(wOutputConstReg, int32_t(wOutput));

        AscendC::Reg::RegTensor<uint32_t> initial2DRegGrad;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegGradOne;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndex;
        AscendC::Reg::RegTensor<uint32_t> initial2DRegIndexOne;
        AscendC::Reg::RegTensor<uint32_t> parallelRegGrad;
        AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;

        AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::DataCopy(initial2DRegGrad, helpAddr + 2 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegGradOne, helpAddr + 3 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndex, helpAddr + 6 * V_REG_SIZE / sizeof(uint32_t));
        AscendC::Reg::DataCopy(initial2DRegIndexOne, helpAddr + 7 * V_REG_SIZE / sizeof(uint32_t));

        uint32_t highArgmaxOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxAligned;
        uint32_t highIndexOffset = highBlockConcurrentCount * highConcurrentCount * hArgmaxActual * wArgmaxActual;
        uint32_t highOutputOffset = highBlockConcurrentCount * highConcurrentCount * hOutputActual * wOutputAligned;
        for (uint16_t hProBatchIdx = 0; hProBatchIdx < hRemainTail; hProBatchIdx++) {
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                int32_t offset = (wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned +
                                  highArgmaxOffset);
                int32_t indexOffset = (wBatchIdx + (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxActual +
                                       highIndexOffset);
                AscendC::Reg::Adds(parallelRegGrad, initial2DRegGrad, offset, allMaskU32);
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndex, indexOffset, allMaskU32);
                DoMulNCNchw<T1, int32_t, int32_t, IS_CHECK_RANGE, IS_OVERLAP>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask6, wOutputConstReg, curHIndex,
                    curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual,
                    wFullBatchCount);
            }

            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                int32_t offset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                  (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxAligned + highArgmaxOffset);
                int32_t indexOffset = (wBatchIdx + wProBatchSize * wFullBatchCount +
                                       (hFullBatchCount * hProBatchSize + hProBatchIdx) * wArgmaxActual +
                                       highIndexOffset);
                AscendC::Reg::Adds(parallelRegGrad, initial2DRegGradOne, offset, allMaskU32);
                AscendC::Reg::Adds(parallelRegIndex, initial2DRegIndexOne, indexOffset, allMaskU32);
                DoMulNCNchw<T1, int32_t, int32_t, IS_CHECK_RANGE, IS_OVERLAP>(
                    yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad, mask7, wOutputConstReg, curHIndex,
                    curWIndex, wOutputAligned, highOutputOffset, zeroConstReg, wMaxReg, hMaxReg, highOutputPlaneActual,
                    1);
            }
        }
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::BackwardComputeInt(
    __local_mem__ T1* yAddr, __local_mem__ T1* gradAddr, __local_mem__ int32_t* argmaxAddr)
{
    // 整数累加按原 dtype 位宽回绕，等价于先 int64 累加再截断到原 dtype。
    const uint16_t hArgmax = static_cast<uint16_t>(hArgmaxActual_);
    const uint16_t wArgmax = static_cast<uint16_t>(wArgmaxActual_);
    const int64_t wArgmaxAligned = wArgmaxAligned_;
    const uint16_t highAxis = static_cast<uint16_t>(highAxisActual_);
    const uint16_t wPro = static_cast<uint16_t>(curWProBatchSize_ > 0 ? curWProBatchSize_ : 1);
    // r 的实际处理行数在 AICore 侧算好，VF 内只做固定边界遍历。
    const uint16_t rowResidues = wPro < wArgmax ? wPro : wArgmax;
    const bool isOverlap = isOverlap_;
    const int32_t wFull = static_cast<int32_t>(wOutput_);
    const int32_t wOutAligned = static_cast<int32_t>(wOutputAligned_);
    const int32_t wOutActual = static_cast<int32_t>(wOutputActual_);
    const int32_t hOutActual = static_cast<int32_t>(hOutputActual_);
    const int32_t curH = static_cast<int32_t>(hAxisIndex_ * hOutputInner_);
    const int32_t curW = static_cast<int32_t>(wAxisIndex_ * wOutputInner_);
    const int32_t ncStride = hOutActual * wOutAligned;

    uint32_t wDivMagic = 0;
    uint32_t wDivShift = 0;
    GetUintDivMagicAndShift<uint32_t>(wDivMagic, wDivShift, static_cast<uint32_t>(wFull));

    if (isOverlap) {
        this->template BackwardComputeIntSimdVf<true>(yAddr, gradAddr, argmaxAddr, hArgmax, wArgmax, wArgmaxAligned,
                                                      highAxis, wPro, rowResidues, wFull, wOutAligned, wOutActual,
                                                      hOutActual, curH, curW, ncStride, wDivMagic, wDivShift);
    } else {
        this->template BackwardComputeIntSimdVf<false>(yAddr, gradAddr, argmaxAddr, hArgmax, wArgmax, wArgmaxAligned,
                                                       highAxis, wPro, rowResidues, wFull, wOutAligned, wOutActual,
                                                       hOutActual, curH, curW, ncStride, wDivMagic, wDivShift);
    }
}

template <typename T1, const uint32_t IS_CHECK_RANGE>
template <const bool IS_OVERLAP>
__simd_vf__ inline void MaxPoolGradNCHWBackwardBase<T1, IS_CHECK_RANGE>::BackwardComputeIntSimdVf(
    __ubuf__ T1* yAddr, __ubuf__ T1* gradAddr, __ubuf__ int32_t* argmaxAddr, uint16_t hArgmax, uint16_t wArgmax,
    int32_t wArgmaxAligned, uint16_t highAxis, uint16_t wPro, uint16_t rowResidues, int32_t wFull, int32_t wOutAligned,
    int32_t wOutActual, int32_t hOutActual, int32_t curH, int32_t curW, int32_t ncStride, uint32_t wDivMagic,
    uint32_t wDivShift)
{
    Reg::RegTensor<int32_t> laneIdx;
    Reg::RegTensor<int32_t> wLocal;
    Reg::RegTensor<int32_t> argOff;
    Reg::RegTensor<int32_t> gradOff;
    Reg::RegTensor<int32_t> targetIdx;
    Reg::RegTensor<int32_t> wIdx;
    Reg::RegTensor<int32_t> hLocal;
    Reg::RegTensor<int32_t> wLocalIn;
    Reg::RegTensor<int32_t> yOff;
    Reg::RegTensor<uint32_t> divResult;
    Reg::RegTensor<uint32_t> wDivMagicReg;
    Reg::MaskReg allMask = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(wDivMagicReg, wDivMagic);

    for (uint16_t nc = 0; nc < highAxis; nc++) {
        const int64_t argBase = static_cast<int64_t>(nc) * hArgmax * wArgmax;
        const int64_t gradBase = static_cast<int64_t>(nc) * hArgmax * wArgmaxAligned;
        for (uint16_t h = 0; h < hArgmax; h++) {
            const int64_t argRowBase = argBase + static_cast<int64_t>(h) * wArgmax;
            const int64_t gradRowBase = gradBase + static_cast<int64_t>(h) * wArgmaxAligned;
            for (uint16_t r = 0; r < rowResidues; r++) {
                // 同一余数组内各 chunk 的目标地址互不重叠，屏障只需在组间/行间切换时插入。
                // 统一按 IS_OVERLAP 模板参数无条件执行，避免 VF 内出现运行时 if；首轮多一次屏障不影响正确性。
                if constexpr (IS_OVERLAP) {
                    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                }
                const uint16_t cols = static_cast<uint16_t>((wArgmax - r + wPro - 1) / wPro);
                const uint16_t chunks = static_cast<uint16_t>((cols + NCHW_INT_INDEX_LANES - 1) / NCHW_INT_INDEX_LANES);
                for (uint16_t chunk = 0; chunk < chunks; chunk++) {
                    const uint16_t c0 = static_cast<uint16_t>(chunk * NCHW_INT_INDEX_LANES);
                    Reg::Arange(laneIdx, static_cast<int32_t>(0));
                    Reg::Adds(laneIdx, laneIdx, static_cast<int32_t>(c0), allMask);
                    Reg::Muls(wLocal, laneIdx, static_cast<int32_t>(wPro), allMask);
                    Reg::Adds(wLocal, wLocal, static_cast<int32_t>(r), allMask);
                    Reg::Adds(argOff, wLocal, static_cast<int32_t>(argRowBase), allMask);
                    Reg::Adds(gradOff, wLocal, static_cast<int32_t>(gradRowBase), allMask);

                    // 用向量比较生成有效 lane 掩码，替代标量 count/UpdateMask。
                    Reg::MaskReg mask32;
                    Reg::Compares<int32_t, CMPMODE::LT>(mask32, laneIdx, static_cast<int32_t>(cols), allMask);
                    Reg::DataCopyGather(targetIdx, argmaxAddr, (Reg::RegTensor<uint32_t>&)argOff, mask32);

                    // argmax 中是全局 h/w 扁平偏移，用 fast-div 换算到当前 block 的 y 缓冲布局。
                    PoolUtils::Index::FastDivImplVF(divResult, (Reg::RegTensor<uint32_t>&)targetIdx, wDivMagicReg,
                                                    static_cast<int16_t>(wDivShift), mask32);
                    Reg::Muls(wIdx, (Reg::RegTensor<int32_t>&)divResult, wFull, mask32);
                    Reg::Sub(wIdx, targetIdx, wIdx, mask32);
                    Reg::Adds(hLocal, (Reg::RegTensor<int32_t>&)divResult, -curH, mask32);
                    Reg::Adds(wLocalIn, wIdx, -curW, mask32);
                    Reg::Muls(yOff, hLocal, wOutAligned, mask32);
                    Reg::Add(yOff, yOff, wLocalIn, mask32);
                    Reg::Adds(yOff, yOff, static_cast<int32_t>(nc) * ncStride, mask32);

                    if constexpr (sizeof(T1) == 1) {
                        Reg::RegTensor<uint16_t> yOffU16;
                        Reg::RegTensor<uint16_t> gradOffU16;
                        Reg::Pack(yOffU16, yOff);
                        Reg::Pack(gradOffU16, gradOff);
                        // Preserve the b32 lane validity when narrowing indices. The upper
                        // half of a packed register does not contain additional input lanes.
                        Reg::MaskReg mask16;
                        Reg::Pack(mask16, mask32);
                        if constexpr (IS_CHECK_RANGE == 1) {
                            Reg::RegTensor<uint16_t> hLocU16;
                            Reg::RegTensor<uint16_t> wLocU16;
                            Reg::Pack(hLocU16, hLocal);
                            Reg::Pack(wLocU16, wLocalIn);
                            PoolUtils::Compute::FilterMaskVF<int16_t>(
                                mask16, (Reg::RegTensor<int16_t>&)hLocU16, (Reg::RegTensor<int16_t>&)wLocU16,
                                static_cast<int16_t>(hOutActual), static_cast<int16_t>(wOutActual));
                        }
                        Reg::RegTensor<int16_t> yReg;
                        Reg::RegTensor<int16_t> gradReg;
                        Reg::DataCopyGather(yReg, yAddr, yOffU16, mask16);
                        Reg::DataCopyGather(gradReg, gradAddr, gradOffU16, mask16);
                        // b8 的 scatter 按 b16 掩码取每个 lane 的低字节写回，等价于模 256 回绕，
                        // 无需 Cast/Pack 改变 lane 映射。
                        Reg::Add(yReg, yReg, gradReg, mask16);
                        Reg::DataCopyScatter((__local_mem__ int8_t*)yAddr, (Reg::RegTensor<int8_t>&)yReg, yOffU16,
                                             mask16);
                    } else if constexpr (sizeof(T1) == 2) {
                        Reg::RegTensor<uint16_t> yOffU16;
                        Reg::RegTensor<uint16_t> gradOffU16;
                        Reg::Pack(yOffU16, yOff);
                        Reg::Pack(gradOffU16, gradOff);
                        // Preserve the b32 lane validity when narrowing indices. The upper
                        // half of a packed register does not contain additional input lanes.
                        Reg::MaskReg mask16;
                        Reg::Pack(mask16, mask32);
                        if constexpr (IS_CHECK_RANGE == 1) {
                            Reg::RegTensor<uint16_t> hLocU16;
                            Reg::RegTensor<uint16_t> wLocU16;
                            Reg::Pack(hLocU16, hLocal);
                            Reg::Pack(wLocU16, wLocalIn);
                            PoolUtils::Compute::FilterMaskVF<int16_t>(
                                mask16, (Reg::RegTensor<int16_t>&)hLocU16, (Reg::RegTensor<int16_t>&)wLocU16,
                                static_cast<int16_t>(hOutActual), static_cast<int16_t>(wOutActual));
                        }
                        Reg::RegTensor<T1> yReg;
                        Reg::RegTensor<T1> gradReg;
                        Reg::DataCopyGather(yReg, yAddr, yOffU16, mask16);
                        Reg::DataCopyGather(gradReg, gradAddr, gradOffU16, mask16);
                        Reg::Add(yReg, yReg, gradReg, mask16);
                        Reg::DataCopyScatter(yAddr, yReg, yOffU16, mask16);
                    } else if constexpr (sizeof(T1) == 4) {
                        if constexpr (IS_CHECK_RANGE == 1) {
                            PoolUtils::Compute::FilterMaskVF<int32_t>(mask32, hLocal, wLocalIn, hOutActual, wOutActual);
                        }
                        Reg::RegTensor<T1> yReg;
                        Reg::RegTensor<T1> gradReg;
                        Reg::DataCopyGather(yReg, yAddr, (Reg::RegTensor<uint32_t>&)yOff, mask32);
                        Reg::DataCopyGather(gradReg, gradAddr, (Reg::RegTensor<uint32_t>&)gradOff, mask32);
                        Reg::Add(yReg, yReg, gradReg, mask32);
                        Reg::DataCopyScatter(yAddr, yReg, (Reg::RegTensor<uint32_t>&)yOff, mask32);
                    } else {
                        if constexpr (IS_CHECK_RANGE == 1) {
                            PoolUtils::Compute::FilterMaskVF<int32_t>(mask32, hLocal, wLocalIn, hOutActual, wOutActual);
                        }
                        Reg::RegTensor<T1, Reg::RegTraitNumTwo> yReg;
                        Reg::RegTensor<T1, Reg::RegTraitNumTwo> gradReg;
                        Reg::DataCopyGather(yReg, yAddr, (Reg::RegTensor<uint32_t>&)yOff, mask32);
                        Reg::DataCopyGather(gradReg, gradAddr, (Reg::RegTensor<uint32_t>&)gradOff, mask32);
                        Reg::Add(yReg, yReg, gradReg, mask32);
                        Reg::DataCopyScatter(yAddr, yReg, (Reg::RegTensor<uint32_t>&)yOff, mask32);
                    }
                }
            }
        }
    }
}

} // namespace MaxPoolGradNCHWBackwardBaseNameSpace
#endif // MAX_POOL_GRAD_NCHW_BACKWARD_BASE_H_
