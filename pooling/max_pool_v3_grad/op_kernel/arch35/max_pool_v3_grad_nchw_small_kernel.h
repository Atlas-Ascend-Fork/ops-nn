/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool_v3_grad_nchw_small_kernel.h
 * \brief MaxPoolV3Grad NCHW small-kernel template without the PIPE_ALL barrier.
 */

#ifndef MAX_POOL_V3_GRAD_NCHW_SMALL_KERNEL_H
#define MAX_POOL_V3_GRAD_NCHW_SMALL_KERNEL_H

#include <limits>
#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../pool_grad_common/arch35/max_pool_grad_struct.h"
#include "../pool_grad_common/arch35/max_pool_grad_nchw_backward_base.h"
#include "pool_utils/arch35/compute/max_pool_negative_value.h"
#include "pool_utils/arch35/compute/pool_grad_scatter_compute.h"
#include "pool_utils/arch35/index/max_pool_with_argmax_index.h"
#include "pool_utils/arch35/data_move/pool_calc_buffer_data_move.h"
#include "../pool_3d_common/arch35/pool_big_kernel_utils.h"
#include "../pool_3d_common/arch35/pool_3d_common.h"
#include "simt_api/asc_simt.h"

namespace MaxPoolV3GradNCHWSmallKernelNameSpace {
using namespace AscendC;
using MaxPoolGradNCHWNameSpace::ConvertIndexInt32FastDivVF;
using MaxPoolGradNCHWNameSpace::ConvertIndexNcInt32FastDivVF;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;

constexpr uint32_t BUFFER_NUM = 2;
// input/grad 队列在当前顺序结构下不构成跨迭代预取，单缓冲即可；output 保留双缓冲以重叠 MTE3。
constexpr uint32_t IO_BUFFER_NUM = 1;
constexpr int64_t RATIO = 2;
constexpr uint16_t UINT8_HIGH_MASK = 0x00FF;
constexpr uint16_t UINT8_SIGN_BIT = 0x0080;
constexpr int16_t INT8_SIGN_OFFSET = -128;

/*
 * 功能：整数 SIMD 前向计算中 gather 目标类型与比较类型的映射。
 * 说明：b8 的 gather 只支持落入 b16 目标，因此 8 位数据统一提升到 16 位比较；
 *       64 位类型需要使用双寄存器 RegTraitNumTwo。
 */
template <typename T>
struct NchwIntVecTraits {
    using ComputeT = T;
    static constexpr bool MASK_8BIT = false;
    static constexpr bool SIGN_8BIT = false;
};

template <>
struct NchwIntVecTraits<int8_t> {
    using ComputeT = int16_t;
    static constexpr bool MASK_8BIT = true;
    static constexpr bool SIGN_8BIT = true;
};

template <>
struct NchwIntVecTraits<uint8_t> {
    using ComputeT = uint16_t;
    static constexpr bool MASK_8BIT = true;
    static constexpr bool SIGN_8BIT = false;
};

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
class PoolGradNCHWSmallKernelV3
    : public MaxPoolGradNCHWBackwardBaseNameSpace::MaxPoolGradNCHWBackwardBase<TYPE_ORIG_X, IS_CHECK_RANGE> {
    using Base = MaxPoolGradNCHWBackwardBaseNameSpace::MaxPoolGradNCHWBackwardBase<TYPE_ORIG_X, IS_CHECK_RANGE>;

public:
    __aicore__ inline PoolGradNCHWSmallKernelV3(TPipe& pipeIn,
                                                const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void ForwardScalarCompute();
    template <bool IS_PAD>
    __simd_callee__ inline void ConvertIndexWithoutPadAlignInt32VF(Reg::RegTensor<int32_t>& srcReg,
                                                                   uint32_t wStrideOffset, int32_t left, int32_t wInput,
                                                                   int32_t hIndexBase, Reg::RegTensor<int32_t>& dstReg,
                                                                   int32_t ncInputOffset, uint32_t magic,
                                                                   uint32_t shift);
    template <bool IS_PAD>
    __simd_callee__ inline void ConvertIndexWithoutPadAlignNcInt32(
        Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, int32_t left, int32_t wInput, int32_t hIndexBase,
        Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount, int32_t inputNcSize,
        uint32_t magicNc, uint32_t shiftNc, uint32_t magicWStride, uint32_t shiftWStride);

    __simd_callee__ inline void ProcessW(__local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset,
                                         uint16_t wStrideOffset, Reg::RegTensor<int32_t>& indexReg, uint16_t hKernel,
                                         uint16_t wKernel, uint16_t repeatElem, Reg::RegTensor<int32_t>& maxIndexReg,
                                         uint32_t hDilation, uint32_t wDilation);
    __simd_callee__ inline void ProcessWFloat(__local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset,
                                              uint16_t wStrideOffset, Reg::RegTensor<int32_t>& indexReg,
                                              uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
                                              Reg::RegTensor<int32_t>& maxIndexReg, uint32_t hDilation,
                                              uint32_t wDilation);
    __simd_callee__ inline void ProcessWInt(__local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset,
                                            uint16_t wStrideOffset, Reg::RegTensor<int32_t>& indexReg, uint16_t hKernel,
                                            uint16_t wKernel, uint16_t repeatElem, Reg::RegTensor<int32_t>& maxIndexReg,
                                            uint32_t hDilation, uint32_t wDilation);

    template <bool IS_PAD>
    __simd_vf__ inline void SingleRowGatherSimdVf(__ubuf__ TYPE_ORIG_X* computeAddr, __ubuf__ int32_t* argmaxAddr,
                                                  uint16_t loopW, uint16_t repeatsElem, uint16_t tailRepeatsElem,
                                                  uint16_t hKernel, uint16_t wKernel, uint32_t wStride,
                                                  uint32_t highAxisActual, uint32_t hOutputActual,
                                                  uint32_t wOutputActual, uint32_t hInputActualPad,
                                                  uint32_t wInputActualAlignedPad, uint32_t hDilation,
                                                  uint32_t wDilation, int32_t left, int32_t hIndexBase, int32_t wInput,
                                                  uint32_t magic, uint32_t shift, uint32_t strideH);
    template <bool IS_PAD>
    __simd_vf__ inline void MultiRowGatherSimdVf(__ubuf__ TYPE_ORIG_X* computeAddr, __ubuf__ int32_t* argmaxAddr,
                                                 uint32_t wOutputActual, uint16_t hKernel, uint16_t wKernel,
                                                 uint32_t wStride, uint32_t rate2D, uint16_t hBatchCount,
                                                 uint16_t hLoopTimes, uint16_t repeatsElem, uint16_t tailRepeatsElem,
                                                 uint32_t highAxisActual, uint32_t hInputActualPad,
                                                 uint32_t wInputActualAlignedPad, uint32_t hDilation,
                                                 uint32_t wDilation, int32_t left, int32_t hIndexBase, int32_t wInput,
                                                 uint32_t magic, uint32_t shift);
    __simd_vf__ inline void DupAndCopyToCalcBufferSimdVf(__ubuf__ TYPE_ORIG_X* dstAddr, __ubuf__ TYPE_ORIG_X* srcAddr,
                                                         uint16_t highAxis, uint16_t hRows, uint16_t hPad,
                                                         uint16_t wPad, uint16_t loopCols, uint16_t tailCols,
                                                         uint32_t repeatElm, uint32_t srcBatchStride,
                                                         uint32_t srcRowStride, uint32_t dstBatchStride,
                                                         uint32_t dstRowStride, uint32_t wInputPadActualAlign,
                                                         uint16_t downOffset);
    template <bool IS_PAD>
    __simd_vf__ inline void MultiNcGatherSimdVf(__ubuf__ TYPE_ORIG_X* computeAddr, __ubuf__ int32_t* argmaxAddr,
                                                uint16_t ncLoopTimes, uint32_t ncBatchCount, uint16_t repeatsElem,
                                                uint16_t tailRepeatsElem, uint16_t hKernel, uint16_t wKernel,
                                                uint16_t wInputActualAlignedPad, int32_t rate3D, int32_t num2D,
                                                int32_t rate2D, int32_t wOutputActual, int32_t wStride, int32_t left,
                                                int32_t wInput, int32_t hIndexBase, int32_t inputNcSize,
                                                uint32_t hDilation, uint32_t wDilation, uint32_t magicNc,
                                                uint32_t shiftNc, uint32_t magicWStride, uint32_t shiftWStride);

    template <bool IS_PAD>
    __aicore__ inline void ForwardGather(__local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ int32_t* argmaxAddr);
    template <bool IS_PAD>
    __aicore__ inline void MultiRowGather(__local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ int32_t* argmaxAddr);
    template <bool IS_PAD>
    __aicore__ inline void SingleRowGather(__local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ int32_t* argmaxAddr);
    template <bool IS_PAD>
    __aicore__ inline void MultiNcGather(__local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ int32_t* argmaxAddr);
    __aicore__ inline void DupAndCopyToCalcBuffer(__local_mem__ TYPE_ORIG_X* dstAddr,
                                                  __local_mem__ TYPE_ORIG_X* srcAddr);
    __aicore__ inline void ForwardCopyIn();
    __aicore__ inline void Forward();
    __aicore__ inline void Backward();
    __aicore__ inline void IdentityCopyGradToOutput();

    TPipe& pipe_;
    const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<TPosition::VECCALC> inputCalcBuff_;
    TBuf<TPosition::VECCALC> argmaxBuff_;

    GlobalTensor<TYPE_ORIG_X> xGm_;

    int64_t hInputActualPad_ = 0;
    int64_t wInputActualPad_ = 0;
    int64_t wInputActualAlignedPad_ = 0;
    int64_t leftOffsetToInputLeft_ = 0;
    int64_t rightOffsetToInputRight_ = 0;
    int64_t topOffsetToInputTop_ = 0;
    int64_t downOffsetToInputDown_ = 0;
    int64_t highInputOffset_ = 0;
    int64_t forwardhInputOffset_ = 0;
    int64_t forwardwInputOffset_ = 0;
    int64_t hInputActualNoPad_ = 0;
    int64_t wInputActualNoPad_ = 0;
    int64_t forwardHighAxisIndex_ = 0;
    int64_t forwardhighAxisActual_ = 0;
    int64_t forwardHAxisIndex_ = 0;
    int64_t hOutputReal_ = 0;
    int64_t wOutputReal_ = 0;

    bool isPad_ = false;
    bool isIdentity_ = false;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t V_REG_SIZE = platform::GetVRegSize();
    constexpr static uint16_t vlT1_ = platform::GetVRegSize() / sizeof(TYPE_ORIG_X);
};

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::Init(GM_ADDR orig_x, GM_ADDR orig_y,
                                                                                    GM_ADDR grads, GM_ADDR y)
{
    Base::ParseTilingData(tilingData_);
    Base::blockIdx_ = GetBlockIdx();
    if (Base::blockIdx_ >= Base::usedCoreNum_) {
        return;
    }
    Base::curCoreProcessNum_ = (Base::blockIdx_ + 1 == Base::usedCoreNum_) ? Base::tailCoreProcessNum_ :
                                                                             Base::normalCoreProcessNum_;
    xGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)orig_x);
    Base::gradGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)grads);
    Base::yGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)y);

    isPad_ = tilingData_.isPad != 0;
    isIdentity_ = (tilingData_.hKernel == 1 && tilingData_.wKernel == 1 && tilingData_.hStride == 1 &&
                   tilingData_.wStride == 1 && !isPad_);

    // identity 路径不使用 input/argmax/help 临时区，跳过分配以把 UB 让给更大的 tile。
    if (!isIdentity_) {
        pipe_.InitBuffer(inputQue_, IO_BUFFER_NUM, tilingData_.inputBufferSize);
        if (isPad_) {
            pipe_.InitBuffer(inputCalcBuff_, tilingData_.inputBufferSize);
        }
        pipe_.InitBuffer(argmaxBuff_, Base::argmaxBufferSize_);
        if constexpr (std::negation<std::is_integral<TYPE_ORIG_X>>::value) {
            pipe_.InitBuffer(Base::helpBuf_, MaxPoolGradNCHWBackwardBaseNameSpace::HELP_BUFFER);
        }
    }

    pipe_.InitBuffer(Base::outputQue_, BUFFER_NUM, Base::outputBufferSize_);
    // identity 也走 grad 队列 + UB->UB 拷贝：GM->UB 由 MTE2 完成，必须经 grad 队列同步后
    // 再由 Vector 写入 output 队列，避免 MTE3 读到未完成的 MTE2 数据。
    pipe_.InitBuffer(Base::gradQue_, IO_BUFFER_NUM, Base::gradBufferSize_);
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::Forward()
{
    LocalTensor<TYPE_ORIG_X> inputLocal = inputQue_.template DeQue<TYPE_ORIG_X>();
    __local_mem__ TYPE_ORIG_X* inputQueAddr = (__local_mem__ TYPE_ORIG_X*)inputLocal.GetPhyAddr();
    __local_mem__ TYPE_ORIG_X* computeAddr = inputQueAddr;
    if (isPad_) {
        LocalTensor<TYPE_ORIG_X> caclBuffLocal = inputCalcBuff_.template Get<TYPE_ORIG_X>();
        __local_mem__ TYPE_ORIG_X* inputBuffAddr = (__local_mem__ TYPE_ORIG_X*)caclBuffLocal.GetPhyAddr();
        if constexpr (std::is_integral<TYPE_ORIG_X>::value) {
            constexpr TYPE_ORIG_X padValue = std::is_signed<TYPE_ORIG_X>::value ?
                                                 std::numeric_limits<TYPE_ORIG_X>::lowest() :
                                                 static_cast<TYPE_ORIG_X>(0);
            int32_t padCount = static_cast<int32_t>(Base::highAxisActual_ * hInputActualPad_ * wInputActualAlignedPad_);
            Duplicate(caclBuffLocal, padValue, padCount);
        }
        DupAndCopyToCalcBuffer(inputBuffAddr, inputQueAddr);
        computeAddr = inputBuffAddr;
    }
    LocalTensor<int32_t> argmaxLocal = argmaxBuff_.template Get<int32_t>();
    __local_mem__ int32_t* argmaxAddr = (__local_mem__ int32_t*)argmaxLocal.GetPhyAddr();

    if (isPad_) {
        this->template ForwardGather<true>(computeAddr, argmaxAddr);
    } else {
        this->template ForwardGather<false>(computeAddr, argmaxAddr);
    }

    inputQue_.FreeTensor(inputLocal);
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::Backward()
{
    LocalTensor<int32_t> argmaxLocal = argmaxBuff_.template Get<int32_t>();
    __local_mem__ int32_t* argmaxAddr = (__local_mem__ int32_t*)argmaxLocal.GetPhyAddr();
    Base::BackwardComputeDispatch(argmaxAddr);
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ForwardScalarCompute()
{
    forwardHighAxisIndex_ = Base::highAxisIndex_;
    forwardhighAxisActual_ = Base::highAxisActual_;

    forwardHAxisIndex_ = Base::hAxisIndex_;
    hOutputReal_ = Base::hArgmaxActual_;

    wOutputReal_ = Base::wArgmaxActual_;

    hInputActualPad_ = (hOutputReal_ - 1) * tilingData_.hStride + (tilingData_.hKernel - 1) * tilingData_.dilationH + 1;
    wInputActualPad_ = (wOutputReal_ - 1) * tilingData_.wStride + (tilingData_.wKernel - 1) * tilingData_.dilationW + 1;

    wInputActualAlignedPad_ = CeilDivision(wInputActualPad_, BLOCK_SIZE / sizeof(TYPE_ORIG_X)) *
                              (BLOCK_SIZE / sizeof(TYPE_ORIG_X));
    int64_t inputPlaneSize = tilingData_.hOutput * tilingData_.wOutput;
    highInputOffset_ = Base::highAxisIndex_ * tilingData_.highAxisInner * inputPlaneSize;
    forwardhInputOffset_ = Base::hArgmaxActualStart_ * tilingData_.hStride * tilingData_.wOutput;
    forwardwInputOffset_ = Base::wArgmaxActualStart_ * tilingData_.wStride;

    if (isPad_) {
        int64_t tRelBoundDistance = Base::hArgmaxActualStart_ * tilingData_.hStride - tilingData_.padH;
        int64_t bRelBoundDistance = Base::hArgmaxActualStart_ * tilingData_.hStride +
                                    (hOutputReal_ - 1) * tilingData_.hStride +
                                    (tilingData_.hKernel - 1) * tilingData_.dilationH + 1 - tilingData_.hOutput -
                                    tilingData_.padH;
        int64_t lRelBoundDistance = Base::wArgmaxActualStart_ * tilingData_.wStride - tilingData_.padW;
        int64_t rRelBoundDistance = Base::wArgmaxActualStart_ * tilingData_.wStride +
                                    (wOutputReal_ - 1) * tilingData_.wStride +
                                    (tilingData_.wKernel - 1) * tilingData_.dilationW + 1 - tilingData_.wOutput -
                                    tilingData_.padW;
        leftOffsetToInputLeft_ = lRelBoundDistance >= 0 ? 0 : -lRelBoundDistance;
        rightOffsetToInputRight_ = rRelBoundDistance >= 0 ? rRelBoundDistance : 0;
        topOffsetToInputTop_ = tRelBoundDistance >= 0 ? 0 : -tRelBoundDistance;
        downOffsetToInputDown_ = bRelBoundDistance >= 0 ? bRelBoundDistance : 0;
        hInputActualNoPad_ = hInputActualPad_ - topOffsetToInputTop_ - downOffsetToInputDown_;
        wInputActualNoPad_ = wInputActualPad_ - leftOffsetToInputLeft_ - rightOffsetToInputRight_;
        forwardhInputOffset_ = topOffsetToInputTop_ == 0 ?
                                   forwardhInputOffset_ - tilingData_.padH * tilingData_.wOutput :
                                   0;
        forwardwInputOffset_ = leftOffsetToInputLeft_ == 0 ? forwardwInputOffset_ - tilingData_.padW : 0;
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ForwardCopyIn()
{
    LocalTensor<TYPE_ORIG_X> xLocal = inputQue_.template AllocTensor<TYPE_ORIG_X>();
    int64_t xGmOffset = highInputOffset_ + forwardhInputOffset_ + forwardwInputOffset_;

    LoopModeParams loopModeParams;
    if (isPad_) {
        int64_t wInputActualAlignedNoPad = CeilDivision(wInputActualNoPad_, BLOCK_SIZE / sizeof(TYPE_ORIG_X)) *
                                           (BLOCK_SIZE / sizeof(TYPE_ORIG_X));
        loopModeParams.loop1Size = Base::highAxisActual_;
        loopModeParams.loop1SrcStride = static_cast<uint64_t>(Base::hOutput_) * static_cast<uint64_t>(Base::wOutput_) *
                                        sizeof(TYPE_ORIG_X);
        loopModeParams.loop1DstStride = hInputActualNoPad_ * wInputActualAlignedNoPad * sizeof(TYPE_ORIG_X);
        loopModeParams.loop2Size = 1;
        loopModeParams.loop2SrcStride = 0;
        loopModeParams.loop2DstStride = 0;
    } else {
        loopModeParams.loop1Size = Base::highAxisActual_;
        loopModeParams.loop1SrcStride = static_cast<uint64_t>(Base::hOutput_) * static_cast<uint64_t>(Base::wOutput_) *
                                        sizeof(TYPE_ORIG_X);
        loopModeParams.loop1DstStride = hInputActualPad_ * wInputActualAlignedPad_ * sizeof(TYPE_ORIG_X);
        loopModeParams.loop2Size = 1;
        loopModeParams.loop2SrcStride = 0;
        loopModeParams.loop2DstStride = 0;
    }
    SetLoopModePara(loopModeParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPadExtParams<TYPE_ORIG_X> padParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams;
    if (isPad_) {
        copyParams.blockCount = static_cast<uint16_t>(hInputActualNoPad_);
        copyParams.blockLen = static_cast<uint32_t>(wInputActualNoPad_ * sizeof(TYPE_ORIG_X));
        const int64_t srcGap = Base::wOutput_ - wInputActualNoPad_;
        const int64_t clampedSrcGap = srcGap > 0 ? srcGap : 0;
        copyParams.srcStride = static_cast<uint32_t>(clampedSrcGap * sizeof(TYPE_ORIG_X));
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
    } else {
        copyParams.blockCount = static_cast<uint16_t>(hInputActualPad_);
        copyParams.blockLen = static_cast<uint32_t>(wInputActualPad_ * sizeof(TYPE_ORIG_X));
        const int64_t srcGap = Base::wOutput_ - wInputActualPad_;
        const int64_t clampedSrcGap = srcGap > 0 ? srcGap : 0;
        copyParams.srcStride = static_cast<uint32_t>(clampedSrcGap * sizeof(TYPE_ORIG_X));
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
    }
    DataCopyPad(xLocal, xGm_[xGmOffset], copyParams, padParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inputQue_.EnQue(xLocal);
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::Process()
{
    if (Base::blockIdx_ >= Base::usedCoreNum_) {
        return;
    }

    // k=1,s=1 且无 pad 时 out_grad 与 orig_input 完全同形，可直接拷贝；
    // 有 pad 时池化输出比输入大，grad 不能按 identity 路径读。
    for (int64_t loopNum = 0; loopNum < Base::curCoreProcessNum_; loopNum++) {
        Base::ScalarCompute(loopNum);
        // DataCopyPad 的目的行按 32B 对齐；64-bit 数据下该对齐为 4 个元素，
        // 不能沿用 base 中与 int32 对齐取 max 得到的 8。
        constexpr int64_t rowAlignElem = BLOCK_SIZE / sizeof(TYPE_ORIG_X);
        Base::wArgmaxAligned_ = CeilDivision(Base::wArgmaxActual_, rowAlignElem) * rowAlignElem;
        Base::wOutputAligned_ = CeilDivision(Base::wOutputActual_, rowAlignElem) * rowAlignElem;
        if (Base::hArgmaxActual_ <= 0 || Base::wArgmaxActual_ <= 0) {
            Base::ProcessNoArgmaxBlock();
            continue;
        }
        if (isIdentity_) {
            Base::CopyInGrad();
            IdentityCopyGradToOutput();
            Base::CopyOut();
        } else {
            ForwardScalarCompute();
            ForwardCopyIn();
            // grad 搬入只依赖 ScalarCompute 的偏移，提前发起以与 forward 的 gather 计算重叠。
            Base::CopyInGrad();
            Forward();
            Backward();
            Base::CopyOut();
        }
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__simd_callee__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ConvertIndexWithoutPadAlignInt32VF(
    Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, int32_t left, int32_t wInput, int32_t hIndexBase,
    Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset, uint32_t magic, uint32_t shift)
{
    if constexpr (IS_PAD) {
        ConvertIndexInt32FastDivVF<1>(srcReg, wStrideOffset, left, wInput, hIndexBase, dstReg, ncInputOffset, magic,
                                      shift);
    } else {
        ConvertIndexInt32FastDivVF<0>(srcReg, wStrideOffset, left, wInput, hIndexBase, dstReg, ncInputOffset, magic,
                                      shift);
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ProcessW(
    __local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset, uint16_t wStrideOffset, Reg::RegTensor<int32_t>& indexReg,
    uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem, Reg::RegTensor<int32_t>& maxIndexReg, uint32_t hDilation,
    uint32_t wDilation)
{
    if constexpr (std::is_integral<TYPE_ORIG_X>::value) {
        ProcessWInt(computeAddr, hOffset, wStrideOffset, indexReg, hKernel, wKernel, repeatElem, maxIndexReg, hDilation,
                    wDilation);
    } else {
        ProcessWFloat(computeAddr, hOffset, wStrideOffset, indexReg, hKernel, wKernel, repeatElem, maxIndexReg,
                      hDilation, wDilation);
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ProcessWFloat(
    __local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset, uint16_t wStrideOffset, Reg::RegTensor<int32_t>& indexReg,
    uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem, Reg::RegTensor<int32_t>& maxIndexReg, uint32_t hDilation,
    uint32_t wDilation)
{
    Reg::RegTensor<int32_t> indexWithOffset;
    Reg::RegTensor<TYPE_ORIG_X> calcReg;
    Reg::RegTensor<uint16_t> indexConvert;
    uint32_t maskCount = repeatElem;
    Reg::MaskReg allMaskU32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg gatherMask = Reg::UpdateMask<TYPE_ORIG_X>(maskCount);
    Reg::RegTensor<TYPE_ORIG_X> maxReg;
    Reg::MaskReg neMask;
    Reg::MaskReg gtMask;
    Reg::MaskReg tmpMask;
    PoolUtils::Compute::DuplicateNegInfRegVF<TYPE_ORIG_X>(maxReg);

    Reg::Adds(maxIndexReg, indexReg, hOffset, allMaskU32);
    for (int32_t hIndex = 0; hIndex < hKernel; hIndex++) {
        for (int32_t wIndex = 0; wIndex < wKernel; wIndex++) {
            int32_t relIndex = hIndex * wStrideOffset * hDilation + wIndex * wDilation;
            int32_t offset = static_cast<int32_t>(hOffset + relIndex);
            Reg::Adds(indexWithOffset, indexReg, offset, allMaskU32);
            if constexpr (std::is_same<TYPE_ORIG_X, float>::value) {
                Reg::DataCopyGather(calcReg, computeAddr, (Reg::RegTensor<uint32_t>&)indexWithOffset, gatherMask);
            } else {
                Reg::Pack(indexConvert, indexWithOffset);
                Reg::DataCopyGather(calcReg, computeAddr, indexConvert, gatherMask);
            }

            Reg::Compare<TYPE_ORIG_X, CMPMODE::GT>(gtMask, calcReg, maxReg, gatherMask);
            Reg::Compare<TYPE_ORIG_X, CMPMODE::NE>(neMask, calcReg, calcReg, gatherMask);
            Reg::MaskOr(gtMask, gtMask, neMask, gatherMask);

            if constexpr (sizeof(int32_t) / sizeof(TYPE_ORIG_X) == 1) {
                Reg::Select(maxIndexReg, indexWithOffset, maxIndexReg, gtMask);
            } else {
                Reg::MaskUnPack(tmpMask, gtMask);
                Reg::Select(maxIndexReg, indexWithOffset, maxIndexReg, tmpMask);
            }
            Reg::Max(maxReg, maxReg, calcReg, gatherMask);
        }
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ProcessWInt(
    __local_mem__ TYPE_ORIG_X* computeAddr, int32_t hOffset, uint16_t wStrideOffset, Reg::RegTensor<int32_t>& indexReg,
    uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem, Reg::RegTensor<int32_t>& maxIndexReg, uint32_t hDilation,
    uint32_t wDilation)
{
    static_assert(std::is_integral<TYPE_ORIG_X>::value, "ProcessWInt only supports integral dtypes");
    using Traits = NchwIntVecTraits<TYPE_ORIG_X>;
    using ComputeT = typename Traits::ComputeT;
    constexpr bool IS_B64 = sizeof(TYPE_ORIG_X) == sizeof(int64_t);
    using CalcReg = std::conditional_t<IS_B64, Reg::RegTensor<ComputeT, Reg::RegTraitNumTwo>, Reg::RegTensor<ComputeT>>;

    Reg::RegTensor<int32_t> indexWithOffset;
    Reg::RegTensor<uint16_t> indexConvert;
    Reg::RegTensor<uint16_t> highMaskReg;
    Reg::RegTensor<uint16_t> signXorReg;
    CalcReg calcReg;
    CalcReg maxReg;
    uint32_t maskCount = repeatElem;
    Reg::MaskReg allMaskU32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg gatherMask;
    if constexpr (IS_B64) {
        gatherMask = Reg::UpdateMask<ComputeT, Reg::RegTraitNumTwo>(maskCount);
    } else {
        gatherMask = Reg::UpdateMask<ComputeT>(maskCount);
    }
    Reg::MaskReg gtMask;
    Reg::MaskReg selMask;

    Reg::Duplicate(maxReg, std::numeric_limits<ComputeT>::lowest());
    if constexpr (Traits::MASK_8BIT) {
        Reg::Duplicate(highMaskReg, UINT8_HIGH_MASK);
    }
    if constexpr (Traits::SIGN_8BIT) {
        Reg::Duplicate(signXorReg, UINT8_SIGN_BIT);
    }

    Reg::Adds(maxIndexReg, indexReg, hOffset, allMaskU32);
    for (uint16_t hIndex = 0; hIndex < hKernel; hIndex++) {
        for (uint16_t wIndex = 0; wIndex < wKernel; wIndex++) {
            int32_t relIndex = static_cast<int32_t>(hIndex) * wStrideOffset * hDilation +
                               static_cast<int32_t>(wIndex) * wDilation;
            int32_t offset = static_cast<int32_t>(hOffset + relIndex);
            Reg::Adds(indexWithOffset, indexReg, offset, allMaskU32);

            if constexpr (sizeof(TYPE_ORIG_X) == 1) {
                Reg::Pack(indexConvert, indexWithOffset);
                Reg::DataCopyGather(calcReg, computeAddr, indexConvert, gatherMask);
                if constexpr (Traits::MASK_8BIT) {
                    // gather 的 b8->b16 扩展方向未在接口层约定，统一截到 0..255 后再做数值映射。
                    Reg::And(calcReg, calcReg, (Reg::RegTensor<ComputeT>&)highMaskReg, gatherMask);
                }
                if constexpr (Traits::SIGN_8BIT) {
                    // 0..255 先异或符号位再减 128，等价于按 int8 解释并符号扩展。
                    Reg::Xor(calcReg, calcReg, (Reg::RegTensor<ComputeT>&)signXorReg, gatherMask);
                    Reg::Adds(calcReg, calcReg, ComputeT(INT8_SIGN_OFFSET), gatherMask);
                }
            } else if constexpr (sizeof(TYPE_ORIG_X) == 2) {
                Reg::Pack(indexConvert, indexWithOffset);
                Reg::DataCopyGather(calcReg, computeAddr, indexConvert, gatherMask);
            } else {
                Reg::DataCopyGather(calcReg, computeAddr, (Reg::RegTensor<uint32_t>&)indexWithOffset, gatherMask);
            }

            Reg::Compare<ComputeT, CMPMODE::GT>(gtMask, calcReg, maxReg, gatherMask);
            if constexpr (IS_B64) {
                Reg::Select(maxIndexReg, indexWithOffset, maxIndexReg, gtMask);
            } else if constexpr (sizeof(TYPE_ORIG_X) <= 2) {
                Reg::MaskUnPack(selMask, gtMask);
                Reg::Select(maxIndexReg, indexWithOffset, maxIndexReg, selMask);
            } else {
                Reg::Select(maxIndexReg, indexWithOffset, maxIndexReg, gtMask);
            }
            Reg::Max(maxReg, maxReg, calcReg, gatherMask);
        }
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__simd_callee__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ConvertIndexWithoutPadAlignNcInt32(
    Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, int32_t left, int32_t wInput, int32_t hIndexBase,
    Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount, int32_t inputNcSize,
    uint32_t magicNc, uint32_t shiftNc, uint32_t magicWStride, uint32_t shiftWStride)
{
    if constexpr (IS_PAD) {
        ConvertIndexNcInt32FastDivVF<1>(srcReg, wStrideOffset, left, wInput, hIndexBase, dstReg, ncInputOffset,
                                        ncOutputCount, inputNcSize, magicNc, shiftNc, magicWStride, shiftWStride);
    } else {
        ConvertIndexNcInt32FastDivVF<0>(srcReg, wStrideOffset, left, wInput, hIndexBase, dstReg, ncInputOffset,
                                        ncOutputCount, inputNcSize, magicNc, shiftNc, magicWStride, shiftWStride);
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ForwardGather(
    __local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ int32_t* argmaxAddr)
{
    if (Base::wArgmaxActual_ * RATIO > Base::vlArgmax_) {
        this->template SingleRowGather<IS_PAD>(computeAddr, argmaxAddr);
    } else if (Base::hArgmaxActual_ * Base::wArgmaxActual_ * RATIO > Base::vlArgmax_) {
        this->template MultiRowGather<IS_PAD>(computeAddr, argmaxAddr);
    } else {
        this->template MultiNcGather<IS_PAD>(computeAddr, argmaxAddr);
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::SingleRowGather(
    __local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ int32_t* argmaxAddr)
{
    uint16_t loopW = static_cast<uint16_t>(Base::wArgmaxActual_) / Base::vlArgmax_;
    uint16_t repeatsElem = Base::vlArgmax_;
    uint16_t tailRepeatsElem = static_cast<uint16_t>(Base::wArgmaxActual_) - loopW * Base::vlArgmax_;
    if (tailRepeatsElem == 0) {
        loopW = loopW - 1;
        tailRepeatsElem = repeatsElem;
    }
    uint16_t hKernel = Base::kernelH_;
    uint16_t wKernel = Base::kernelW_;
    uint32_t wStride = Base::strideW_;
    uint32_t highAxisActual = static_cast<uint32_t>(forwardhighAxisActual_);
    uint32_t hOutputActual = static_cast<uint32_t>(Base::hArgmaxActual_);
    uint32_t wOutputActual = static_cast<uint32_t>(Base::wArgmaxActual_);
    uint32_t hInputActualPad = static_cast<uint32_t>(hInputActualPad_);
    uint32_t wInputActualAlignedPad = static_cast<uint32_t>(wInputActualAlignedPad_);
    uint32_t hDilation = Base::dilationH_;
    uint32_t wDilation = Base::dilationW_;
    int32_t left = static_cast<int32_t>(Base::wArgmaxActualStart_ * wStride - Base::padW_);
    int32_t hIndexBase = static_cast<int32_t>(Base::hArgmaxActualStart_ * Base::strideH_ - Base::padH_);
    int32_t wInput = static_cast<int32_t>(Base::wOutput_);

    uint32_t magic = 0;
    uint32_t shift = 0;
    GetUintDivMagicAndShift<uint32_t>(magic, shift, wInputActualAlignedPad);

    this->template SingleRowGatherSimdVf<IS_PAD>(computeAddr, argmaxAddr, loopW, repeatsElem, tailRepeatsElem, hKernel,
                                                 wKernel, wStride, highAxisActual, hOutputActual, wOutputActual,
                                                 hInputActualPad, wInputActualAlignedPad, hDilation, wDilation, left,
                                                 hIndexBase, wInput, magic, shift, Base::strideH_);
    return;
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__simd_vf__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::SingleRowGatherSimdVf(
    __ubuf__ TYPE_ORIG_X* computeAddr, __ubuf__ int32_t* argmaxAddr, uint16_t loopW, uint16_t repeatsElem,
    uint16_t tailRepeatsElem, uint16_t hKernel, uint16_t wKernel, uint32_t wStride, uint32_t highAxisActual,
    uint32_t hOutputActual, uint32_t wOutputActual, uint32_t hInputActualPad, uint32_t wInputActualAlignedPad,
    uint32_t hDilation, uint32_t wDilation, int32_t left, int32_t hIndexBase, int32_t wInput, uint32_t magic,
    uint32_t shift, uint32_t strideH)
{
    Reg::RegTensor<int32_t> indexRegInit;
    PoolUtils::Index::GenInitial1DIndicesVF<int32_t>(indexRegInit, static_cast<int64_t>(wStride));

    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        for (uint16_t hLoop = 0; hLoop < static_cast<uint16_t>(hOutputActual); hLoop++) {
            Reg::RegTensor<int32_t> maxIndexReg;
            Reg::RegTensor<int32_t> maxIndexConvertReg;
            Reg::UnalignReg u1;
            int32_t ncInputOffset = nc * hInputActualPad * wInputActualAlignedPad;
            int32_t ncOutOffset = nc * hOutputActual * wOutputActual;
            int32_t vfMaxAddrOffset = ncOutOffset + hLoop * wOutputActual;
            __local_mem__ int32_t* argmaxAddrLocal = argmaxAddr + vfMaxAddrOffset;
            for (uint16_t wLoop = 0; wLoop < loopW; wLoop++) {
                int32_t wOffset = ncInputOffset + hLoop * wInputActualAlignedPad * strideH +
                                  wLoop * repeatsElem * wStride;
                ProcessW(computeAddr, wOffset, static_cast<uint16_t>(wInputActualAlignedPad), indexRegInit, hKernel,
                         wKernel, repeatsElem, maxIndexReg, hDilation, wDilation);
                this->template ConvertIndexWithoutPadAlignInt32VF<IS_PAD>(
                    maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase,
                    maxIndexConvertReg, ncInputOffset, magic, shift);
                Reg::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
                Reg::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
            }
            int32_t wOffsetTail = ncInputOffset + hLoop * wInputActualAlignedPad * strideH +
                                  loopW * repeatsElem * wStride;
            ProcessW(computeAddr, wOffsetTail, static_cast<uint16_t>(wInputActualAlignedPad), indexRegInit, hKernel,
                     wKernel, tailRepeatsElem, maxIndexReg, hDilation, wDilation);
            this->template ConvertIndexWithoutPadAlignInt32VF<IS_PAD>(
                maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase,
                maxIndexConvertReg, ncInputOffset, magic, shift);
            Reg::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
            Reg::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::MultiRowGather(
    __local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ int32_t* argmaxAddr)
{
    uint32_t wOutputActual = static_cast<uint32_t>(Base::wArgmaxActual_);
    uint16_t hKernel = Base::kernelH_;
    uint16_t wKernel = Base::kernelW_;
    uint32_t wStride = Base::strideW_;
    uint32_t rate2D = wInputActualAlignedPad_ * Base::strideH_;
    uint16_t hBatchCount = Base::vlArgmax_ / wOutputActual;
    uint16_t hLoopTimes = static_cast<uint16_t>(Base::hArgmaxActual_) / hBatchCount;
    uint16_t hTail = static_cast<uint16_t>(Base::hArgmaxActual_) - hLoopTimes * hBatchCount;
    if (hTail == 0) {
        hLoopTimes = hLoopTimes - 1;
        hTail = hBatchCount;
    }
    uint16_t repeatsElem = hBatchCount * wOutputActual;
    uint16_t tailRepeatsElem = hTail * wOutputActual;
    uint32_t highAxisActual = static_cast<uint32_t>(forwardhighAxisActual_);
    uint32_t hInputActualPad = static_cast<uint32_t>(hInputActualPad_);
    uint32_t wInputActualAlignedPad = static_cast<uint32_t>(wInputActualAlignedPad_);
    uint32_t hOutputActual = static_cast<uint32_t>(Base::hArgmaxActual_);
    uint32_t hDilation = Base::dilationH_;
    uint32_t wDilation = Base::dilationW_;
    int32_t left = static_cast<int32_t>(Base::wArgmaxActualStart_ * wStride - Base::padW_);
    int32_t hIndexBase = static_cast<int32_t>(Base::hArgmaxActualStart_ * Base::strideH_ - Base::padH_);
    int32_t wInput = static_cast<int32_t>(Base::wOutput_);

    uint32_t magic = 0;
    uint32_t shift = 0;
    GetUintDivMagicAndShift<uint32_t>(magic, shift, wInputActualAlignedPad);

    this->template MultiRowGatherSimdVf<IS_PAD>(computeAddr, argmaxAddr, wOutputActual, hKernel, wKernel, wStride,
                                                rate2D, hBatchCount, hLoopTimes, repeatsElem, tailRepeatsElem,
                                                highAxisActual, hInputActualPad, wInputActualAlignedPad, hDilation,
                                                wDilation, left, hIndexBase, wInput, magic, shift);
    return;
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__simd_vf__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::MultiRowGatherSimdVf(
    __ubuf__ TYPE_ORIG_X* computeAddr, __ubuf__ int32_t* argmaxAddr, uint32_t wOutputActual, uint16_t hKernel,
    uint16_t wKernel, uint32_t wStride, uint32_t rate2D, uint16_t hBatchCount, uint16_t hLoopTimes,
    uint16_t repeatsElem, uint16_t tailRepeatsElem, uint32_t highAxisActual, uint32_t hInputActualPad,
    uint32_t wInputActualAlignedPad, uint32_t hDilation, uint32_t wDilation, int32_t left, int32_t hIndexBase,
    int32_t wInput, uint32_t magic, uint32_t shift)
{
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<int32_t> maxIndexReg;
    Reg::RegTensor<int32_t> maxIndexConvertReg;
    Reg::UnalignReg u1;
    __local_mem__ int32_t* argmaxAddrLocal = argmaxAddr;
    PoolUtils::Index::GenGatterIndex2DVF<int32_t>(indexReg, static_cast<int32_t>(rate2D),
                                                  static_cast<int32_t>(wOutputActual), static_cast<int32_t>(wStride));
    for (uint16_t nc = 0; nc < static_cast<uint16_t>(highAxisActual); nc++) {
        int32_t ncInputOffset = nc * hInputActualPad * wInputActualAlignedPad;
        for (uint16_t hLoop = 0; hLoop < hLoopTimes; hLoop++) {
            int32_t wOffset = ncInputOffset + hLoop * hBatchCount * rate2D;
            ProcessW(computeAddr, wOffset, static_cast<uint16_t>(wInputActualAlignedPad), indexReg, hKernel, wKernel,
                     repeatsElem, maxIndexReg, hDilation, wDilation);
            this->template ConvertIndexWithoutPadAlignInt32VF<IS_PAD>(
                maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase,
                maxIndexConvertReg, ncInputOffset, magic, shift);
            Reg::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
            Reg::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
        }
        int32_t wOffsetTail = ncInputOffset + hLoopTimes * hBatchCount * rate2D;
        ProcessW(computeAddr, wOffsetTail, static_cast<uint16_t>(wInputActualAlignedPad), indexReg, hKernel, wKernel,
                 tailRepeatsElem, maxIndexReg, hDilation, wDilation);
        this->template ConvertIndexWithoutPadAlignInt32VF<IS_PAD>(
            maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase, maxIndexConvertReg,
            ncInputOffset, magic, shift);
        Reg::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
        Reg::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__simd_vf__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::MultiNcGatherSimdVf(
    __ubuf__ TYPE_ORIG_X* computeAddr, __ubuf__ int32_t* argmaxAddr, uint16_t ncLoopTimes, uint32_t ncBatchCount,
    uint16_t repeatsElem, uint16_t tailRepeatsElem, uint16_t hKernel, uint16_t wKernel, uint16_t wInputActualAlignedPad,
    int32_t rate3D, int32_t num2D, int32_t rate2D, int32_t wOutputActual, int32_t wStride, int32_t left, int32_t wInput,
    int32_t hIndexBase, int32_t inputNcSize, uint32_t hDilation, uint32_t wDilation, uint32_t magicNc, uint32_t shiftNc,
    uint32_t magicWStride, uint32_t shiftWStride)
{
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<int32_t> maxIndexReg;
    Reg::RegTensor<int32_t> maxIndexConvertReg;
    Reg::UnalignReg u1;
    __local_mem__ int32_t* argmaxAddrLocal = argmaxAddr;
    // 3D gather 索引与 nc batch 无关，整个 tile 只生成一次；argmax 按 batch 连续写入。
    PoolUtils::Index::GenGatterIndex3DVF<int32_t>(indexReg, rate3D, num2D, rate2D, wOutputActual, wStride);

    for (uint16_t nc = 0; nc < ncLoopTimes; nc++) {
        uint32_t ncInputOffset = nc * ncBatchCount * static_cast<uint32_t>(rate3D);
        int32_t hOffset = static_cast<int32_t>(nc) * static_cast<int32_t>(ncBatchCount) * rate3D;
        ProcessW((__local_mem__ TYPE_ORIG_X*)computeAddr, hOffset, wInputActualAlignedPad, indexReg, hKernel, wKernel,
                 repeatsElem, maxIndexReg, hDilation, wDilation);
        this->template ConvertIndexWithoutPadAlignNcInt32<IS_PAD>(
            maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase, maxIndexConvertReg,
            static_cast<int32_t>(ncInputOffset), num2D, inputNcSize, magicNc, shiftNc, magicWStride, shiftWStride);
        Reg::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, repeatsElem);
        Reg::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
    }

    uint32_t ncInputOffsetTail = ncLoopTimes * ncBatchCount * static_cast<uint32_t>(rate3D);
    int32_t hOffsetTail = static_cast<int32_t>(ncLoopTimes) * static_cast<int32_t>(ncBatchCount) * rate3D;
    ProcessW((__local_mem__ TYPE_ORIG_X*)computeAddr, hOffsetTail, wInputActualAlignedPad, indexReg, hKernel, wKernel,
             tailRepeatsElem, maxIndexReg, hDilation, wDilation);
    this->template ConvertIndexWithoutPadAlignNcInt32<IS_PAD>(
        maxIndexReg, static_cast<uint32_t>(wInputActualAlignedPad), left, wInput, hIndexBase, maxIndexConvertReg,
        static_cast<int32_t>(ncInputOffsetTail), num2D, inputNcSize, magicNc, shiftNc, magicWStride, shiftWStride);
    Reg::DataCopyUnAlign(argmaxAddrLocal, maxIndexConvertReg, u1, tailRepeatsElem);
    Reg::DataCopyUnAlignPost(argmaxAddrLocal, u1, 0);
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool IS_PAD>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::MultiNcGather(
    __local_mem__ TYPE_ORIG_X* computeAddr, __local_mem__ int32_t* argmaxAddr)
{
    uint16_t wKernel = Base::kernelW_;
    uint16_t hKernel = Base::kernelH_;
    uint32_t wStride = Base::strideW_;
    // rate3D/rate2D 是 calcbuffer 平面/行跨度，极端 stride+kernel 下可能超过 16 bit。
    int32_t rate3D = static_cast<int32_t>(hInputActualPad_ * wInputActualAlignedPad_);
    uint16_t num2D = static_cast<uint16_t>(Base::hArgmaxActual_ * Base::wArgmaxActual_);
    int32_t rate2D = static_cast<int32_t>(Base::strideH_ * wInputActualAlignedPad_);
    uint16_t wOutputActual = static_cast<uint16_t>(Base::wArgmaxActual_);
    uint16_t eachBatchCount = static_cast<uint16_t>(Base::hArgmaxActual_ * Base::wArgmaxActual_);
    uint16_t ncBatchCount = Base::vlArgmax_ / eachBatchCount;
    uint16_t ncLoopTimes = static_cast<uint16_t>(forwardhighAxisActual_) / ncBatchCount;
    uint16_t ncTail = static_cast<uint16_t>(forwardhighAxisActual_) - ncLoopTimes * ncBatchCount;
    if (ncTail == 0) {
        ncLoopTimes = ncLoopTimes - 1;
        ncTail = ncBatchCount;
    }
    uint16_t repeatsElem = ncBatchCount * eachBatchCount;
    uint16_t tailRepeatsElem = ncTail * eachBatchCount;
    uint32_t wInputActualAlignedPad = static_cast<uint32_t>(wInputActualAlignedPad_);
    uint32_t hDilation = Base::dilationH_;
    uint32_t wDilation = Base::dilationW_;
    int32_t left = static_cast<int32_t>(Base::wArgmaxActualStart_ * wStride - Base::padW_);
    int32_t hIndexBase = static_cast<int32_t>(Base::hArgmaxActualStart_ * Base::strideH_ - Base::padH_);
    int32_t wInput = static_cast<int32_t>(Base::wOutput_);
    int32_t inputNcSize = static_cast<int32_t>(rate3D);

    uint32_t magicWStride = 0;
    uint32_t shiftWStride = 0;
    uint32_t magicNc = 0;
    uint32_t shiftNc = 0;
    GetUintDivMagicAndShift<uint32_t>(magicWStride, shiftWStride, wInputActualAlignedPad);
    GetUintDivMagicAndShift<uint32_t>(magicNc, shiftNc, static_cast<uint32_t>(num2D));

    this->template MultiNcGatherSimdVf<IS_PAD>(
        (__ubuf__ TYPE_ORIG_X*)computeAddr, (__ubuf__ int32_t*)argmaxAddr, ncLoopTimes,
        static_cast<uint32_t>(ncBatchCount), repeatsElem, tailRepeatsElem, hKernel, wKernel,
        static_cast<uint16_t>(wInputActualAlignedPad), rate3D, static_cast<int32_t>(num2D), rate2D,
        static_cast<int32_t>(wOutputActual), static_cast<int32_t>(wStride), left, wInput, hIndexBase, inputNcSize,
        hDilation, wDilation, magicNc, shiftNc, magicWStride, shiftWStride);
    return;
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::DupAndCopyToCalcBuffer(
    __local_mem__ TYPE_ORIG_X* dstAddr, __local_mem__ TYPE_ORIG_X* srcAddr)
{
    constexpr uint32_t rowAlignElem = BLOCK_SIZE / sizeof(TYPE_ORIG_X);
    uint32_t wInputPadActualAlign = CeilDivision(wInputActualPad_, rowAlignElem) * rowAlignElem;
    uint16_t hPad = topOffsetToInputTop_;
    uint16_t wPad = leftOffsetToInputLeft_;
    uint16_t hRows = hInputActualNoPad_;
    uint16_t wCols = wInputActualNoPad_;
    uint16_t highAxis = Base::highAxisActual_;
    uint32_t srcBatchStride = hInputActualNoPad_ * CeilDivision(wInputActualNoPad_, rowAlignElem) * rowAlignElem;
    uint32_t dstBatchStride = (hRows + hPad + downOffsetToInputDown_) * wInputPadActualAlign;
    uint32_t srcRowStride = CeilDivision(wInputActualNoPad_, rowAlignElem) * rowAlignElem;
    uint32_t dstRowStride = wInputPadActualAlign;
    uint32_t repeatElm = rowAlignElem;
    uint16_t loopCols = wCols / repeatElm;
    uint16_t tailCols = wCols % repeatElm;
    DupAndCopyToCalcBufferSimdVf(dstAddr, srcAddr, highAxis, hRows, hPad, wPad, loopCols, tailCols, repeatElm,
                                 srcBatchStride, srcRowStride, dstBatchStride, dstRowStride, wInputPadActualAlign,
                                 downOffsetToInputDown_);
};

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::DupAndCopyToCalcBufferSimdVf(
    __ubuf__ TYPE_ORIG_X* dstAddr, __ubuf__ TYPE_ORIG_X* srcAddr, uint16_t highAxis, uint16_t hRows, uint16_t hPad,
    uint16_t wPad, uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride,
    uint32_t srcRowStride, uint32_t dstBatchStride, uint32_t dstRowStride, uint32_t wInputPadActualAlign,
    uint16_t downOffset)
{
    Reg::RegTensor<TYPE_ORIG_X> v0;
    if constexpr (std::negation<std::is_integral<TYPE_ORIG_X>>::value) {
        // 整数 pad 继续走 Forward 里的高层 Duplicate，避免 16-bit Reg 填充的布局差异。
        PoolUtils::Compute::DuplicateNegInfRegVF<TYPE_ORIG_X>(v0);
        Reg::MaskReg preg = Reg::CreateMask<TYPE_ORIG_X, Reg::MaskPattern::ALL>();
        uint32_t padTotal = static_cast<uint32_t>(highAxis) * (hRows + hPad + downOffset) * wInputPadActualAlign;
        uint16_t padFull = static_cast<uint16_t>(padTotal / repeatElm);
        uint16_t padTail = static_cast<uint16_t>(padTotal % repeatElm);
        __ubuf__ TYPE_ORIG_X* padAddr = dstAddr;
        for (uint16_t i = 0; i < padFull; i++) {
            Reg::StoreAlign<TYPE_ORIG_X, Reg::PostLiteral::POST_MODE_UPDATE>(padAddr, v0, repeatElm, preg);
        }
        // 尾批用 mask 处理：padTail=0 时 UpdateMask 产生全 0 mask，store 自然无效，避免 VF 内运行时 if。
        uint32_t padTailCount = padTail;
        preg = Reg::UpdateMask<TYPE_ORIG_X>(padTailCount);
        Reg::StoreAlign<TYPE_ORIG_X, Reg::PostLiteral::POST_MODE_UPDATE>(padAddr, v0, repeatElm, preg);
    }
    Reg::UnalignRegForStore u0;
    for (uint16_t i = 0; i < highAxis; i++) {
        for (uint16_t j = 0; j < hRows; j++) {
            __ubuf__ TYPE_ORIG_X* curSrcAddr = srcAddr + i * srcBatchStride + j * srcRowStride;
            __ubuf__ TYPE_ORIG_X* curDstAddr = dstAddr + i * dstBatchStride + (j + hPad) * dstRowStride + wPad;
            for (uint16_t k = 0; k < loopCols; k++) {
                Reg::LoadAlign<TYPE_ORIG_X, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                Reg::StoreUnAlign(curDstAddr, v0, u0, repeatElm);
            }
            Reg::LoadAlign<TYPE_ORIG_X, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
            Reg::StoreUnAlign(curDstAddr, v0, u0, tailCols);
            Reg::StoreUnAlignPost(curDstAddr, u0, 0);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void PoolGradNCHWSmallKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::IdentityCopyGradToOutput()
{
    // identity：out_grad 与 grad 同形。grad 先经 VECIN 队列完成 MTE2 同步，再由 Vector 写入
    // output 队列，保证 MTE3 不会读到未完成的 GM->UB 数据。
    LocalTensor<TYPE_ORIG_X> gradLocal = Base::gradQue_.template DeQue<TYPE_ORIG_X>();
    LocalTensor<TYPE_ORIG_X> yLocal = Base::outputQue_.template AllocTensor<TYPE_ORIG_X>();
    uint32_t calCount = static_cast<uint32_t>(Base::highAxisActual_ * Base::hOutputActual_ * Base::wOutputAligned_);

    AscendC::Copy(yLocal, gradLocal, calCount);

    Base::outputQue_.EnQue(yLocal);
    Base::gradQue_.FreeTensor(gradLocal);
}

} // namespace MaxPoolV3GradNCHWSmallKernelNameSpace
#endif // MAX_POOL_V3_GRAD_NCHW_SMALL_KERNEL_H
