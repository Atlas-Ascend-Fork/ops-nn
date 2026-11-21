/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL_V3_GRAD_NHWC_INT_SMALL_KERNEL_H_
#define MAX_POOL_V3_GRAD_NHWC_INT_SMALL_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "max_pool_v3_grad_nhwc_int_impl_scatter.h"

namespace MaxPoolV3GradNHWCIntNameSpace {

using namespace AscendC;

// Small kernels, including the large-C path, load the complete tile and overlapping window halo.
template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
class MaxPoolV3GradNHWCSmallKernel : public MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE> {
    using Base = MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>;

public:
    __aicore__ inline MaxPoolV3GradNHWCSmallKernel() {}

    __aicore__ inline void Init(GM_ADDR origX, GM_ADDR origY, GM_ADDR grad, GM_ADDR y,
                                const MaxPoolV3GradNHWCTilingData& tilingData, TPipe& pipe);
    __aicore__ inline void Process();

    using Base::argmaxBuff_;
    using Base::argmaxBufferSize_;
    using Base::BLOCK_SIZE;
    using Base::blockIdx_;
    using Base::cAligned_;
    using Base::cAxisGradOffset_;
    using Base::cAxisIndex_;
    using Base::cDim_;
    using Base::cOutputActual_;
    using Base::curCoreProcessNum_;
    using Base::gradGm_;
    using Base::gradQue_;
    using Base::hArgmaxActual_;
    using Base::hArgmaxActualEnd_;
    using Base::hArgmaxActualStart_;
    using Base::hAxisIndex_;
    using Base::hOutputActual_;
    using Base::isPad_;
    using Base::mte3ToMte2Event_;
    using Base::nAxisIndex_;
    using Base::nOutputActual_;
    using Base::outputQue_;
    using Base::tilingData_;
    using Base::V_REG_SIZE;
    using Base::vlT2_;
    using Base::vToMte2Event_;
    using Base::wArgmaxActual_;
    using Base::wArgmaxActualEnd_;
    using Base::wArgmaxActualStart_;
    using Base::wAxisIndex_;
    using Base::wOutputActual_;
    using Base::yGm_;

private:
    __aicore__ inline void ForwardScalarCompute();
    __aicore__ inline void ForwardCopyIn();
    __aicore__ inline void Forward();
    __aicore__ inline void ForwardComputeTile();

    __simd_callee__ inline void MaxPoolSingleChannelWithArgmax(__ubuf__ INDEX_T* argmaxAddr, __ubuf__ T* srcAddr,
                                                               uint16_t kH, uint16_t kW, uint32_t rowStride,
                                                               uint32_t colStride, uint16_t repeatElms, int32_t curInH,
                                                               int32_t curInW, int32_t initialIndex,
                                                               int32_t wOutputPlane);
    __simd_vf__ inline void ProcessBigCPosition(__ubuf__ INDEX_T* argmaxAddr, __ubuf__ T* srcAddr, uint16_t cLoop,
                                                uint16_t tailNum, uint16_t vl, uint16_t hKernel, uint16_t wKernel,
                                                uint32_t rowStride, uint32_t colStride, int32_t curInH, int32_t curInW,
                                                int32_t initialIndex, int32_t wOutputPlane);
    __aicore__ inline void ChunkCGather(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void SingleRowGather(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultiRowGather(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr);
    __aicore__ inline void MultiNcGather(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr);

    template <bool IS_PAD>
    __simd_vf__ inline void ProcessWNHWC(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem,
                                         uint16_t cChunk, uint16_t hKernel, uint16_t wKernel, uint32_t colStride,
                                         uint32_t rowStride, uint32_t wBaseStep, int32_t hOrigin, int32_t wOrigin,
                                         int32_t wStride, int32_t hOutputPlane, int32_t wOutputPlane);

    template <bool IS_PAD>
    __simd_vf__ inline void ProcessWNHWC2D(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem,
                                           uint16_t cChunk, uint16_t wOut, uint16_t hKernel, uint16_t wKernel,
                                           uint32_t colStride, uint32_t rowStride, uint32_t wBaseStep,
                                           uint32_t hBaseStep, int32_t hOrigin, int32_t wOrigin, int32_t wStride,
                                           int32_t hStride, int32_t hOutputPlane, int32_t wOutputPlane);

    template <bool IS_PAD>
    __simd_vf__ inline void ProcessWNHWC3D(__ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem,
                                           uint16_t cChunk, uint16_t wOut, uint16_t hOut, uint16_t hKernel,
                                           uint16_t wKernel, uint32_t colStride, uint32_t rowStride, uint32_t wBaseStep,
                                           uint32_t hBaseStep, uint32_t ncBaseStep, int32_t hOrigin, int32_t wOrigin,
                                           int32_t wStride, int32_t hStride, int32_t hOutputPlane,
                                           int32_t wOutputPlane);

    __simd_callee__ inline void ProcessWNHWCKernel(__ubuf__ T* xAddr, uint32_t colStride, uint32_t rowStride,
                                                   Reg::RegTensor<int32_t>& indexReg, uint16_t hKernel,
                                                   uint16_t wKernel, uint16_t repeatElem,
                                                   Reg::RegTensor<INDEX_T>& argmaxHStart,
                                                   Reg::RegTensor<INDEX_T>& argmaxWStart,
                                                   Reg::RegTensor<INDEX_T>& argmaxHRes,
                                                   Reg::RegTensor<INDEX_T>& argmaxWRes);

    template <bool IS_PAD>
    __simd_callee__ inline void ComposeAndStoreArgmax(Reg::RegTensor<INDEX_T>& argmaxHRes,
                                                      Reg::RegTensor<INDEX_T>& argmaxWRes, __ubuf__ INDEX_T* argmaxAddr,
                                                      uint16_t repeatElem, int32_t hOutputPlane, int32_t wOutputPlane);

    __simd_vf__ inline void DupCalcLowestVf(__ubuf__ T* calcAddr, uint64_t totalElems, uint32_t vl);
    __simd_vf__ inline void CopyValidToCalcVf(__ubuf__ T* calcAddr, __ubuf__ T* srcAddr, uint32_t nActual,
                                              uint32_t hNoPad, uint32_t wNoPadRowElems, uint32_t hPad,
                                              uint32_t wPadRowElems, uint32_t leftElems, uint32_t top, uint32_t vl);

    template <typename... Args>
    __aicore__ inline void DispatchProcessWNHWC(Args... args)
    {
        if (isPad_) {
            ProcessWNHWC<true>(args...);
        } else {
            ProcessWNHWC<false>(args...);
        }
    }

    template <typename... Args>
    __aicore__ inline void DispatchProcessWNHWC2D(Args... args)
    {
        if (isPad_) {
            ProcessWNHWC2D<true>(args...);
        } else {
            ProcessWNHWC2D<false>(args...);
        }
    }

    template <typename... Args>
    __aicore__ inline void DispatchProcessWNHWC3D(Args... args)
    {
        if (isPad_) {
            ProcessWNHWC3D<true>(args...);
        } else {
            ProcessWNHWC3D<false>(args...);
        }
    }

    GlobalTensor<T> xGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<TPosition::VECCALC> inputCalcBuff_;

    TBufPool<TPosition::VECCALC> forwardBufPool_;
    TBufPool<TPosition::VECCALC> backwardBufPool_;
    int64_t totalStageBufferSize_ = 0;

    bool isBigC_ = false;

    __ubuf__ T* xForwardAddr_ = nullptr;

    int64_t inputStrideW_ = 0;
    int64_t inputStrideH_ = 0;
    int64_t forwardInputOffset_ = 0;

    uint32_t ncBaseStep_ = 0;
    uint32_t hBaseStep_ = 0;
    int64_t hInputActualPad_ = 0;
    int64_t wInputActualPad_ = 0;

    int64_t leftOffsetToInputLeft_ = 0;
    int64_t rightOffsetToInputRight_ = 0;
    int64_t topOffsetToInputTop_ = 0;
    int64_t downOffsetToInputDown_ = 0;
    int64_t hInputActualNoPad_ = 0;
    int64_t wInputActualNoPad_ = 0;
};

} // namespace MaxPoolV3GradNHWCIntNameSpace
#endif // MAX_POOL_V3_GRAD_NHWC_INT_SMALL_KERNEL_H_
