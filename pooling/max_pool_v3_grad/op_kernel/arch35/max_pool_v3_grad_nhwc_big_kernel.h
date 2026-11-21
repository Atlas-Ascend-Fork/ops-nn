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
 * \file max_pool_v3_grad_nhwc_big_kernel.h
 * \brief NHWC forward implementation for pooling kernels with at least 128 elements.
 */

#ifndef MAX_POOL_V3_GRAD_NHWC_BIG_KERNEL_H_
#define MAX_POOL_V3_GRAD_NHWC_BIG_KERNEL_H_

#include "max_pool_v3_grad_nhwc_impl_scatter.h"
#include "pool_utils/arch35/index/max_pool_with_argmax_index.h"
#include "pool_utils/arch35/compute/max_pool_negative_value.h"

namespace MaxPoolV3GradNHWCNameSpace {

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
class MaxPoolV3GradNHWCBigKernel : public MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE> {
    using Base = MaxPoolV3GradNHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>;

public:
    __aicore__ inline MaxPoolV3GradNHWCBigKernel() {}

    __aicore__ inline void Init(GM_ADDR origX, GM_ADDR origY, GM_ADDR grad, GM_ADDR y,
                                const MaxPoolV3GradNHWCTilingData& tilingData, TPipe& pipe);
    __aicore__ inline void Process();

    using Base::argmaxBuff_;
    using Base::argmaxBufferSize_;
    using Base::blockIdx_;
    using Base::cAligned_;
    using Base::cAxisGradOffset_;
    using Base::cDim_;
    using Base::cOutputActual_;
    using Base::curCoreProcessNum_;
    using Base::hArgmaxActual_;
    using Base::hArgmaxActualStart_;
    using Base::helpBuf_;
    using Base::nAxisIndex_;
    using Base::nOutputActual_;
    using Base::tilingData_;
    using Base::wArgmaxActual_;
    using Base::wArgmaxActualStart_;

private:
    __aicore__ inline void ForwardBigKernel();
    __aicore__ inline void CalcRowKernelSize(int64_t nIdx, int64_t hIdx, int64_t& curKH, int64_t& rowOriginFlat,
                                             int64_t& rowInOffset);
    __aicore__ inline void CalcKernelSize(int64_t nIdx, int64_t hIdx, int64_t wIdx, int64_t& curKH, int64_t& curKW,
                                          int64_t& curInOffset, int64_t& curOriginIndex);
    __aicore__ inline void NoSplitKernelProcess(int64_t curKH, int64_t curKW, int64_t curInOffset,
                                                int64_t curOriginIndex, int64_t argmaxOffset);
    __aicore__ inline void SplitKernelProcess(int64_t curKH, int64_t curKW, int64_t curInOffset, int64_t curOriginIndex,
                                              int64_t argmaxOffset);
    __aicore__ inline void InitMergeBuffer(int64_t argmaxOffset, int64_t curOriginIndex);
    template <bool MERGE = false>
    __aicore__ inline void ComputeSingleArgmax(__local_mem__ T* xSrcBase, int64_t rowStride, int64_t curKW,
                                               int64_t curKH, int64_t curOriginIndex, int64_t argmaxOffset);
    __aicore__ inline void CopyInMultiRows_2D(int64_t offset, int64_t blockLen, int64_t blockCount);

    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<TPosition::VECCALC> maxValBuf_;
    int64_t maxCount_ = 1;

    TBufPool<TPosition::VECCALC> forwardBufPool_;
    TBufPool<TPosition::VECCALC> backwardBufPool_;
    int64_t totalStageBufferSize_ = 0;
};

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::Init(
    GM_ADDR origX, GM_ADDR origY, GM_ADDR grad, GM_ADDR y, const MaxPoolV3GradNHWCTilingData& tilingData, TPipe& pipe)
{
    (void)origY;
    Base::ParseTilingData(tilingData);
    Base::vToMte2Event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    Base::mte3ToMte2Event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    Base::blockIdx_ = GetBlockIdx();
    Base::curCoreProcessNum_ = (Base::blockIdx_ == tilingData_.base.usedCoreNum - 1) ?
                                   tilingData_.base.tailCoreProcessNum :
                                   tilingData_.base.normalCoreProcessNum;

    xGm_.SetGlobalBuffer((__gm__ T*)origX);
    Base::gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    Base::yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe.InitBuffer(Base::argmaxBuff_, tilingData_.base.argmaxBufferSize);
    pipe.InitBuffer(Base::helpBuf_, HELP_BUFFER);

    maxCount_ = tilingData_.base.inputBufferSize / static_cast<int64_t>(sizeof(T));
    // Size the merge buffer from the largest channel tile, matching the host budget.
    // cAligned_ is available only after ScalarCompute.
    const int64_t blockElems = platform::GetUbBlockSize() / sizeof(T);
    const int64_t cAlignedMax = (tilingData.cOutputInner + blockElems - 1) / blockElems * blockElems;
    int64_t mergeBufSize = cAlignedMax * static_cast<int64_t>(sizeof(T));
    int64_t forwardSize = tilingData_.base.inputBufferSize * BUFFER_NUM + mergeBufSize;
    int64_t backwardSize = tilingData_.base.gradBufferSize + tilingData_.base.outputBufferSize;
    totalStageBufferSize_ = (forwardSize > backwardSize) ? forwardSize : backwardSize;

    pipe.InitBufPool(forwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_));
    pipe.InitBufPool(backwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_), forwardBufPool_);

    forwardBufPool_.InitBuffer(inputQue_, BUFFER_NUM, static_cast<uint32_t>(tilingData_.base.inputBufferSize));
    forwardBufPool_.InitBuffer(maxValBuf_, static_cast<uint32_t>(mergeBufSize));
    backwardBufPool_.InitBuffer(Base::outputQue_, 1, static_cast<uint32_t>(tilingData_.base.outputBufferSize));
    backwardBufPool_.InitBuffer(Base::gradQue_, 1, static_cast<uint32_t>(tilingData_.base.gradBufferSize));
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::Process()
{
    if (blockIdx_ >= tilingData_.base.usedCoreNum) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; ++loopNum) {
        Base::ScalarCompute(loopNum);

        if (hArgmaxActual_ == 0 || wArgmaxActual_ == 0) {
            Base::ProcessNoArgmaxBlock();
            Base::Mte3Drain();
            continue;
        }

        ForwardBigKernel();

        Base::VDrainToMte2();

        Base::CopyInGrad();
        Base::Backward();
        Base::CopyOut();
        Base::Mte3Drain();
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::ForwardBigKernel()
{
    LocalTensor<INDEX_T> argmaxLocal = argmaxBuff_.template Get<INDEX_T>();
    Duplicate<INDEX_T>(argmaxLocal, static_cast<INDEX_T>(-1),
                       static_cast<uint32_t>(argmaxBufferSize_ / sizeof(INDEX_T)));
    PipeBarrier<PIPE_V>();

    for (int64_t nIdx = 0; nIdx < nOutputActual_; nIdx++) {
        for (int64_t hIdx = 0; hIdx < hArgmaxActual_; hIdx++) {
            for (int64_t wIdx = 0; wIdx < wArgmaxActual_; wIdx++) {
                int64_t curKH = 0;
                int64_t curKW = 0;
                int64_t curInOffset = 0;
                int64_t curOriginIndex = 0;
                CalcKernelSize(nIdx, hIdx, wIdx, curKH, curKW, curInOffset, curOriginIndex);
                if (curKH <= 0 || curKW <= 0) {
                    continue;
                }
                const int64_t argmaxOffset = (nIdx * hArgmaxActual_ + hIdx) * wArgmaxActual_ * cOutputActual_ +
                                             wIdx * cOutputActual_;
                if (curKH * curKW * cAligned_ <= maxCount_) {
                    NoSplitKernelProcess(curKH, curKW, curInOffset, curOriginIndex, argmaxOffset);
                } else {
                    SplitKernelProcess(curKH, curKW, curInOffset, curOriginIndex, argmaxOffset);
                }
            }
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::CalcRowKernelSize(
    int64_t nIdx, int64_t hIdx, int64_t& curKH, int64_t& rowOriginFlat, int64_t& rowInOffset)
{
    const int64_t hOut = hArgmaxActualStart_ + hIdx;

    int64_t curOriginH = hOut * tilingData_.base.hStride - tilingData_.base.padH;
    curKH = tilingData_.base.hKernel;

    if (curOriginH < 0) {
        curKH += curOriginH;
        curOriginH = 0;
    }
    if (curOriginH + curKH > tilingData_.base.hOutput) {
        curKH = tilingData_.base.hOutput - curOriginH;
    }

    rowOriginFlat = curOriginH * tilingData_.base.wOutput;
    const int64_t nOffset = (nAxisIndex_ * tilingData_.base.highAxisInner + nIdx) * tilingData_.base.hOutput *
                            tilingData_.base.wOutput;
    rowInOffset = (nOffset + rowOriginFlat) * cDim_ + cAxisGradOffset_;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::CalcKernelSize(
    int64_t nIdx, int64_t hIdx, int64_t wIdx, int64_t& curKH, int64_t& curKW, int64_t& curInOffset,
    int64_t& curOriginIndex)
{
    CalcRowKernelSize(nIdx, hIdx, curKH, curOriginIndex, curInOffset);

    const int64_t wOut = wArgmaxActualStart_ + wIdx;
    int64_t curOriginW = wOut * tilingData_.base.wStride - tilingData_.base.padW;
    curKW = tilingData_.base.wKernel;
    if (curOriginW < 0) {
        curKW += curOriginW;
        curOriginW = 0;
    }
    if (curOriginW + curKW > tilingData_.base.wOutput) {
        curKW = tilingData_.base.wOutput - curOriginW;
    }

    curOriginIndex += curOriginW;
    curInOffset += curOriginW * cDim_;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::CopyInMultiRows_2D(int64_t offset,
                                                                                                  int64_t blockLen,
                                                                                                  int64_t blockCount)
{
    LocalTensor<T> xLocal = inputQue_.template AllocTensor<T>();
    // Load window rows [H][W][C] from GM into UB [H][W][cAligned].
    DataCopyPadExtParams<T> padExtParams = {false, 0, 0, 0};
    DataCopyExtParams extParams;
    extParams.blockCount = static_cast<uint32_t>(blockLen);
    extParams.blockLen = static_cast<uint32_t>(cOutputActual_ * sizeof(T));
    extParams.srcStride = static_cast<uint32_t>((cDim_ - cOutputActual_) * sizeof(T));
    // UB blocks advance with aligned block lengths; no extra destination stride is needed.
    extParams.dstStride = 0;
    extParams.rsv = 0;
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop1Size = static_cast<uint32_t>(blockCount);
    loopParams.loop1SrcStride = static_cast<uint64_t>(tilingData_.base.wOutput * cDim_ * sizeof(T));
    loopParams.loop2SrcStride = static_cast<uint64_t>(blockCount * blockLen * cAligned_ * sizeof(T));
    loopParams.loop1DstStride = static_cast<uint64_t>(blockLen * cAligned_ * sizeof(T));
    loopParams.loop2DstStride = static_cast<uint64_t>(blockCount * blockLen * cAligned_ * sizeof(T));
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inputQue_.template EnQue<T>(xLocal);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::NoSplitKernelProcess(
    int64_t curKH, int64_t curKW, int64_t curInOffset, int64_t curOriginIndex, int64_t argmaxOffset)
{
    CopyInMultiRows_2D(curInOffset, curKW, curKH);
    LocalTensor<T> xLocal = inputQue_.template DeQue<T>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    const int64_t rowStride = curKW * cAligned_;
    ComputeSingleArgmax<false>(xLocalAddr, rowStride, curKW, curKH, curOriginIndex, argmaxOffset);
    inputQue_.template FreeTensor<T>(xLocal);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::SplitKernelProcess(
    int64_t curKH, int64_t curKW, int64_t curInOffset, int64_t curOriginIndex, int64_t argmaxOffset)
{
    InitMergeBuffer(argmaxOffset, curOriginIndex);
    if (curKW <= 0 || curKH <= 0 || maxCount_ <= 0) {
        return;
    }

    if (curKW * cAligned_ <= maxCount_) {
        const int64_t hFactor = maxCount_ / (curKW * cAligned_);
        for (int64_t hStart = 0; hStart < curKH; hStart += hFactor) {
            const int64_t curh = (curKH - hStart < hFactor) ? curKH - hStart : hFactor;
            const int64_t rowAdvance = hStart * tilingData_.base.wOutput;
            CopyInMultiRows_2D(curInOffset + rowAdvance * cDim_, curKW, curh);
            LocalTensor<T> xLocal = inputQue_.template DeQue<T>();
            __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
            const int64_t rowStride = curKW * cAligned_;
            ComputeSingleArgmax<true>(xLocalAddr, rowStride, curKW, curh, curOriginIndex + rowAdvance, argmaxOffset);
            inputQue_.template FreeTensor<T>(xLocal);
        }
    } else {
        const int64_t wFactor = maxCount_ / cAligned_;
        for (int64_t h = 0; h < curKH; h++) {
            const int64_t rowAdvance = h * tilingData_.base.wOutput;
            for (int64_t wStart = 0; wStart < curKW; wStart += wFactor) {
                const int64_t curw = (curKW - wStart < wFactor) ? curKW - wStart : wFactor;
                CopyInMultiRows_2D(curInOffset + (rowAdvance + wStart) * cDim_, curw, 1);
                LocalTensor<T> xLocal = inputQue_.template DeQue<T>();
                __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
                const int64_t rowStride = curw * cAligned_;
                ComputeSingleArgmax<true>(xLocalAddr, rowStride, curw, 1, curOriginIndex + rowAdvance + wStart,
                                          argmaxOffset);
                inputQue_.template FreeTensor<T>(xLocal);
            }
        }
    }
}

template <typename T, typename INDEX_T>
__simd_vf__ inline void InitMergeBufferVF(__ubuf__ T* maxValAddr, __ubuf__ INDEX_T* argmaxAddr, uint32_t vc,
                                          int32_t curOriginIndex)
{
    Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::RegTensor<T> negInfReg;
    Reg::RegTensor<int32_t> initIdxReg;
    Reg::UnalignRegForStore u0;
    PoolUtils::Compute::DuplicateNegInfRegVF<T>(negInfReg);
    Reg::Duplicate(initIdxReg, static_cast<int32_t>(curOriginIndex), maskAll);
    __ubuf__ T* maxDst = maxValAddr;
    Reg::StoreUnAlign(maxDst, negInfReg, u0, static_cast<uint32_t>(vc));
    Reg::StoreUnAlignPost(maxDst, u0, 0);
    __ubuf__ uint32_t* idxDst = (__ubuf__ uint32_t*)argmaxAddr;
    Reg::RegTensor<uint32_t>& initStoreU32 = (Reg::RegTensor<uint32_t>&)initIdxReg;
    Reg::StoreUnAlign<uint32_t>(idxDst, initStoreU32, u0, static_cast<uint32_t>(vc));
    Reg::StoreUnAlignPost<uint32_t>(idxDst, u0, 0);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::InitMergeBuffer(int64_t argmaxOffset,
                                                                                               int64_t curOriginIndex)
{
    __ubuf__ INDEX_T* argmaxAddr = (__ubuf__ INDEX_T*)argmaxBuff_.template Get<INDEX_T>().GetPhyAddr();
    __ubuf__ T* maxValAddr = (__ubuf__ T*)maxValBuf_.template Get<T>().GetPhyAddr();
    constexpr int64_t vlT = platform::GetVRegSize() / sizeof(T);
    constexpr int64_t vlIdx = platform::GetVRegSize() / sizeof(int32_t);
    constexpr int64_t vlStep = vlT < vlIdx ? vlT : vlIdx;
    for (int64_t c0 = 0; c0 < cOutputActual_; c0 += vlStep) {
        const int64_t vc = (cOutputActual_ - c0 < vlStep) ? cOutputActual_ - c0 : vlStep;
        InitMergeBufferVF(maxValAddr + c0, argmaxAddr + argmaxOffset + c0, static_cast<uint32_t>(vc),
                          static_cast<int32_t>(curOriginIndex));
    }
}

template <typename T, typename INDEX_T, bool MERGE>
__simd_vf__ inline void ComputeSingleArgmaxVF(__ubuf__ T* xSrcBase, __ubuf__ INDEX_T* argmaxAddr,
                                              __ubuf__ T* maxValAddr, int32_t rowStride, uint16_t curKW, uint16_t curKH,
                                              int32_t curOriginIndex, int32_t cAlignedC, uint32_t vc, int32_t inW)
{
    Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::RegTensor<T> xReg;
    Reg::RegTensor<T> maxReg;
    Reg::RegTensor<int32_t> idxReg;
    Reg::RegTensor<int32_t> candIdx;
    Reg::MaskReg gtMask;
    Reg::MaskReg nanMask;
    Reg::MaskReg updMask;
    if constexpr (MERGE) {
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::UnalignRegForLoad ul0;
        __ubuf__ T* maxSrc = maxValAddr;
        Reg::LoadUnAlignPre(ul0, maxSrc);
        Reg::LoadUnAlign(maxReg, ul0, maxSrc, static_cast<uint32_t>(vc));
        __ubuf__ int32_t* idxSrc = (__ubuf__ int32_t*)argmaxAddr;
        Reg::LoadUnAlignPre(ul0, idxSrc);
        Reg::LoadUnAlign(idxReg, ul0, idxSrc, static_cast<uint32_t>(vc));
    } else {
        PoolUtils::Compute::DuplicateNegInfRegVF<T>(maxReg);
        Reg::Duplicate(idxReg, static_cast<int32_t>(curOriginIndex), maskAll);
    }
    Reg::RegTensor<int32_t> hFlatReg;
    for (uint16_t h = 0; h < static_cast<uint16_t>(curKH); h++) {
        const int32_t hFlat = static_cast<int32_t>(curOriginIndex) + h * inW;
        Reg::Duplicate(hFlatReg, hFlat, maskAll);
        for (uint16_t w = 0; w < static_cast<uint16_t>(curKW); w++) {
            auto srcAddr = xSrcBase + h * rowStride + w * cAlignedC;
            Reg::DataCopy(xReg, srcAddr);
            Reg::Compare<T, CMPMODE::NE>(nanMask, xReg, xReg, maskAll);
            Reg::Compare<T, CMPMODE::GT>(gtMask, xReg, maxReg, maskAll);
            Reg::MaskXor(updMask, gtMask, nanMask, maskAll);
            Reg::Max(maxReg, xReg, maxReg, maskAll);
            Reg::Adds(candIdx, hFlatReg, static_cast<int32_t>(w), maskAll);
            if constexpr (sizeof(int32_t) / sizeof(T) == 2) {
                Reg::MaskReg updMaskB32;
                Reg::UnPack(updMaskB32, updMask);
                Reg::Select(idxReg, candIdx, idxReg, updMaskB32);
            } else {
                Reg::Select(idxReg, candIdx, idxReg, updMask);
            }
        }
    }
    Reg::UnalignRegForStore u0;
    Reg::RegTensor<uint32_t>& idxStoreU32 = (Reg::RegTensor<uint32_t>&)idxReg;
    __ubuf__ uint32_t* idxDst = (__ubuf__ uint32_t*)argmaxAddr;
    if constexpr (MERGE) {
        Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
        __ubuf__ T* maxDst = maxValAddr;
        Reg::StoreUnAlign(maxDst, maxReg, u0, static_cast<uint32_t>(vc));
        Reg::StoreUnAlignPost(maxDst, u0, 0);
    }
    Reg::StoreUnAlign<uint32_t>(idxDst, idxStoreU32, u0, static_cast<uint32_t>(vc));
    Reg::StoreUnAlignPost<uint32_t>(idxDst, u0, 0);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
template <bool MERGE>
__aicore__ inline void MaxPoolV3GradNHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::ComputeSingleArgmax(
    __local_mem__ T* xSrcBase, int64_t rowStride, int64_t curKW, int64_t curKH, int64_t curOriginIndex,
    int64_t argmaxOffset)
{
    __ubuf__ INDEX_T* argmaxAddr = (__ubuf__ INDEX_T*)argmaxBuff_.template Get<INDEX_T>().GetPhyAddr();
    __ubuf__ T* maxValAddr = (__ubuf__ T*)maxValBuf_.template Get<T>().GetPhyAddr();
    const int64_t ubBlockElems = platform::GetUbBlockSize() / sizeof(T);
    const int64_t cAlignedC = (cOutputActual_ + ubBlockElems - 1) / ubBlockElems * ubBlockElems;
    constexpr int64_t vlT = platform::GetVRegSize() / sizeof(T);
    constexpr int64_t vlIdx = platform::GetVRegSize() / sizeof(int32_t);
    constexpr int64_t vlStep = vlT < vlIdx ? vlT : vlIdx;
    for (int64_t c0 = 0; c0 < cOutputActual_; c0 += vlStep) {
        const int64_t vc = cOutputActual_ - c0 < vlStep ? cOutputActual_ - c0 : vlStep;
        ComputeSingleArgmaxVF<T, INDEX_T, MERGE>(
            xSrcBase + c0, argmaxAddr + argmaxOffset + c0, maxValAddr + c0, static_cast<int32_t>(rowStride),
            static_cast<uint16_t>(curKW), static_cast<uint16_t>(curKH), static_cast<int32_t>(curOriginIndex),
            static_cast<int32_t>(cAlignedC), static_cast<uint32_t>(vc), static_cast<int32_t>(tilingData_.base.wOutput));
    }
}

} // namespace MaxPoolV3GradNHWCNameSpace

#endif // MAX_POOL_V3_GRAD_NHWC_BIG_KERNEL_H_
