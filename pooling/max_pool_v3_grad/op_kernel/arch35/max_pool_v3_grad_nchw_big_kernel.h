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
 * \file max_pool_v3_grad_nchw_big_kernel.h
 * \brief MaxPoolV3Grad NCHW big-kernel template, 与小核模板(max_pool_v3_grad_nchw_small_kernel.h)对称。
 */

#ifndef MAX_POOL_V3_GRAD_NCHW_BIG_KERNEL_H_
#define MAX_POOL_V3_GRAD_NCHW_BIG_KERNEL_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../pool_grad_common/arch35/max_pool_grad_struct.h"
#include "../pool_grad_common/arch35/max_pool_grad_nchw_big_kernel.h"

namespace MaxPoolV3GradNCHWBigKernelNameSpace {
using namespace AscendC;
using MaxPoolGradNCHWBigKernelNameSpace::MaxPoolGradNCHWBigKernel;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;
using PoolBigKernelUtils::CalcRealIndex;
using PoolBigKernelUtils::CalcRealIndexVF;
using PoolBigKernelUtils::LoadOneElement;
using PoolBigKernelUtils::LoadOneElementVF;
using PoolBigKernelUtils::ReduceMaxWithIndex;
using PoolBigKernelUtils::ReduceMaxWithIndexVF;
using PoolBigKernelUtils::StoreOneElement;
using PoolBigKernelUtils::StoreOneElementVF;
using PoolUtils::Compute::FilterMaskVF;
using PoolUtils::Index::GenInitial1DIndicesVF;

constexpr int32_t INT_REPEAT_ELEMS = 64; // lane按32bit槽解包, 单次vector恒处理64个元素
constexpr uint16_t UINT8_HIGH_MASK = 0x00FF;
constexpr uint16_t UINT8_SIGN_BIT = 0x0080;
constexpr size_t DTYPE_WIDTH_B8 = sizeof(int8_t);
constexpr size_t DTYPE_WIDTH_B16 = sizeof(int16_t);
constexpr size_t DTYPE_WIDTH_B32 = sizeof(int32_t);
constexpr size_t DTYPE_WIDTH_B64 = sizeof(int64_t);

constexpr AscendC::Reg::CastTrait castTraitIntWiden = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

__simd_callee__ inline void ArgmaxPadIndexVf(AscendC::Reg::RegTensor<int32_t>& index, AscendC::Reg::MaskReg& maskAll32)
{
    AscendC::Reg::RegTensor<int32_t> idxZeroReg;
    AscendC::Reg::Duplicate(idxZeroReg, static_cast<int32_t>(0));
    AscendC::Reg::MaskReg negIdxMask;
    AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::GE>(negIdxMask, index, idxZeroReg, maskAll32);
    AscendC::Reg::Select(index, index, idxZeroReg, negIdxMask);
}

template <typename T>
__simd_callee__ inline void SentinelFilterVf(AscendC::Reg::MaskReg& preg, AscendC::Reg::RegTensor<T>& hLocReg,
                                             AscendC::Reg::RegTensor<T>& wLocReg)
{
    AscendC::Reg::MaskReg gtMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<T> zeroConstReg;
    AscendC::Reg::Duplicate(zeroConstReg, static_cast<T>(0));
    AscendC::Reg::Compare<T, AscendC::CMPMODE::GE>(gtMask, hLocReg, zeroConstReg, gtMask);
    AscendC::Reg::Compare<T, AscendC::CMPMODE::GE>(gtMask, wLocReg, zeroConstReg, gtMask);
    AscendC::Reg::MaskAnd(preg, preg, gtMask, allMask);
}

template <typename T>
struct NchwIntBigVecTraits {
    using WideT = T;
    static constexpr bool MASK_8BIT = false;
    static constexpr bool SIGN_8BIT = false;
    static constexpr T WideMin() { return T(0); }
    static constexpr T MinVal() { return T(0); }
};

template <>
struct NchwIntBigVecTraits<int8_t> {
    using WideT = int32_t;
    static constexpr bool MASK_8BIT = true;
    static constexpr bool SIGN_8BIT = true;
    static constexpr int32_t WideMin() { return INT32_MIN; }
    static constexpr int8_t MinVal() { return INT8_MIN; }
};

template <>
struct NchwIntBigVecTraits<uint8_t> {
    using WideT = uint32_t;
    static constexpr bool MASK_8BIT = true;
    static constexpr bool SIGN_8BIT = false;
    static constexpr uint32_t WideMin() { return 0; }
    static constexpr uint8_t MinVal() { return 0; }
};

template <>
struct NchwIntBigVecTraits<int16_t> {
    using WideT = int32_t;
    static constexpr bool MASK_8BIT = false;
    static constexpr bool SIGN_8BIT = false;
    static constexpr int32_t WideMin() { return INT32_MIN; }
    static constexpr int16_t MinVal() { return INT16_MIN; }
};

template <>
struct NchwIntBigVecTraits<uint16_t> {
    using WideT = uint32_t;
    static constexpr bool MASK_8BIT = false;
    static constexpr bool SIGN_8BIT = false;
    static constexpr uint32_t WideMin() { return 0; }
    static constexpr uint16_t MinVal() { return 0; }
};

template <>
struct NchwIntBigVecTraits<int32_t> {
    using WideT = int32_t;
    static constexpr bool MASK_8BIT = false;
    static constexpr bool SIGN_8BIT = false;
    static constexpr int32_t WideMin() { return INT32_MIN; }
    static constexpr int32_t MinVal() { return INT32_MIN; }
};

template <>
struct NchwIntBigVecTraits<uint32_t> {
    using WideT = uint32_t;
    static constexpr bool MASK_8BIT = false;
    static constexpr bool SIGN_8BIT = false;
    static constexpr uint32_t WideMin() { return 0; }
    static constexpr uint32_t MinVal() { return 0; }
};

template <>
struct NchwIntBigVecTraits<int64_t> {
    using WideT = int64_t; // 64位: b64双寄存器比较域
    static constexpr bool MASK_8BIT = false;
    static constexpr bool SIGN_8BIT = false;
    static constexpr int64_t WideMin() { return INT64_MIN; }
    static constexpr int64_t MinVal() { return INT64_MIN; }
};

template <>
struct NchwIntBigVecTraits<uint64_t> {
    using WideT = uint64_t;
    static constexpr bool MASK_8BIT = false;
    static constexpr bool SIGN_8BIT = false;
    static constexpr uint64_t WideMin() { return 0; }
    static constexpr uint64_t MinVal() { return 0; }
};

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
class PoolGradNCHWBigKernelV3 : public MaxPoolGradNCHWBigKernel<TYPE_ORIG_X, IS_CHECK_RANGE> {
    using Base = MaxPoolGradNCHWBigKernel<TYPE_ORIG_X, IS_CHECK_RANGE>;
    using Traits = NchwIntBigVecTraits<TYPE_ORIG_X>;
    using WideT = typename Traits::WideT;
    // b8归一化的中间b16域
    using MidT = std::conditional_t<std::is_same_v<WideT, uint32_t>, uint16_t, int16_t>;

public:
    __aicore__ inline PoolGradNCHWBigKernelV3(TPipe& pipeIn,
                                              const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y)
    {
        if constexpr (std::is_integral<TYPE_ORIG_X>::value) {
            Base::ParseTilingData(tilingData_);
            Base::blockIdx_ = GetBlockIdx();
            if (Base::blockIdx_ >= Base::usedCoreNum_) {
                return;
            }

            Base::curCoreProcessNum_ = (Base::blockIdx_ + 1 == Base::usedCoreNum_) ? Base::tailCoreProcessNum_ :
                                                                                     Base::normalCoreProcessNum_;

            Base::xGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)orig_x);
            Base::gradGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)grads);
            Base::yGm_.SetGlobalBuffer((__gm__ TYPE_ORIG_X*)y);

            Base::inHW_ = Base::hOutput_ * Base::wOutput_;
            int64_t maxCountBySize = Base::inputBufferSize_ / static_cast<int64_t>(sizeof(TYPE_ORIG_X));
            Base::maxCount_ = (maxCountBySize < 1) ? 1 : maxCountBySize;

            const int64_t forwardStageBufferSize = Base::inputBufferSize_ *
                                                       MaxPoolGradNCHWBigKernelNameSpace::BK_BUFFER_NUM +
                                                   MaxPoolGradNCHWBigKernelNameSpace::MERGE_BUF_ALIGN;
            const int64_t backwardStageBufferSize = Base::gradBufferSize_ *
                                                        MaxPoolGradNCHWBigKernelNameSpace::BK_BUFFER_NUM +
                                                    Base::outputBufferSize_ *
                                                        MaxPoolGradNCHWBigKernelNameSpace::BK_BUFFER_NUM;
            Base::totalStageBufferSize_ = (forwardStageBufferSize > backwardStageBufferSize) ? forwardStageBufferSize :
                                                                                               backwardStageBufferSize;

            pipe_.InitBuffer(Base::argmaxBuf_, Base::argmaxBufferSize_);
            pipe_.InitBuffer(Base::helpBuf_, MaxPoolGradNCHWBackwardBaseNameSpace::HELP_BUFFER);
            pipe_.InitBufPool(Base::forwardBufPool_, static_cast<uint32_t>(Base::totalStageBufferSize_));
            pipe_.InitBufPool(Base::backwardBufPool_, static_cast<uint32_t>(Base::totalStageBufferSize_),
                              Base::forwardBufPool_);
        } else {
            Base::Init(orig_x, orig_y, grads, y, pipe_, tilingData_);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (std::is_integral<TYPE_ORIG_X>::value) {
            ProcessInt();
        } else {
            Base::Process();
        }
    }

private:
    TPipe& pipe_;
    const MaxPoolGradWithArgmaxNCHWTilingCommonData& tilingData_;

private:
    __aicore__ inline void DrainMte3ForPoolReuse()
    {
        event_t eventMte3Drain = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventMte3Drain);
        WaitFlag<HardEvent::MTE3_S>(eventMte3Drain);
    }

    __aicore__ inline void ProcessInt()
    {
        if (Base::blockIdx_ >= Base::usedCoreNum_) {
            return;
        }

        for (int64_t loopNum = 0; loopNum < Base::curCoreProcessNum_; ++loopNum) {
            Base::ScalarCompute(loopNum);

            if constexpr (sizeof(TYPE_ORIG_X) == DTYPE_WIDTH_B64) {
                // b64: y/grad tile行距须与DataCopyPad按32B块对齐的实际行距一致,
                // 即按输入dtype粒度(BLOCK_SIZE/sizeof)对齐; 基类max()公式按index dtype粒度
                // (8元素)得到136/16, 实测CopyOut/CopyInGrad行距132/12, 差值即W方向整体偏移
                // 与grad槽位错配。<=32bit dtype下两公式恒等, 本修正仅影响64bit dtype。
                const int32_t inputBlk = static_cast<int32_t>(Base::BLOCK_SIZE /
                                                              static_cast<int64_t>(sizeof(TYPE_ORIG_X)));
                Base::wOutputAligned_ = CeilDivision(static_cast<int32_t>(Base::wOutputActual_), inputBlk) * inputBlk;
                Base::wArgmaxAligned_ = CeilDivision(static_cast<int32_t>(Base::wArgmaxActual_), inputBlk) * inputBlk;
            }

            if (Base::hArgmaxActual_ <= 0 || Base::wArgmaxActual_ <= 0) {
                Base::InitBackwardStageBuffers();
                Base::ProcessNoArgmaxBlock();
                Base::backwardBufPool_.Reset();
                DrainMte3ForPoolReuse();
                continue;
            }

            Base::InitForwardStageBuffers();
            ForwardComputeTileInt();
            Base::forwardBufPool_.Reset();

            Base::InitBackwardStageBuffers();
            Base::CopyInGrad();
            BackwardScatterInt();
            Base::CopyOut();
            DrainMte3ForPoolReuse();
            Base::backwardBufPool_.Reset();
        }
    }

    __aicore__ inline void ForwardComputeTileInt()
    {
        LocalTensor<int32_t> argmaxLocal = Base::argmaxBuf_.template Get<int32_t>();
        Duplicate(argmaxLocal, static_cast<int32_t>(-1), Base::argmaxBufferSize_ / sizeof(int32_t));

        const int64_t tileHW = Base::hArgmaxActual_ * Base::wArgmaxActual_;
        for (int64_t highIdx = 0; highIdx < Base::highAxisActual_; ++highIdx) {
            for (int64_t hIdx = 0; hIdx < Base::hArgmaxActual_; ++hIdx) {
                for (int64_t wIdx = 0; wIdx < Base::wArgmaxActual_; ++wIdx) {
                    int64_t curkH = 0;
                    int64_t curkW = 0;
                    int64_t curInOffset = 0;
                    int64_t curOriginIndex = 0;
                    Base::CalcKernelSize(highIdx, hIdx, wIdx, curkH, curkW, curInOffset, curOriginIndex);

                    if (curkH <= 0 || curkW <= 0) {
                        continue;
                    }

                    const int64_t bufferOffset = highIdx * tileHW + hIdx * Base::wArgmaxActual_ + wIdx;
                    if (curkH * curkW <= Base::maxCount_) {
                        NoSplitKernelProcessInt(curkH, curkW, curInOffset, curOriginIndex, bufferOffset);
                    } else {
                        SplitKernelProcessInt(curkH, curkW, curInOffset, curOriginIndex, bufferOffset);
                    }
                }
            }
        }
    }

    __aicore__ inline void NoSplitKernelProcessInt(int64_t curkH, int64_t curkW, int64_t curInOffset,
                                                   int64_t curOriginIndex, int64_t bufferOffset)
    {
        Base::CopyInMultiRows(curInOffset, curkW, curkH);
        ComputeArgmax<false, false>(curkW * curkH, curkW, curOriginIndex, bufferOffset);
    }

    __aicore__ inline void SplitKernelProcessInt(int64_t curkH, int64_t curkW, int64_t curInOffset,
                                                 int64_t curOriginIndex, int64_t bufferOffset)
    {
        InitMergeBufferInt(bufferOffset, curOriginIndex);

        if (curkW <= 0 || curkH <= 0 || Base::maxCount_ <= 0) {
            return;
        }

        if (curkW <= Base::maxCount_) {
            const int64_t hFactor = Base::maxCount_ / ((curkW > 0) ? curkW : 1);
            const int64_t hLoops = CeilDivision(static_cast<int32_t>(curkH), static_cast<int32_t>(hFactor));
            const int64_t hTail = curkH - (hLoops - 1) * hFactor;

            int64_t inputOffset = curInOffset;
            int64_t kernelOffset = curOriginIndex;
            for (int64_t hLoop = 0; hLoop < hLoops; ++hLoop) {
                const int64_t curhFactor = (hLoop == hLoops - 1) ? hTail : hFactor;
                Base::CopyInMultiRows(inputOffset, curkW, curhFactor);
                ComputeArgmax<true, false>(curkW * curhFactor, ((curkW > 0) ? curkW : 1), kernelOffset, bufferOffset);
                inputOffset += curhFactor * Base::wOutput_;
                kernelOffset += curhFactor * Base::wOutput_;
            }
        } else {
            const int64_t hLoops = curkH;
            const int64_t wFactor = Base::maxCount_;
            const int64_t wLoops = CeilDivision(static_cast<int32_t>(curkW), static_cast<int32_t>(wFactor));
            const int64_t wTail = curkW - (wLoops - 1) * wFactor;

            for (int64_t hLoop = 0; hLoop < hLoops; ++hLoop) {
                int64_t inputOffset = curInOffset + hLoop * Base::wOutput_;
                int64_t kernelOffset = curOriginIndex + hLoop * Base::wOutput_;
                for (int64_t wLoop = 0; wLoop < wLoops; ++wLoop) {
                    const int64_t curFactor = (wLoop == wLoops - 1) ? wTail : wFactor;
                    Base::CopyInSingleRow(inputOffset, curFactor);
                    ComputeArgmax<true, true>(curFactor, ((curkW > 0) ? curkW : 1), kernelOffset, bufferOffset);
                    inputOffset += curFactor;
                    kernelOffset += curFactor;
                }
            }
        }
    }

    __aicore__ inline void InitMergeBufferInt(int64_t bufferOffset, int64_t initIndex)
    {
        LocalTensor<int32_t> argmaxLocal = Base::argmaxBuf_.template Get<int32_t>();
        __local_mem__ int32_t* argmaxAddr = (__local_mem__ int32_t*)argmaxLocal.GetPhyAddr();
        __local_mem__ WideT* maxValAddr = (__local_mem__ WideT*)Base::maxValBuf_.template Get<WideT>().GetPhyAddr();

        // 与其后的MERGE读同为V pipe, 按序执行无需barrier
        InitMergeBufferSimdVf((__ubuf__ int32_t*)argmaxAddr, (__ubuf__ WideT*)maxValAddr,
                              static_cast<uint32_t>(bufferOffset), static_cast<int32_t>(initIndex),
                              static_cast<WideT>(Traits::WideMin()));
    }

    /*
     * 功能：单个chunk的argmax重算, 全dtype向量化。
     *       b8/b16走b16比较域(b8先归一化), b32走b32域, b64走RegTraitNumTwo双寄存器b64域。
     */
    template <bool MERGE, bool SPLITKW>
    __aicore__ inline void ComputeArgmax(int64_t dataCount, int64_t curKw, int64_t curOriginIndex, int64_t bufferOffset)
    {
        ComputeArgmaxVec<MERGE, SPLITKW>(dataCount, curKw, curOriginIndex, bufferOffset);
    }

    template <bool MERGE, bool SPLITKW>
    __aicore__ inline void ComputeArgmaxVec(int64_t dataCount, int64_t curKw, int64_t curOriginIndex,
                                            int64_t bufferOffset)
    {
        LocalTensor<TYPE_ORIG_X> xLocal = Base::inputQue_.template DeQue<TYPE_ORIG_X>();
        __local_mem__ TYPE_ORIG_X* xLocalAddr = (__local_mem__ TYPE_ORIG_X*)xLocal.GetPhyAddr();

        LocalTensor<int32_t> argmaxLocal = Base::argmaxBuf_.template Get<int32_t>();
        __local_mem__ int32_t* argmaxAddr = (__local_mem__ int32_t*)argmaxLocal.GetPhyAddr();

        __local_mem__ WideT* maxValAddr = (__local_mem__ WideT*)Base::maxValBuf_.template Get<WideT>().GetPhyAddr();

        const uint16_t repeatTimes = static_cast<uint16_t>(
            CeilDivision(static_cast<int32_t>(dataCount), INT_REPEAT_ELEMS));

        // 按dtype位宽编译期分发到对应比较域的VF实现(aicore宿主直接调用simd_vf)
        if constexpr (sizeof(TYPE_ORIG_X) <= DTYPE_WIDTH_B16) {
            ComputeArgmaxB16DomainSimdVf<MERGE, SPLITKW>(
                (__ubuf__ TYPE_ORIG_X*)xLocalAddr, (__ubuf__ int32_t*)argmaxAddr, (__ubuf__ WideT*)maxValAddr,
                dataCount, curKw, curOriginIndex, static_cast<uint32_t>(bufferOffset), repeatTimes, Base::wOutput_);
        } else if constexpr (sizeof(TYPE_ORIG_X) == DTYPE_WIDTH_B32) {
            ComputeArgmaxB32DomainSimdVf<MERGE, SPLITKW>(
                (__ubuf__ TYPE_ORIG_X*)xLocalAddr, (__ubuf__ int32_t*)argmaxAddr, (__ubuf__ WideT*)maxValAddr,
                dataCount, curKw, curOriginIndex, static_cast<uint32_t>(bufferOffset), repeatTimes, Base::wOutput_,
                static_cast<WideT>(Traits::WideMin()));
        } else {
            ComputeArgmaxB64DomainSimdVf<MERGE, SPLITKW>(
                (__ubuf__ TYPE_ORIG_X*)xLocalAddr, (__ubuf__ int32_t*)argmaxAddr, (__ubuf__ WideT*)maxValAddr,
                dataCount, curKw, curOriginIndex, static_cast<uint32_t>(bufferOffset), repeatTimes, Base::wOutput_,
                static_cast<WideT>(Traits::WideMin()));
        }

        // VF读input须排空后才能FreeTensor供下下个窗口MTE2复写(V->S同步:
        // S侧等待V队列越过标记, clipped_swiglu同款事件模式; 此前PipeBarrier的位置)
        event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
        Base::inputQue_.FreeTensor(xLocal);
    }

    template <bool MERGE, bool SPLITKW>
    __simd_vf__ inline void ComputeArgmaxB16DomainSimdVf(__ubuf__ TYPE_ORIG_X* xLocalAddr, __ubuf__ int32_t* argmaxAddr,
                                                         __ubuf__ WideT* maxValAddr, int64_t dataCount, int64_t curKw,
                                                         int64_t curOriginIndex, uint32_t bufferOffset,
                                                         uint16_t repeatTimes, int64_t wOutput);

    /*
     * 功能: b32域argmax重算(u32索引gather, b32域运行最大值), 归约复用ReduceMaxWithIndexVF。
     */
    template <bool MERGE, bool SPLITKW>
    __simd_vf__ inline void ComputeArgmaxB32DomainSimdVf(__ubuf__ TYPE_ORIG_X* xLocalAddr, __ubuf__ int32_t* argmaxAddr,
                                                         __ubuf__ WideT* maxValAddr, int64_t dataCount, int64_t curKw,
                                                         int64_t curOriginIndex, uint32_t bufferOffset,
                                                         uint16_t repeatTimes, int64_t wOutput, WideT wideMin);

    template <bool MERGE, bool SPLITKW>
    __simd_vf__ inline void ComputeArgmaxB64DomainSimdVf(__ubuf__ TYPE_ORIG_X* xLocalAddr, __ubuf__ int32_t* argmaxAddr,
                                                         __ubuf__ WideT* maxValAddr, int64_t dataCount, int64_t curKw,
                                                         int64_t curOriginIndex, uint32_t bufferOffset,
                                                         uint16_t repeatTimes, int64_t wOutput, WideT wideMin);

    __aicore__ inline void BackwardScatterInt()
    {
        const uint32_t calcCount = static_cast<uint32_t>(Base::outputBufferSize_) / sizeof(TYPE_ORIG_X);

        LocalTensor<TYPE_ORIG_X> yLocal = Base::outputQue_.template AllocTensor<TYPE_ORIG_X>();
        Duplicate(yLocal, static_cast<TYPE_ORIG_X>(0), calcCount);

        LocalTensor<TYPE_ORIG_X> gradLocal = Base::gradQue_.template DeQue<TYPE_ORIG_X>();
        LocalTensor<int32_t> argmaxLocal = Base::argmaxBuf_.template Get<int32_t>();
        __local_mem__ TYPE_ORIG_X* yAddr = (__local_mem__ TYPE_ORIG_X*)yLocal.GetPhyAddr();
        __local_mem__ TYPE_ORIG_X* gradAddr = (__local_mem__ TYPE_ORIG_X*)gradLocal.GetPhyAddr();
        __local_mem__ int32_t* argmaxAddr = (__local_mem__ int32_t*)argmaxLocal.GetPhyAddr();

        BackwardComputeInt(yAddr, gradAddr, argmaxAddr);

        Base::outputQue_.EnQue(yLocal);
        Base::gradQue_.FreeTensor(gradLocal);
    }

    __aicore__ inline void BackwardComputeInt(__local_mem__ TYPE_ORIG_X* yAddr, __local_mem__ TYPE_ORIG_X* gradAddr,
                                              __local_mem__ int32_t* argmaxAddr)
    {
        const int64_t wOutput = Base::wOutput_;
        const int64_t hOutputActual = Base::hOutputActual_;
        const int64_t wOutputActual = Base::wOutputActual_;
        const int64_t wOutputAligned = Base::wOutputAligned_;
        const uint16_t highAxisActual = static_cast<uint16_t>(Base::highAxisActual_);
        const int64_t curHIndex = Base::hAxisIndex_ * Base::hOutputInner_;
        const int64_t curWIndex = Base::wAxisIndex_ * Base::wOutputInner_;
        const int64_t wArgmaxActual = Base::wArgmaxActual_;
        const int64_t wArgmaxAligned = Base::wArgmaxAligned_;
        const uint16_t hArgmaxActual = static_cast<uint16_t>(Base::hArgmaxActual_);
        const uint16_t wProBatchSize = static_cast<uint16_t>(Base::curWProBatchSize_ > 0 ? Base::curWProBatchSize_ : 1);

        const uint32_t wFullBatchCount = static_cast<uint32_t>(wArgmaxActual / wProBatchSize);
        const uint16_t computeSize = INT_REPEAT_ELEMS;
        const uint16_t repeatTimes = static_cast<uint16_t>(wFullBatchCount / computeSize);
        const int64_t wRemain = wArgmaxActual - repeatTimes * wProBatchSize * computeSize;
        const uint32_t wRemainBatchCount = static_cast<uint32_t>(wRemain / wProBatchSize);
        const uint16_t wRemainTail = static_cast<uint16_t>(wRemain % wProBatchSize);

        if (Base::isOverlap_) {
            BackwardComputeIntSimdVf<true>((__ubuf__ TYPE_ORIG_X*)yAddr, (__ubuf__ TYPE_ORIG_X*)gradAddr,
                                           (__ubuf__ int32_t*)argmaxAddr, highAxisActual, hArgmaxActual,
                                           static_cast<int32_t>(wArgmaxActual), static_cast<int32_t>(wArgmaxAligned),
                                           wProBatchSize, repeatTimes, wRemainBatchCount, wRemainTail,
                                           static_cast<int32_t>(wOutput), static_cast<int32_t>(wOutputAligned),
                                           static_cast<int32_t>(hOutputActual), static_cast<int32_t>(wOutputActual),
                                           static_cast<int32_t>(curHIndex), static_cast<int32_t>(curWIndex));
        } else {
            BackwardComputeIntSimdVf<false>((__ubuf__ TYPE_ORIG_X*)yAddr, (__ubuf__ TYPE_ORIG_X*)gradAddr,
                                            (__ubuf__ int32_t*)argmaxAddr, highAxisActual, hArgmaxActual,
                                            static_cast<int32_t>(wArgmaxActual), static_cast<int32_t>(wArgmaxAligned),
                                            wProBatchSize, repeatTimes, wRemainBatchCount, wRemainTail,
                                            static_cast<int32_t>(wOutput), static_cast<int32_t>(wOutputAligned),
                                            static_cast<int32_t>(hOutputActual), static_cast<int32_t>(wOutputActual),
                                            static_cast<int32_t>(curHIndex), static_cast<int32_t>(curWIndex));
        }
    }

    __simd_vf__ inline void InitMergeBufferSimdVf(__ubuf__ int32_t* argmaxAddr, __ubuf__ WideT* maxValAddr,
                                                  uint32_t bufferOffset, int32_t initIndex, WideT wideMin);

    template <const bool IS_OVERLAP>
    __simd_vf__ inline void BackwardComputeIntSimdVf(
        __ubuf__ TYPE_ORIG_X* yAddr, __ubuf__ TYPE_ORIG_X* gradAddr, __ubuf__ int32_t* argmaxAddr,
        uint16_t highAxisActual, uint16_t hArgmaxActual, int32_t wArgmaxActual, int32_t wArgmaxAligned,
        uint16_t wProBatchSize, uint16_t repeatTimes, uint32_t wRemainBatchCount, uint16_t wRemainTail, int32_t wOutput,
        int32_t wOutputAligned, int32_t hOutputActual, int32_t wOutputActual, int32_t curHIndex, int32_t curWIndex);

    template <const bool IS_OVERLAP>
    __simd_callee__ inline void DoScatterSingle(uint32_t argmaxMaskCount, __ubuf__ TYPE_ORIG_X* yAddr,
                                                __ubuf__ TYPE_ORIG_X* gradAddr, __ubuf__ int32_t* argmaxAddr,
                                                AscendC::Reg::RegTensor<uint32_t>& parallelRegIndex,
                                                AscendC::Reg::RegTensor<uint32_t>& parallelRegGrad,
                                                AscendC::Reg::RegTensor<int32_t>& wOutputConstReg, int32_t curHIndex,
                                                int32_t curWIndex, int32_t wOutputAligned, int32_t highOutputOffset,
                                                int32_t hOutputActual, int32_t wOutputActual);
};

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void PoolGradNCHWBigKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::InitMergeBufferSimdVf(
    __ubuf__ int32_t* argmaxAddr, __ubuf__ WideT* maxValAddr, uint32_t bufferOffset, int32_t initIndex, WideT wideMin)
{
    // argmax槽恒为i32域; maxVal按各dtype前向MERGE的比较域存储:
    // b8→MidT(b16归一化域), b16→原dtype, b32→WideT, b64→双寄存器b64域
    AscendC::Reg::MaskReg pregOneIdx = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();

    AscendC::Reg::RegTensor<int32_t> initResIndex;
    AscendC::Reg::Duplicate(initResIndex, initIndex);
    StoreOneElementVF<int32_t, int32_t>(argmaxAddr, initResIndex, pregOneIdx, bufferOffset);

    if constexpr (sizeof(TYPE_ORIG_X) == DTYPE_WIDTH_B8) {
        AscendC::Reg::RegTensor<MidT> minValReg;
        AscendC::Reg::Duplicate(minValReg, std::numeric_limits<MidT>::lowest());
        AscendC::Reg::MaskReg pregOneDom = AscendC::Reg::CreateMask<MidT, AscendC::Reg::MaskPattern::VL1>();
        StoreOneElementVF<MidT, MidT, WideT>(maxValAddr, minValReg, pregOneDom, 0);
    } else if constexpr (sizeof(TYPE_ORIG_X) == DTYPE_WIDTH_B16) {
        AscendC::Reg::RegTensor<TYPE_ORIG_X> minValReg;
        AscendC::Reg::Duplicate(minValReg, Traits::MinVal());
        AscendC::Reg::MaskReg pregOneDom = AscendC::Reg::CreateMask<TYPE_ORIG_X, AscendC::Reg::MaskPattern::VL1>();
        StoreOneElementVF<TYPE_ORIG_X, TYPE_ORIG_X, WideT>(maxValAddr, minValReg, pregOneDom, 0);
    } else if constexpr (sizeof(TYPE_ORIG_X) == DTYPE_WIDTH_B32) {
        AscendC::Reg::RegTensor<WideT> minValReg;
        AscendC::Reg::Duplicate(minValReg, wideMin);
        StoreOneElementVF<WideT, WideT>(maxValAddr, minValReg, pregOneIdx, 0);
    } else {
        // b64: 双寄存器Duplicate后单元素StoreUnAlign(与前向MERGE的读域一致)
        AscendC::Reg::RegTensor<WideT, AscendC::Reg::RegTraitNumTwo> minValReg;
        AscendC::Reg::Duplicate(minValReg, wideMin);
        AscendC::Reg::UnalignRegForStore u0;
        __ubuf__ WideT* dstAddr = (__ubuf__ WideT*)maxValAddr;
        AscendC::Reg::StoreUnAlign(dstAddr, minValReg, u0, 1);
        AscendC::Reg::StoreUnAlignPost(dstAddr, u0, 0);
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool MERGE, bool SPLITKW>
__simd_vf__ inline void PoolGradNCHWBigKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ComputeArgmaxB16DomainSimdVf(
    __ubuf__ TYPE_ORIG_X* xLocalAddr, __ubuf__ int32_t* argmaxAddr, __ubuf__ WideT* maxValAddr, int64_t dataCount,
    int64_t curKw, int64_t curOriginIndex, uint32_t bufferOffset, uint16_t repeatTimes, int64_t wOutput)
{
    AscendC::Reg::RegTensor<int32_t> resIndex;
    AscendC::Reg::RegTensor<int32_t> index;
    AscendC::Reg::RegTensor<int32_t> realResIndex;
    AscendC::Reg::MaskReg maskAll32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    // argmax槽的单元素读写恒为i32域(VL1); maxVal的域随分支而定
    AscendC::Reg::MaskReg pregOne = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();

    AscendC::Reg::Duplicate(resIndex, static_cast<int32_t>(-1));
    AscendC::Reg::Arange(index, 0);

    // ===== b8/b16: b16比较域(小核ProcessWInt同构; b8经u16索引gather到MidT后归一化) =====
    using DomT = MidT;
    AscendC::Reg::RegTensor<DomT> resVal;
    AscendC::Reg::MaskReg maskAllDom = AscendC::Reg::CreateMask<DomT, AscendC::Reg::MaskPattern::ALL>();
    if constexpr (sizeof(TYPE_ORIG_X) == DTYPE_WIDTH_B8) {
        AscendC::Reg::Duplicate(resVal, std::numeric_limits<DomT>::lowest());
    } else {
        AscendC::Reg::Duplicate(resVal, Traits::MinVal());
    }

    // b8归一化常量(b16域, 循环外预置)
    AscendC::Reg::RegTensor<DomT> highMaskReg;
    AscendC::Reg::RegTensor<DomT> signXorReg;
    if constexpr (Traits::MASK_8BIT) {
        AscendC::Reg::Duplicate(highMaskReg, DomT(UINT8_HIGH_MASK));
    }
    if constexpr (Traits::SIGN_8BIT) {
        AscendC::Reg::Duplicate(signXorReg, DomT(UINT8_SIGN_BIT));
    }

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        const uint32_t validCount = static_cast<uint32_t>(dataCount - static_cast<int64_t>(i) * INT_REPEAT_ELEMS >
                                                                  INT_REPEAT_ELEMS ?
                                                              INT_REPEAT_ELEMS :
                                                              dataCount - static_cast<int64_t>(i) * INT_REPEAT_ELEMS);
        uint32_t maskNum = validCount;
        AscendC::Reg::MaskReg p16 = AscendC::Reg::UpdateMask<DomT>(maskNum);

        AscendC::Reg::RegTensor<int32_t> seqIdx;
        AscendC::Reg::Arange(seqIdx, 0);
        AscendC::Reg::Adds(seqIdx, seqIdx, static_cast<int32_t>(i) * INT_REPEAT_ELEMS, maskAll32);
        AscendC::Reg::RegTensor<uint16_t> seqU16;
        AscendC::Reg::Pack(seqU16, seqIdx);

        AscendC::Reg::RegTensor<DomT> x16;
        AscendC::Reg::DataCopyGather(x16, xLocalAddr, seqU16, p16);
        if constexpr (Traits::MASK_8BIT) {
            // gather的b8->b16扩展方向未在接口层约定, 统一截到0..255
            AscendC::Reg::And(x16, x16, highMaskReg, p16);
        }
        if constexpr (Traits::SIGN_8BIT) {
            // 0..255先异或符号位再减128, 等价于按int8解释并符号扩展(与小核一致)
            AscendC::Reg::Xor(x16, x16, signXorReg, p16);
            AscendC::Reg::Adds(x16, x16, DomT(-128), p16);
        }

        // b16域严格大于比较, 掩码经MaskUnPack到b32粒度更新索引(小核同构)
        AscendC::Reg::MaskReg gtMask16;
        AscendC::Reg::Compare<DomT, CMPMODE::GT>(gtMask16, x16, resVal, p16);
        AscendC::Reg::MaskReg gtMask32;
        AscendC::Reg::MaskUnPack(gtMask32, gtMask16);
        AscendC::Reg::Select(resIndex, index, resIndex, gtMask32);
        // Select回写, 避免Max对尾块掩码外lane的ZEROING清零
        AscendC::Reg::Select(resVal, x16, resVal, gtMask16);
        AscendC::Reg::Adds(index, index, INT_REPEAT_ELEMS, maskAll32);
    }

    // 水平argmax: b16域取max, 等值lane经MaskUnPack后对索引Reduce[MIN](first-wins)
    AscendC::Reg::RegTensor<DomT> maxScalar;
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(maxScalar, resVal, maskAllDom);
    AscendC::Reg::RegTensor<DomT> maxBcast;
    AscendC::Reg::Duplicate(maxBcast, maxScalar, maskAllDom);
    AscendC::Reg::MaskReg eqMask16;
    AscendC::Reg::Compare<DomT, CMPMODE::EQ>(eqMask16, resVal, maxBcast, maskAllDom);
    AscendC::Reg::MaskReg eqMask32;
    AscendC::Reg::MaskUnPack(eqMask32, eqMask16);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MIN>(index, resIndex, eqMask32);

    ArgmaxPadIndexVf(index, maskAll32);
    CalcRealIndexVF<int32_t, SPLITKW>(realResIndex, index, curKw, wOutput, curOriginIndex);

    if constexpr (MERGE) {
        AscendC::Reg::RegTensor<int32_t> lastResIndex;
        LoadOneElementVF<int32_t, int32_t>(argmaxAddr, lastResIndex, pregOne, bufferOffset);

        AscendC::Reg::RegTensor<DomT> lastVal;
        AscendC::Reg::MaskReg pregOneDom = AscendC::Reg::CreateMask<DomT, AscendC::Reg::MaskPattern::VL1>();
        LoadOneElementVF<DomT, DomT, WideT>(maxValAddr, lastVal, pregOneDom, 0);

        AscendC::Reg::MaskReg selMask16;
        AscendC::Reg::Compare<DomT, CMPMODE::GT>(selMask16, maxBcast, lastVal, maskAllDom);
        AscendC::Reg::MaskReg selMask32;
        AscendC::Reg::MaskUnPack(selMask32, selMask16);
        AscendC::Reg::Select(maxBcast, maxBcast, lastVal, selMask16);
        AscendC::Reg::Select(realResIndex, realResIndex, lastResIndex, selMask32);

        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_LOAD, AscendC::Reg::MemType::VEC_STORE>();
    }

    StoreOneElementVF<int32_t, int32_t>(argmaxAddr, realResIndex, pregOne, bufferOffset);
    if constexpr (MERGE) {
        AscendC::Reg::MaskReg pregOneDom = AscendC::Reg::CreateMask<DomT, AscendC::Reg::MaskPattern::VL1>();
        StoreOneElementVF<DomT, DomT, WideT>(maxValAddr, maxBcast, pregOneDom, 0);
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool MERGE, bool SPLITKW>
__simd_vf__ inline void PoolGradNCHWBigKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ComputeArgmaxB32DomainSimdVf(
    __ubuf__ TYPE_ORIG_X* xLocalAddr, __ubuf__ int32_t* argmaxAddr, __ubuf__ WideT* maxValAddr, int64_t dataCount,
    int64_t curKw, int64_t curOriginIndex, uint32_t bufferOffset, uint16_t repeatTimes, int64_t wOutput, WideT wideMin)
{
    AscendC::Reg::RegTensor<int32_t> resIndex;
    AscendC::Reg::RegTensor<int32_t> index;
    AscendC::Reg::RegTensor<int32_t> realResIndex;
    AscendC::Reg::MaskReg maskAll32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    // argmax槽的单元素读写恒为i32域(VL1); maxVal的域随分支而定
    AscendC::Reg::MaskReg pregOne = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();

    AscendC::Reg::Duplicate(resIndex, static_cast<int32_t>(-1));
    AscendC::Reg::Arange(index, 0);

    // ===== b32: b32比较域(u32索引gather) =====
    AscendC::Reg::RegTensor<WideT> res;
    AscendC::Reg::RegTensor<WideT> vd0;
    AscendC::Reg::MaskReg cmpMaskReg;
    AscendC::Reg::MaskReg maskAll = AscendC::Reg::CreateMask<WideT, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(res, wideMin);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        const uint32_t validCount = static_cast<uint32_t>(dataCount - static_cast<int64_t>(i) * INT_REPEAT_ELEMS >
                                                                  INT_REPEAT_ELEMS ?
                                                              INT_REPEAT_ELEMS :
                                                              dataCount - static_cast<int64_t>(i) * INT_REPEAT_ELEMS);
        uint32_t maskNum = validCount;
        AscendC::Reg::MaskReg p0 = AscendC::Reg::UpdateMask<WideT>(maskNum);

        // 顺序索引gather加载: AddrReg/LoadAlign属scalar侧指令, VF内语义未定义
        AscendC::Reg::RegTensor<int32_t> seqIdx;
        AscendC::Reg::Arange(seqIdx, 0);
        AscendC::Reg::Adds(seqIdx, seqIdx, static_cast<int32_t>(i) * INT_REPEAT_ELEMS, maskAll32);
        AscendC::Reg::DataCopyGather(vd0, xLocalAddr, (AscendC::Reg::RegTensor<uint32_t>&)seqIdx, p0);

        // 整型无NaN: 仅严格大于比较, 平局保留先扫描到的位置(first-wins)。
        // 运行最大值必须用Select回写: Max对掩码外lane按ZEROING清零,
        // 尾块(p0<64)会把未掩码lane的running max清零导致max丢失
        AscendC::Reg::Compare<WideT, CMPMODE::GT>(cmpMaskReg, vd0, res, p0);
        AscendC::Reg::Select(res, vd0, res, cmpMaskReg);
        AscendC::Reg::Select(resIndex, index, resIndex, cmpMaskReg);
        AscendC::Reg::Adds(index, index, INT_REPEAT_ELEMS, maskAll32);
    }

    ReduceMaxWithIndexVF<WideT>(res, index, res, resIndex, static_cast<int32_t>(-1));

    CalcRealIndexVF<int32_t, SPLITKW>(realResIndex, index, curKw, wOutput, curOriginIndex);

    if constexpr (MERGE) {
        AscendC::Reg::RegTensor<int32_t> lastResIndex;
        LoadOneElementVF<int32_t, int32_t>(argmaxAddr, lastResIndex, pregOne, bufferOffset);

        AscendC::Reg::RegTensor<WideT> lastRes;
        LoadOneElementVF<WideT, WideT>(maxValAddr, lastRes, pregOne, 0);

        AscendC::Reg::MaskReg selReg;
        AscendC::Reg::Compare<WideT, CMPMODE::GT>(selReg, res, lastRes, maskAll);

        AscendC::Reg::Select(res, res, lastRes, selReg);
        AscendC::Reg::Select(realResIndex, realResIndex, lastResIndex, selReg);

        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_LOAD, AscendC::Reg::MemType::VEC_STORE>();
    }

    StoreOneElementVF<int32_t, int32_t>(argmaxAddr, realResIndex, pregOne, bufferOffset);
    if constexpr (MERGE) {
        StoreOneElementVF<WideT, WideT>(maxValAddr, res, pregOne, 0);
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <bool MERGE, bool SPLITKW>
__simd_vf__ inline void PoolGradNCHWBigKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::ComputeArgmaxB64DomainSimdVf(
    __ubuf__ TYPE_ORIG_X* xLocalAddr, __ubuf__ int32_t* argmaxAddr, __ubuf__ WideT* maxValAddr, int64_t dataCount,
    int64_t curKw, int64_t curOriginIndex, uint32_t bufferOffset, uint16_t repeatTimes, int64_t wOutput, WideT wideMin)
{
    AscendC::Reg::RegTensor<int32_t> resIndex;
    AscendC::Reg::RegTensor<int32_t> index;
    AscendC::Reg::RegTensor<int32_t> realResIndex;
    AscendC::Reg::MaskReg maskAll32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    // argmax槽的单元素读写恒为i32域(VL1); maxVal的域随分支而定
    AscendC::Reg::MaskReg pregOne = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();

    AscendC::Reg::Duplicate(resIndex, static_cast<int32_t>(-1));
    AscendC::Reg::Arange(index, 0);

    // ===== b64: RegTraitNumTwo双寄存器b64比较域(小核ProcessWInt/max_pool_v3前向同构) =====
    // 掩码经UpdateMask<WideT, RegTraitNumTwo>生成plt_b32位形(b32粒度), 64个b64元素
    // 恰对应64个i32 lane; b64比较掩码可直接作用于i32索引Select(小核同构)。
    // i32-ALL掩码(64bit)与b64双寄存器全量掩码位形一致(max_pool_v3 ReduceMaxAll同构)。
    using DualReg = AscendC::Reg::RegTensor<WideT, AscendC::Reg::RegTraitNumTwo>;
    DualReg res;
    DualReg vd0;
    AscendC::Reg::MaskReg cmpMaskReg;

    AscendC::Reg::Duplicate(res, wideMin);

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        const uint32_t validCount = static_cast<uint32_t>(dataCount - static_cast<int64_t>(i) * INT_REPEAT_ELEMS >
                                                                  INT_REPEAT_ELEMS ?
                                                              INT_REPEAT_ELEMS :
                                                              dataCount - static_cast<int64_t>(i) * INT_REPEAT_ELEMS);
        uint32_t maskNum = validCount;
        AscendC::Reg::MaskReg p64 = AscendC::Reg::UpdateMask<WideT, AscendC::Reg::RegTraitNumTwo>(maskNum);

        AscendC::Reg::RegTensor<int32_t> seqIdx;
        AscendC::Reg::Arange(seqIdx, 0);
        AscendC::Reg::Adds(seqIdx, seqIdx, static_cast<int32_t>(i) * INT_REPEAT_ELEMS, maskAll32);
        AscendC::Reg::DataCopyGather(vd0, xLocalAddr, (AscendC::Reg::RegTensor<uint32_t>&)seqIdx, p64);

        // 严格大于first-wins, Select回写(同b32路径, 避免Max对尾块掩码外lane清零)
        AscendC::Reg::Compare<WideT, CMPMODE::GT>(cmpMaskReg, vd0, res, p64);
        AscendC::Reg::Select(res, vd0, res, cmpMaskReg);
        AscendC::Reg::Select(resIndex, index, resIndex, cmpMaskReg);
        AscendC::Reg::Adds(index, index, INT_REPEAT_ELEMS, maskAll32);
    }

    // 水平argmax: b64域Reduce[MAX]后经reg->reg Duplicate广播(等值比较),
    // 等值lane对i32索引Reduce[MIN](first-wins), 掩码位形与i32 lane一一对应
    DualReg maxDual;
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(maxDual, res, maskAll32);
    DualReg maxBcast;
    AscendC::Reg::Duplicate(maxBcast, maxDual, maskAll32);
    AscendC::Reg::MaskReg eqMask;
    AscendC::Reg::Compare<WideT, CMPMODE::EQ>(eqMask, res, maxBcast, maskAll32);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MIN>(index, resIndex, eqMask);

    ArgmaxPadIndexVf(index, maskAll32);
    CalcRealIndexVF<int32_t, SPLITKW>(realResIndex, index, curKw, wOutput, curOriginIndex);

    if constexpr (MERGE) {
        AscendC::Reg::RegTensor<int32_t> lastResIndex;
        LoadOneElementVF<int32_t, int32_t>(argmaxAddr, lastResIndex, pregOne, bufferOffset);

        // b64单元素load: LoadOneElementVF模板不含双寄存器类型,
        // 按其else分支语义显式展开(LoadUnAlign支持b64)
        DualReg lastRes;
        {
            AscendC::Reg::UnalignRegForLoad u0;
            __ubuf__ WideT* srcAddr = (__ubuf__ WideT*)maxValAddr;
            AscendC::Reg::LoadUnAlignPre(u0, srcAddr);
            AscendC::Reg::LoadUnAlign(lastRes, u0, srcAddr, 1);
        }

        AscendC::Reg::MaskReg selReg;
        AscendC::Reg::Compare<WideT, CMPMODE::GT>(selReg, maxBcast, lastRes, maskAll32);

        AscendC::Reg::Select(maxBcast, maxBcast, lastRes, selReg);
        AscendC::Reg::Select(realResIndex, realResIndex, lastResIndex, selReg);

        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_LOAD, AscendC::Reg::MemType::VEC_STORE>();
    }

    StoreOneElementVF<int32_t, int32_t>(argmaxAddr, realResIndex, pregOne, bufferOffset);
    if constexpr (MERGE) {
        // b64单元素store(max_pool_v3 StoreUnAlign双寄存器count=1同构)
        {
            AscendC::Reg::UnalignRegForStore u0;
            __ubuf__ WideT* dstAddr = (__ubuf__ WideT*)maxValAddr;
            AscendC::Reg::StoreUnAlign(dstAddr, maxBcast, u0, 1);
            AscendC::Reg::StoreUnAlignPost(dstAddr, u0, 0);
        }
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <const bool IS_OVERLAP>
__simd_vf__ inline void PoolGradNCHWBigKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::BackwardComputeIntSimdVf(
    __ubuf__ TYPE_ORIG_X* yAddr, __ubuf__ TYPE_ORIG_X* gradAddr, __ubuf__ int32_t* argmaxAddr, uint16_t highAxisActual,
    uint16_t hArgmaxActual, int32_t wArgmaxActual, int32_t wArgmaxAligned, uint16_t wProBatchSize, uint16_t repeatTimes,
    uint32_t wRemainBatchCount, uint16_t wRemainTail, int32_t wOutput, int32_t wOutputAligned, int32_t hOutputActual,
    int32_t wOutputActual, int32_t curHIndex, int32_t curWIndex)
{
    const uint16_t computeSize = INT_REPEAT_ELEMS;
    const uint32_t one = 1;
    const uint32_t all = computeSize;

    AscendC::Reg::RegTensor<int32_t> wOutputConstReg;
    AscendC::Reg::Duplicate(wOutputConstReg, wOutput);

    AscendC::Reg::RegTensor<uint32_t> initialRegIndex;
    AscendC::Reg::RegTensor<uint32_t> parallelRegIndex;
    AscendC::Reg::RegTensor<uint32_t> parallelRegGrad;

    AscendC::Reg::MaskReg allMaskU32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();

    GenInitial1DIndicesVF((AscendC::Reg::RegTensor<int32_t>&)initialRegIndex, wProBatchSize);

    for (uint16_t highIdx = 0; highIdx < highAxisActual; ++highIdx) {
        const uint32_t highArgmaxOffset = highIdx * hArgmaxActual * wArgmaxAligned;
        const uint32_t highIndexOffset = highIdx * hArgmaxActual * wArgmaxActual;
        const uint32_t highOutputOffset = highIdx * hOutputActual * wOutputAligned;
        for (uint16_t hIdx = 0; hIdx < hArgmaxActual; hIdx++) {
            for (uint16_t wRepeatIdx = 0; wRepeatIdx < repeatTimes; wRepeatIdx++) {
                for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                    const uint32_t gradBase = wBatchIdx + wRepeatIdx * computeSize * wProBatchSize +
                                              hIdx * wArgmaxAligned + highArgmaxOffset;
                    const uint32_t argBase = wBatchIdx + wRepeatIdx * computeSize * wProBatchSize +
                                             hIdx * wArgmaxActual + highIndexOffset;
                    AscendC::Reg::Adds(parallelRegGrad, initialRegIndex, gradBase, allMaskU32);
                    AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, argBase, allMaskU32);
                    DoScatterSingle<IS_OVERLAP>(all, yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad,
                                                wOutputConstReg, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                                hOutputActual, wOutputActual);
                }
            }
            // 尾段整batch, 用不满mask
            for (uint16_t wBatchIdx = 0; wBatchIdx < wProBatchSize; wBatchIdx++) {
                const uint32_t gradBase = wBatchIdx + repeatTimes * computeSize * wProBatchSize +
                                          hIdx * wArgmaxAligned + highArgmaxOffset;
                const uint32_t argBase = wBatchIdx + repeatTimes * computeSize * wProBatchSize + hIdx * wArgmaxActual +
                                         highIndexOffset;
                AscendC::Reg::Adds(parallelRegGrad, initialRegIndex, gradBase, allMaskU32);
                AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, argBase, allMaskU32);
                DoScatterSingle<IS_OVERLAP>(wRemainBatchCount, yAddr, gradAddr, argmaxAddr, parallelRegIndex,
                                            parallelRegGrad, wOutputConstReg, curHIndex, curWIndex, wOutputAligned,
                                            highOutputOffset, hOutputActual, wOutputActual);
            }

            // 尾段零散点
            for (uint16_t wBatchIdx = 0; wBatchIdx < wRemainTail; wBatchIdx++) {
                const uint32_t gradBase = wBatchIdx + wRemainBatchCount * wProBatchSize +
                                          repeatTimes * computeSize * wProBatchSize + hIdx * wArgmaxAligned +
                                          highArgmaxOffset;
                const uint32_t argBase = wBatchIdx + wRemainBatchCount * wProBatchSize +
                                         repeatTimes * computeSize * wProBatchSize + hIdx * wArgmaxActual +
                                         highIndexOffset;
                AscendC::Reg::Adds(parallelRegGrad, initialRegIndex, gradBase, allMaskU32);
                AscendC::Reg::Adds(parallelRegIndex, initialRegIndex, argBase, allMaskU32);
                DoScatterSingle<IS_OVERLAP>(one, yAddr, gradAddr, argmaxAddr, parallelRegIndex, parallelRegGrad,
                                            wOutputConstReg, curHIndex, curWIndex, wOutputAligned, highOutputOffset,
                                            hOutputActual, wOutputActual);
            }
        }
    }
}

template <typename TYPE_ORIG_X, const uint32_t IS_CHECK_RANGE>
template <const bool IS_OVERLAP>
__simd_callee__ inline void PoolGradNCHWBigKernelV3<TYPE_ORIG_X, IS_CHECK_RANGE>::DoScatterSingle(
    uint32_t argmaxMaskCount, __ubuf__ TYPE_ORIG_X* yAddr, __ubuf__ TYPE_ORIG_X* gradAddr, __ubuf__ int32_t* argmaxAddr,
    AscendC::Reg::RegTensor<uint32_t>& parallelRegIndex, AscendC::Reg::RegTensor<uint32_t>& parallelRegGrad,
    AscendC::Reg::RegTensor<int32_t>& wOutputConstReg, int32_t curHIndex, int32_t curWIndex, int32_t wOutputAligned,
    int32_t highOutputOffset, int32_t hOutputActual, int32_t wOutputActual)
{
    AscendC::Reg::RegTensor<int32_t> argmaxReg;
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;

    uint32_t argmaxMask = argmaxMaskCount;
    AscendC::Reg::MaskReg pregArgmax = AscendC::Reg::UpdateMask<int32_t>(argmaxMask);

    AscendC::Reg::DataCopyGather(argmaxReg, argmaxAddr, parallelRegIndex, pregArgmax);

    // argmax全局h/w扁平偏移换算到当前tile的y缓冲布局(IndexConvNchw的int32/非MUL_NC路径,
    // 该公共头暂无VF变体, 按位逐行内联)
    AscendC::Reg::RegTensor<int32_t> hTmpIndexReg;
    AscendC::Reg::RegTensor<int32_t> wTmpIndexReg;
    AscendC::Reg::MaskReg allMaskI32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Div(hTmpIndexReg, argmaxReg, wOutputConstReg, allMaskI32);
    AscendC::Reg::Adds(hIndexReg, hTmpIndexReg, -curHIndex, allMaskI32);
    AscendC::Reg::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMaskI32);
    AscendC::Reg::Sub(wTmpIndexReg, argmaxReg, wTmpIndexReg, allMaskI32);
    AscendC::Reg::Adds(wIndexReg, wTmpIndexReg, -curWIndex, allMaskI32);
    AscendC::Reg::Muls((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, hIndexReg, wOutputAligned, allMaskI32);
    AscendC::Reg::Add((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                      wIndexReg, allMaskI32);
    AscendC::Reg::Adds((AscendC::Reg::RegTensor<int32_t>&)argmaxReg, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg,
                       highOutputOffset, allMaskI32);

    if constexpr (sizeof(TYPE_ORIG_X) == DTYPE_WIDTH_B8) {
        // 8位: b16掩码过滤 + b16 lane相加 + 低字节散射
        AscendC::Reg::RegTensor<uint16_t> yOffU16;
        AscendC::Reg::RegTensor<uint16_t> gradOffU16;
        AscendC::Reg::Pack(yOffU16, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg);
        AscendC::Reg::Pack(gradOffU16, (AscendC::Reg::RegTensor<int32_t>&)parallelRegGrad);
        uint32_t count16 = argmaxMaskCount;
        AscendC::Reg::MaskReg mask16 = AscendC::Reg::UpdateMask<int16_t>(count16);
        AscendC::Reg::RegTensor<uint16_t> hLocU16;
        AscendC::Reg::RegTensor<uint16_t> wLocU16;
        AscendC::Reg::Pack(hLocU16, hIndexReg);
        AscendC::Reg::Pack(wLocU16, wIndexReg);
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMaskVF<int16_t>(mask16, (AscendC::Reg::RegTensor<int16_t>&)hLocU16,
                                  (AscendC::Reg::RegTensor<int16_t>&)wLocU16, static_cast<int16_t>(hOutputActual),
                                  static_cast<int16_t>(wOutputActual));
        } else {
            SentinelFilterVf<int16_t>(mask16, (AscendC::Reg::RegTensor<int16_t>&)hLocU16,
                                      (AscendC::Reg::RegTensor<int16_t>&)wLocU16);
        }
        if constexpr (IS_OVERLAP) {
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }

        AscendC::Reg::RegTensor<int16_t> yReg;
        AscendC::Reg::RegTensor<int16_t> gradReg;
        AscendC::Reg::DataCopyGather(yReg, yAddr, yOffU16, mask16);
        AscendC::Reg::DataCopyGather(gradReg, gradAddr, gradOffU16, mask16);
        // b8 的 scatter 按 b16 掩码取每个 lane 的低字节写回，等价于模 256 回绕，
        // 无需 Cast/Pack 改变 lane 映射(与小核一致)。
        AscendC::Reg::Add(yReg, yReg, gradReg, mask16);
        AscendC::Reg::DataCopyScatter((__ubuf__ int8_t*)yAddr, (AscendC::Reg::RegTensor<int8_t>&)yReg, yOffU16, mask16);
    } else if constexpr (sizeof(TYPE_ORIG_X) == DTYPE_WIDTH_B16) {
        // 16位: u16索引gather + 原位宽回绕累加
        AscendC::Reg::RegTensor<uint16_t> yOffU16;
        AscendC::Reg::RegTensor<uint16_t> gradOffU16;
        AscendC::Reg::Pack(yOffU16, (AscendC::Reg::RegTensor<int32_t>&)argmaxReg);
        AscendC::Reg::Pack(gradOffU16, (AscendC::Reg::RegTensor<int32_t>&)parallelRegGrad);
        uint32_t count16 = argmaxMaskCount;
        AscendC::Reg::MaskReg mask16 = AscendC::Reg::UpdateMask<TYPE_ORIG_X>(count16);
        {
            AscendC::Reg::RegTensor<uint16_t> hLocU16;
            AscendC::Reg::RegTensor<uint16_t> wLocU16;
            AscendC::Reg::Pack(hLocU16, hIndexReg);
            AscendC::Reg::Pack(wLocU16, wIndexReg);
            if constexpr (IS_CHECK_RANGE == 1) {
                FilterMaskVF<int16_t>(mask16, (AscendC::Reg::RegTensor<int16_t>&)hLocU16,
                                      (AscendC::Reg::RegTensor<int16_t>&)wLocU16, static_cast<int16_t>(hOutputActual),
                                      static_cast<int16_t>(wOutputActual));
            } else {
                SentinelFilterVf<int16_t>(mask16, (AscendC::Reg::RegTensor<int16_t>&)hLocU16,
                                          (AscendC::Reg::RegTensor<int16_t>&)wLocU16);
            }
        }
        if constexpr (IS_OVERLAP) {
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }

        AscendC::Reg::RegTensor<TYPE_ORIG_X> yReg;
        AscendC::Reg::RegTensor<TYPE_ORIG_X> gradReg;
        AscendC::Reg::DataCopyGather(yReg, yAddr, yOffU16, mask16);
        AscendC::Reg::DataCopyGather(gradReg, gradAddr, gradOffU16, mask16);
        AscendC::Reg::Add(yReg, yReg, gradReg, mask16);
        AscendC::Reg::DataCopyScatter(yAddr, yReg, yOffU16, mask16);
    } else if constexpr (sizeof(TYPE_ORIG_X) == DTYPE_WIDTH_B32) {
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMaskVF<int32_t>(pregArgmax, hIndexReg, wIndexReg, hOutputActual, wOutputActual);
        } else {
            SentinelFilterVf<int32_t>(pregArgmax, hIndexReg, wIndexReg);
        }
        if constexpr (IS_OVERLAP) {
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }

        AscendC::Reg::RegTensor<TYPE_ORIG_X> yReg;
        AscendC::Reg::RegTensor<TYPE_ORIG_X> gradReg;
        AscendC::Reg::DataCopyGather(yReg, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
        AscendC::Reg::DataCopyGather(gradReg, gradAddr, parallelRegGrad, pregArgmax);
        AscendC::Reg::Add(yReg, yReg, gradReg, pregArgmax);
        AscendC::Reg::DataCopyScatter(yAddr, yReg, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
    } else {
        // 64位: RegTraitNumTwo双寄存器承载, u32索引gather/scatter(与小核后向同构);
        // 掩码为plt_b32位形(b32粒度), FilterMaskVF<int32_t>可直接作用于该掩码
        uint32_t count64 = argmaxMaskCount;
        AscendC::Reg::MaskReg mask64 = AscendC::Reg::UpdateMask<TYPE_ORIG_X, AscendC::Reg::RegTraitNumTwo>(count64);
        if constexpr (IS_CHECK_RANGE == 1) {
            FilterMaskVF<int32_t>(mask64, hIndexReg, wIndexReg, hOutputActual, wOutputActual);
        } else {
            SentinelFilterVf<int32_t>(mask64, hIndexReg, wIndexReg);
        }
        if constexpr (IS_OVERLAP) {
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }

        AscendC::Reg::RegTensor<TYPE_ORIG_X, AscendC::Reg::RegTraitNumTwo> yReg;
        AscendC::Reg::RegTensor<TYPE_ORIG_X, AscendC::Reg::RegTraitNumTwo> gradReg;
        AscendC::Reg::DataCopyGather(yReg, yAddr, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, mask64);
        AscendC::Reg::DataCopyGather(gradReg, gradAddr, parallelRegGrad, mask64);
        AscendC::Reg::Add(yReg, yReg, gradReg, mask64);
        AscendC::Reg::DataCopyScatter(yAddr, yReg, (AscendC::Reg::RegTensor<uint32_t>&)argmaxReg, mask64);
    }
}

} // namespace MaxPoolV3GradNCHWBigKernelNameSpace

#endif // MAX_POOL_V3_GRAD_NCHW_BIG_KERNEL_H_
