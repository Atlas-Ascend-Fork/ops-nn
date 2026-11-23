/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_bp_func.h
 * \brief
 */

#ifndef CONV3D_BP_FUNC_ADVANCE_H
#define CONV3D_BP_FUNC_ADVANCE_H

#include "conv3d_bp_common_func.h"
#include "conv3d_bp_large_attribute_func.h"
#include "conv3d_bp_kernel_split_func.h"
#include "../../../inc/macro.h"

#include "conv3d_bp_func_arch35_init_params.h"
#include "conv3d_bp_func_arch35_buffer_tque.h"
#include "conv3d_bp_func_arch35_compute.h"

DECLARE_CHECK_IMPL(Init);
DECLARE_CHECK_IMPL(SetOutBackprop);
DECLARE_CHECK_IMPL(SetWeight);
DECLARE_CHECK_IMPL(SetBias);
DECLARE_CHECK_IMPL(SetScale);
DECLARE_CHECK_IMPL(SetSingleShape);
DECLARE_CHECK_IMPL(SetStartIdx);
DECLARE_CHECK_IMPL(SetBatchCoreIdx);
DECLARE_CHECK_IMPL(FreeBiasTensor);
DECLARE_CHECK_SYNC_IMPL(Iterate);
DECLARE_CHECK_SYNC_IMPL(IterateAll);
DECLARE_CHECK_SYNC_IMPL(GetTensorC);
DECLARE_CHECK_SYNC_IMPL(VecPreProcess);
DECLARE_CHECK_SYNC_IMPL(VecPostProcess);
DECLARE_CHECK_IMPL(End);
namespace Convolution3DBackpropFunc {

template <class Intf>
struct Init {
    // 定义call函数的默认重载函数，支持任意类型任意数量的参数
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const Conv3DBackpropInputArch35TilingData& tiling,
                                       const bool hasBias, const bool hasSecondOutput = false)
    {
        self->ctx.hasBias_ = hasBias;
#ifdef DTYPE_Y1
        self->ctx.hasSecondOutput_ = hasSecondOutput;
#endif
        // kernel侧通过ctx持有tiling指针，供后续全流程访问，避免向kernel内部逐层传递引用
        self->ctx.tiling_ = &(tiling);
        self->ctx.isLocalW_ = self->ctx.tiling_->enableLocalW == 1;
        self->ctx.curEnableFullLoad_ = self->ctx.tiling_->enableFullLoad;
        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(true);
        }
        CheckTiling<Intf>(self);
        InitParams<Intf>(self);
        InitTque<Intf>(self, hasBias);
        // kernel拆分、切Dk、C04场景，Init()先空下一个CrossCoreSetFlag()来保证End()里下的CrossCoreWaitFlag()可以成对
        CrossCoreSetHeadForMix<Intf>(self);
    }
};

template <class Intf>
struct FreeBiasTensor {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self)
    {
        if ASCEND_IS_AIC_SCALAR {
            self->ctx.biasL1Que_.FreeTensor(self->ctx.biasL1Buf_);
            self->ctx.biasBTQue_.FreeTensor(self->ctx.biasBTBuf_);
        }
    }
};

template <class Intf>
struct SetWeight {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::SrcBT>& weight)
    {
        self->ctx.weightGlobal_ = weight;
    }
};

template <class Intf>
struct SetOutBackprop {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::SrcAT>& outBackprop)
    {
        self->ctx.outBackPropGlobal_ = outBackprop;
    }
};

template <class Intf>
struct SetBias {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::BiasT>& bias)
    {
        self->ctx.biasGlobal_ = bias;
    }
};

template <class Intf>
struct SetScale {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::ScaleT0>& scale)
    {
        self->ctx.scale0Global_ = scale;
    }
};

template <class Intf>
struct SetSingleShape {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, uint64_t singleShapeM, uint64_t singleShapeK, uint32_t singleShapeN,
                                       uint32_t singleShapeD)
    {
        if (self->ctx.singleShapeDin_ != singleShapeD) {
            self->ctx.singleShapeDin_ = singleShapeD;
        }

        if (self->ctx.singleShapeM_ != singleShapeM) {
            self->ctx.singleShapeM_ = singleShapeM;
            InitStepMParams<Intf>(self);
        }

        if (self->ctx.singleShapeCin_ != singleShapeN) {
            self->ctx.singleShapeCin_ = singleShapeN;
            InitStepNParams<Intf>(self);
        }

        if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
            self->ctx.singleShapeCout_ = singleShapeK / self->ctx.singleShapeHWk_;
            InitStepKParams<Intf>(self);
        } else {
            if (self->ctx.singleShapeCout_ * self->ctx.singleShapeHWk_ != singleShapeK) {
                self->ctx.singleShapeCout_ = singleShapeK / self->ctx.singleShapeHWk_;
                InitStepKParams<Intf>(self);
            }
        }

        if (self->ctx.tiling_->enlarge > 1) {
            self->ctx.curEnlarge = self->ctx.singleShapeCout_ / (self->ctx.tiling_->cout / self->ctx.tiling_->oriGroup);
            self->ctx.curEnlargeCin1_ = DivCeil16(self->ctx.curEnlarge *
                                                  (self->ctx.tiling_->cin / self->ctx.tiling_->oriGroup));
        }
    }
};

template <class Intf>
struct SetStartIdx {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, uint32_t curDinStartIdx, int64_t curMStartIdx,
                                       int32_t curCinStartIdx, int32_t curCoutStartIdx)
    {
        // 需要增加入参的startDinIdx判断，否则无法确定padding和补0问题
        // 增加入参，传参判断
        self->ctx.curDinStartIdx_ = curDinStartIdx;
        self->ctx.curMStartIdx_ = curMStartIdx;
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
            self->ctx.curHoStartIdx_ = static_cast<int32_t>(curMStartIdx / self->ctx.tiling_->wi) -
                                       self->ctx.tiling_->backpropPadUp;
        } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            self->ctx.curBackPropPadUp_ = self->ctx.tiling_->backpropPadUp >> 1;
            self->ctx.curHoStartIdx_ = static_cast<int32_t>(curMStartIdx / (self->ctx.splitWi_)) -
                                       self->ctx.curBackPropPadUp_;
        } else {
            self->ctx.curHoStartIdx_ = static_cast<int32_t>(curMStartIdx / self->ctx.tiling_->wi) -
                                       self->ctx.curBackPropPadUp_;
        }
        self->ctx.curCinStartIdx_ = curCinStartIdx;
        self->ctx.curCoutStartIdx_ = curCoutStartIdx;
    }
};

template <class Intf>
struct SetBatchCoreIdx {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, uint32_t batchCoreIdx) { self->ctx.curBatchCoreIdx_ = batchCoreIdx; }
};

template <class Intf>
static __aicore__ inline void InitFirstIterState(Intf* self)
{
    if (EnableVecGroupEnlarge(self)) {
        // 初始化 AIV transdata 迭代器。
        self->ctx.groupIterIdx_ = 0;
    }
    self->ctx.needComputeFlag_ = true;
    self->ctx.curMIdx_ = 0;
    self->ctx.curNIdx_ = 0;
    self->ctx.curDinIdx_ = self->ctx.curDinStartIdx_;
    self->ctx.curDkIdx_ = 0;
    self->ctx.curHoIdx_ = self->ctx.curHoStartIdx_;
    self->ctx.isFirstIter_ = false;
    self->ctx.isLoadA1_ = true;
    self->ctx.isFreeA1_ = false;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        self->ctx.rearrangeHIndex_ = 0;
        self->ctx.rearrangeWIndex_ = 0;
        self->ctx.splitHIndex_ = self->ctx.splitHStartIndex_;
        self->ctx.splitWIndex_ = self->ctx.splitWStartIndex_;
        if (self->ctx.tiling_->dk > 1) { // dk等于1时，全载时无需重复加载B矩阵
            self->ctx.isLoadB1_ = true;
            self->ctx.isFreeB1_ = false;
        }
    } else {
        if (!self->ctx.isB1FullLoadFlag_ || self->ctx.tiling_->dk > 1 || Intf::conv3dConfig.enableC04Flag ||
            !self->ctx.curEnableFullLoad_ || self->ctx.tiling_->group != 1) {
            self->ctx.isLoadB1_ = true;
            self->ctx.isFreeB1_ = false;
        }
    }
}

template <class Intf>
static __aicore__ inline bool ProcessKernelSplitIteration(Intf* self, bool hasBias)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (!IterateForKernelSplit<Intf>(self)) {
            return false;
        }
        UpdateFullLoadL1Status<Intf>(self);
        if (unlikely(self->ctx.tiling_->hk == 1)) {
            if (unlikely(hasBias && self->ctx.needComputeFlag_)) {
                self->ctx.needComputeFlag_ = false;
                Compute<Intf, true>(self);
            }
            return true;
        }
        if (unlikely(hasBias)) {
            Compute<Intf, true>(self);
        } else {
            Compute<Intf, false>(self);
        }
        return true;
    }
    return false;
}

template <class Intf>
static __aicore__ inline bool AdvanceIterIndices(Intf* self)
{
    if (++self->ctx.curMIdx_ >= self->ctx.mIter_) {
        self->ctx.curMIdx_ = 0;
        if (++self->ctx.curNIdx_ >= self->ctx.nIter_) {
            self->ctx.curNIdx_ = 0;
            if (++self->ctx.curDinIdx_ >= self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_) {
                return false;
            }
        }
    }
    return true;
}

template <class Intf, bool sync>
struct Iterate {
    // 一次iterate计算(baseM, baseN, baseD)，当前baseD=1
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline bool call(Intf* self, bool enPartialSum, bool hasBias)
    {
        /*
        |   <---------singleShapeM------->        |
        |  <---L1A_ping--->  |  <---L1A_pong--->  |
        |_L0A1_|_L0A2_|_L0A3_|_L0A4_|_L0A5_|_L0A6_|
        ↑  <--curStepM_-->    |                    ↑
        curML0Idx_            ↑                  mIter_
        curML1Idx_        next_curML1Idx

        |   <---------singleShapeN------->        |
        |  <---L1B_ping--->  |  <---L1B_pong--->  |
        |_L0B1_|_L0B2_|_L0B3_|_L0B4_|_L0B5_|_L0B6_|
        ↑  <--curStepN_-->    |                    ↑
        curNL0Idx_            ↑                   nIter_
        curNL1Idx_       next_curNL1Idx

        order_N表示L1上驻留B循环A，顺序为L1A_ping * L1B_ping, L1A_pong * L1B_ping，L1A_ping * L1B_pong，L1A_pong *
        L1B_pong L0上也是驻留B，循环A order_N: L0A1*L0B1, L0A2*L0B1, L0A3*L0B1, L0A1*L0B2 …………
        L0A3*L0B3，L0A4*L0B1，L0A5*L0B1 …… L0A6*L0B6 order_M: L0A1*L0B1, L0A1*L0B2, L0A1*L0B3, L0A2*L0B1 …………
        L0A3*L0B3，L0A1*L0B4，L0A1*L0B5 …… L0A6*L0B6
        */
        // 更新idx，用L1、L1step、L0三个指针控制走位和计算offset，表示计算第几个mL0 * baseN
        if (unlikely(hasBias && !self->ctx.enableSplitK_)) {
            self->ctx.computeBiasOnce_ = false;
        }
        if (unlikely(self->ctx.isFirstIter_)) {
            InitFirstIterState<Intf>(self);
        } else {
            if (ProcessKernelSplitIteration<Intf>(self, hasBias)) {
                return true;
            }
            if (!AdvanceIterIndices<Intf>(self)) {
                return false;
            }
        }

        UpdateL1ComputeInfo<Intf>(self);
        if (unlikely(hasBias)) {
            Compute<Intf, true>(self);
        } else {
            Compute<Intf, false>(self);
        }
        return true;
    }
};

template <class Intf, bool sync>
struct IterateAll {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output, uint8_t enAtomic,
                                       bool fullLoadBiasFlag_, bool freeBiasFlag_)
    {
        bool hasBias = self->ctx.hasBias_;
        if (unlikely(hasBias && self->ctx.tiling_->isBiasFullLoad)) {
            if (freeBiasFlag_) {
                if ASCEND_IS_AIC_SCALAR {
                    self->ctx.biasL1Que_.FreeTensor(self->ctx.biasL1Buf_);
                    self->ctx.biasBTQue_.FreeTensor(self->ctx.biasBTBuf_);
                }
            }
            if (fullLoadBiasFlag_) {
                Convolution3DBackpropFunc::FullLoadBias<Intf>(self);
            }
        }
        SetDequantScale<Intf>(self);
        if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_SPLIT_KERNEL_HW) {
            if (self->ctx.enableSplitK_) {
                CalcSplitK_<Intf, sync>(self, enAtomic, output, hasBias);
            } else {
                while (self->template Iterate<sync>(false, hasBias)) {
                    self->template VecPreProcess<sync>(output, enAtomic);
                    self->template GetTensorC<sync>(output, enAtomic);
                    self->template VecPostProcess<sync>(output, enAtomic);
                }
            }
        } else {
            self->template IterateAllForKernelSplit<sync>(output, enAtomic);
        }

        if ASCEND_IS_AIC_SCALAR {
            if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
                if (self->ctx.tiling_->quantMode0 ==
                    static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                    self->ctx.scale0L1Que_.FreeTensor(self->ctx.scale0L1Buf_);
                }
            }
#ifdef DTYPE_Y1
            using Intf1 = Convolution3DBackprop::Output1Intf<Intf>;
            if constexpr (Intf1::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
                if (self->ctx.hasSecondOutput_ &&
                    self->ctx.tiling_->quantMode1 ==
                        static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                    self->ctx.scale1L1Que_.FreeTensor(self->ctx.scale1L1Buf_);
                }
            }
#endif
        }
        self->ctx.isFirstIter_ = true;
    }
};

template <class Intf, bool sync>
struct GetTensorC {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                       uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        LoadL0c2Gm<Intf>(self, output, enAtomic, enSequentialWrite);
    }
};

template <class Intf, bool sync>
struct VecPreProcess {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                       uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        if (unlikely(self->ctx.enableSplitDk_)) {
            if ASCEND_IS_AIC_SCALAR {
                // 反向event(实际只需要保证vector核内的时序,不加反向event时cube到vector之间的event有可能会下超过16次,导致报错)
                CvCrossCoreWait<Intf, PIPE_MTE3, PIPE_FIX>(self, FLAG_FIXP_ID);
            }
        }
    }
};

template <class Intf, bool sync>
struct VecPostProcess {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                       uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            Rearrange2Gm<Intf>(self, output, enAtomic, enSequentialWrite);
        }
        if (unlikely(self->ctx.enableSplitDk_)) {
            if ASCEND_IS_AIC_SCALAR {
                CvCrossCoreSet<Intf, PIPE_FIX, PIPE_MTE2>(self, FLAG_MTE2_VEC_ID);
            }
            if ASCEND_IS_AIV_SCALAR {
                if (GetSubBlockIdx() != 0) {
                    return;
                }
                CvCrossCoreWait<Intf, PIPE_FIX, PIPE_MTE2>(self, FLAG_MTE2_VEC_ID);
                CastToDstType<Intf>(self, output, enAtomic, enSequentialWrite);
                CvCrossCoreSet<Intf, PIPE_MTE3, PIPE_FIX>(self, FLAG_FIXP_ID);
            }
        }
        // workSpace累加UB cast搬出
        if (self->ctx.useUbAccumForSplitK_) {
            if ASCEND_IS_AIC_SCALAR {
                // splitK场景最后一段KSegment且需要计算搬出
                if (self->ctx.isLastKSegment_ && self->ctx.needComputeFlag_) {
                    CvCrossCoreSet<Intf, PIPE_FIX, PIPE_MTE2>(self, FLAG_MTE2_VEC_ID);
                }
            }
        }
    }
};

template <class Intf>
struct End {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, Convolution3DBackpropFunc);
    static __aicore__ inline void call(Intf* self)
    {
        CrossCoreWaitTailForMix<Intf>(self);

        self->ctx.inQueL1A_.FreeAllEvent();
        self->ctx.inQueL1B_.FreeAllEvent();

        self->ctx.l0cPing_.FreeAllEvent();
        if (self->ctx.tiling_->cl0Pbuffer > 1) {
            self->ctx.l0cPong_.FreeAllEvent();
        }

        if (unlikely(self->ctx.hasBias_)) {
            self->ctx.biasL1Que_.FreeAllEvent();
            self->ctx.biasBTQue_.FreeAllEvent();
        }

        if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                self->ctx.scale0L1Que_.FreeAllEvent();
            }
        }
#ifdef DTYPE_Y1
        using Intf1 = Convolution3DBackprop::Output1Intf<Intf>;
        if constexpr (Intf1::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if (self->ctx.hasSecondOutput_ &&
                self->ctx.tiling_->quantMode1 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                self->ctx.scale1L1Que_.FreeAllEvent();
            }
        }
#endif

        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(false);
        }
    }
};
} // namespace Convolution3DBackpropFunc
#endif
