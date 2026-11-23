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
 * \file conv3d_bp_func_arch35_init_params.h
 * \brief Iteration order and tiling parameter initialization helpers for conv3d backprop input
 */

#ifndef CONV3D_BP_FUNC_ARCH35_INIT_PARAMS_H
#define CONV3D_BP_FUNC_ARCH35_INIT_PARAMS_H

#include "conv3d_bp_common_func.h"
#include "conv3d_bp_large_attribute_func.h"
#include "conv3d_bp_kernel_split_func.h"
#include "../../../inc/macro.h"

namespace Convolution3DBackpropFunc {

enum class IterateOrder : uint8_t {
    ORDER_M = 0,
    ORDER_N,
    ORDER_K,
    UNDEF,
};

template <class Intf>
__aicore__ inline void CheckTiling(Intf* self)
{
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.tiling_->batch > 0), "orignal batch is %d , which should be larger than 0",
                   self->ctx.tiling_->batch);
    ascendc_assert((self->ctx.tiling_->cin > 0), "orignal cin is %d , which should be larger than 0",
                   self->ctx.tiling_->cin);
    ascendc_assert((self->ctx.tiling_->cout > 0), "orignal cout is %d , which should be larger than 0",
                   self->ctx.tiling_->cout);
    ascendc_assert((self->ctx.tiling_->ho > 0), "orignal ho is %d , which should be larger than 0",
                   self->ctx.tiling_->ho);
    ascendc_assert((self->ctx.tiling_->wo > 0), "orignal wo is %d , which should be larger than 0",
                   self->ctx.tiling_->wo);
    ascendc_assert((self->ctx.tiling_->hi > 0), "orignal hi is %d , which should be larger than 0",
                   self->ctx.tiling_->hi);
    ascendc_assert((self->ctx.tiling_->wi > 0), "orignal wi is %d , which should be larger than 0",
                   self->ctx.tiling_->wi);
    ascendc_assert((self->ctx.tiling_->hk > 0), "orignal hk is %d , which should be larger than 0",
                   self->ctx.tiling_->hk);
    ascendc_assert((self->ctx.tiling_->wk > 0), "orignal wk is %d , which should be larger than 0",
                   self->ctx.tiling_->wk);
    ascendc_assert((self->ctx.tiling_->singleCoreBatch > 0), "singleCoreBatch is %d , which should be larger than 0",
                   self->ctx.tiling_->singleCoreBatch);
    ascendc_assert((self->ctx.tiling_->singleCoreCout > 0), "singleCoreCout is %d , which should be larger than 0",
                   self->ctx.tiling_->singleCoreCout);
    ascendc_assert((self->ctx.tiling_->singleCoreCin > 0), "singleCoreCin is %d , which should be larger than 0",
                   self->ctx.tiling_->singleCoreCin);
    ascendc_assert((self->ctx.tiling_->baseM > 0), "baseM is %d , which should be larger than 0",
                   self->ctx.tiling_->baseM);
    ascendc_assert((self->ctx.tiling_->baseK > 0), "baseK is %d , which should be larger than 0",
                   self->ctx.tiling_->baseK);
    ascendc_assert((self->ctx.tiling_->baseN > 0), "baseN is %d , which should be larger than 0",
                   self->ctx.tiling_->baseN);
    ascendc_assert((self->ctx.tiling_->stepKa > 0), "stepKa is %d , which should be larger than 0",
                   self->ctx.tiling_->stepKa);
    ascendc_assert((self->ctx.tiling_->stepKb > 0), "stepKb is %d , which should be larger than 0",
                   self->ctx.tiling_->stepKb);
    ascendc_assert((self->ctx.tiling_->dk % self->ctx.tiling_->singleIterateDk == 0),
                   "tiling_->dk should be divisible by tiling_->singleIterateDk");
#endif
}

template <class Intf>
__aicore__ inline void InitStepMParams(Intf* self)
{
    self->ctx.mIter_ = DivCeil(self->ctx.singleShapeM_, self->ctx.tiling_->baseM);
    self->ctx.tailM_ = self->ctx.singleShapeM_ - (self->ctx.mIter_ - 1) * self->ctx.tiling_->baseM;
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.mIter_ > 0), "self->ctx.mIter_ is %d , which should be larger than 0", self->ctx.mIter_);
#endif
}

template <class Intf>
__aicore__ inline void InitStepKParams(Intf* self)
{
    uint64_t tmpSingleCoreK = 0;
    // kernel拆分后，K方向去除无效计算
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        tmpSingleCoreK = self->ctx.tiling_->group == 1 ? static_cast<uint64_t>(self->ctx.tiling_->cout1) *
                                                             self->ctx.splitHkWkC0List_[self->ctx.splitIndex_] :
                                                         static_cast<uint64_t>(self->ctx.tiling_->cout1G) *
                                                             self->ctx.splitHkWkC0List_[self->ctx.splitIndex_];
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            RecalcStepForKernelSplit<Intf>(self); // 当pad大于0时，子kernel顺序会变化，stepka需要重新计算
        }
    } else {
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            tmpSingleCoreK = self->ctx.singleShapeHWk_ * C04_COUT_SIZE;
        } else {
            uint32_t singleShapeCout1 = DivCeilC0<Intf>(self, self->ctx.singleShapeCout_);
            tmpSingleCoreK = singleShapeCout1 * self->ctx.singleShapeHWk_ * self->ctx.tiling_->c0;
        }
    }
    self->ctx.kIter_ = DivCeil(tmpSingleCoreK, self->ctx.tiling_->baseK);
    self->ctx.tailK_ = tmpSingleCoreK - (self->ctx.kIter_ - 1) * self->ctx.tiling_->baseK;
    self->ctx.kIterStepKaTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKa_) - 1) * self->ctx.curStepKa_;
    self->ctx.kIterStepKbTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKb_) - 1) * self->ctx.curStepKb_;
    self->ctx.stepKaTail = self->ctx.kIter_ - self->ctx.kIterStepKaTail;
    self->ctx.stepKbTail = self->ctx.kIter_ - self->ctx.kIterStepKbTail;
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.kIter_ > 0), "self->ctx.kIter_ is %d , which should be larger than 0", self->ctx.kIter_);
#endif
}

template <class Intf>
__aicore__ inline void InitStepNParams(Intf* self)
{
    self->ctx.nIter_ = DivCeil(self->ctx.singleShapeCin_, self->ctx.tiling_->baseN);
    self->ctx.tailN_ = self->ctx.singleShapeCin_ - (self->ctx.nIter_ - 1) * self->ctx.tiling_->baseN;
#ifdef __CCE_KT_TEST__
    ascendc_assert((self->ctx.nIter_ > 0), "self->ctx.nIter_ is %d , which should be larger than 0", self->ctx.nIter_);
#endif
}

template <class Intf>
static __aicore__ inline void InitParamsForSplitDkSplit(Intf* self)
{
    self->ctx.enableSplitDk_ = self->ctx.tiling_->singleIterateDk != self->ctx.tiling_->dk;
#ifdef __CCE_KT_TEST__
    ascendc_assert(
        (!self->ctx.enableSplitDk_ || (self->ctx.tiling_->iterateOrder == static_cast<int>(IterateOrder::ORDER_K))),
        "iterateOrder must be ORDER_K when splitDK.");
#endif
}

template <class Intf>
__aicore__ inline void InitFullLoadFlag(Intf* self)
{
    if constexpr (Intf::conv3dConfig.loadB1Condition != TPL_GM_TO_L1) {
        self->ctx.isB1FullLoadFlag_ = false;
        self->ctx.isA1FullLoadFlag_ = false;
        return;
    }
    if (unlikely(UseLocalWWindow<Intf>(self))) {
        self->ctx.isB1FullLoadFlag_ = false;
        self->ctx.isA1FullLoadFlag_ = false;
        return;
    }
    bool isDoutDkFullLoad = self->ctx.tiling_->dk == 1 || self->ctx.tiling_->dout == 1;
    bool baseB1Cond = self->ctx.tiling_->baseN >= self->ctx.tiling_->singleCoreCin && self->ctx.tiling_->enlarge == 1;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        bool needReloadA1 = self->ctx.tiling_->wk == 3 ||
                            ((self->ctx.tiling_->backpropPadLeft == 0 || self->ctx.tiling_->backpropPadLeft == 2) &&
                             (self->ctx.tiling_->wk == 4 || self->ctx.tiling_->wk == 2));
        self->ctx.isA1FullLoadFlag_ = self->ctx.kSCoutFullLoad_ && self->ctx.tiling_->al1Pbuffer == 1 &&
                                      isDoutDkFullLoad && !needReloadA1;
        self->ctx.isB1FullLoadFlag_ = self->ctx.kSCoutFullLoad_ && baseB1Cond && self->ctx.tiling_->bl1Pbuffer == 1 &&
                                      isDoutDkFullLoad;
    } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        self->ctx.isA1FullLoadFlag_ = false;
        self->ctx.isB1FullLoadFlag_ = baseB1Cond && isDoutDkFullLoad &&
                                      self->ctx.tiling_->baseK * self->ctx.curStepKb_ >=
                                          self->ctx.tiling_->singleCoreCout * self->ctx.splitHkWkList_[0];
    } else {
        self->ctx.isA1FullLoadFlag_ = false;
        self->ctx.isB1FullLoadFlag_ = baseB1Cond &&
                                      self->ctx.tiling_->baseK * self->ctx.curStepKb_ >=
                                          self->ctx.tiling_->singleCoreCout * self->ctx.singleShapeHWk_ &&
                                      self->ctx.tiling_->bl1Pbuffer == 1 && self->ctx.tiling_->singleIterateDk == 1;
    }
}

template <class Intf>
__aicore__ inline void InitC04Params(Intf* self)
{
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        if ASCEND_IS_AIV_SCALAR {
            self->ctx.c04LoadToB1IterIdx_ = 0;
            uint32_t c04UbBufSize = (AscendC::TOTAL_UB_SIZE - AscendC::VECTOR_REG_WIDTH - MASK_REG_WIDTH -
                                     AscendC::ONE_BLOCK_SIZE) >>
                                    1;
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))
            if constexpr (std::is_same<typename Intf::SrcT, bfloat16_t>::value ||
                          std::is_same<typename Intf::SrcT, half>::value) {
                c04UbBufSize = BIT16_GATHER_MAX_BUF_SIZE;
            }
#endif
            uint32_t c04UbPixNum = DivDtypeByte<typename Intf::SrcBT>(c04UbBufSize);
            uint32_t tmpVal1 = self->ctx.dkHkWk_ << C04_SHIFT_SIZE;
            uint32_t tmpVal2 = AlignUp((self->ctx.hkWk_ << C04_SHIFT_SIZE), self->ctx.tiling_->c0);
            uint32_t tmpVal3 = tmpVal1 > tmpVal2 ? tmpVal1 : tmpVal2;
            self->ctx.vecBlockN_ = ((c04UbPixNum / tmpVal3) >> 4)
                                   << 4; // >> 4 等价于除BLOCK_CUBE，<< 4 等价于乘BLOCK_CUBE
        }
    }
}

template <class Intf>
__aicore__ inline void InitParamsPart3(Intf* self)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202) || __DAV_35_FAMILY__
    self->ctx.dstB2Stride_ = 0;
    self->ctx.startAddrOffset_ = 0;
    self->ctx.headWi_ = 0;
    self->ctx.midHi_ = 0;
    self->ctx.tailWi_ = 0;
    self->ctx.curBackPropPadUp_ = 0;
    self->ctx.localWoStart_ = 0;
    self->ctx.localL1W_ = 0;
    self->ctx.localPadLeft_ = 0;
    self->ctx.localPadRight_ = 0;
#endif
    if (EnableVecGroupEnlarge(self)) {
        self->ctx.groupIterIdx_ = 0;
    }
    if (self->ctx.tiling_->enlarge > 1) {
        self->ctx.curEnlarge = 0;
    }
}
template <class Intf>
__aicore__ inline void InitParamsPart2(Intf* self)
{
    self->ctx.mIter_ = 0;
    self->ctx.nIter_ = 0;
    self->ctx.kIter_ = 0;
    self->ctx.tailM_ = 0;
    self->ctx.tailN_ = 0;
    self->ctx.tailK_ = 0;
    self->ctx.curMIdx_ = 0;
    self->ctx.curNIdx_ = 0;
    self->ctx.curLoadKbl1_ = 0;
    self->ctx.curDinIdx_ = 0;
    self->ctx.curDkIdx_ = 0;
    self->ctx.kIterStepKaTail = 0;
    self->ctx.kIterStepKbTail = 0;
    self->ctx.stepKaTail = 0;
    self->ctx.stepKbTail = 0;
    self->ctx.curHoIdx_ = 0;
    self->ctx.baseB1Offset_ = 0;
    self->ctx.blockBaseN_ = 0;
    self->ctx.baseUseM_ = 0;
    self->ctx.baseUseN_ = 0;
    self->ctx.baseUseK_ = 0;
    self->ctx.curEnlargeCin1_ = 0;
    self->ctx.curDinStartIdx_ = 0;
    self->ctx.curHoStartIdx_ = 0;
    self->ctx.curCinStartIdx_ = 0;
    self->ctx.curCoutStartIdx_ = 0;
    self->ctx.rearrangeHIndex_ = 0;
    self->ctx.rearrangeWIndex_ = 0;
    self->ctx.splitHIndex_ = 0;
    self->ctx.splitWIndex_ = 0;
    self->ctx.curMStartIdx_ = 0;
    self->ctx.curHkIdx_ = 0;
    self->ctx.curWkIdx_ = 0;
    self->ctx.curWoLeftIdx_ = 0;
    self->ctx.curWoRightIdx_ = 0;
    self->ctx.isLoadA1_ = true;
    self->ctx.isLoadB1_ = true;
    self->ctx.isFreeA1_ = false;
    self->ctx.isFreeB1_ = false;
    self->ctx.isLastKSegment_ = false;
    self->ctx.needComputeFlag_ = true;
    self->ctx.realMSize_ = 0;
    self->ctx.computeBiasOnce_ = false;
}

template <class Intf>
static __aicore__ inline void InitParamsForNormal(Intf* self)
{
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        self->ctx.channelSize_ = C04_COUT_SIZE;
    } else {
        uint32_t curChannelSize = DivHkWk<Intf>(self, self->ctx.curStepKa_ * self->ctx.tiling_->baseK);
        uint32_t alignedCout = AlignUpC0<Intf>(self, self->ctx.tiling_->singleCoreCout);
        self->ctx.channelSize_ = curChannelSize > alignedCout ? alignedCout : curChannelSize;
    }
    uint32_t mL1Size = self->ctx.tiling_->baseM;
    mL1Size = mL1Size > self->ctx.tiling_->singleCoreM ? self->ctx.tiling_->singleCoreM : mL1Size;
    self->ctx.curHoSize_ = CalFmapH<Intf>(self, mL1Size);
}

template <class Intf>
__aicore__ inline void InitSingleShapeParams(Intf* self)
{
    // 初始化为0，避免decache功能开启后全局变量不会初始化为0，出现随机值导致未知问题
    self->ctx.hkWk_ = static_cast<uint64_t>(self->ctx.tiling_->hk) * self->ctx.tiling_->wk;
    self->ctx.dkHkWk_ = static_cast<uint64_t>(self->ctx.tiling_->dk) * self->ctx.hkWk_;
    self->ctx.khDilation_ = (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH + 1;
    self->ctx.kwDilation_ = (self->ctx.tiling_->wk - 1) * self->ctx.tiling_->dilationW + 1;
    self->ctx.woExpand_ = (self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW + 1;
    self->ctx.cin1G_ = (self->ctx.tiling_->cinG + 15) >> 4;
    self->ctx.singleShapeDin_ = 0;
    self->ctx.singleShapeM_ = 0;
    self->ctx.singleShapeCin_ = 0;
    self->ctx.singleShapeCout_ = 0;
    self->ctx.groupNzUbBytes_ = 0;
    self->ctx.vselrScatterPadElems_ = 0;
    self->ctx.singleShapeHWk_ = self->ctx.hkWk_;
    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK) {
        self->ctx.singleShapeHWk_ = self->ctx.tiling_->wk;
    } else if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
        self->ctx.singleShapeHWk_ = 1;
    }
}

template <class Intf>
__aicore__ inline void InitParams(Intf* self)
{
    InitSingleShapeParams<Intf>(self);
    self->ctx.l0cPingPongFlag_ = 1;
    self->ctx.isFirstIter_ = true;
    self->ctx.curStepKa_ = self->ctx.tiling_->stepKa;
    self->ctx.curStepKb_ = self->ctx.tiling_->stepKb;
    InitParamsForSplitDkSplit<Intf>(self);
    self->ctx.enableSplitK_ = self->ctx.tiling_->enableSplitK;
    self->ctx.useUbAccumForSplitK_ = self->ctx.tiling_->useUbAccumForSplitK;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        InitParamsForKernelSplitHW<Intf>(self);
    } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        InitParamsForKernelSplitH<Intf>(self);
    } else {
        InitParamsForNormal<Intf>(self);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202) || __DAV_35_FAMILY__
    self->ctx.hoWo_ = static_cast<uint64_t>(self->ctx.tiling_->ho) * self->ctx.tiling_->wo;
    self->ctx.doHoWo_ = self->ctx.tiling_->dout * self->ctx.hoWo_;
    self->ctx.hiWi_ = static_cast<uint64_t>(self->ctx.tiling_->hi) * self->ctx.tiling_->wi;
    self->ctx.diHiWi_ = self->ctx.tiling_->di * self->ctx.hiWi_;
    self->ctx.realWoSize_ = self->ctx.tiling_->wo;
    InitFullLoadFlag<Intf>(self);
    InitC04Params<Intf>(self);
#endif
    self->ctx.hoExpand_ = (self->ctx.tiling_->ho - 1) * self->ctx.tiling_->strideH + 1;
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        self->ctx.hoExpand_ = self->ctx.tiling_->ho;
    }
    self->ctx.l0PingPongFlag_ = 0;
    self->ctx.l0aPingPongAddr_ = DivDtypeByte<typename Intf::SrcAT>(TOTAL_L0A_SIZE >> 1);
    self->ctx.l0bPingPongAddr_ = DivDtypeByte<typename Intf::SrcBT>(TOTAL_L0B_SIZE >> 1);
    self->ctx.enableL0PingPong_ = (self->ctx.tiling_->al0Pbuffer - 1) & (self->ctx.tiling_->bl0Pbuffer - 1);
    InitLoadToA2Params<Intf>(self);
    InitLoadToB2Params<Intf>(self);
    InitParamsPart2<Intf>(self);
    InitParamsPart3<Intf>(self);
}

} // namespace Convolution3DBackpropFunc

#endif
