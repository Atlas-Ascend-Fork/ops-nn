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
 * \file conv3d_bp_func_arch35_compute.h
 * \brief Matrix compute and iteration helpers for conv3d backprop input
 */

#ifndef CONV3D_BP_FUNC_ARCH35_COMPUTE_H
#define CONV3D_BP_FUNC_ARCH35_COMPUTE_H

#include "conv3d_bp_common_func.h"
#include "conv3d_bp_large_attribute_func.h"
#include "conv3d_bp_kernel_split_func.h"
#include "../../../inc/macro.h"

namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline void FreeFullLoadL1Tensor(Intf* self)
{
    if ASCEND_IS_AIC_SCALAR {
        // 全载场景，如果Tensor已经被载入了，进行释放
        if (self->ctx.isB1FullLoadFlag_ && self->ctx.isFreeB1_ && !self->ctx.isLoadB1_) {
            self->ctx.inQueL1B_.FreeTensor(self->ctx.cacheB1Buf_);
            self->ctx.isLoadB1_ = true;
        }

        if (self->ctx.isA1FullLoadFlag_ && self->ctx.isFreeA1_ && !self->ctx.isLoadA1_) {
            self->ctx.inQueL1A_.FreeTensor(self->ctx.cacheA1Buf_);
            self->ctx.isLoadA1_ = true;
        }
    }
}

template <class Intf, bool hasBias>
static __aicore__ inline void ComputeForNoTilingHWk(Intf* self, LocalTensor<typename Intf::SrcAT>& l0a,
                                                    LocalTensor<typename Intf::SrcBT>& l0b,
                                                    LocalTensor<typename Intf::L0cT>& l0c, uint8_t& l0PingPongFlag)
{
    bool isFirstDk = (self->ctx.curDkIdx_ == 0);
    const uint32_t strideD = self->ctx.tiling_->strideD;
    const int64_t strideDMulDout = self->ctx.tiling_->dout * strideD;
    const bool strideDgtOne = strideD > 1;
    for (uint64_t curInnerKdIdx = self->ctx.curDkIdx_;
         curInnerKdIdx < self->ctx.curDkIdx_ + self->ctx.tiling_->singleIterateDk; curInnerKdIdx++) {
        if (EnableVecGroupEnlarge(self)) {
            self->ctx.groupIterIdx_ = 0;
        }

        int64_t dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront -
                       curInnerKdIdx * self->ctx.tiling_->dilationD;
        if (dTmp < 0 || (strideDgtOne && (dTmp % strideD > 0)) || dTmp >= strideDMulDout) {
            continue;
        }
        int64_t curDoutIdx = strideDgtOne ? (dTmp / strideD) : dTmp;
        if (self->ctx.kSCoutFullLoad_) {
            ComputeForKIter<Intf, hasBias, true>(self, l0a, l0b, l0c, curInnerKdIdx, curDoutIdx, isFirstDk,
                                                 l0PingPongFlag);
        } else {
            ComputeForKIter<Intf, hasBias, false>(self, l0a, l0b, l0c, curInnerKdIdx, curDoutIdx, isFirstDk,
                                                  l0PingPongFlag);
        }
        isFirstDk = false;
    }
    if (hasBias) {
        if (!self->ctx.computeBiasOnce_) {
            ComputeForBias<Intf, hasBias>(self, l0a, l0b, l0c, isFirstDk, l0PingPongFlag);
        }
    }
}

template <class Intf>
static __aicore__ inline void ComputeStart(Intf* self, LocalTensor<typename Intf::L0cT>& l0c)
{
    if ASCEND_IS_AIC_SCALAR {
        InitMmadParams<Intf>(self);
        UpdateLoadToA2ParamsM<Intf>(self);
        UpdateLoadToB2ParamsN<Intf>(self);
        if (self->ctx.l0cPingPongFlag_) {
            l0c = self->ctx.l0cPing_.template AllocTensor<typename Intf::L0cT>();
        } else {
            l0c = self->ctx.l0cPong_.template AllocTensor<typename Intf::L0cT>();
        }
    }

    CrossCoreSetHead<Intf>(self);
}

template <class Intf>
static __aicore__ inline void ComputeEnd(Intf* self, LocalTensor<typename Intf::L0cT>& l0c, uint8_t l0PingPongFlag)
{
    if ASCEND_IS_AIC_SCALAR {
        if (self->ctx.l0cPingPongFlag_) {
            self->ctx.l0cPing_.EnQue(l0c);
        } else {
            self->ctx.l0cPong_.EnQue(l0c);
        }
    }

    self->ctx.l0PingPongFlag_ = l0PingPongFlag;
    CrossCoreWaitTail<Intf>(self);
}

template <class Intf, bool hasBias>
static __aicore__ inline void NoNeedComputeHandle(Intf* self)
{
    if (!hasBias) {
        FreeFullLoadL1Tensor<Intf>(self);
        return;
    }

    LocalTensor<typename Intf::SrcAT> l0a;
    LocalTensor<typename Intf::SrcBT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    ComputeStart<Intf>(self, l0c);
    uint8_t l0PingPongFlag = self->ctx.l0PingPongFlag_;
    // 有bias场景 加bias再退出
    ComputeForBias<Intf, hasBias>(self, l0a, l0b, l0c, 1, l0PingPongFlag);
    self->ctx.needComputeFlag_ = true;
    FreeFullLoadL1Tensor<Intf>(self);
    ComputeEnd<Intf>(self, l0c, l0PingPongFlag);
}

template <class Intf, bool hasBias>
static __aicore__ inline void Compute(Intf* self)
{
    if (!self->ctx.needComputeFlag_) {
        NoNeedComputeHandle<Intf, hasBias>(self);
        return;
    }

    LocalTensor<typename Intf::SrcAT> l0a;
    LocalTensor<typename Intf::SrcBT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    ComputeStart<Intf>(self, l0c);
    uint8_t l0PingPongFlag = self->ctx.l0PingPongFlag_;
    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1) {
        ComputeForNoTilingHWk<Intf, hasBias>(self, l0a, l0b, l0c, l0PingPongFlag);
    } else {
        ComputeForTilingHkWk<Intf, hasBias>(self, l0a, l0b, l0c, l0PingPongFlag);
    }
    ComputeEnd<Intf>(self, l0c, l0PingPongFlag);
}

template <class Intf>
static __aicore__ inline void UpdateLocalWLoadInfo(Intf* self, uint64_t globalMStart)
{
    const int64_t strideW = self->ctx.tiling_->strideW;
    const int64_t wo = self->ctx.tiling_->wo;
    const int64_t expandedW = (wo - 1) * strideW + 1;
    const int64_t needStart = static_cast<int64_t>(globalMStart) - self->ctx.tiling_->backpropPadLeft;
    const int64_t needEnd = static_cast<int64_t>(globalMStart) + self->ctx.baseUseM_ -
                            self->ctx.tiling_->backpropPadLeft + self->ctx.kwDilation_ - 1;
    const int64_t clippedStart = needStart > 0 ? needStart : 0;
    const int64_t clippedEnd = needEnd < expandedW ? needEnd : expandedW;
    int64_t firstWo = (clippedStart + strideW - 1) / strideW;
    int64_t lastWo = (clippedEnd + strideW - 1) / strideW;
    firstWo = firstWo < wo ? firstWo : wo - 1;
    lastWo = lastWo <= wo ? lastWo : wo;
    lastWo = lastWo > firstWo ? lastWo : firstWo + 1;

    const int64_t realWo = lastWo - firstWo;
    const int64_t expandedOrigin = firstWo * strideW;
    const int64_t localL1W = (realWo - 1) * strideW + 1;
    const int64_t localPadLeft = expandedOrigin - needStart;
    const int64_t localPadRight = needEnd - expandedOrigin - localL1W;
    self->ctx.localWoStart_ = static_cast<uint32_t>(firstWo);
    self->ctx.realWoSize_ = static_cast<uint32_t>(realWo);
    self->ctx.localL1W_ = static_cast<uint32_t>(localL1W);
    self->ctx.localPadLeft_ = static_cast<uint32_t>(localPadLeft > 0 ? localPadLeft : 0);
    self->ctx.localPadRight_ = static_cast<uint32_t>(localPadRight > 0 ? localPadRight : 0);
    self->ctx.load3d_.mStartPt = 0;
}

template <class Intf>
static __aicore__ inline void UpdateCurHoSize(Intf* self)
{
    // 要从当前核hoStartIdx开始算，普通场景可以跨行，要从GM起点开始计算
    uint32_t curWi = self->ctx.tiling_->wi;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        curWi = self->ctx.splitWi_;
    }

    uint64_t curMIdx = self->ctx.curMStartIdx_ + self->ctx.curMIdx_ * self->ctx.tiling_->baseM;
    uint32_t curHiIdx;
    uint32_t curWiIdx;
    if (curMIdx == 0) {
        curHiIdx = 0;
        curWiIdx = 0;
    } else {
        curHiIdx = curMIdx / curWi;
        curWiIdx = curMIdx - curHiIdx * curWi;
    }
    self->ctx.load3d_.mStartPt = curWiIdx;
    uint32_t hiCal = DivCeil(self->ctx.baseUseM_ + curWiIdx, curWi);
    uint32_t endHoIdx = 0;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
        self->ctx.curHoIdx_ = static_cast<int32_t>(curHiIdx) - self->ctx.tiling_->backpropPadUp;
        endHoIdx = self->ctx.curHoIdx_ + (hiCal + (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH);
    } else {
        self->ctx.curHoIdx_ = static_cast<int32_t>(curHiIdx) - self->ctx.curBackPropPadUp_;
        endHoIdx = self->ctx.curHoIdx_ +
                   (hiCal + (self->ctx.splitHkList_[self->ctx.splitIndex_] - 1) * self->ctx.tiling_->dilationH);
    }
    UpdateCurHoSizeCore<Intf>(self, endHoIdx);
    if (unlikely(UseLocalWWindow<Intf>(self))) {
        UpdateLocalWLoadInfo<Intf>(self, curMIdx);
    }
}

template <class Intf>
static __aicore__ inline void UpdateKComputeStatus(Intf* self)
{
    if (!self->ctx.needComputeFlag_) {
        // 其他方向要跳过计算，K可以不判断了
        return;
    }
    const uint32_t strideD = self->ctx.tiling_->strideD;
    const int64_t strideDThreshold = self->ctx.tiling_->dout * strideD;
    const bool strideDgtOne = strideD > 1;
    const int64_t DkThreshold = self->ctx.curDkIdx_ + self->ctx.tiling_->singleIterateDk;
    bool isKNeedCompute = false;
    for (uint64_t curKdIdx = self->ctx.curDkIdx_; curKdIdx < DkThreshold; curKdIdx++) {
        // 由于膨胀卷积使dk的位置发生改变，求解dout_idx时，dk_idx需乘上膨胀系数再参与计算，才能求取正确的索引
        int64_t dTmp = self->ctx.curDinIdx_ + self->ctx.tiling_->padFront - curKdIdx * self->ctx.tiling_->dilationD;
        if (dTmp < 0 || (strideDgtOne && (dTmp % strideD > 0)) || dTmp >= strideDThreshold) {
            continue;
        }
        isKNeedCompute = true;
        break;
    }

    self->ctx.needComputeFlag_ = isKNeedCompute;
    if constexpr (Intf::conv3dConfig.loadB1Condition != TPL_GM_TO_L1) {
        UpdateHkComputeStatus<Intf>(self);
    }
}

template <class Intf>
static __aicore__ inline bool CheckFreeA1ForKernelSplit(Intf* self)
{
    // 负 W pad 时各子 kernel 的裁剪区间不同，切换后需重新加载 A1。
    if (self->ctx.tiling_->backpropPadLeft < 0 || self->ctx.tiling_->backpropPadRight < 0) {
        return true;
    }
    // 反向 pad 为 0 的情况。
    if (self->ctx.tiling_->backpropPadUp == 0) {
        for (int splitIndex = 0; splitIndex < self->ctx.tiling_->strideW * self->ctx.tiling_->strideH; splitIndex++) {
            if (self->ctx.subPadUpList_[splitIndex] < 0) {
                return true;
            }
        }
    }
    if (self->ctx.tiling_->backpropPadLeft == 0) {
        for (int splitIndex = 0; splitIndex < self->ctx.tiling_->strideW * self->ctx.tiling_->strideH; splitIndex++) {
            if (self->ctx.subPadLeftList_[splitIndex] < 0) {
                return true;
            }
        }
    }

    return false;
}

template <class Intf>
static __aicore__ inline void UpdateFullLoadL1Status(Intf* self)
{
    if (!self->ctx.isB1FullLoadFlag_ && !self->ctx.isA1FullLoadFlag_) {
        return;
    }
    bool isLastMNIter = self->ctx.curMIdx_ == self->ctx.mIter_ - 1 && self->ctx.curNIdx_ == self->ctx.nIter_ - 1;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        bool isLastRearrangeHWIter = self->ctx.rearrangeWIndex_ == self->ctx.tiling_->strideW - 1 &&
                                     self->ctx.rearrangeHIndex_ == self->ctx.tiling_->strideH - 1;
        self->ctx.isFreeA1_ = isLastRearrangeHWIter || CheckFreeA1ForKernelSplit(self);
        self->ctx.isFreeB1_ = isLastMNIter && self->ctx.tiling_->dk > 1 && isLastRearrangeHWIter;
    } else {
        bool isLastDIter = (self->ctx.tiling_->dk == 1 &&
                            self->ctx.curDinIdx_ == self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_ - 1) ||
                           self->ctx.tiling_->dk > 1;
        bool suppressB1Free = self->ctx.isB1FullLoadFlag_ && self->ctx.tiling_->dk == 1 &&
                              self->ctx.curEnableFullLoad_ && !Intf::conv3dConfig.enableC04Flag &&
                              self->ctx.tiling_->group == 1;
        self->ctx.isFreeB1_ = isLastMNIter && isLastDIter && !suppressB1Free;
    }
}

template <class Intf>
static __aicore__ inline void UpdateL1ComputeInfo(Intf* self)
{
    self->ctx.baseUseM_ = (self->ctx.curMIdx_ + 1 == self->ctx.mIter_) ? self->ctx.tailM_ : self->ctx.tiling_->baseM;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_NO_SPLIT_KERNEL) {
        uint32_t Hd = (self->ctx.tiling_->hk - 1) * self->ctx.tiling_->dilationH + 1;
        uint32_t Wd = (self->ctx.tiling_->wk - 1) * self->ctx.tiling_->dilationW + 1;
        // load3d指令的padList[3]固定为255
        uint32_t backpropPadDown = self->ctx.tiling_->backpropPadDown > 255 ? self->ctx.tiling_->backpropPadDown : 255;
        uint32_t Hp = self->ctx.hoExpand_ + self->ctx.tiling_->backpropPadUp + backpropPadDown;
        uint32_t Wp = (self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW + 1 +
                      self->ctx.tiling_->backpropPadLeft + self->ctx.tiling_->backpropPadRight;
        uint64_t M = (Hp - Hd + 1) * (Wp - Wd + 1);
        if (unlikely(M < self->ctx.baseUseM_) && self->ctx.mIter_ == 1) {
            self->ctx.baseUseM_ = M;
        }
    }
    self->ctx.baseUseN_ = (self->ctx.curNIdx_ + 1 == self->ctx.nIter_) ? self->ctx.tailN_ : self->ctx.tiling_->baseN;
    self->ctx.needComputeFlag_ = true;
    UpdateCurHoSize<Intf>(self);
    UpdateKComputeStatus<Intf>(self);
    UpdateFullLoadL1Status<Intf>(self);
}

template <class Intf>
static __aicore__ inline void UpdateSplitKTail(Intf* self, uint32_t kIdx)
{
    if (kIdx + self->ctx.tiling_->kSegment < self->ctx.tiling_->coutG) {
        return;
    }
    self->ctx.isLastKSegment_ = true;
    self->ctx.singleShapeCout_ = self->ctx.tiling_->kSegmentTail;
    uint64_t tmpSingleCoreK = 0;
    uint32_t singleShapeCout1 = DivCeilC0<Intf>(self, self->ctx.singleShapeCout_);
    tmpSingleCoreK = singleShapeCout1 * self->ctx.singleShapeHWk_ * self->ctx.tiling_->c0;

    self->ctx.kIter_ = DivCeil(tmpSingleCoreK, self->ctx.tiling_->baseK);
    self->ctx.tailK_ = tmpSingleCoreK - (self->ctx.kIter_ - 1) * self->ctx.tiling_->baseK;
    self->ctx.kIterStepKaTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKa_) - 1) * self->ctx.curStepKa_;
    self->ctx.kIterStepKbTail = (DivCeil(self->ctx.kIter_, self->ctx.curStepKb_) - 1) * self->ctx.curStepKb_;

    self->ctx.stepKaTail = self->ctx.kIter_ - self->ctx.kIterStepKaTail;
    self->ctx.stepKbTail = self->ctx.kIter_ - self->ctx.kIterStepKbTail;
}

template <class Intf>
static __aicore__ inline void CalcInWorkspace(Intf* self, const GlobalTensor<typename Intf::DstT>& output)
{
    // 无需在workspace上累加的场景不需要搬出
    if (!self->ctx.useUbAccumForSplitK_) {
        return;
    }
    // 无需计算场景不需要搬出
    if (!self->ctx.needComputeFlag_) {
        return;
    }
    if ASCEND_IS_AIV_SCALAR {
        CvCrossCoreWait<Intf, PIPE_FIX, PIPE_MTE2>(self, FLAG_MTE2_VEC_ID);
        AccumulateSegmentOnWorkspace<Intf>(self, output);
    }
}

template <class Intf>
static __aicore__ inline void SetDequantScale(Intf* self)
{
    bool hasVectorScale = false;
    if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        hasVectorScale = self->ctx.tiling_->quantMode0 ==
                         static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT);
    }
#ifdef DTYPE_Y1
    using Intf1 = Convolution3DBackprop::Output1Intf<Intf>;
    if constexpr (Intf1::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        hasVectorScale = hasVectorScale || (self->ctx.hasSecondOutput_ &&
                                            self->ctx.tiling_->quantMode1 ==
                                                static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT));
    }
#endif
    if (hasVectorScale) {
        event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventId);
        WaitFlag<HardEvent::FIX_MTE2>(eventId);
        Convolution3DBackpropFunc::FullLoadToScaleL1<Intf>(self);
    }
}

template <class Intf, bool sync>
static __aicore__ inline void CalcSplitK_(Intf* self, uint8_t& enAtomic,
                                          const GlobalTensor<typename Intf::DstT>& output, bool hasBias)
{
    // calRound多轮场景重置标志位
    self->ctx.isLastKSegment_ = false;
    // bias 需计算singlecoreM部分
    int32_t computeBiasTimes = (self->ctx.tiling_->singleCoreM - 1) / self->ctx.tiling_->baseM;

    if (unlikely(hasBias)) {
        self->ctx.computeBiasOnce_ = false;
    }
    for (uint32_t kIdx = self->ctx.curCoutStartIdx_; kIdx < self->ctx.tiling_->coutG;
         kIdx += self->ctx.tiling_->kSegment) {
        self->ctx.curCoutStartIdx_ = kIdx;
        UpdateSplitKTail<Intf>(self, kIdx);
        // fp32: 使用atomic累加到GM; 非fp32: 使用atomic累加到workspace
        enAtomic = (kIdx == 0) ? 0 : 1;
        uint32_t iterCnt = 0;
        while (self->template Iterate<sync>(false, hasBias)) {
            self->ctx.realMSize_ = (iterCnt++ != 0) ? self->ctx.tiling_->singleCoreM : self->ctx.baseUseM_;
            self->template GetTensorC<sync>(output, enAtomic);
            if (unlikely(hasBias && computeBiasTimes > 0)) {
                self->ctx.computeBiasOnce_ = false;
                computeBiasTimes -= 1;
            }
        }
        self->template VecPostProcess<sync>(output, enAtomic);
        self->ctx.isFirstIter_ = true;
    }
    CalcInWorkspace<Intf>(self, output);
}

} // namespace Convolution3DBackpropFunc

#endif
