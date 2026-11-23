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
 * \file conv3d_bp_func_arch35_buffer_tque.h
 * \brief Buffer and queue initialization helpers for conv3d backprop input
 */

#ifndef CONV3D_BP_FUNC_ARCH35_BUFFER_TQUE_H
#define CONV3D_BP_FUNC_ARCH35_BUFFER_TQUE_H

#include "conv3d_bp_common_func.h"
#include "conv3d_bp_large_attribute_func.h"
#include "conv3d_bp_kernel_split_func.h"
#include "../../../inc/macro.h"

namespace Convolution3DBackpropFunc {

template <class Intf>
__aicore__ inline void CalcMatrixByteSize(Intf* self, uint32_t& aMatrixByteSize, uint32_t& bMatrixByteSize)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        uint32_t hoSize = CalFmapHMaxForKernelSplit(self, self->ctx.tiling_->baseM);
        hoSize = self->ctx.tiling_->ho < hoSize ? self->ctx.tiling_->ho : hoSize;
        aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.channelSize_ * sizeof(typename Intf::SrcAT);
        if (self->ctx.kSCoutFullLoad_) {
            bMatrixByteSize = self->ctx.tiling_->baseN * self->ctx.channelSize_ * self->ctx.singleShapeHWk_ *
                              sizeof(typename Intf::SrcBT);
        } else {
            bMatrixByteSize = self->ctx.tiling_->baseN * self->ctx.curStepKb_ * self->ctx.tiling_->baseK *
                              sizeof(typename Intf::SrcBT);
        }
    } else if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        uint32_t hoSize = self->ctx.tiling_->ho < self->ctx.curHoSize_ ? self->ctx.tiling_->ho : self->ctx.curHoSize_;
        aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.tiling_->strideW * self->ctx.channelSize_ *
                          sizeof(typename Intf::SrcAT);
        bMatrixByteSize = self->ctx.tiling_->baseN * self->ctx.curStepKb_ * self->ctx.tiling_->baseK *
                          sizeof(typename Intf::SrcBT);
    } else {
        uint32_t hoSize = self->ctx.hoExpand_ < static_cast<uint64_t>(self->ctx.curHoSize_) ?
                              static_cast<uint32_t>(self->ctx.hoExpand_) :
                              self->ctx.curHoSize_;
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.tiling_->strideW * C04_COUT_SIZE *
                              sizeof(typename Intf::SrcAT);
        } else if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
            // 不加载L1不加载完整的hkwk时, wo无需搬一整行
            aMatrixByteSize = self->ctx.tiling_->baseM *
                              DivHkWk<Intf>(self, self->ctx.curStepKa_ * self->ctx.tiling_->baseK) *
                              sizeof(typename Intf::SrcAT);
        } else if (unlikely(UseLocalWWindow<Intf>(self))) {
            uint32_t localL1W = self->ctx.tiling_->baseM + self->ctx.kwDilation_ - 1;
            aMatrixByteSize = localL1W * DivHkWk<Intf>(self, self->ctx.curStepKa_ * self->ctx.tiling_->baseK) *
                              sizeof(typename Intf::SrcAT);
        } else {
            aMatrixByteSize = hoSize * self->ctx.tiling_->wo * self->ctx.tiling_->strideW *
                              DivHkWk<Intf>(self, self->ctx.curStepKa_ * self->ctx.tiling_->baseK) *
                              sizeof(typename Intf::SrcAT);
        }
        bMatrixByteSize = self->ctx.tiling_->baseN * self->ctx.curStepKb_ * self->ctx.tiling_->baseK *
                          sizeof(typename Intf::SrcBT);
    }
}

template <class Intf>
__aicore__ inline uint32_t CalcBiasL1Size(Intf* self)
{
    // 64: BT Buffer 需要64B对齐，加63后右移6位代替乘64
    uint32_t biasSize = self->ctx.tiling_->isBiasFullLoad ?
                            (DivCeil(self->ctx.tiling_->singleCoreCin * sizeof(typename Intf::BiasT), 64) << 6) :
                            (DivCeil(self->ctx.tiling_->baseN * sizeof(typename Intf::BiasT), 64) << 6);
    return AlignUp(biasSize, ONE_BLK_SIZE);
}

template <class Intf>
__aicore__ inline uint32_t CalcScaleBufSize(Intf* self)
{
    uint32_t scaleSize = DivCeil(self->ctx.tiling_->singleCoreCin * sizeof(typename Intf::ScaleT0), ONE_BLK_SIZE) *
                         ONE_BLK_SIZE;
    return AlignUp(scaleSize, ONE_BLK_SIZE);
}

template <class Intf>
__aicore__ inline void InitBiasTque(Intf* self)
{
    uint32_t biasSize = CalcBiasL1Size<Intf>(self);
    self->ctx.pipe_.InitBuffer(self->ctx.biasL1Que_, 1, biasSize);
    self->ctx.pipe_.InitBuffer(self->ctx.biasBTQue_, 1, biasSize);
}

template <class Intf>
__aicore__ inline uint32_t CalcScaleL1Size(Intf* self)
{
    uint32_t totalScaleSize = 0;
    uint32_t scaleSize = CalcScaleBufSize<Intf>(self);
    if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            totalScaleSize += scaleSize;
        }
    }
#ifdef DTYPE_Y1
    using Intf1 = Convolution3DBackprop::Output1Intf<Intf>;
    if constexpr (Intf1::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (self->ctx.hasSecondOutput_ &&
            self->ctx.tiling_->quantMode1 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            totalScaleSize += scaleSize;
        }
    }
#endif
    return totalScaleSize;
}

template <class Intf>
__aicore__ inline void InitScaleTque(Intf* self)
{
    uint32_t scaleSize = CalcScaleBufSize<Intf>(self);
    if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            self->ctx.pipe_.InitBuffer(self->ctx.scale0L1Que_, 1, scaleSize);
        }
    }
#ifdef DTYPE_Y1
    using Intf1 = Convolution3DBackprop::Output1Intf<Intf>;
    if constexpr (Intf1::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (self->ctx.hasSecondOutput_) {
            if (self->ctx.tiling_->quantMode1 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                self->ctx.pipe_.InitBuffer(self->ctx.scale1L1Que_, 1, scaleSize);
            }
        }
    }
#endif
}

template <class Intf>
__aicore__ inline void InitScaleL1Tque(Intf* self, uint32_t scaleOffset)
{
    uint32_t scaleSize = CalcScaleBufSize<Intf>(self);
    if constexpr (Intf::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (self->ctx.tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            self->ctx.pipe_.InitBuffer(self->ctx.scale0L1Que_, Std::make_tuple(scaleOffset, scaleSize));
            scaleOffset += scaleSize;
        }
    }
#ifdef DTYPE_Y1
    using Intf1 = Convolution3DBackprop::Output1Intf<Intf>;
    if constexpr (Intf1::Config::fType::format != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (self->ctx.hasSecondOutput_ &&
            self->ctx.tiling_->quantMode1 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            self->ctx.pipe_.InitBuffer(self->ctx.scale1L1Que_, Std::make_tuple(scaleOffset, scaleSize));
        }
    }
#endif
}

template <class Intf>
__aicore__ inline void InitL0TBuf(Intf* self)
{
    if (self->ctx.tiling_->cl0Pbuffer > 1) {
        uint32_t l0cSize = TOTAL_L0C_SIZE >> 1;
        self->ctx.pipe_.InitBuffer(self->ctx.l0cPing_, 1, l0cSize);
        self->ctx.pipe_.InitBuffer(self->ctx.l0cPong_, 1, l0cSize);
    } else {
        self->ctx.pipe_.InitBuffer(self->ctx.l0cPing_, 1, TOTAL_L0C_SIZE);
    }
    self->ctx.pipe_.InitBuffer(self->ctx.l0aBuf_, TOTAL_L0A_SIZE);
    self->ctx.pipe_.InitBuffer(self->ctx.l0bBuf_, TOTAL_L0B_SIZE);
}

// UB→L1B 场景须严格保持原始 InitBuffer 调用顺序（b1UbPing/Pong 先于 AIC 侧分配），
// 否则 AIV/AIC 两侧 B1 地址错位引发精度问题
template <class Intf>
__aicore__ inline void InitTqueForUbToL1B(Intf* self, const bool hasBias, uint32_t aMatrixByteSize,
                                          uint32_t bMatrixByteSize)
{
    self->ctx.pipe_.InitBuffer(self->ctx.b1UbPing_, bMatrixByteSize);
    if (self->ctx.tiling_->bl1Pbuffer > 1) {
        self->ctx.pipe_.InitBuffer(self->ctx.b1UbPong_, bMatrixByteSize);
    }
    if ASCEND_IS_AIC_SCALAR {
        self->ctx.pipe_.InitBuffer(self->ctx.inQueL1A_, self->ctx.tiling_->al1Pbuffer, aMatrixByteSize);
        InitL0TBuf<Intf>(self);
        if (unlikely(hasBias)) {
            InitBiasTque(self);
        }
        InitScaleTque(self);
    }
}

// 普通场景（GM→L1B 直载）bank 分离布局，偏移 ONE_BLK_SIZE 对齐：
//   bank0: A1-ping | B1-ping | bias | scale0/scale1；bank1: A1-pong | B1-pong
//   bias/scale 溢出 bank0 时 pong 区顺延
template <class Intf>
__aicore__ inline void InitTqueBankSeparated(Intf* self, const bool hasBias, uint32_t aMatrixByteSize,
                                             uint32_t bMatrixByteSize)
{
    bool a1DoubleBuf = self->ctx.tiling_->al1Pbuffer > 1;
    bool b1DoubleBuf = self->ctx.tiling_->bl1Pbuffer > 1;
    constexpr uint32_t halfL1Size = TOTAL_L1_SIZE / 2;
    uint32_t aAligned = AlignUp(aMatrixByteSize, ONE_BLK_SIZE);
    uint32_t bAligned = AlignUp(bMatrixByteSize, ONE_BLK_SIZE);
    uint32_t biasSize = hasBias ? CalcBiasL1Size<Intf>(self) : 0;
    uint32_t scaleSize = CalcScaleL1Size<Intf>(self);

    uint32_t l1A1PingOffset = 0;
    uint32_t l1B1PingOffset = aAligned;
    uint32_t l1BiasOffset = l1B1PingOffset + bAligned;
    uint32_t l1ScaleOffset = l1BiasOffset + biasSize;
    uint32_t l1PongOffset = halfL1Size;
    if (l1ScaleOffset + scaleSize > l1PongOffset) {
        l1PongOffset = l1ScaleOffset + scaleSize;
    }
#ifdef __CCE_KT_TEST__
    if (a1DoubleBuf || b1DoubleBuf) {
        uint32_t pongEnd = l1PongOffset + (a1DoubleBuf ? aAligned : 0) + (b1DoubleBuf ? bAligned : 0);
        ascendc_assert((pongEnd <= TOTAL_L1_SIZE), "bank-separated l1 layout exceeds limit");
    } else {
        ascendc_assert((l1ScaleOffset + scaleSize <= TOTAL_L1_SIZE), "bank-separated l1 layout exceeds limit");
    }
#endif

    if ASCEND_IS_AIC_SCALAR {
        if (unlikely(hasBias)) {
            self->ctx.pipe_.InitBuffer(self->ctx.biasL1Que_, Std::make_tuple(l1BiasOffset, biasSize));
            self->ctx.pipe_.InitBuffer(self->ctx.biasBTQue_, 1, biasSize);
        }
        InitScaleL1Tque<Intf>(self, l1ScaleOffset);
        if (a1DoubleBuf) {
            self->ctx.pipe_.InitBuffer(self->ctx.inQueL1A_, Std::make_tuple(l1A1PingOffset, aMatrixByteSize),
                                       Std::make_tuple(l1PongOffset, aMatrixByteSize));
        } else {
            self->ctx.pipe_.InitBuffer(self->ctx.inQueL1A_, Std::make_tuple(l1A1PingOffset, aMatrixByteSize));
        }
        if (b1DoubleBuf) {
            self->ctx.pipe_.InitBuffer(self->ctx.inQueL1B_, Std::make_tuple(l1B1PingOffset, bMatrixByteSize),
                                       Std::make_tuple(l1PongOffset + (a1DoubleBuf ? aAligned : 0), bMatrixByteSize));
        } else {
            self->ctx.pipe_.InitBuffer(self->ctx.inQueL1B_, Std::make_tuple(l1B1PingOffset, bMatrixByteSize));
        }
        InitL0TBuf<Intf>(self);
    }
}

template <class Intf>
__aicore__ inline void InitTque(Intf* self, const bool hasBias)
{
    uint32_t bMatrixByteSize = 0;
    uint32_t aMatrixByteSize = 0;
    CalcMatrixByteSize(self, aMatrixByteSize, bMatrixByteSize);
#ifdef __CCE_KT_TEST__
    uint32_t usedBufferSize = aMatrixByteSize * self->ctx.tiling_->al1Pbuffer +
                              bMatrixByteSize * self->ctx.tiling_->bl1Pbuffer;
    ascendc_assert((usedBufferSize <= TOTAL_L1_SIZE), "l1 size exceeds limit");
#endif

    // 路由须与 B1 搬运路径一致：fractal_z 直搬（loadB1FractalZ==1）走 inQueL1B_，不能进 b1UbPing 路径
    if (EnableVecGroupEnlarge(self) || Intf::conv3dConfig.enableC04Flag) {
        InitTqueForUbToL1B<Intf>(self, hasBias, aMatrixByteSize, bMatrixByteSize);
    } else {
        InitTqueBankSeparated<Intf>(self, hasBias, aMatrixByteSize, bMatrixByteSize);
    }

    InitUbByteSize(self);
}

} // namespace Convolution3DBackpropFunc

#endif
