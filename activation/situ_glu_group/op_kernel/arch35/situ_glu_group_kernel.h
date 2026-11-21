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
 * \file situ_glu_group_kernel.h
 * \brief SituGluGroup Arch35 (Ascend 950/350) kernel (VF activation + VF group-index reduce).
 *
 * y = beta * tanh(gate / beta) * sigmoid(gate) * up'  (up' = linear_beta * tanh(up / linear_beta)
 * when linear_beta > 0, else up), optionally scaled per row by weight and truncated to the first
 * realBs rows, where realBs = min(total, bs) and total is sum(group_index) in count mode or
 * group_index[-1] in cumsum mode (a zero total writes no rows).
 */

#ifndef SITU_GLU_GROUP_KERNEL_H
#define SITU_GLU_GROUP_KERNEL_H

#include "kernel_operator.h"
#include "situ_glu_group_tiling_data.h"
#include "situ_glu_group_tiling_key.h"

namespace NsSituGluGroup {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint16_t FOUR_UNFOLD = 4;
constexpr uint32_t TBUF_POOL_DEPTH = 16;
// Smallest column block (in bytes) handed to one core after the splitD axis is cut. A block
// shorter than one cacheline would turn every row into a tiny MTE2 burst, so keep the column
// granularity at 512B (Ascend 950/350 cacheline) or larger.
constexpr int64_t MIN_COL_BYTES = 512;
constexpr int64_t KERNEL_INT64_MAX = 9223372036854775807LL;
// Upper bound for one DataCopyPad descriptor. A fused row tile can be tens of KB wide, and a single
// over-long burst is rejected by the MTE2 front end (aicore error), so contiguous tiles are emitted
// in <= 32KB chunks.
constexpr int64_t MAX_COPY_BURST_BYTES = 32 * 1024;

// Element offset of the auxiliary negative-value probe. It is placed one 32B block after the sum
// so the second StoreAlign keeps the 32B alignment requirement for every supported element type.
template <typename T>
__aicore__ inline constexpr uint32_t GroupIndexMinSlot()
{
    return static_cast<uint32_t>(BLOCK_SIZE) / static_cast<uint32_t>(sizeof(T));
}
// Dispatcher mask bits: bit0 = weight present, bit1 = linear_beta active.
constexpr int32_t MASK_HAS_WEIGHT = 0b01;
constexpr int32_t MASK_HAS_LINEAR_BETA = 0b10;
// How a tile is moved into UB:
//   SPLIT  : gate and up copied separately with one burst per row (the default; also the only
//            option when column splitting hands this core a slice of splitD, and for splitD == 1).
//   NARROW : fp32 splitD == 2; the tile is copied as one contiguous burst and Reg::Gather rebuilds
//            gate/up. (The DIST_DINTLV variant for splitD == 1 was removed: its 16-bit unpack and
//            per-row weight layouts were incorrect on device.)
constexpr int32_t COPY_MODE_SPLIT = 0;
constexpr int32_t COPY_MODE_NARROW = 1;
// 16-bit splitD == 1: each row is one 32-bit word holding [gate(低16位), up(高16位)].
constexpr int32_t COPY_MODE_PAIR16 = 2;
// 16-bit splitD >= 2: convert the whole row tile to float in UB, then reuse the fp32 packed gather.
constexpr int32_t COPY_MODE_NARROW_F32 = 3;
// The pair16 load reads a full 64-word vector, so the last block over-reads up to 63 words; require
// enough row words in the (double-buffered) tile buffer to stay inside it.
constexpr int64_t PAIR16_MIN_ROW_FACTOR = 5;

__aicore__ inline int32_t CeilDiv(int32_t a, int b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

// int64 overload: row counts / d extents are int64_t on the host side; using the int32 version
// silently truncates very large shapes.
__aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Gcd(int64_t a, int64_t b)
{
    if (a < 0) {
        a = -a;
    }
    if (b < 0) {
        b = -b;
    }
    while (b != 0) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a == 0 ? 1 : a;
}

template <typename T>
__aicore__ inline int64_t RoundUp(int64_t num)
{
    int64_t elemNum = BLOCK_SIZE / static_cast<int64_t>(sizeof(T));
    return (num + elemNum - 1) / elemNum * elemNum;
}

// Up-align an element count to `align`. Implemented with a remainder so that num + align - 1 can
// never overflow the signed int64 range; the guard keeps the addition safe for absurd inputs.
__aicore__ inline int64_t AlignUpElems(int64_t num, int64_t align)
{
    if (align <= 1 || num <= 0) {
        return num;
    }
    const int64_t remainder = num % align;
    if (remainder == 0) {
        return num;
    }
    if (num > KERNEL_INT64_MAX - align) {
        return num;
    }
    return num - remainder + align;
}

// ---------------------------------------------------------------------------
// VF (RegBase) activation path, modeled on swiglu_group (arch35).
// The whole row stays in vector registers: half/bf16 are unpacked/packed in
// registers, weight is broadcast with DIST_BRC_B32, so there is no per-row
// scalar GetValue and no fp32 UB scratch buffer.
// ---------------------------------------------------------------------------
constexpr int32_t VL_FP32 = 64;
constexpr float TANH_CLAMP = 8.8f;
constexpr float TANH_TWO = 2.0f;
// Stable tanh: the naive (exp(2x) - 1) / (exp(2x) + 1) loses all precision near
// zero (e.g. x = 1e-8 makes fp32 exp(2e-8) round to 1.0f, so exp(2x) - 1 becomes
// 0 and a later multiply by a large beta turns an expected ~1 into 0). For
// |x| < TANH_LINEAR_THRESHOLD, tanh(x) is replaced by x; the truncation error is
// |tanh(x) - x| <= x^3 / 3, i.e. <= 4.2e-8 at the 5e-3 boundary (relative
// <= 8.3e-6). The naive form is kept for the remaining range, where it has no
// cancellation and its relative error stays around 1.3e-5 at the boundary.
constexpr float TANH_LINEAR_THRESHOLD = 5e-3f;
constexpr float FLOAT_MAX = 3.402823466e+38F;

// 1/beta or 1/linear_beta can overflow to +-inf when beta/linear_beta is a
// subnormal (e.g. 1e-45). A later 0 * inf would turn a legal zero input into
// NaN, so clamp the reciprocal to the largest finite float; the tanh clamp then
// saturates exactly as the unclamped formula would.
__aicore__ inline float SafeReciprocal(float value)
{
    const float inv = 1.0f / value;
    if (inv > FLOAT_MAX) {
        return FLOAT_MAX;
    }
    if (inv < -FLOAT_MAX) {
        return -FLOAT_MAX;
    }
    return inv;
}

constexpr AscendC::Reg::CastTrait castTraitB162B32Even = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::UNKNOWN};

constexpr AscendC::Reg::CastTrait castTraitB322B16Even = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void LoadInputData(RegTensor<float>& dst, __ubuf__ T* src, MaskReg preg, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::LoadAlign(dst, src + srcOffset);
    } else {
        RegTensor<T> tmp;
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        AscendC::Reg::Cast<float, T, castTraitB162B32Even>(dst, tmp, preg);
    }
}

template <typename T>
__simd_callee__ inline void StoreOutputData(__ubuf__ T* dst, RegTensor<float>& src, MaskReg preg, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        AscendC::Reg::StoreAlign(dst + dstOffset, src, preg);
    } else {
        RegTensor<T> tmp;
        AscendC::Reg::Cast<T, float, castTraitB322B16Even>(tmp, src, preg);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, preg);
    }
}

// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1), with a linear near-zero branch
// tanh(x) ~= x for |x| < TANH_LINEAR_THRESHOLD. Both the gate and the
// linear_beta up path call this helper; the linear branch prevents the naive
// exp form's catastrophic cancellation from being amplified by beta/linear_beta.
// NOTE: inOut is read as the input x and then overwritten with the naive tanh
// intermediate, so callers must not rely on its value after the call.
__simd_callee__ inline void VFTanhVf(RegTensor<float>& dst, RegTensor<float>& inOut, RegTensor<float>& tmp,
                                     MaskReg preg)
{
    AscendC::Reg::Mins(dst, inOut, TANH_CLAMP, preg);
    AscendC::Reg::Maxs(dst, dst, -TANH_CLAMP, preg);
    // tmp = exp(2x)
    AscendC::Reg::Muls(tmp, dst, TANH_TWO, preg);
    AscendC::Reg::Exp(tmp, tmp, preg);
    // inOut = naive tanh = (exp(2x) - 1) / (exp(2x) + 1)
    AscendC::Reg::Adds(inOut, tmp, -1.0f, preg);
    AscendC::Reg::Adds(tmp, tmp, 1.0f, preg);
    AscendC::Reg::Div(inOut, inOut, tmp, preg);
    // Select tanh(x) ~= x for |x| < threshold.
    AscendC::Reg::Abs(tmp, dst, preg);
    MaskReg cmpMask;
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(cmpMask, tmp, TANH_LINEAR_THRESHOLD, preg);
    AscendC::Reg::Select<float>(dst, dst, inOut, cmpMask);
}

// sigmoid(x) = 1 / (1 + exp(-x))
__simd_callee__ inline void VFSigmoidVf(RegTensor<float>& dst, RegTensor<float>& src, RegTensor<float>& tmp,
                                        MaskReg preg)
{
    AscendC::Reg::Muls(dst, src, -1.0f, preg);
    AscendC::Reg::Exp(dst, dst, preg);
    AscendC::Reg::Adds(dst, dst, 1.0f, preg);
    AscendC::Reg::Duplicate(tmp, 1.0f, preg);
    AscendC::Reg::Div(dst, tmp, dst, preg);
}

template <typename T, bool hasWeight, bool hasLinearBeta>
__simd_callee__ inline void VFSituComputeVf(__ubuf__ T* yAddr, RegTensor<float>& gate, RegTensor<float>& up,
                                            RegTensor<float>& weight, uint32_t yOffset, MaskReg preg, float beta,
                                            float invBeta, float linearBeta, float invLinearBeta)
{
    RegTensor<float> y;
    RegTensor<float> tanhGate;
    RegTensor<float> sigGate;
    RegTensor<float> tmp;
    RegTensor<float> upOut;

    // situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    AscendC::Reg::Muls(tmp, gate, invBeta, preg);
    VFTanhVf(tanhGate, tmp, sigGate, preg);
    VFSigmoidVf(sigGate, gate, tmp, preg);
    AscendC::Reg::Muls(tanhGate, tanhGate, beta, preg);
    AscendC::Reg::Mul(y, tanhGate, sigGate, preg);

    // up' = linearBeta * tanh(up / linearBeta) when linearBeta > 0, else up
    if constexpr (hasLinearBeta) {
        AscendC::Reg::Muls(tmp, up, invLinearBeta, preg);
        VFTanhVf(upOut, tmp, sigGate, preg);
        AscendC::Reg::Muls(upOut, upOut, linearBeta, preg);
        AscendC::Reg::Mul(y, y, upOut, preg);
    } else {
        AscendC::Reg::Mul(y, y, up, preg);
    }

    if constexpr (hasWeight) {
        AscendC::Reg::Mul(y, y, weight, preg);
    }
    StoreOutputData<T>(yAddr, y, preg, yOffset);
}

// Gate/up fetched from UB at a shared offset: each half lives in its own packed buffer.
template <typename T, bool hasWeight, bool hasLinearBeta>
__simd_callee__ inline void VFSituRowVf(__ubuf__ T* yAddr, __ubuf__ T* gateAddr, __ubuf__ T* upAddr,
                                        RegTensor<float>& weight, uint32_t srcOffset, MaskReg preg, float beta,
                                        float invBeta, float linearBeta, float invLinearBeta)
{
    RegTensor<float> gate;
    RegTensor<float> up;
    LoadInputData<T>(gate, gateAddr, preg, srcOffset);
    LoadInputData<T>(up, upAddr, preg, srcOffset);
    VFSituComputeVf<T, hasWeight, hasLinearBeta>(yAddr, gate, up, weight, srcOffset, preg, beta, invBeta, linearBeta,
                                                 invLinearBeta);
}

template <typename T, bool hasWeight, bool hasLinearBeta, bool singleLoop>
__simd_vf__ inline void VFSituVf(__ubuf__ T* yAddr, __ubuf__ T* gateAddr, __ubuf__ T* upAddr,
                                 __ubuf__ float* weightAddr, uint16_t loopCount, uint32_t sregNum, uint32_t curColAlign,
                                 uint16_t curRowNum, float beta, float invBeta, float linearBeta, float invLinearBeta)
{
    MaskReg preg;
    RegTensor<float> weight;
    if constexpr (singleLoop) {
        uint32_t sreg = sregNum;
        preg = AscendC::Reg::UpdateMask<float>(sreg);
        for (uint16_t i = 0; i < curRowNum; i++) {
            // Load the per-row weight once; it is reused for every d-chunk of this row.
            if constexpr (hasWeight) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, weightAddr + i);
            }
            VFSituRowVf<T, hasWeight, hasLinearBeta>(yAddr, gateAddr, upAddr, weight, i * curColAlign, preg, beta,
                                                     invBeta, linearBeta, invLinearBeta);
        }
    } else {
        for (uint16_t i = 0; i < curRowNum; i++) {
            // Load the per-row weight once; it is reused for every d-chunk of this row.
            if constexpr (hasWeight) {
                AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, weightAddr + i);
            }
            uint32_t sreg = sregNum;
            for (uint16_t j = 0; j < loopCount; j++) {
                preg = AscendC::Reg::UpdateMask<float>(sreg);
                VFSituRowVf<T, hasWeight, hasLinearBeta>(yAddr, gateAddr, upAddr, weight, j * VL_FP32 + i * curColAlign,
                                                         preg, beta, invBeta, linearBeta, invLinearBeta);
            }
        }
    }
}

template <typename T, bool hasWeight, bool hasLinearBeta>
__aicore__ inline void VFSitu(const LocalTensor<T>& yLocal, const LocalTensor<T>& gateLocal,
                              const LocalTensor<T>& upLocal, const LocalTensor<float>& weightLocal, uint16_t curRowNum,
                              uint32_t curColNum, float beta, float linearBeta)
{
    __ubuf__ T* yAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* gateAddr = (__ubuf__ T*)gateLocal.GetPhyAddr();
    __ubuf__ T* upAddr = (__ubuf__ T*)upLocal.GetPhyAddr();
    __ubuf__ float* weightAddr = hasWeight ? (__ubuf__ float*)weightLocal.GetPhyAddr() : nullptr;
    uint16_t loopCount = static_cast<uint16_t>(CeilDiv(curColNum, VL_FP32));
    uint32_t sregNum = curColNum;
    uint32_t curColAlign = static_cast<uint32_t>(RoundUp<T>(static_cast<int64_t>(curColNum)));
    float invBeta = SafeReciprocal(beta);
    float invLinearBeta = hasLinearBeta ? SafeReciprocal(linearBeta) : 0.0f;
    if (loopCount == 1) {
        AscendC::VF_CALL<VFSituVf<T, hasWeight, hasLinearBeta, true>>(yAddr, gateAddr, upAddr, weightAddr, loopCount,
                                                                      sregNum, curColAlign, curRowNum, beta, invBeta,
                                                                      linearBeta, invLinearBeta);
    } else {
        AscendC::VF_CALL<VFSituVf<T, hasWeight, hasLinearBeta, false>>(yAddr, gateAddr, upAddr, weightAddr, loopCount,
                                                                       sregNum, curColAlign, curRowNum, beta, invBeta,
                                                                       linearBeta, invLinearBeta);
    }
}

// Packed narrow rows (fp32 splitD * sizeof(float) < 32B, i.e. splitD in [1, 7]): a row half is only
// a few bytes, so the per-row MTE2 burst is 4~28B and MTE2 collapses. Here the whole row tile is
// copied as one contiguous burst and Reg::Gather rebuilds gate/up (and the per-row weight) inside
// the vector pipe, so the layout conversion happens on the vector pipe instead of the DMA.
//
// The index tables are built once in Init (element offsets, period splitD inside a 2*splitD row):
//   gate  : (i / splitD) * (2 * splitD) + (i % splitD) + gateHalf
//   up    : same + upHalf
//   weight: i / splitD (lane -> row within the block)
// and each block only adds its row base. The weight table lets the packed vector carry one weight
// per lane (row), so per-row weight is applied correctly instead of a single broadcast value.
template <typename T, bool hasWeight, bool hasLinearBeta>
__simd_vf__ inline void VFSituNarrowVf(__ubuf__ T* yAddr, __ubuf__ T* xAddr, __ubuf__ void* idxAddr,
                                       __ubuf__ float* weightAddr, uint32_t rowNum, uint32_t splitD,
                                       uint32_t rowsPerIter, float beta, float invBeta, float linearBeta,
                                       float invLinearBeta)
{
    constexpr uint32_t VL = static_cast<uint32_t>(VL_FP32);
    // Two tables of 32-bit element offsets: [0, VL) gate, [VL, 2*VL) up.
    RegTensor<uint32_t> index;
    RegTensor<uint32_t> indexUpTable;
    RegTensor<uint32_t> weightIndexTable;
    RegTensor<uint32_t> indexUpd;
    RegTensor<uint32_t> indexUpUpd;
    RegTensor<uint32_t> weightIndex;
    RegTensor<float> gate;
    RegTensor<float> up;
    RegTensor<float> weight;
    AscendC::Reg::LoadAlign(index, (__ubuf__ uint32_t*)idxAddr);
    AscendC::Reg::LoadAlign(indexUpTable, (__ubuf__ uint32_t*)idxAddr + VL);
    if constexpr (hasWeight) {
        AscendC::Reg::LoadAlign(weightIndexTable, (__ubuf__ uint32_t*)idxAddr + 2u * VL);
    }
    for (uint32_t base = 0; base < rowNum; base += rowsPerIter) {
        uint32_t left = (rowNum - base) * splitD;
        uint32_t active = left > VL ? VL : left;
        MaskReg preg = AscendC::Reg::UpdateMask<float>(active);
        const uint32_t rowBase = base * 2u * splitD;
        AscendC::Reg::Adds(indexUpd, index, rowBase, preg);
        AscendC::Reg::Gather(gate, xAddr, indexUpd, preg);
        AscendC::Reg::Adds(indexUpUpd, indexUpTable, rowBase, preg);
        AscendC::Reg::Gather(up, xAddr, indexUpUpd, preg);
        if constexpr (hasWeight) {
            // One weight per lane (lane j belongs to row base + j/splitD), gathered from the row
            // weight tile. A DIST_BRC broadcast here would apply the first row's weight to all rows.
            AscendC::Reg::Adds(weightIndex, weightIndexTable, base, preg);
            AscendC::Reg::Gather(weight, weightAddr, weightIndex, preg);
        }
        VFSituComputeVf<T, hasWeight, hasLinearBeta>(yAddr, gate, up, weight, base * splitD, preg, beta, invBeta,
                                                     linearBeta, invLinearBeta);
    }
}

template <typename T, bool hasWeight, bool hasLinearBeta>
__aicore__ inline void VFSituNarrow(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                    const LocalTensor<uint32_t>& idxLocal, const LocalTensor<float>& weightLocal,
                                    uint32_t rowNum, uint32_t splitD, uint32_t rowsPerIter, float beta,
                                    float linearBeta)
{
    __ubuf__ T* yAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* xAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ void* idxAddr = (__ubuf__ void*)idxLocal.GetPhyAddr();
    __ubuf__ float* weightAddr = hasWeight ? (__ubuf__ float*)weightLocal.GetPhyAddr() : nullptr;
    float invBeta = SafeReciprocal(beta);
    float invLinearBeta = hasLinearBeta ? SafeReciprocal(linearBeta) : 0.0f;
    AscendC::VF_CALL<VFSituNarrowVf<T, hasWeight, hasLinearBeta>>(
        yAddr, xAddr, idxAddr, weightAddr, rowNum, splitD, rowsPerIter, beta, invBeta, linearBeta, invLinearBeta);
}

template <typename T>
__aicore__ inline void SituGluGroupNarrowDispatcher(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                                    const LocalTensor<uint32_t>& idxLocal,
                                                    const LocalTensor<float>& weightLocal, uint32_t rowNum,
                                                    uint32_t splitD, uint32_t rowsPerIter, float beta, float linearBeta,
                                                    int32_t maskBit)
{
    if (maskBit == 0) {
        VFSituNarrow<T, false, false>(yLocal, xLocal, idxLocal, weightLocal, rowNum, splitD, rowsPerIter, beta,
                                      linearBeta);
    } else if (maskBit == MASK_HAS_WEIGHT) {
        VFSituNarrow<T, true, false>(yLocal, xLocal, idxLocal, weightLocal, rowNum, splitD, rowsPerIter, beta,
                                     linearBeta);
    } else if (maskBit == MASK_HAS_LINEAR_BETA) {
        VFSituNarrow<T, false, true>(yLocal, xLocal, idxLocal, weightLocal, rowNum, splitD, rowsPerIter, beta,
                                     linearBeta);
    } else {
        VFSituNarrow<T, true, true>(yLocal, xLocal, idxLocal, weightLocal, rowNum, splitD, rowsPerIter, beta,
                                    linearBeta);
    }
}

// Convert a 16-bit row tile to float using the proven LoadInputData + StoreAlign<float> path, so the
// packed gather can then run on the float layout (no b16 Gather/Store layout guesswork).
template <typename T>
__simd_vf__ inline void VFConvertToF32Vf(__ubuf__ float* dstAddr, __ubuf__ T* srcAddr, uint32_t elems)
{
    RegTensor<float> f;
    for (uint32_t off = 0; off < elems; off += static_cast<uint32_t>(VL_FP32)) {
        uint32_t remaining = elems - off;
        uint32_t cnt = remaining > static_cast<uint32_t>(VL_FP32) ? static_cast<uint32_t>(VL_FP32) : remaining;
        MaskReg preg = AscendC::Reg::UpdateMask<float>(cnt);
        LoadInputData<T>(f, srcAddr, preg, off);
        AscendC::Reg::StoreAlign(dstAddr + off, f, preg);
    }
}

// Packed gather on the float copy, storing y as T (the original dtype).
template <typename T, bool hasWeight, bool hasLinearBeta>
__simd_vf__ inline void VFSituNarrowF32Vf(__ubuf__ T* yAddr, __ubuf__ float* xAddr, __ubuf__ void* idxAddr,
                                          __ubuf__ float* weightAddr, uint32_t rowNum, uint32_t splitD,
                                          uint32_t rowsPerIter, float beta, float invBeta, float linearBeta,
                                          float invLinearBeta)
{
    constexpr uint32_t VL = static_cast<uint32_t>(VL_FP32);
    RegTensor<uint32_t> index;
    RegTensor<uint32_t> indexUpTable;
    RegTensor<uint32_t> weightIndexTable;
    RegTensor<uint32_t> indexUpd;
    RegTensor<uint32_t> indexUpUpd;
    RegTensor<uint32_t> weightIndex;
    RegTensor<float> gate;
    RegTensor<float> up;
    RegTensor<float> weight;
    AscendC::Reg::LoadAlign(index, (__ubuf__ uint32_t*)idxAddr);
    AscendC::Reg::LoadAlign(indexUpTable, (__ubuf__ uint32_t*)idxAddr + VL);
    if constexpr (hasWeight) {
        AscendC::Reg::LoadAlign(weightIndexTable, (__ubuf__ uint32_t*)idxAddr + 2u * VL);
    }
    for (uint32_t base = 0; base < rowNum; base += rowsPerIter) {
        uint32_t left = (rowNum - base) * splitD;
        uint32_t active = left > VL ? VL : left;
        MaskReg preg = AscendC::Reg::UpdateMask<float>(active);
        const uint32_t rowBase = base * 2u * splitD;
        AscendC::Reg::Adds(indexUpd, index, rowBase, preg);
        AscendC::Reg::Gather(gate, xAddr, indexUpd, preg);
        AscendC::Reg::Adds(indexUpUpd, indexUpTable, rowBase, preg);
        AscendC::Reg::Gather(up, xAddr, indexUpUpd, preg);
        if constexpr (hasWeight) {
            AscendC::Reg::Adds(weightIndex, weightIndexTable, base, preg);
            AscendC::Reg::Gather(weight, weightAddr, weightIndex, preg);
        }
        VFSituComputeVf<T, hasWeight, hasLinearBeta>(yAddr, gate, up, weight, base * splitD, preg, beta, invBeta,
                                                     linearBeta, invLinearBeta);
    }
}

template <typename T, bool hasWeight, bool hasLinearBeta>
__aicore__ inline void VFSituNarrowF32(const LocalTensor<T>& yLocal, const LocalTensor<float>& xF32Local,
                                       const LocalTensor<uint32_t>& idxLocal, const LocalTensor<float>& weightLocal,
                                       uint32_t rowNum, uint32_t splitD, uint32_t rowsPerIter, float beta,
                                       float linearBeta)
{
    __ubuf__ T* yAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ float* xAddr = (__ubuf__ float*)xF32Local.GetPhyAddr();
    __ubuf__ void* idxAddr = (__ubuf__ void*)idxLocal.GetPhyAddr();
    __ubuf__ float* weightAddr = hasWeight ? (__ubuf__ float*)weightLocal.GetPhyAddr() : nullptr;
    float invBeta = SafeReciprocal(beta);
    float invLinearBeta = hasLinearBeta ? SafeReciprocal(linearBeta) : 0.0f;
    AscendC::VF_CALL<VFSituNarrowF32Vf<T, hasWeight, hasLinearBeta>>(
        yAddr, xAddr, idxAddr, weightAddr, rowNum, splitD, rowsPerIter, beta, invBeta, linearBeta, invLinearBeta);
}

template <typename T>
__aicore__ inline void SituGluGroupNarrowF32Dispatcher(const LocalTensor<T>& yLocal,
                                                       const LocalTensor<float>& xF32Local,
                                                       const LocalTensor<uint32_t>& idxLocal,
                                                       const LocalTensor<float>& weightLocal, uint32_t rowNum,
                                                       uint32_t splitD, uint32_t rowsPerIter, float beta,
                                                       float linearBeta, int32_t maskBit)
{
    if (maskBit == 0) {
        VFSituNarrowF32<T, false, false>(yLocal, xF32Local, idxLocal, weightLocal, rowNum, splitD, rowsPerIter, beta,
                                         linearBeta);
    } else if (maskBit == MASK_HAS_WEIGHT) {
        VFSituNarrowF32<T, true, false>(yLocal, xF32Local, idxLocal, weightLocal, rowNum, splitD, rowsPerIter, beta,
                                        linearBeta);
    } else if (maskBit == MASK_HAS_LINEAR_BETA) {
        VFSituNarrowF32<T, false, true>(yLocal, xF32Local, idxLocal, weightLocal, rowNum, splitD, rowsPerIter, beta,
                                        linearBeta);
    } else {
        VFSituNarrowF32<T, true, true>(yLocal, xF32Local, idxLocal, weightLocal, rowNum, splitD, rowsPerIter, beta,
                                       linearBeta);
    }
}

template <typename T>
__aicore__ inline void VFConvertToF32(const LocalTensor<float>& dstLocal, const LocalTensor<T>& srcLocal,
                                      uint32_t elems)
{
    __ubuf__ float* dst = (__ubuf__ float*)dstLocal.GetPhyAddr();
    __ubuf__ T* src = (__ubuf__ T*)srcLocal.GetPhyAddr();
    AscendC::VF_CALL<VFConvertToF32Vf<T>>(dst, src, elems);
}

// 16-bit splitD == 1: every row is a single 32-bit word [gate(低16位), up(高16位)]. The tile is copied
// contiguously, then each word is split by masking/shifting. Both extracted halves then sit in the
// low 16 bits of a 32-bit lane, which is exactly the layout castTraitB162B32Even (RegLayout::ZERO)
// consumes, so no DIST_DINTLV / 16-bit pack-cast is involved.
template <typename T, bool gateIsHigh, bool hasLinearBeta>
__simd_vf__ inline void VFSituPair16Vf(__ubuf__ T* yAddr, __ubuf__ T* xAddr, uint32_t rowNum, float beta, float invBeta,
                                       float linearBeta, float invLinearBeta)
{
    constexpr uint32_t kRowsPerIter = static_cast<uint32_t>(VL_FP32);
    RegTensor<uint32_t> words;
    RegTensor<uint32_t> lo;
    RegTensor<uint32_t> hi;
    RegTensor<uint32_t> lowMask;
    RegTensor<float> gate;
    RegTensor<float> up;
    RegTensor<float> weight; // unused: pair16 is only selected when weight is absent
    MaskReg pregAll = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(lowMask, static_cast<uint32_t>(0xFFFF), pregAll);
    __ubuf__ uint32_t* src = (__ubuf__ uint32_t*)xAddr;
    for (uint32_t base = 0; base < rowNum; base += kRowsPerIter) {
        uint32_t remaining = rowNum - base;
        MaskReg preg = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(words, src + base);
        AscendC::Reg::And(lo, words, lowMask, preg);
        AscendC::Reg::ShiftRights(hi, words, static_cast<int16_t>(16), preg);
        if constexpr (gateIsHigh) {
            AscendC::Reg::Cast<float, T, castTraitB162B32Even>(gate, (RegTensor<T>&)hi, preg);
            AscendC::Reg::Cast<float, T, castTraitB162B32Even>(up, (RegTensor<T>&)lo, preg);
        } else {
            AscendC::Reg::Cast<float, T, castTraitB162B32Even>(gate, (RegTensor<T>&)lo, preg);
            AscendC::Reg::Cast<float, T, castTraitB162B32Even>(up, (RegTensor<T>&)hi, preg);
        }
        VFSituComputeVf<T, false, hasLinearBeta>(yAddr, gate, up, weight, base, preg, beta, invBeta, linearBeta,
                                                 invLinearBeta);
    }
}

template <typename T, bool gateIsHigh, bool hasLinearBeta>
__aicore__ inline void VFSituPair16(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal, uint32_t rowNum,
                                    float beta, float linearBeta)
{
    __ubuf__ T* yAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* xAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    float invBeta = SafeReciprocal(beta);
    float invLinearBeta = hasLinearBeta ? SafeReciprocal(linearBeta) : 0.0f;
    AscendC::VF_CALL<VFSituPair16Vf<T, gateIsHigh, hasLinearBeta>>(yAddr, xAddr, rowNum, beta, invBeta, linearBeta,
                                                                   invLinearBeta);
}

template <typename T>
__aicore__ inline void SituGluGroupPair16Dispatcher(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                                    uint32_t rowNum, bool gateIsHigh, bool hasLinearBeta, float beta,
                                                    float linearBeta)
{
    if (gateIsHigh) {
        if (hasLinearBeta) {
            VFSituPair16<T, true, true>(yLocal, xLocal, rowNum, beta, linearBeta);
        } else {
            VFSituPair16<T, true, false>(yLocal, xLocal, rowNum, beta, linearBeta);
        }
    } else {
        if (hasLinearBeta) {
            VFSituPair16<T, false, true>(yLocal, xLocal, rowNum, beta, linearBeta);
        } else {
            VFSituPair16<T, false, false>(yLocal, xLocal, rowNum, beta, linearBeta);
        }
    }
}

template <typename T>
__aicore__ inline void SituGluGroupDispatcher(const LocalTensor<T>& yLocal, const LocalTensor<T>& gateLocal,
                                              const LocalTensor<T>& upLocal, const LocalTensor<float>& weightLocal,
                                              uint16_t curRowNum, uint32_t curColNum, float beta, float linearBeta,
                                              int32_t maskBit)
{
    if (maskBit == 0) {
        VFSitu<T, false, false>(yLocal, gateLocal, upLocal, weightLocal, curRowNum, curColNum, beta, linearBeta);
    } else if (maskBit == MASK_HAS_WEIGHT) {
        VFSitu<T, true, false>(yLocal, gateLocal, upLocal, weightLocal, curRowNum, curColNum, beta, linearBeta);
    } else if (maskBit == MASK_HAS_LINEAR_BETA) {
        VFSitu<T, false, true>(yLocal, gateLocal, upLocal, weightLocal, curRowNum, curColNum, beta, linearBeta);
    } else {
        VFSitu<T, true, true>(yLocal, gateLocal, upLocal, weightLocal, curRowNum, curColNum, beta, linearBeta);
    }
}

// Saturating accumulation for the count-mode row total. The caller only consumes
// min(trueSum, cap), so the running value is kept inside [0, cap]: the addend is clamped to cap and
// the sum is re-clamped after every step. Both inputs are then bounded by cap, hence sum + x <=
// 2 * cap and no signed int64 add can overflow (Host rejects cap > INT64_MAX / 64).
// A negative user element survives the clamp and is still caught by the independent Min probe /
// Trap path; clamping the sum to zero only keeps that invalid data from corrupting the saturating
// arithmetic. Clamping at cap is exact for min(trueSum, cap) because a partial sum that reaches cap
// can only grow further.
template <typename T>
__simd_callee__ inline void VFAccumulateGroupIndex(RegTensor<T>& sum, RegTensor<T>& x, RegTensor<T>& capReg,
                                                   RegTensor<T>& zeroReg, MaskReg preg)
{
    Min(x, x, capReg, preg);
    Add(sum, sum, x, preg);
    Max(sum, sum, zeroReg, preg);
    Min(sum, sum, capReg, preg);
}

template <typename T>
__simd_callee__ inline void VFClampGroupIndexSum(RegTensor<T>& sum, RegTensor<T>& capReg, RegTensor<T>& zeroReg,
                                                 MaskReg preg)
{
    Max(sum, sum, zeroReg, preg);
    Min(sum, sum, capReg, preg);
}

// group_index helpers, mirrored from swiglu_group_base.h. The reduce always yields the min probe;
// the sum is only accumulated in count mode (withSum=true). The mode-specific total is derived in
// ProcessGroupIndexTiling:
//   count  : realBs = min(sum(group_index), bs)
//   cumsum : realBs = min(group_index[last], bs)
// `cap` is bs (the positive row count that saturates the sum); the caller returns early for an
// empty output, so the accumulator never sees cap <= 0.
template <typename T, bool withUbReduce = false, bool withSum = true>
__simd_vf__ inline void VFProcessGroupIndexSmallVf(__ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t curColNum,
                                                   uint16_t vlLen, uint16_t loopCount, int64_t cap)
{
    RegTensor<T> x;
    RegTensor<T> sum;
    // Running minimum of the masked chunks. Zero is a neutral seed because the operator requires
    // non-negative counts: the reduced minimum stays 0 for valid data and turns negative only when
    // an element is negative.
    RegTensor<T> minValue;
    RegTensor<T> capReg;
    RegTensor<T> zeroReg;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum, static_cast<T>(0), pregMain);
    Duplicate(minValue, static_cast<T>(0), pregMain);
    if constexpr (withSum) {
        Duplicate(capReg, static_cast<T>(cap), pregMain);
        Duplicate(zeroReg, static_cast<T>(0), pregMain);
    }
    uint32_t sreg = curColNum;
    MaskReg pregLoop;
    for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x, xLocalAddr + i * vlLen);
        Adds(x, x, static_cast<T>(0), pregLoop);
        Min(minValue, minValue, x, pregMain);
        if constexpr (withSum) {
            VFAccumulateGroupIndex(sum, x, capReg, zeroReg, pregMain);
        }
    }
    if constexpr (withSum) {
        // Each lane is <= cap, so the cross-lane reduction stays <= 32 * cap (bounded by cap above).
        Reduce<ReduceType::SUM>(sum, sum, pregMain);
        VFClampGroupIndexSum(sum, capReg, zeroReg, pregMain);
    }
    Reduce<ReduceType::MIN>(minValue, minValue, pregMain);
    if constexpr (withUbReduce) {
        RegTensor<T> origin;
        if constexpr (withSum) {
            LoadAlign(origin, yLocalAddr);
            Add(sum, sum, origin, pregMerge);
            VFClampGroupIndexSum(sum, capReg, zeroReg, pregMain);
        }
        LoadAlign(origin, yLocalAddr + GroupIndexMinSlot<T>());
        Min(minValue, minValue, origin, pregMerge);
    }
    if constexpr (withSum) {
        StoreAlign(yLocalAddr, sum, pregMerge);
    }
    StoreAlign(yLocalAddr + GroupIndexMinSlot<T>(), minValue, pregMerge);
}

template <typename T, bool withUbReduce = false, bool withSum = true>
__simd_vf__ inline void VFProcessGroupIndexLargeVf(__ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t vlLen,
                                                   uint16_t fourLoopCount, uint16_t tailLoopNum, uint32_t tailReminder,
                                                   int64_t cap)
{
    RegTensor<T> x0;
    RegTensor<T> x1;
    RegTensor<T> x2;
    RegTensor<T> x3;
    RegTensor<T> sum0;
    RegTensor<T> sum1;
    RegTensor<T> sum2;
    RegTensor<T> sum3;
    // Running minimum of the masked chunks, seeded with the neutral zero (see the small variant).
    RegTensor<T> min0;
    RegTensor<T> min1;
    RegTensor<T> min2;
    RegTensor<T> min3;
    RegTensor<T> capReg;
    RegTensor<T> zeroReg;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum0, static_cast<T>(0), pregMain);
    Duplicate(sum1, static_cast<T>(0), pregMain);
    Duplicate(sum2, static_cast<T>(0), pregMain);
    Duplicate(sum3, static_cast<T>(0), pregMain);
    Duplicate(min0, static_cast<T>(0), pregMain);
    Duplicate(min1, static_cast<T>(0), pregMain);
    Duplicate(min2, static_cast<T>(0), pregMain);
    Duplicate(min3, static_cast<T>(0), pregMain);
    if constexpr (withSum) {
        Duplicate(capReg, static_cast<T>(cap), pregMain);
        Duplicate(zeroReg, static_cast<T>(0), pregMain);
    }
    MaskReg pregLoop;
    for (uint16_t i = 0; i < fourLoopCount; i++) {
        LoadAlign(x0, xLocalAddr + i * FOUR_UNFOLD * vlLen);
        Min(min0, min0, x0, pregMain);
        if constexpr (withSum) {
            VFAccumulateGroupIndex(sum0, x0, capReg, zeroReg, pregMain);
        }
        LoadAlign(x1, xLocalAddr + (i * FOUR_UNFOLD + 1) * vlLen);
        Min(min1, min1, x1, pregMain);
        if constexpr (withSum) {
            VFAccumulateGroupIndex(sum1, x1, capReg, zeroReg, pregMain);
        }
        LoadAlign(x2, xLocalAddr + (i * FOUR_UNFOLD + 2) * vlLen);
        Min(min2, min2, x2, pregMain);
        if constexpr (withSum) {
            VFAccumulateGroupIndex(sum2, x2, capReg, zeroReg, pregMain);
        }
        LoadAlign(x3, xLocalAddr + (i * FOUR_UNFOLD + 3) * vlLen);
        Min(min3, min3, x3, pregMain);
        if constexpr (withSum) {
            VFAccumulateGroupIndex(sum3, x3, capReg, zeroReg, pregMain);
        }
    }
    uint32_t sreg = tailReminder;
    for (uint16_t i = 0; i < tailLoopNum; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x0, xLocalAddr + (fourLoopCount * FOUR_UNFOLD + i) * vlLen);
        Adds(x0, x0, static_cast<T>(0), pregLoop);
        Min(min0, min0, x0, pregMain);
        if constexpr (withSum) {
            VFAccumulateGroupIndex(sum0, x0, capReg, zeroReg, pregMain);
        }
    }
    // Every lane is already <= cap, so the pairwise merges stay <= 2 * cap and cannot overflow.
    if constexpr (withSum) {
        Add(sum0, sum0, sum1, pregMain);
        VFClampGroupIndexSum(sum0, capReg, zeroReg, pregMain);
        Add(sum2, sum2, sum3, pregMain);
        VFClampGroupIndexSum(sum2, capReg, zeroReg, pregMain);
        Add(sum0, sum0, sum2, pregMain);
        VFClampGroupIndexSum(sum0, capReg, zeroReg, pregMain);
    }
    Min(min0, min0, min1, pregMain);
    Min(min2, min2, min3, pregMain);
    Min(min0, min0, min2, pregMain);
    if constexpr (withSum) {
        // Each lane is <= cap, so the cross-lane reduction stays <= 32 * cap (bounded by cap above).
        Reduce<ReduceType::SUM>(sum0, sum0, pregMain);
        VFClampGroupIndexSum(sum0, capReg, zeroReg, pregMain);
    }
    Reduce<ReduceType::MIN>(min0, min0, pregMain);
    if constexpr (withUbReduce) {
        RegTensor<T> origin;
        if constexpr (withSum) {
            LoadAlign(origin, yLocalAddr);
            Add(sum0, sum0, origin, pregMerge);
            VFClampGroupIndexSum(sum0, capReg, zeroReg, pregMain);
        }
        LoadAlign(origin, yLocalAddr + GroupIndexMinSlot<T>());
        Min(min0, min0, origin, pregMerge);
    }
    if constexpr (withSum) {
        StoreAlign(yLocalAddr, sum0, pregMerge);
    }
    StoreAlign(yLocalAddr + GroupIndexMinSlot<T>(), min0, pregMerge);
}

template <typename T, bool withUbReduce = false, bool withSum = true>
__aicore__ inline void VFProcessGroupIndex(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                           uint16_t curColNum, int64_t cap)
{
    __ubuf__ T* yLocalAddr = (__ubuf__ T*)yLocal.GetPhyAddr();
    __ubuf__ T* xLocalAddr = (__ubuf__ T*)xLocal.GetPhyAddr();
    uint16_t vlLen = REPEAT_SIZE / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint16_t fullBlocks = curColNum / vlLen;
    uint16_t fourLoopCount = fullBlocks / FOUR_UNFOLD;
    uint16_t tailLoopNum = loopCount - fourLoopCount * FOUR_UNFOLD;
    uint32_t tailReminder = curColNum - fourLoopCount * vlLen * FOUR_UNFOLD;
    if (loopCount < FOUR_UNFOLD) {
        AscendC::VF_CALL<VFProcessGroupIndexSmallVf<T, withUbReduce, withSum>>(yLocalAddr, xLocalAddr, curColNum, vlLen,
                                                                               loopCount, cap);
    } else {
        AscendC::VF_CALL<VFProcessGroupIndexLargeVf<T, withUbReduce, withSum>>(
            yLocalAddr, xLocalAddr, vlLen, fourLoopCount, tailLoopNum, tailReminder, cap);
    }
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst,
                              const uint32_t copyLen, uint32_t srcStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = nBurst;
    dataCopyExtParams.blockLen = copyLen * sizeof(T);
    dataCopyExtParams.srcStride = srcStride * sizeof(T);
    dataCopyExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCopyExtParams, dataCopyPadExtParams);
}

// Row tiling derived from the real row count and the number of ROW blocks (blockNum), not from the
// launched block count: the remaining blocks are handed to the column split in ComputeTaskTiling,
// and the runtime may also enable fewer blocks than the host blockDim (core limiting / fused
// scheduling), so rows assigned to blocks that do not exist would be silently skipped.
__aicore__ inline void ComputeBlockTiling(const SituGluGroupTilingData* tilingData, int64_t realBs, int64_t blockNum,
                                          int64_t& usedCoreNums, int64_t& rowOfFormerBlock, int64_t& rowOfTailBlock,
                                          int64_t& rowLoopOfFormerBlock, int64_t& rowLoopOfTailBlock,
                                          int64_t& tailRowFactorOfFormerBlock, int64_t& tailRowFactorOfTailBlock)
{
    if (realBs <= 0 || blockNum <= 0) {
        usedCoreNums = 0;
        return;
    }

    rowOfFormerBlock = CeilDiv(realBs, blockNum);
    int64_t blocksNeeded = CeilDiv(realBs, rowOfFormerBlock);
    usedCoreNums = blocksNeeded < blockNum ? blocksNeeded : blockNum;
    rowOfTailBlock = realBs - (usedCoreNums - 1) * rowOfFormerBlock;

    rowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock, tilingData->rowFactor);
    rowLoopOfTailBlock = CeilDiv(rowOfTailBlock, tilingData->rowFactor);
    tailRowFactorOfFormerBlock = rowOfFormerBlock % tilingData->rowFactor == 0 ?
                                     tilingData->rowFactor :
                                     rowOfFormerBlock % tilingData->rowFactor;
    tailRowFactorOfTailBlock = rowOfTailBlock % tilingData->rowFactor == 0 ? tilingData->rowFactor :
                                                                             rowOfTailBlock % tilingData->rowFactor;
}

// Column split. Row-only parallelism leaves cores idle whenever realBs is smaller than the
// launched block count (dim=0 / 1-D input makes bs == 1, and group_index truncation can push
// realBs even lower). In that case the leftover parallelism is taken from the splitD axis:
//   totalBlocks = rowBlockNum * colBlockNum, every core owns one (row block, column block) pair.
// Column blocks are 32B aligned and at least MIN_COL_BYTES long so each core still issues
// reasonably sized MTE2 bursts; the last column block absorbs the remainder.
template <typename T>
__aicore__ inline void ComputeColTiling(int64_t realBs, int64_t splitD, int64_t totalBlocks, int64_t& colBlockNum,
                                        int64_t& colLenOfBlock, int64_t& tailColLen)
{
    colBlockNum = 1;
    colLenOfBlock = splitD;
    tailColLen = splitD;
    if (realBs <= 0 || splitD <= 0 || totalBlocks <= realBs) {
        return;
    }
    const int64_t alignElems = static_cast<int64_t>(BLOCK_SIZE) / static_cast<int64_t>(sizeof(T));
    const int64_t minColElems = MIN_COL_BYTES / static_cast<int64_t>(sizeof(T));
    const int64_t wantColBlocks = totalBlocks / realBs;
    const int64_t maxColBlocks = splitD / minColElems;
    const int64_t target = wantColBlocks < maxColBlocks ? wantColBlocks : maxColBlocks;
    if (target <= 1) {
        return;
    }
    const int64_t colLen = AlignUpElems(CeilDiv(splitD, target), alignElems);
    if (colLen < alignElems || colLen >= splitD) {
        return;
    }
    const int64_t colBlocks = CeilDiv(splitD, colLen);
    const int64_t tailLen = splitD - (colBlocks - 1) * colLen;
    if (colBlocks < 1 || colBlocks > totalBlocks || tailLen <= 0) {
        // The aligned block length did not tile splitD within the block budget: stay on one
        // column block instead of leaving a column range unassigned.
        return;
    }
    colBlockNum = colBlocks;
    colLenOfBlock = colLen;
    tailColLen = tailLen;
}

// Row tiling derived from the real row count and the number of ROW blocks; the column split above
// consumes the remaining blocks. Both the no-group path (realBs = bs) and the group_index path
// (realBs = min(sum, bs)) go through here so the two paths partition rows identically.
template <typename T>
__aicore__ inline void ComputeTaskTiling(const SituGluGroupTilingData* tilingData, int64_t realBs, int64_t totalBlocks,
                                         int64_t& colBlockNum, int64_t& colLenOfBlock, int64_t& tailColLen,
                                         int64_t& usedCoreNums, int64_t& rowOfFormerBlock, int64_t& rowOfTailBlock,
                                         int64_t& rowLoopOfFormerBlock, int64_t& rowLoopOfTailBlock,
                                         int64_t& tailRowFactorOfFormerBlock, int64_t& tailRowFactorOfTailBlock)
{
    usedCoreNums = 0;
    colBlockNum = 0;
    colLenOfBlock = 0;
    tailColLen = 0;
    if (realBs <= 0 || totalBlocks <= 0) {
        return;
    }
    ComputeColTiling<T>(realBs, tilingData->splitD, totalBlocks, colBlockNum, colLenOfBlock, tailColLen);
    int64_t rowBlockNum = totalBlocks / colBlockNum;
    if (rowBlockNum < 1) {
        rowBlockNum = 1;
    }
    if (rowBlockNum > realBs) {
        rowBlockNum = realBs;
    }
    ComputeBlockTiling(tilingData, realBs, rowBlockNum, usedCoreNums, rowOfFormerBlock, rowOfTailBlock,
                       rowLoopOfFormerBlock, rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);
}

// Resolve the real row count (realBs) from group_index when it is present. When group_index is
// absent every row is produced, so realBs is just bs. The returned value only describes the row
// extent; the row/column task split happens afterwards in ComputeTaskTiling.
template <typename TBufPoolType>
__aicore__ inline void ProcessGroupIndexTiling(GM_ADDR groupIndex, const SituGluGroupTilingData* tilingData,
                                               TBufPoolType& tBufPool, TQue<QuePosition::VECIN, 1>& groupIndexQue,
                                               TBuf<QuePosition::VECCALC>& groupIndexSumBuf,
                                               GlobalTensor<int64_t>& groupIndexGm, LocalTensor<int64_t>& groupSumLocal,
                                               int64_t& realBs)
{
    realBs = 0;
    // Empty output (zero row count or zero cut extent): no tile plan is consumed, and running the
    // plan math would evaluate CeilDiv(bs, blockNum) on a bs that d == 0 leaves unbounded.
    if (tilingData->bs <= 0 || tilingData->splitD <= 0) {
        return;
    }
    // Follow the host-side presence decision: a non-null pointer with a null shape is treated as
    // absent by Tiling (no groupIndexLoop/groupIndexFactor), so the kernel must take the default
    // path instead of reading an uninitialized group sum.
    if (tilingData->hasGroupIndex == 0) {
        realBs = tilingData->bs;
        return;
    }
    if (groupIndex == nullptr) {
        // Tiling sets hasGroupIndex only when the group_index shape is present and non-empty, so a
        // null GM address is an inconsistent launch. bs > 0 is guaranteed by the empty-output guard
        // above, so the output would be left unwritten while the caller still sees success; fail
        // loudly instead of returning a silent no-op.
        KERNEL_LOG(KERNEL_ERROR,
                   "SituGluGroup: tiling reports group_index present but its GM address is null, bs is %ld",
                   static_cast<int64_t>(tilingData->bs));
        // Trap does not return; no fallthrough reset is needed here.
        AscendC::Trap();
    }

    groupIndexGm.SetGlobalBuffer((__gm__ int64_t*)groupIndex, static_cast<uint64_t>(tilingData->groupIndexElemNum));
    tBufPool.InitBuffer(groupIndexQue, DOUBLE_BUFFER_NUM,
                        RoundUp<int64_t>(tilingData->groupIndexFactor) * sizeof(int64_t));
    // Two 32B slots: slot 0 holds sum(group_index), slot 1 holds the negative-value probe.
    tBufPool.InitBuffer(groupIndexSumBuf, GROUP_INDEX_SUM_SLOT_NUM * BLOCK_SIZE);
    groupSumLocal = groupIndexSumBuf.Get<int64_t>();
    // cumsum mode only needs the negative probe and the last cumulative offset; skip the per-element
    // sum accumulation when the sum is not consumed.
    const bool needSum = (tilingData->groupListType != 0);
    // Count mode only consumes min(sum(group_index), bs). The VF accumulator saturates at bs so no
    // signed int64 add can overflow; the empty-output guard above keeps bs positive here.
    const int64_t sumCap = tilingData->bs;
    // V->S edge: the VF reduce writes groupSumLocal on the V pipe and the scalar GetValue below must
    // observe it. Arch35 (Ascend 950/350) provides Mutex::Lock/Unlock, which is preferred over the ISASI
    // SetFlag/WaitFlag pair because it is version-portable and cannot leak an event token on the
    // early-return paths.
    AscendC::MutexID mutexId = AscendC::AllocMutexID();
    AscendC::Mutex::Lock<PIPE_V>(mutexId);
    for (int64_t idx = 0; idx < tilingData->groupIndexLoop; idx++) {
        int64_t curGroupIndexFactor = (idx == tilingData->groupIndexLoop - 1) ? tilingData->tailGroupIndexFactor :
                                                                                tilingData->groupIndexFactor;
        LocalTensor<int64_t> groupIndexLocal = groupIndexQue.template AllocTensor<int64_t>();
        CopyIn(groupIndexGm[idx * tilingData->groupIndexFactor], groupIndexLocal, 1, curGroupIndexFactor);
        groupIndexQue.template EnQue(groupIndexLocal);
        groupIndexLocal = groupIndexQue.template DeQue<int64_t>();
        if (idx == 0) {
            if (needSum) {
                VFProcessGroupIndex<int64_t, false, true>(groupSumLocal, groupIndexLocal, curGroupIndexFactor, sumCap);
            } else {
                VFProcessGroupIndex<int64_t, false, false>(groupSumLocal, groupIndexLocal, curGroupIndexFactor, sumCap);
            }
        } else {
            if (needSum) {
                VFProcessGroupIndex<int64_t, true, true>(groupSumLocal, groupIndexLocal, curGroupIndexFactor, sumCap);
            } else {
                VFProcessGroupIndex<int64_t, true, false>(groupSumLocal, groupIndexLocal, curGroupIndexFactor, sumCap);
            }
        }
        groupIndexQue.template FreeTensor(groupIndexLocal);
    }
    AscendC::Mutex::Unlock<PIPE_V>(mutexId);
    AscendC::Mutex::Lock<PIPE_S>(mutexId);
    int64_t groupMin = groupSumLocal.GetValue(GroupIndexMinSlot<int64_t>());
    int64_t groupSum = needSum ? groupSumLocal.GetValue(0) : 0;
    AscendC::Mutex::Unlock<PIPE_S>(mutexId);
    AscendC::ReleaseMutexID(mutexId);
    // group_list_type selects how group_index is interpreted:
    //   0 = cumsum: each element is a cumulative row end offset, so the last element gives realBs;
    //   1 = count : each element is a per-group row count, so their sum gives realBs.
    // In count mode groupSum is already saturated at bs, so it equals min(trueSum, bs) for any
    // non-negative input and no int64 accumulated value can overflow.
    // Both modes must reject negative elements: a negative value would corrupt realBs and the
    // operator would silently leave rows untouched, so fail explicitly instead. groupMin is seeded
    // at 0, so it only turns negative when an element is actually negative (a zero element keeps it
    // at 0, which is legal).
    // A zero total (cumsum last == 0, or count sum == 0) is legal and yields realBs = 0: no row is
    // written, matching SwigluGroup and the golden (realBs = min(total, rows)).
    if (tilingData->groupListType == 0) {
        // The VF reduce above only yields min in cumsum mode, so read the last cumulative offset
        // from GM with the scalar pipe to recover the total row count.
        const int64_t lastIndex = tilingData->groupIndexElemNum - 1;
        int64_t groupLast = groupIndexGm.GetValue(static_cast<uint64_t>(lastIndex));
        if (groupMin < 0) {
            KERNEL_LOG(KERNEL_ERROR,
                       "SituGluGroup: cumsum group_index elements must be non-negative, got min %ld and last %ld",
                       static_cast<int64_t>(groupMin), static_cast<int64_t>(groupLast));
            // Trap does not return; no fallthrough reset is needed here.
            AscendC::Trap();
        }
        realBs = groupLast > tilingData->bs ? tilingData->bs : groupLast;
    } else {
        if (groupMin < 0) {
            KERNEL_LOG(KERNEL_ERROR,
                       "SituGluGroup: count group_index elements must be non-negative, got min %ld and sum %ld",
                       static_cast<int64_t>(groupMin), static_cast<int64_t>(groupSum));
            // Trap does not return; no fallthrough reset is needed here.
            AscendC::Trap();
        }
        realBs = groupSum > tilingData->bs ? tilingData->bs : groupSum;
    }
    tBufPool.Reset();
}

template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
__aicore__ inline void CopyOut(const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm,
                               const uint16_t nBurst, const uint32_t copyLen, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad<T, mode>(outputGm, outputTensor, dataCopyParams);
}

// Contiguous tile copies emitted as <= 32KB bursts. A fused row tile is rowFactor rows wide, so a
// single descriptor can span tens of KB and is rejected by the MTE2 front end (aicore error).
template <typename T>
__aicore__ inline void CopyInContig(const GlobalTensor<T>& srcGm, const LocalTensor<T>& dstLocal, int64_t elems)
{
    const int64_t maxElems = MAX_COPY_BURST_BYTES / static_cast<int64_t>(sizeof(T));
    int64_t done = 0;
    while (done < elems) {
        int64_t chunk = elems - done;
        if (chunk > maxElems) {
            chunk = maxElems;
        }
        CopyIn(srcGm[done], dstLocal[done], 1, static_cast<uint32_t>(chunk));
        done += chunk;
    }
}

template <typename T>
__aicore__ inline void CopyOutContig(const LocalTensor<T>& srcLocal, const GlobalTensor<T>& dstGm, int64_t elems)
{
    const int64_t maxElems = MAX_COPY_BURST_BYTES / static_cast<int64_t>(sizeof(T));
    int64_t done = 0;
    while (done < elems) {
        int64_t chunk = elems - done;
        if (chunk > maxElems) {
            chunk = maxElems;
        }
        CopyOut(srcLocal[done], dstGm[done], 1, static_cast<uint32_t>(chunk));
        done += chunk;
    }
}

template <typename T>
class SituGluGroupKernel {
public:
    __aicore__ inline SituGluGroupKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y,
                                const SituGluGroupTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe_ = pipePtr;
        tilingData_ = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        pipe_->InitBufPool(tBufPool, tilingData_->ubSize);
        // rowOfTailBlock is only consumed while deriving rowLoopOfTailBlock/tailRowFactorOfTailBlock
        // inside the tile computation, so it stays a local instead of a class member.
        int64_t rowOfTailBlock = 0;
        // realBs is the row extent after group_index truncation (== bs when group_index is absent).
        int64_t realBs = 0;
        ProcessGroupIndexTiling(groupIndex, tilingData_, tBufPool, groupIndexQue, groupIndexSumBuf, groupIndexGm,
                                groupSumLocal, realBs);
        // GetBlockNum() is the total task budget: rows first, then the splitD axis for whatever
        // parallelism is left over (bs < blockDim). Using the launched block count instead of the
        // host coreNum keeps core limiting / fused scheduling from assigning rows or columns to
        // blocks that do not exist.
        ComputeTaskTiling<T>(tilingData_, realBs, static_cast<int64_t>(GetBlockNum()), colBlockNum_, colLenOfBlock_,
                             tailColLen_, usedCoreNums_, rowOfFormerBlock_, rowOfTailBlock, rowLoopOfFormerBlock_,
                             rowLoopOfTailBlock_, tailRowFactorOfFormerBlock_, tailRowFactorOfTailBlock_);

        // Follow the Tiling decision: Tiling sets hasWeight only when the weight shape is present
        // and included in the UB budget. The kernel still guards the pointer so an inconsistent
        // present-but-null weight can never reach DataCopyPad with a null GM address.
        if (tilingData_->hasWeight != 0) {
            if (weight == nullptr) {
                // Tiling sets hasWeight only when numel(weight) == bs, so a null GM address is an
                // inconsistent launch unless weight is the legitimate empty tensor (bs == 0). With
                // bs > 0 the output would be left unwritten while the caller still sees success.
                if (tilingData_->bs > 0) {
                    KERNEL_LOG(KERNEL_ERROR,
                               "SituGluGroup: tiling reports weight present but its GM address is null, bs is %ld",
                               static_cast<int64_t>(tilingData_->bs));
                    // Trap does not return; no fallthrough reset is needed here.
                    AscendC::Trap();
                }
                // bs == 0: weight is an empty tensor and y has no elements to write.
                usedCoreNums_ = 0;
                return;
            }
            hasWeight_ = true;
            weightGm.SetGlobalBuffer((__gm__ float*)weight);
            tBufPool.InitBuffer(weightQue, DOUBLE_BUFFER_NUM, RoundUp<float>(tilingData_->rowFactor) * sizeof(float));
        }

        hasLinearBeta_ = (tilingData_->linearBeta > 0.0f);
        // activateLeft == 1: gate is the front half [0, splitD); otherwise gate is the back half.
        gateHalfOffset_ = (tilingData_->activateLeft == 1) ? 0 : tilingData_->splitD;
        upHalfOffset_ = (tilingData_->activateLeft == 1) ? tilingData_->splitD : 0;
        // Decide the copy layout BEFORE allocating, because it selects which queues exist.
        // PACKED narrow path: when one row half (splitD) is below a 32B block, the per-row MTE2 burst
        // is only a few bytes. The whole row tile is copied as one contiguous burst and the layout is
        // rebuilt inside the vector pipe:
        //   - fp32 splitD in [1, 7]  -> Reg::Gather (with per-row weight)
        //   - 16-bit splitD == 1     -> split each 32-bit word into low/high 16-bit halves
        // Every other shape uses the two-buffer SPLIT path. (The DIST_DINTLV variant for splitD == 1
        // was removed: its 16-bit unpack / per-row weight layouts were incorrect on device.)
        copyMode_ = COPY_MODE_SPLIT;
        if (colBlockNum_ == 1 && tilingData_->dFactor >= tilingData_->splitD && tilingData_->splitD > 0) {
            if (IsSameType<T, float>::value && tilingData_->splitD * static_cast<int64_t>(sizeof(float)) < BLOCK_SIZE) {
                copyMode_ = COPY_MODE_NARROW;
                narrowSplitD_ = static_cast<uint32_t>(tilingData_->splitD);
                // The packed block advances by rowsPerIter rows, so every y StoreAlign offset is
                // base * splitD. Pick rowsPerIter as a multiple of the rows that fill one 32B block
                // (blockElems / gcd(splitD, blockElems)) so base * splitD stays 32B aligned; otherwise
                // the vector store to a misaligned UB address faults on device.
                const int64_t blockElems = static_cast<int64_t>(BLOCK_SIZE) / static_cast<int64_t>(sizeof(float));
                const int64_t maxRows = static_cast<int64_t>(VL_FP32) / static_cast<int64_t>(narrowSplitD_);
                const int64_t alignRows = blockElems / Gcd(static_cast<int64_t>(narrowSplitD_), blockElems); // >= 1
                int64_t rowsPerIter = (maxRows / alignRows) * alignRows;
                if (rowsPerIter < 1) {
                    rowsPerIter = 1;
                }
                narrowRowsPerIter_ = static_cast<uint32_t>(rowsPerIter);
            } else if (!IsSameType<T, float>::value && tilingData_->splitD == 1 && !hasWeight_ &&
                       tilingData_->rowFactor >= PAIR16_MIN_ROW_FACTOR) {
                copyMode_ = COPY_MODE_PAIR16;
            } else if (!IsSameType<T, float>::value && tilingData_->splitD >= 2 &&
                       tilingData_->splitD * static_cast<int64_t>(sizeof(T)) < BLOCK_SIZE) {
                // Convert the 16-bit tile to float and reuse the fp32 gather. Only when the y
                // StoreAlign offsets can stay 32B aligned (rowsPerIter * splitD % blockElems == 0).
                const int64_t blockElems = static_cast<int64_t>(BLOCK_SIZE) / static_cast<int64_t>(sizeof(T));
                const int64_t maxRows = static_cast<int64_t>(VL_FP32) / static_cast<int64_t>(tilingData_->splitD);
                const int64_t alignRows = blockElems / Gcd(static_cast<int64_t>(tilingData_->splitD), blockElems);
                if (alignRows > 0 && alignRows <= maxRows) {
                    int64_t rowsPerIter = (maxRows / alignRows) * alignRows;
                    if (rowsPerIter < 1) {
                        rowsPerIter = 1;
                    }
                    copyMode_ = COPY_MODE_NARROW_F32;
                    narrowSplitD_ = static_cast<uint32_t>(tilingData_->splitD);
                    narrowRowsPerIter_ = static_cast<uint32_t>(rowsPerIter);
                }
            }
        }

        if (copyMode_ == COPY_MODE_SPLIT) {
            tBufPool.InitBuffer(gateQue, DOUBLE_BUFFER_NUM,
                                tilingData_->rowFactor * RoundUp<T>(tilingData_->dFactor) * sizeof(T));
            tBufPool.InitBuffer(upQue, DOUBLE_BUFFER_NUM,
                                tilingData_->rowFactor * RoundUp<T>(tilingData_->dFactor) * sizeof(T));
        } else {
            // One fused tile buffer holding every row (gate + up) back to back.
            tBufPool.InitBuffer(xQue, DOUBLE_BUFFER_NUM,
                                2 * tilingData_->rowFactor * RoundUp<T>(tilingData_->dFactor) * sizeof(T));
        }
        if (copyMode_ == COPY_MODE_NARROW || copyMode_ == COPY_MODE_NARROW_F32) {
            // Three VL-sized tables: gate offsets, up offsets, per-lane weight row offsets.
            tBufPool.InitBuffer(narrowIdxBuf, static_cast<int32_t>(3 * VL_FP32) * sizeof(uint32_t));
            BuildNarrowIndex();
        }
        if (copyMode_ == COPY_MODE_NARROW_F32) {
            // Float copy of the whole row tile (the 16-bit tile is converted before the gather).
            const int64_t f32Elems = tilingData_->rowFactor * 2 * tilingData_->splitD;
            tBufPool.InitBuffer(narrowF32Buf, f32Elems * sizeof(float));
        }
        tBufPool.InitBuffer(yQue, DOUBLE_BUFFER_NUM,
                            tilingData_->rowFactor * RoundUp<T>(tilingData_->dFactor) * sizeof(T));
    }

    // Element offsets of one gather block (another form of "tiling data": the layout depends only on
    // splitD, never on the row index, and every block only adds its row base):
    //   table0 (gate): (i / splitD) * (2 * splitD) + (i % splitD) + gateHalf
    //   table1 (up)  : same row base + upHalf
    //   table2 (weight): i / splitD  (lane -> row offset within the packed block)
    __aicore__ inline void BuildNarrowIndex()
    {
        const uint32_t splitD = narrowSplitD_;
        const uint32_t gateHalf = static_cast<uint32_t>(gateHalfOffset_);
        const uint32_t upHalf = static_cast<uint32_t>(upHalfOffset_);
        const uint32_t vl = static_cast<uint32_t>(VL_FP32);
        LocalTensor<uint32_t> idx = narrowIdxBuf.Get<uint32_t>();
        for (uint32_t i = 0; i < vl; i++) {
            const uint32_t rowBase = (i / splitD) * (2u * splitD) + (i % splitD);
            idx.SetValue(i, rowBase + gateHalf);
            idx.SetValue(vl + i, rowBase + upHalf);
            idx.SetValue(2u * vl + i, i / splitD);
        }
    }

    __aicore__ inline void Process()
    {
        if (tilingData_->bs == 0 || tilingData_->splitD == 0) {
            // Empty input (zero row count or zero cut-dim extent): nothing to compute.
            return;
        }
        if (usedCoreNums_ <= 0 || colBlockNum_ <= 0 || tilingData_->dFactor <= 0) {
            return;
        }
        // Task grid: blockIdx = rowBlockIdx * colBlockNum + colBlockIdx. The column split exists so
        // that cores stay busy when there are fewer row blocks than launched blocks (bs == 1 for
        // dim=0 / 1-D inputs, or a small realBs after group_index truncation).
        const int64_t taskIdx = GetBlockIdx();
        const int64_t rowBlockIdx = taskIdx / colBlockNum_;
        if (rowBlockIdx >= usedCoreNums_) {
            // Over-provisioned launch: this block owns no rows.
            return;
        }
        const int64_t colBlockIdx = taskIdx % colBlockNum_;
        const int64_t colStart = colBlockIdx * colLenOfBlock_;
        const int64_t colLen = (colBlockIdx == colBlockNum_ - 1) ? tailColLen_ : colLenOfBlock_;
        if (colLen <= 0) {
            return;
        }
        // Per-core d loop: the host dLoop/dFactor plan assumes every core walks the whole splitD,
        // while a column block only owns [colStart, colStart + colLen). dFactor is never larger than
        // the host value, so the queue buffers sized by rowFactor * RoundUp(dFactor) still fit.
        const int64_t dFactor = tilingData_->dFactor;
        const int64_t dLoopOfBlock = CeilDiv(colLen, dFactor);
        const int64_t tailDOfBlock = colLen - (dLoopOfBlock - 1) * dFactor;

        int64_t rowOuterLoop = (rowBlockIdx == usedCoreNums_ - 1) ? rowLoopOfTailBlock_ : rowLoopOfFormerBlock_;
        int64_t tailRowFactor = (rowBlockIdx == usedCoreNums_ - 1) ? tailRowFactorOfTailBlock_ :
                                                                     tailRowFactorOfFormerBlock_;
        int64_t rowGmBase = rowBlockIdx * rowOfFormerBlock_;
        // The narrow mode copies every row from its start (no column slice in this mode).
        int64_t xRowGmBaseOffset = rowGmBase * tilingData_->d + colStart;
        int64_t gateGmBaseOffset = xRowGmBaseOffset + gateHalfOffset_;
        int64_t upGmBaseOffset = xRowGmBaseOffset + upHalfOffset_;
        int64_t yGmBaseOffset = rowGmBase * tilingData_->splitD + colStart;
        int64_t weightGmBaseOffset = rowGmBase;
        // Elements copied per row tile in the narrow mode: a whole row (d = 2 * splitD).
        const int64_t fusedRowElems = tilingData_->d;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData_->rowFactor;
            if (hasWeight_) {
                weightLocal = weightQue.template AllocTensor<float>();
                CopyIn(weightGm[weightGmBaseOffset + rowOuterIdx * tilingData_->rowFactor], weightLocal, 1,
                       curRowFactor);
                weightQue.template EnQue(weightLocal);
                weightLocal = weightQue.template DeQue<float>();
            }

            if (copyMode_ != COPY_MODE_SPLIT) {
                // One contiguous MTE2 burst for the whole tile; the layout is rebuilt in the VF.
                int64_t xTileGmOffset = xRowGmBaseOffset + rowOuterIdx * tilingData_->rowFactor * tilingData_->d;
                gateLocal = xQue.template AllocTensor<T>();
                CopyInContig(xGm[xTileGmOffset], gateLocal, curRowFactor * fusedRowElems);
                xQue.template EnQue(gateLocal);
                gateLocal = xQue.template DeQue<T>();

                yLocal = yQue.template AllocTensor<T>();
                if (copyMode_ == COPY_MODE_NARROW) {
                    ComputeVfNarrow(yLocal, gateLocal, curRowFactor);
                } else if (copyMode_ == COPY_MODE_NARROW_F32) {
                    ComputeVfNarrowF32(yLocal, gateLocal, curRowFactor);
                } else {
                    ComputeVfPair16(yLocal, gateLocal, curRowFactor);
                }
                xQue.template FreeTensor(gateLocal);

                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                // y rows are packed back to back in the tile, so the tile leaves as contiguous bursts.
                CopyOutContig(yLocal, yGm[yGmBaseOffset + rowOuterIdx * tilingData_->rowFactor * tilingData_->splitD],
                              curRowFactor * tilingData_->splitD);
                yQue.template FreeTensor(yLocal);
                if (hasWeight_) {
                    weightQue.template FreeTensor(weightLocal);
                }
                continue;
            }

            for (int64_t dLoopIdx = 0; dLoopIdx < dLoopOfBlock; dLoopIdx++) {
                int64_t curDFactor = (dLoopIdx == dLoopOfBlock - 1) ? tailDOfBlock : dFactor;
                int64_t xBaseOffset = rowOuterIdx * tilingData_->rowFactor * tilingData_->d + dLoopIdx * dFactor;
                gateLocal = gateQue.template AllocTensor<T>();
                CopyIn(xGm[gateGmBaseOffset + xBaseOffset], gateLocal, curRowFactor, curDFactor,
                       tilingData_->d - curDFactor);
                gateQue.template EnQue(gateLocal);
                gateLocal = gateQue.template DeQue<T>();

                upLocal = upQue.template AllocTensor<T>();
                CopyIn(xGm[upGmBaseOffset + xBaseOffset], upLocal, curRowFactor, curDFactor,
                       tilingData_->d - curDFactor);
                upQue.template EnQue(upLocal);
                upLocal = upQue.template DeQue<T>();

                yLocal = yQue.template AllocTensor<T>();

                ComputeVf(yLocal, gateLocal, upLocal, weightLocal, curRowFactor, curDFactor);

                gateQue.template FreeTensor(gateLocal);
                upQue.template FreeTensor(upLocal);

                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                CopyOut(yLocal,
                        yGm[yGmBaseOffset + rowOuterIdx * tilingData_->rowFactor * tilingData_->splitD +
                            dLoopIdx * dFactor],
                        curRowFactor, curDFactor, tilingData_->splitD - curDFactor);
                yQue.template FreeTensor(yLocal);
            }
            if (hasWeight_) {
                weightQue.template FreeTensor(weightLocal);
            }
        }
    }

    __aicore__ inline void ComputeVfNarrow(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                           int64_t curRowFactor)
    {
        // The narrow path is float32 only (the gather index is 32-bit there); the 16-bit
        // instantiations must not reference it, or the gather static_assert fires for them.
        if constexpr (IsSameType<T, float>::value) {
            int32_t maskBit = (hasLinearBeta_ ? MASK_HAS_LINEAR_BETA : 0) | (hasWeight_ ? MASK_HAS_WEIGHT : 0);
            LocalTensor<uint32_t> idx = narrowIdxBuf.Get<uint32_t>();
            SituGluGroupNarrowDispatcher<T>(yLocal, xLocal, idx, weightLocal, static_cast<uint32_t>(curRowFactor),
                                            narrowSplitD_, narrowRowsPerIter_, tilingData_->beta,
                                            tilingData_->linearBeta, maskBit);
        }
    }

    __aicore__ inline void ComputeVfPair16(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                           int64_t curRowFactor)
    {
        // 16-bit splitD == 1 only (pair16 is never selected for fp32). The runtime activate_left
        // picks which 16-bit half is gate.
        if constexpr (!IsSameType<T, float>::value) {
            const bool gateIsHigh = (gateHalfOffset_ > upHalfOffset_);
            SituGluGroupPair16Dispatcher<T>(yLocal, xLocal, static_cast<uint32_t>(curRowFactor), gateIsHigh,
                                            hasLinearBeta_, tilingData_->beta, tilingData_->linearBeta);
        }
    }

    __aicore__ inline void ComputeVfNarrowF32(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                              int64_t curRowFactor)
    {
        if constexpr (!IsSameType<T, float>::value) {
            LocalTensor<float> xf = narrowF32Buf.Get<float>();
            const uint32_t elems = static_cast<uint32_t>(curRowFactor) * narrowSplitD_ * 2u;
            VFConvertToF32<T>(xf, xLocal, elems);
            LocalTensor<uint32_t> idx = narrowIdxBuf.Get<uint32_t>();
            int32_t maskBit = (hasLinearBeta_ ? MASK_HAS_LINEAR_BETA : 0) | (hasWeight_ ? MASK_HAS_WEIGHT : 0);
            SituGluGroupNarrowF32Dispatcher<T>(yLocal, xf, idx, weightLocal, static_cast<uint32_t>(curRowFactor),
                                               narrowSplitD_, narrowRowsPerIter_, tilingData_->beta,
                                               tilingData_->linearBeta, maskBit);
        }
    }

    __aicore__ inline void ComputeVf(const LocalTensor<T>& yLocal, const LocalTensor<T>& gateLocal,
                                     const LocalTensor<T>& upLocal, const LocalTensor<float>& weightLocal,
                                     int64_t curRowFactor, int64_t curDFactor)
    {
        int32_t maskBit = (hasLinearBeta_ ? MASK_HAS_LINEAR_BETA : 0) | (hasWeight_ ? MASK_HAS_WEIGHT : 0);
        SituGluGroupDispatcher<T>(yLocal, gateLocal, upLocal, weightLocal, static_cast<uint16_t>(curRowFactor),
                                  static_cast<uint32_t>(curDFactor), tilingData_->beta, tilingData_->linearBeta,
                                  maskBit);
    }

private:
    TPipe* pipe_;
    const SituGluGroupTilingData* tilingData_;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<int64_t> groupIndexGm;

    TQue<QuePosition::VECIN, 1> gateQue;
    TQue<QuePosition::VECIN, 1> upQue;
    // The narrow mode copies the whole-row tile into this single buffer.
    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECIN, 1> weightQue;

    TQue<QuePosition::VECIN, 1> groupIndexQue;
    TBuf<QuePosition::VECCALC> groupIndexSumBuf;
    // Gather index table for the narrow-row mode (element offsets, rebuilt once per kernel).
    TBuf<QuePosition::VECCALC> narrowIdxBuf;
    // Float copy of the 16-bit row tile for the narrow_f32 path.
    TBuf<QuePosition::VECCALC> narrowF32Buf;
    TBufPool<QuePosition::VECCALC, TBUF_POOL_DEPTH> tBufPool;

    LocalTensor<T> gateLocal;
    LocalTensor<T> upLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> weightLocal;

    LocalTensor<int64_t> groupSumLocal;

    bool hasWeight_ = false;
    bool hasLinearBeta_ = false;
    int64_t gateHalfOffset_ = 0;
    int64_t upHalfOffset_ = 0;
    int32_t copyMode_ = COPY_MODE_SPLIT;
    uint32_t narrowSplitD_ = 0;
    uint32_t narrowRowsPerIter_ = 0;

    int64_t tailRowFactorOfTailBlock_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    // Row blocks and, when bs < blockDim, the additional column blocks along splitD.
    int64_t usedCoreNums_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t colBlockNum_ = 0;
    int64_t colLenOfBlock_ = 0;
    int64_t tailColLen_ = 0;
};

} // namespace NsSituGluGroup

#endif // SITU_GLU_GROUP_KERNEL_H
