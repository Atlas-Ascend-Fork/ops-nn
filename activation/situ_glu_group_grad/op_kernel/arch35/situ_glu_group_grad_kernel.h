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
 * \file situ_glu_group_grad_kernel.h
 * \brief SituGluGroupGrad Arch35 (Ascend 950) kernel (VF backward + VF group-index reduce).
 *
 * Forward:  y = w * situ_a(gate) * up'(up)
 * Backward, with t = tanh(gate/beta), s = sigmoid(gate), situ_a = beta*t*s,
 * up' = linear_beta*tanh(up/linear_beta) (linear_beta > 0) else up:
 * grad_gate = grad_y * up' * s * [(1 - t^2) + beta*t*(1 - s)]   (* w when weight present)
 * grad_up   = grad_y * situ_a * (1 - tanh^2(up/linear_beta))     (linear_beta > 0) (* w)
 * grad_up   = grad_y * situ_a                                    (linear_beta <= 0) (* w)
 * grad_w[n] = sum_h grad_y[n,h] * up'[n,h] * situ_a[n,h]         (unweighted, only when weight present)
 * grad_x = [grad_gate | grad_up] laid out like x (gate/up halves chosen by activate_left). Only the first
 * realBs = min(groupRows, bs) rows are computed and written when group_index is present, where groupRows is
 * group_index[-1] in cumsum mode (group_list_type 0, the default) and sum(group_index) in count mode (1).
 */

#ifndef SITU_GLU_GROUP_GRAD_KERNEL_H
#define SITU_GLU_GROUP_GRAD_KERNEL_H

#include "kernel_operator.h"

namespace SituGluGroupGradOp {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t REPEAT_SIZE = 256;
constexpr uint16_t FOUR_UNFOLD = 4;
constexpr uint32_t TBUF_POOL_DEPTH = 16;
// Rows per grad_weight zero-fill DMA on the d == 0 early-exit path (4 KB of float).
constexpr int64_t GRADW_ZERO_FILL_ROWS = 1024;
// Dispatcher mask bits: bit0 = weight present (also drives grad_weight), bit1 = linear_beta active.
constexpr int32_t MASK_HAS_WEIGHT = 0b01;
constexpr int32_t MASK_HAS_LINEAR_BETA = 0b10;
// group_list_type: 0 (default) = cumsum, group_index[-1] is the total row count;
// 1 = count, sum(group_index) is the total row count.
constexpr int64_t GROUP_LIST_TYPE_COUNT = 1;

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

template <typename T>
__aicore__ inline int64_t RoundUp(int64_t num)
{
    int64_t elemNum = BLOCK_SIZE / static_cast<int64_t>(sizeof(T));
    return (num + elemNum - 1) / elemNum * elemNum;
}

// ---------------------------------------------------------------------------
// VF (RegBase) backward path, modeled on the SituGluGroup Arch35 forward kernel.
// The whole row stays in vector registers: half/bf16 are unpacked/packed in
// registers, weight is broadcast with DIST_B32, so there is no per-row scalar
// GetValue and no fp32 UB scratch buffer for the activation itself.
// ---------------------------------------------------------------------------
constexpr int32_t VL_FP32 = 64;
constexpr float TANH_CLAMP = 8.8f;
constexpr float TANH_TWO = 2.0f;
// |x| below this switches to the Taylor branch: (exp(2x) - 1) cancels badly for small x
// (exp(2x) rounds to 1, losing ~log10(1/x) digits), e.g. rel err ~4e-4 at x = 6e-5.
constexpr float TANH_SMALL = 0.0625f;
constexpr float TANH_POLY_C1 = 1.0f / 3.0f;
constexpr float TANH_POLY_C2 = 2.0f / 15.0f;

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

// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1); clamp to [-8.8, 8.8] keeps exp finite.
// For |x| < TANH_SMALL use x - x^3/3 + 2x^5/15 (rel err < 1e-8 on that range), because the
// exp form's subtract-one step loses most significant digits as x -> 0.
__simd_callee__ inline void VFTanhVf(RegTensor<float>& dst, RegTensor<float>& src, RegTensor<float>& tmp, MaskReg preg)
{
    RegTensor<float> poly;
    // small-|x| Taylor branch: x * (1 - x^2/3 + 2x^4/15)
    AscendC::Reg::Mul(tmp, src, src, preg);
    AscendC::Reg::Muls(poly, tmp, -TANH_POLY_C2, preg);
    AscendC::Reg::Adds(poly, poly, TANH_POLY_C1, preg);
    AscendC::Reg::Mul(poly, poly, tmp, preg);
    AscendC::Reg::Muls(poly, poly, -1.0f, preg);
    AscendC::Reg::Adds(poly, poly, 1.0f, preg);
    AscendC::Reg::Mul(poly, poly, src, preg);

    // large-|x| branch: (exp(2x) - 1) / (exp(2x) + 1)
    AscendC::Reg::Mins(dst, src, TANH_CLAMP, preg);
    AscendC::Reg::Maxs(dst, dst, -TANH_CLAMP, preg);
    AscendC::Reg::Muls(tmp, dst, TANH_TWO, preg);
    AscendC::Reg::Exp(dst, tmp, preg);
    AscendC::Reg::Adds(tmp, dst, -1.0f, preg);
    AscendC::Reg::Adds(dst, dst, 1.0f, preg);
    AscendC::Reg::Div(dst, tmp, dst, preg);

    MaskReg small;
    AscendC::Reg::Abs(tmp, src, preg);
    AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(small, tmp, TANH_SMALL, preg);
    // src1 aliases dst so inactive/small-false lanes keep the exp-branch value.
    AscendC::Reg::Select<float>(dst, poly, dst, small);
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

// One (row, VL-chunk) of the backward. grad_gate / grad_up are stored to UB; the unweighted
// grad_weight integrand (partial = grad_y * up' * situ_a) is returned in `partial` for the caller
// to reduce. Every op carries preg, whose ZEROING merge clears the lanes past the valid column
// count, so the per-row reduce never sums UB tail garbage (fp32 LoadAlign itself is unmasked).
template <typename T, bool hasWeight, bool hasLinearBeta>
__simd_callee__ inline void VFSituGradRowVf(__ubuf__ T* gradGateAddr, __ubuf__ T* gradUpAddr, __ubuf__ T* gradYAddr,
                                            __ubuf__ T* gateAddr, __ubuf__ T* upAddr, RegTensor<float>& weight,
                                            RegTensor<float>& partial, uint32_t srcOffset, MaskReg preg, float beta,
                                            float invBeta, float linearBeta, float invLinearBeta)
{
    RegTensor<float> gradY;
    RegTensor<float> gate;
    RegTensor<float> up;
    RegTensor<float> t;     // tanh(gate / beta)
    RegTensor<float> s;     // sigmoid(gate)
    RegTensor<float> situA; // beta * t * s
    RegTensor<float> tU;    // tanh(up / linear_beta)
    RegTensor<float> tmp;
    RegTensor<float> gradSituA;
    RegTensor<float> gradUpPrime;
    RegTensor<float> gradGate;
    RegTensor<float> gradUp;

    LoadInputData<T>(gradY, gradYAddr, preg, srcOffset);
    LoadInputData<T>(gate, gateAddr, preg, srcOffset);
    LoadInputData<T>(up, upAddr, preg, srcOffset);

    // t = tanh(gate / beta); s = sigmoid(gate)
    AscendC::Reg::Muls(tmp, gate, invBeta, preg);
    VFTanhVf(t, tmp, s, preg);
    VFSigmoidVf(s, gate, tmp, preg);

    // situ_a = beta * t * s
    AscendC::Reg::Mul(situA, t, s, preg);
    AscendC::Reg::Muls(situA, situA, beta, preg);

    // up' = linear_beta * tanh(up / linear_beta) when linear_beta > 0, else up. After this block
    // `up` always holds up'; tU holds tanh(up/linear_beta) only on the linear-beta path.
    if constexpr (hasLinearBeta) {
        AscendC::Reg::Muls(tmp, up, invLinearBeta, preg);
        VFTanhVf(tU, tmp, gradGate, preg);
        AscendC::Reg::Muls(up, tU, linearBeta, preg);
    }

    // grad_situ_a = grad_y * up'; grad_up_prime = grad_y * situ_a
    AscendC::Reg::Mul(gradSituA, gradY, up, preg);
    AscendC::Reg::Mul(gradUpPrime, gradY, situA, preg);

    // partial = grad_y * up' * situ_a (unweighted grad_weight integrand)
    if constexpr (hasWeight) {
        AscendC::Reg::Mul(partial, gradSituA, situA, preg);
    }

    // grad_gate = grad_situ_a * s * [(1 - t^2) + beta*t*(1 - s)]
    AscendC::Reg::Mul(situA, t, t, preg);
    AscendC::Reg::Muls(situA, situA, -1.0f, preg);
    AscendC::Reg::Adds(situA, situA, 1.0f, preg); // situA = 1 - t^2
    AscendC::Reg::Muls(tmp, s, -1.0f, preg);
    AscendC::Reg::Adds(tmp, tmp, 1.0f, preg); // tmp = 1 - s
    AscendC::Reg::Mul(tmp, tmp, t, preg);
    AscendC::Reg::Muls(tmp, tmp, beta, preg);   // tmp = beta * t * (1 - s)
    AscendC::Reg::Add(situA, situA, tmp, preg); // situA = bracket
    AscendC::Reg::Mul(gradGate, gradSituA, s, preg);
    AscendC::Reg::Mul(gradGate, gradGate, situA, preg);

    // grad_up = grad_up_prime * (1 - t_u^2) (linear_beta > 0) else grad_up_prime
    if constexpr (hasLinearBeta) {
        AscendC::Reg::Mul(tmp, tU, tU, preg);
        AscendC::Reg::Muls(tmp, tmp, -1.0f, preg);
        AscendC::Reg::Adds(tmp, tmp, 1.0f, preg); // tmp = 1 - t_u^2
        AscendC::Reg::Mul(gradUp, gradUpPrime, tmp, preg);
    } else {
        AscendC::Reg::Muls(gradUp, gradUpPrime, 1.0f, preg);
    }

    if constexpr (hasWeight) {
        AscendC::Reg::Mul(gradGate, gradGate, weight, preg);
        AscendC::Reg::Mul(gradUp, gradUp, weight, preg);
    }
    StoreOutputData<T>(gradGateAddr, gradGate, preg, srcOffset);
    StoreOutputData<T>(gradUpAddr, gradUp, preg, srcOffset);
}

template <typename T, bool hasWeight, bool hasLinearBeta, bool singleLoop>
__simd_vf__ inline void VFSituGradVf(__ubuf__ T* gradGateAddr, __ubuf__ T* gradUpAddr, __ubuf__ T* gradYAddr,
                                     __ubuf__ T* gateAddr, __ubuf__ T* upAddr, __ubuf__ float* weightAddr,
                                     __ubuf__ float* gradWAccAddr, uint16_t loopCount, uint32_t sregNum,
                                     uint32_t curColAlign, uint16_t curRowNum, bool firstChunk, float beta,
                                     float invBeta, float linearBeta, float invLinearBeta)
{
    MaskReg pregMain = CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<float, AscendC::Reg::MaskPattern::VL1>();
    // RegTensors are declared once at function scope (the VF register allocator resolves them);
    // they are reused across rows / VL-chunks exactly like the SituGluGroup forward kernel.
    RegTensor<float> weight;
    RegTensor<float> partialAcc;
    RegTensor<float> partial;
    RegTensor<float> rowSum;
    RegTensor<float> origin;
    MaskReg preg;
    for (uint16_t i = 0; i < curRowNum; i++) {
        // Load the per-row weight once; it is reused for every VL-chunk of this row. partialAcc is
        // the per-row running sum of the unweighted grad_weight integrand across the VL-chunks.
        if constexpr (hasWeight) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(weight, weightAddr + i);
            AscendC::Reg::Duplicate(partialAcc, 0.0f, pregMain);
        }
        uint32_t remaining = sregNum;
        if constexpr (singleLoop) {
            uint32_t cnt = remaining > static_cast<uint32_t>(VL_FP32) ? static_cast<uint32_t>(VL_FP32) : remaining;
            preg = AscendC::Reg::UpdateMask<float>(cnt);
            VFSituGradRowVf<T, hasWeight, hasLinearBeta>(gradGateAddr, gradUpAddr, gradYAddr, gateAddr, upAddr, weight,
                                                         partial, i * curColAlign, preg, beta, invBeta, linearBeta,
                                                         invLinearBeta);
            if constexpr (hasWeight) {
                AscendC::Reg::Add(partialAcc, partialAcc, partial, pregMain);
            }
        } else {
            for (uint16_t j = 0; j < loopCount; j++) {
                uint32_t cnt = remaining > static_cast<uint32_t>(VL_FP32) ? static_cast<uint32_t>(VL_FP32) : remaining;
                preg = AscendC::Reg::UpdateMask<float>(cnt);
                VFSituGradRowVf<T, hasWeight, hasLinearBeta>(gradGateAddr, gradUpAddr, gradYAddr, gateAddr, upAddr,
                                                             weight, partial, j * VL_FP32 + i * curColAlign, preg, beta,
                                                             invBeta, linearBeta, invLinearBeta);
                if constexpr (hasWeight) {
                    AscendC::Reg::Add(partialAcc, partialAcc, partial, pregMain);
                }
                remaining = (remaining > static_cast<uint32_t>(VL_FP32)) ? (remaining - VL_FP32) : 0;
            }
        }
        // grad_weight[row] accumulates the per-row partial sum across d-chunks; the first chunk
        // overwrites, later chunks add into the running value held in the UB accumulator.
        if constexpr (hasWeight) {
            AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rowSum, partialAcc, pregMain);
            if (!firstChunk) {
                AscendC::Reg::Load(origin, gradWAccAddr + i);
                AscendC::Reg::Add(rowSum, rowSum, origin, pregMerge);
            }
            AscendC::Reg::Store(gradWAccAddr + i, rowSum, 1);
        }
    }
}

template <typename T, bool hasWeight, bool hasLinearBeta>
__aicore__ inline void VFSituGrad(const LocalTensor<T>& gradGateLocal, const LocalTensor<T>& gradUpLocal,
                                  const LocalTensor<T>& gradYLocal, const LocalTensor<T>& gateLocal,
                                  const LocalTensor<T>& upLocal, const LocalTensor<float>& weightLocal,
                                  const LocalTensor<float>& gradWAccLocal, uint16_t curRowNum, uint32_t curColNum,
                                  bool firstChunk, float beta, float linearBeta)
{
    __ubuf__ T* gradGateAddr = (__ubuf__ T*)gradGateLocal.GetPhyAddr();
    __ubuf__ T* gradUpAddr = (__ubuf__ T*)gradUpLocal.GetPhyAddr();
    __ubuf__ T* gradYAddr = (__ubuf__ T*)gradYLocal.GetPhyAddr();
    __ubuf__ T* gateAddr = (__ubuf__ T*)gateLocal.GetPhyAddr();
    __ubuf__ T* upAddr = (__ubuf__ T*)upLocal.GetPhyAddr();
    __ubuf__ float* weightAddr = hasWeight ? (__ubuf__ float*)weightLocal.GetPhyAddr() : nullptr;
    __ubuf__ float* gradWAccAddr = hasWeight ? (__ubuf__ float*)gradWAccLocal.GetPhyAddr() : nullptr;
    uint16_t loopCount = static_cast<uint16_t>(CeilDiv(curColNum, VL_FP32));
    uint32_t sregNum = curColNum;
    uint32_t curColAlign = static_cast<uint32_t>(RoundUp<T>(static_cast<int64_t>(curColNum)));
    float invBeta = 1.0f / beta;
    float invLinearBeta = hasLinearBeta ? (1.0f / linearBeta) : 0.0f;
    if (loopCount == 1) {
        AscendC::VF_CALL<VFSituGradVf<T, hasWeight, hasLinearBeta, true>>(
            gradGateAddr, gradUpAddr, gradYAddr, gateAddr, upAddr, weightAddr, gradWAccAddr, loopCount, sregNum,
            curColAlign, curRowNum, firstChunk, beta, invBeta, linearBeta, invLinearBeta);
    } else {
        AscendC::VF_CALL<VFSituGradVf<T, hasWeight, hasLinearBeta, false>>(
            gradGateAddr, gradUpAddr, gradYAddr, gateAddr, upAddr, weightAddr, gradWAccAddr, loopCount, sregNum,
            curColAlign, curRowNum, firstChunk, beta, invBeta, linearBeta, invLinearBeta);
    }
}

template <typename T>
__aicore__ inline void SituGluGroupGradDispatcher(const LocalTensor<T>& gradGateLocal,
                                                  const LocalTensor<T>& gradUpLocal, const LocalTensor<T>& gradYLocal,
                                                  const LocalTensor<T>& gateLocal, const LocalTensor<T>& upLocal,
                                                  const LocalTensor<float>& weightLocal,
                                                  const LocalTensor<float>& gradWAccLocal, uint16_t curRowNum,
                                                  uint32_t curColNum, bool firstChunk, float beta, float linearBeta,
                                                  int32_t maskBit)
{
    if (maskBit == 0) {
        VFSituGrad<T, false, false>(gradGateLocal, gradUpLocal, gradYLocal, gateLocal, upLocal, weightLocal,
                                    gradWAccLocal, curRowNum, curColNum, firstChunk, beta, linearBeta);
    } else if (maskBit == MASK_HAS_WEIGHT) {
        VFSituGrad<T, true, false>(gradGateLocal, gradUpLocal, gradYLocal, gateLocal, upLocal, weightLocal,
                                   gradWAccLocal, curRowNum, curColNum, firstChunk, beta, linearBeta);
    } else if (maskBit == MASK_HAS_LINEAR_BETA) {
        VFSituGrad<T, false, true>(gradGateLocal, gradUpLocal, gradYLocal, gateLocal, upLocal, weightLocal,
                                   gradWAccLocal, curRowNum, curColNum, firstChunk, beta, linearBeta);
    } else {
        VFSituGrad<T, true, true>(gradGateLocal, gradUpLocal, gradYLocal, gateLocal, upLocal, weightLocal,
                                  gradWAccLocal, curRowNum, curColNum, firstChunk, beta, linearBeta);
    }
}

// group_index (count mode) helpers, mirrored from the SituGluGroup forward: realBs = min(sum(group_index), bs).
template <typename T, bool withUbReduce = false>
__simd_vf__ inline void VFProcessGroupIndexSmallVf(__ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t curColNum,
                                                   uint16_t vlLen, uint16_t loopCount)
{
    RegTensor<T> x;
    RegTensor<T> sum;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum, static_cast<T>(0), pregMain);
    uint32_t sreg = curColNum;
    MaskReg pregLoop;
    for (uint16_t i = 0; i < loopCount; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x, xLocalAddr + i * vlLen);
        Adds(x, x, static_cast<T>(0), pregLoop);
        Add(sum, sum, x, pregMain);
    }
    Reduce<ReduceType::SUM>(sum, sum, pregMain);
    if constexpr (withUbReduce) {
        RegTensor<T> origin;
        LoadAlign(origin, yLocalAddr);
        Add(sum, sum, origin, pregMerge);
    }
    StoreAlign(yLocalAddr, sum, pregMerge);
}

template <typename T, bool withUbReduce = false>
__simd_vf__ inline void VFProcessGroupIndexLargeVf(__ubuf__ T* yLocalAddr, __ubuf__ T* xLocalAddr, uint16_t vlLen,
                                                   uint16_t fourLoopCount, uint16_t tailLoopNum, uint32_t tailReminder)
{
    RegTensor<T> x0;
    RegTensor<T> x1;
    RegTensor<T> x2;
    RegTensor<T> x3;
    RegTensor<T> sum0;
    RegTensor<T> sum1;
    RegTensor<T> sum2;
    RegTensor<T> sum3;
    MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    MaskReg pregMerge = CreateMask<T, AscendC::Reg::MaskPattern::VL1>();
    Duplicate(sum0, static_cast<T>(0), pregMain);
    Duplicate(sum1, static_cast<T>(0), pregMain);
    Duplicate(sum2, static_cast<T>(0), pregMain);
    Duplicate(sum3, static_cast<T>(0), pregMain);
    MaskReg pregLoop;
    for (uint16_t i = 0; i < fourLoopCount; i++) {
        LoadAlign(x0, xLocalAddr + i * FOUR_UNFOLD * vlLen);
        Add(sum0, sum0, x0, pregMain);
        LoadAlign(x1, xLocalAddr + (i * FOUR_UNFOLD + 1) * vlLen);
        Add(sum1, sum1, x1, pregMain);
        LoadAlign(x2, xLocalAddr + (i * FOUR_UNFOLD + 2) * vlLen);
        Add(sum2, sum2, x2, pregMain);
        LoadAlign(x3, xLocalAddr + (i * FOUR_UNFOLD + 3) * vlLen);
        Add(sum3, sum3, x3, pregMain);
    }
    uint32_t sreg = tailReminder;
    for (uint16_t i = 0; i < tailLoopNum; i++) {
        pregLoop = UpdateMask<T>(sreg);
        LoadAlign(x0, xLocalAddr + (fourLoopCount * FOUR_UNFOLD + i) * vlLen);
        Adds(x0, x0, static_cast<T>(0), pregLoop);
        Add(sum0, sum0, x0, pregMain);
    }
    Add(sum0, sum0, sum1, pregMain);
    Add(sum2, sum2, sum3, pregMain);
    Add(sum0, sum0, sum2, pregMain);
    Reduce<ReduceType::SUM>(sum0, sum0, pregMain);
    if constexpr (withUbReduce) {
        RegTensor<T> origin;
        LoadAlign(origin, yLocalAddr);
        Add(sum0, sum0, origin, pregMerge);
    }
    StoreAlign(yLocalAddr, sum0, pregMerge);
}

template <typename T, bool withUbReduce = false>
__aicore__ inline void VFProcessGroupIndex(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                           uint16_t curColNum)
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
        AscendC::VF_CALL<VFProcessGroupIndexSmallVf<T, withUbReduce>>(yLocalAddr, xLocalAddr, curColNum, vlLen,
                                                                      loopCount);
    } else {
        AscendC::VF_CALL<VFProcessGroupIndexLargeVf<T, withUbReduce>>(yLocalAddr, xLocalAddr, vlLen, fourLoopCount,
                                                                      tailLoopNum, tailReminder);
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

__aicore__ inline void SetDefaultBlockTiling(const SituGluGroupGradTilingData* tilingData, int64_t& usedCoreNums,
                                             int64_t& rowOfFormerBlock, int64_t& rowOfTailBlock,
                                             int64_t& rowLoopOfFormerBlock, int64_t& rowLoopOfTailBlock,
                                             int64_t& tailRowFactorOfFormerBlock, int64_t& tailRowFactorOfTailBlock)
{
    rowOfFormerBlock = tilingData->rowOfFormerBlock;
    rowOfTailBlock = tilingData->rowOfTailBlock;
    rowLoopOfFormerBlock = tilingData->rowLoopOfFormerBlock;
    rowLoopOfTailBlock = tilingData->rowLoopOfTailBlock;
    tailRowFactorOfFormerBlock = tilingData->tailRowFactorOfFormerBlock;
    tailRowFactorOfTailBlock = tilingData->tailRowFactorOfTailBlock;
    usedCoreNums = GetBlockNum();
}

__aicore__ inline void SetGroupIndexBlockTiling(const SituGluGroupGradTilingData* tilingData, int64_t realBs,
                                                int64_t& usedCoreNums, int64_t& rowOfFormerBlock,
                                                int64_t& rowOfTailBlock, int64_t& rowLoopOfFormerBlock,
                                                int64_t& rowLoopOfTailBlock, int64_t& tailRowFactorOfFormerBlock,
                                                int64_t& tailRowFactorOfTailBlock)
{
    if (realBs <= 0) {
        usedCoreNums = 0;
        return;
    }

    // Split over the block count actually launched (GetBlockNum()), not tilingData->coreNum.
    // Host sets blockDim = coreNum, but the runtime may enable fewer blocks (core limiting /
    // fused scheduling); rows assigned to blocks that do not exist would be silently skipped.
    // This also keeps the core-count source identical to SetDefaultBlockTiling.
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
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

template <typename TBufPoolType>
__aicore__ inline void ProcessGroupIndexTiling(GM_ADDR groupIndex, const SituGluGroupGradTilingData* tilingData,
                                               TBufPoolType& tBufPool, TQue<QuePosition::VECIN, 1>& groupIndexQue,
                                               TBuf<QuePosition::VECCALC>& groupIndexSumBuf,
                                               GlobalTensor<int64_t>& groupIndexGm, LocalTensor<int64_t>& groupSumLocal,
                                               int64_t& usedCoreNums, int64_t& rowOfFormerBlock,
                                               int64_t& rowOfTailBlock, int64_t& rowLoopOfFormerBlock,
                                               int64_t& rowLoopOfTailBlock, int64_t& tailRowFactorOfFormerBlock,
                                               int64_t& tailRowFactorOfTailBlock, int64_t& validRows)
{
    // Follow the host-side presence decision: a non-null pointer with a null shape is treated as
    // absent by Tiling (no groupIndexLoop/groupIndexFactor), so the kernel must take the default
    // path instead of reading an uninitialized group sum.
    if (tilingData->hasGroupIndex == 0) {
        SetDefaultBlockTiling(tilingData, usedCoreNums, rowOfFormerBlock, rowOfTailBlock, rowLoopOfFormerBlock,
                              rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);
        validRows = tilingData->bs;
        return;
    }
    if (groupIndex == nullptr) {
        // Tiling said group_index is present but the runtime pointer is null (inconsistent state).
        // Do not fall back to GetBlockNum(): with group_index the blockDim equals coreNum while the
        // row tiling is based on usedCoreNums, so treating all blocks as active would read/write
        // past bs. Compute nothing instead of risking an out-of-bounds access.
        usedCoreNums = 0;
        validRows = 0;
        return;
    }

    groupIndexGm.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
    int64_t groupRows = 0;
    if (tilingData->groupListType == GROUP_LIST_TYPE_COUNT) {
        // count mode: group_index holds per-group row counts, so the total is their sum.
        tBufPool.InitBuffer(groupIndexQue, DOUBLE_BUFFER_NUM,
                            RoundUp<int64_t>(tilingData->groupIndexFactor) * sizeof(int64_t));
        tBufPool.InitBuffer(groupIndexSumBuf, BLOCK_SIZE);
        groupSumLocal = groupIndexSumBuf.Get<int64_t>();
        // Ascend950 provides Mutex; use it for the V->S edge instead of SetFlag/WaitFlag<V_S>.
        MutexID mutexId = AscendC::AllocMutexID();
        AscendC::Mutex::Lock<PIPE_V>(mutexId);
        for (int64_t idx = 0; idx < tilingData->groupIndexLoop; idx++) {
            int64_t curGroupIndexFactor = (idx == tilingData->groupIndexLoop - 1) ? tilingData->tailGroupIndexFactor :
                                                                                    tilingData->groupIndexFactor;
            LocalTensor<int64_t> groupIndexLocal = groupIndexQue.template AllocTensor<int64_t>();
            CopyIn(groupIndexGm[idx * tilingData->groupIndexFactor], groupIndexLocal, 1, curGroupIndexFactor);
            groupIndexQue.template EnQue(groupIndexLocal);
            groupIndexLocal = groupIndexQue.template DeQue<int64_t>();
            if (idx == 0) {
                VFProcessGroupIndex<int64_t, false>(groupSumLocal, groupIndexLocal, curGroupIndexFactor);
            } else {
                VFProcessGroupIndex<int64_t, true>(groupSumLocal, groupIndexLocal, curGroupIndexFactor);
            }
            groupIndexQue.template FreeTensor(groupIndexLocal);
        }
        AscendC::Mutex::Unlock<PIPE_V>(mutexId);
        AscendC::Mutex::Lock<PIPE_S>(mutexId);
        groupRows = groupSumLocal.GetValue(0);
        AscendC::Mutex::Unlock<PIPE_S>(mutexId);
        AscendC::ReleaseMutexID(mutexId);
        tBufPool.Reset();
    } else {
        // cumsum mode (group_list_type 0, the default): group_index holds cumulative row counts, so
        // its last element already is the total. One scalar GM read replaces the chunked VF reduce,
        // hence no UB buffer and no V->S handshake. Tiling rejects an empty group_index, so the
        // index below is never negative.
        groupRows = groupIndexGm.GetValue(static_cast<uint64_t>(tilingData->groupIndexElemNum - 1));
    }
    int64_t realBs = groupRows > tilingData->bs ? tilingData->bs : groupRows;
    // Rows the op owns an output for: [0, realBs) with group_index, [0, bs) without. Rows past it
    // are padding and are deliberately left untouched, so validRows never exceeds them.
    validRows = realBs > 0 ? realBs : 0;
    SetGroupIndexBlockTiling(tilingData, realBs, usedCoreNums, rowOfFormerBlock, rowOfTailBlock, rowLoopOfFormerBlock,
                             rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);
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

template <typename T>
class SituGluGroupGradKernel {
public:
    __aicore__ inline SituGluGroupGradKernel() {}

    __aicore__ inline void Init(GM_ADDR gradY, GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR gradX,
                                GM_ADDR gradWeight, const SituGluGroupGradTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe_ = pipePtr;
        tilingData_ = tilingDataPtr;

        gradYGm.SetGlobalBuffer((__gm__ T*)gradY);
        xGm.SetGlobalBuffer((__gm__ T*)x);
        gradXGm.SetGlobalBuffer((__gm__ T*)gradX);

        pipe_->InitBufPool(tBufPool, tilingData_->ubSize);
        ProcessGroupIndexTiling(groupIndex, tilingData_, tBufPool, groupIndexQue, groupIndexSumBuf, groupIndexGm,
                                groupSumLocal, usedCoreNums_, rowOfFormerBlock_, rowOfTailBlock_, rowLoopOfFormerBlock_,
                                rowLoopOfTailBlock_, tailRowFactorOfFormerBlock_, tailRowFactorOfTailBlock_,
                                validRows_);

        // Follow the Tiling decision: Tiling sets hasWeight only when the weight shape is present
        // and included in the UB budget. The kernel still guards the pointer so an inconsistent
        // present-but-null weight can never reach DataCopyPad with a null GM address.
        if (tilingData_->hasWeight != 0) {
            // grad_weight is a required output whenever weight is present, so weight and grad_weight
            // are either both valid or the runtime state is inconsistent. Guard both together: a
            // present-but-null pointer would otherwise reach DataCopyPad / the VF grad_weight
            // accumulator with a null GM address or an uninitialized UB buffer, so make it a
            // deterministic no-op instead. This also keeps hasWeight_ == hasGradW_, which the VF
            // dispatcher relies on (the grad_weight reduce shares the MASK_HAS_WEIGHT path).
            if (weight == nullptr || gradWeight == nullptr) {
                usedCoreNums_ = 0;
                return;
            }
            hasWeight_ = true;
            hasGradW_ = true;
            weightGm.SetGlobalBuffer((__gm__ float*)weight);
            gradWeightGm.SetGlobalBuffer((__gm__ float*)gradWeight);
            tBufPool.InitBuffer(weightQue, DOUBLE_BUFFER_NUM, RoundUp<float>(tilingData_->rowFactor) * sizeof(float));
            // grad_weight accumulator (one float per row of the tile). It is a queued VECOUT tensor so the
            // MTE3 grad_weight copy-out is ordered against the vector writes of the next tile; a raw TBuf
            // here let the next tile's VF overwrite the accumulator while DataCopyPad was still reading it
            // (the leading rows of every tile then held the next tile's grad_weight values). A single
            // buffer keeps the UB budget identical to the former raw TBuf; AllocTensor waits for the
            // previous copy-out, so the tile-to-tile overlap the double buffer would allow is not needed.
            // d == 0 (splitD == 0) never reaches the accumulator: Process exits early and ZeroFillGradWeight
            // reuses this buffer as the zero source instead, so it is sized for that fill. The UB budget is
            // unaffected because Tiling pins rowFactor/dFactor to 1 when d == 0, leaving every tile queue at
            // a single cacheline.
            int64_t gradWAccRows = RoundUp<float>(tilingData_->rowFactor);
            if (tilingData_->splitD == 0 && validRows_ > 0) {
                const int64_t zeroFillRows = validRows_ < GRADW_ZERO_FILL_ROWS ? validRows_ : GRADW_ZERO_FILL_ROWS;
                gradWZeroChunkRows_ = RoundUp<float>(zeroFillRows);
                gradWAccRows = gradWZeroChunkRows_;
            }
            tBufPool.InitBuffer(gradWAccQue, 1, gradWAccRows * sizeof(float));
        }

        const int64_t tileBufElems = tilingData_->rowFactor * RoundUp<T>(tilingData_->dFactor);
        tBufPool.InitBuffer(gradYQue, DOUBLE_BUFFER_NUM, tileBufElems * sizeof(T));
        tBufPool.InitBuffer(gateQue, DOUBLE_BUFFER_NUM, tileBufElems * sizeof(T));
        tBufPool.InitBuffer(upQue, DOUBLE_BUFFER_NUM, tileBufElems * sizeof(T));
        tBufPool.InitBuffer(gradGateQue, DOUBLE_BUFFER_NUM, tileBufElems * sizeof(T));
        tBufPool.InitBuffer(gradUpQue, DOUBLE_BUFFER_NUM, tileBufElems * sizeof(T));
        hasLinearBeta_ = (tilingData_->linearBeta > 0.0f);
        // activateLeft == 1: gate is the front half [0, splitD); otherwise gate is the back half.
        gateHalfOffset_ = (tilingData_->activateLeft == 1) ? 0 : tilingData_->splitD;
        upHalfOffset_ = (tilingData_->activateLeft == 1) ? tilingData_->splitD : 0;
    }

    __aicore__ inline void Process()
    {
        if (tilingData_->bs == 0 || tilingData_->splitD == 0) {
            // Empty input (zero row count or zero cut-dim extent): nothing to compute. grad_weight is
            // still owed zeros when d == 0 leaves rows behind (bs == 0 makes it empty as well).
            ZeroFillGradWeight();
            return;
        }
        if (GetBlockIdx() >= usedCoreNums_) {
            return;
        }
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop = (curBlockIdx == usedCoreNums_ - 1) ? rowLoopOfTailBlock_ : rowLoopOfFormerBlock_;
        int64_t tailRowFactor = (curBlockIdx == usedCoreNums_ - 1) ? tailRowFactorOfTailBlock_ :
                                                                     tailRowFactorOfFormerBlock_;
        int64_t rowGmBase = curBlockIdx * rowOfFormerBlock_;
        // x / grad_x are d wide per row; grad_y is splitD wide per row.
        int64_t xGmBaseOffset = rowGmBase * tilingData_->d;
        int64_t gradYGmBaseOffset = rowGmBase * tilingData_->splitD;
        int64_t weightGmBaseOffset = rowGmBase;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData_->rowFactor;
            int64_t rowBase = rowOuterIdx * tilingData_->rowFactor;
            if (hasWeight_) {
                weightLocal = weightQue.template AllocTensor<float>();
                CopyIn(weightGm[weightGmBaseOffset + rowBase], weightLocal, 1, curRowFactor);
                weightQue.template EnQue(weightLocal);
                weightLocal = weightQue.template DeQue<float>();
            }
            if (hasGradW_) {
                // The accumulator is allocated once per tile so the d-chunk loop can accumulate into it;
                // AllocTensor waits for the previous tile's grad_weight copy-out to release the buffer.
                gradWAccLocal = gradWAccQue.template AllocTensor<float>();
            }

            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData_->dLoop; dLoopIdx++) {
                int64_t curDFactor = (dLoopIdx == tilingData_->dLoop - 1) ? tilingData_->tailDFactor :
                                                                            tilingData_->dFactor;
                int64_t dColOffset = dLoopIdx * tilingData_->dFactor;
                int64_t xRowOffset = rowBase * tilingData_->d + dColOffset;
                int64_t gradYRowOffset = rowBase * tilingData_->splitD + dColOffset;

                gradYLocal = gradYQue.template AllocTensor<T>();
                CopyIn(gradYGm[gradYGmBaseOffset + gradYRowOffset], gradYLocal, curRowFactor, curDFactor,
                       tilingData_->splitD - curDFactor);
                gradYQue.template EnQue(gradYLocal);
                gradYLocal = gradYQue.template DeQue<T>();

                gateLocal = gateQue.template AllocTensor<T>();
                CopyIn(xGm[xGmBaseOffset + xRowOffset + gateHalfOffset_], gateLocal, curRowFactor, curDFactor,
                       tilingData_->d - curDFactor);
                gateQue.template EnQue(gateLocal);
                gateLocal = gateQue.template DeQue<T>();

                upLocal = upQue.template AllocTensor<T>();
                CopyIn(xGm[xGmBaseOffset + xRowOffset + upHalfOffset_], upLocal, curRowFactor, curDFactor,
                       tilingData_->d - curDFactor);
                upQue.template EnQue(upLocal);
                upLocal = upQue.template DeQue<T>();

                gradGateLocal = gradGateQue.template AllocTensor<T>();
                gradUpLocal = gradUpQue.template AllocTensor<T>();

                ComputeVf(gradGateLocal, gradUpLocal, gradYLocal, gateLocal, upLocal, weightLocal, gradWAccLocal,
                          curRowFactor, curDFactor, dLoopIdx == 0);

                gradYQue.template FreeTensor(gradYLocal);
                gateQue.template FreeTensor(gateLocal);
                upQue.template FreeTensor(upLocal);

                gradGateQue.template EnQue(gradGateLocal);
                gradGateLocal = gradGateQue.template DeQue<T>();
                CopyOut(gradGateLocal, gradXGm[xGmBaseOffset + xRowOffset + gateHalfOffset_], curRowFactor, curDFactor,
                        tilingData_->d - curDFactor);
                gradGateQue.template FreeTensor(gradGateLocal);

                gradUpQue.template EnQue(gradUpLocal);
                gradUpLocal = gradUpQue.template DeQue<T>();
                CopyOut(gradUpLocal, gradXGm[xGmBaseOffset + xRowOffset + upHalfOffset_], curRowFactor, curDFactor,
                        tilingData_->d - curDFactor);
                gradUpQue.template FreeTensor(gradUpLocal);
            }

            if (hasGradW_) {
                gradWAccQue.template EnQue(gradWAccLocal);
                gradWAccLocal = gradWAccQue.template DeQue<float>();
                CopyOut(gradWAccLocal, gradWeightGm[weightGmBaseOffset + rowBase], 1, curRowFactor);
                gradWAccQue.template FreeTensor(gradWAccLocal);
            }
            if (hasWeight_) {
                weightQue.template FreeTensor(weightLocal);
            }
        }
    }

    // d == 0 makes grad_weight[n] = sum over an empty cut dim, i.e. 0, for every valid row. The main
    // loop cannot produce it (there is no data to load), so the zeros are written here. The rows are
    // partitioned over the blocks actually launched instead of reusing the row tiling, which Tiling
    // degenerates to 0 for the empty case.
    __aicore__ inline void ZeroFillGradWeight()
    {
        if (gradWZeroChunkRows_ == 0) {
            return;
        }
        const int64_t rowsPerBlock = CeilDiv(validRows_, static_cast<int64_t>(GetBlockNum()));
        const int64_t rowBegin = static_cast<int64_t>(GetBlockIdx()) * rowsPerBlock;
        if (rowBegin >= validRows_) {
            return;
        }
        const int64_t blockRowEnd = rowBegin + rowsPerBlock;
        const int64_t rowLimit = blockRowEnd < validRows_ ? blockRowEnd : validRows_;
        for (int64_t rowBase = rowBegin; rowBase < rowLimit; rowBase += gradWZeroChunkRows_) {
            const int64_t rowsLeft = rowLimit - rowBase;
            const int64_t curRows = rowsLeft < gradWZeroChunkRows_ ? rowsLeft : gradWZeroChunkRows_;
            gradWAccLocal = gradWAccQue.template AllocTensor<float>();
            Duplicate(gradWAccLocal, 0.0f, static_cast<int32_t>(curRows));
            gradWAccQue.template EnQue(gradWAccLocal);
            gradWAccLocal = gradWAccQue.template DeQue<float>();
            CopyOut(gradWAccLocal, gradWeightGm[rowBase], 1, static_cast<uint32_t>(curRows));
            gradWAccQue.template FreeTensor(gradWAccLocal);
        }
    }

    __aicore__ inline void ComputeVf(const LocalTensor<T>& gradGateLocal, const LocalTensor<T>& gradUpLocal,
                                     const LocalTensor<T>& gradYLocal, const LocalTensor<T>& gateLocal,
                                     const LocalTensor<T>& upLocal, const LocalTensor<float>& weightLocal,
                                     const LocalTensor<float>& gradWAccLocal, int64_t curRowFactor, int64_t curDFactor,
                                     bool firstChunk)
    {
        int32_t maskBit = (hasLinearBeta_ ? MASK_HAS_LINEAR_BETA : 0) | (hasWeight_ ? MASK_HAS_WEIGHT : 0);
        SituGluGroupGradDispatcher<T>(gradGateLocal, gradUpLocal, gradYLocal, gateLocal, upLocal, weightLocal,
                                      gradWAccLocal, static_cast<uint16_t>(curRowFactor),
                                      static_cast<uint32_t>(curDFactor), firstChunk, tilingData_->beta,
                                      tilingData_->linearBeta, maskBit);
    }

private:
    TPipe* pipe_;
    const SituGluGroupGradTilingData* tilingData_;
    GlobalTensor<T> gradYGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> gradXGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<float> gradWeightGm;
    GlobalTensor<int64_t> groupIndexGm;

    TQue<QuePosition::VECIN, 1> gradYQue;
    TQue<QuePosition::VECIN, 1> gateQue;
    TQue<QuePosition::VECIN, 1> upQue;
    TQue<QuePosition::VECOUT, 1> gradGateQue;
    TQue<QuePosition::VECOUT, 1> gradUpQue;
    TQue<QuePosition::VECIN, 1> weightQue;

    TQue<QuePosition::VECIN, 1> groupIndexQue;
    TBuf<QuePosition::VECCALC> groupIndexSumBuf;
    TQue<QuePosition::VECOUT, 1> gradWAccQue;
    TBufPool<QuePosition::VECCALC, TBUF_POOL_DEPTH> tBufPool;

    LocalTensor<T> gradYLocal;
    LocalTensor<T> gateLocal;
    LocalTensor<T> upLocal;
    LocalTensor<T> gradGateLocal;
    LocalTensor<T> gradUpLocal;
    LocalTensor<float> weightLocal;
    LocalTensor<float> gradWAccLocal;

    LocalTensor<int64_t> groupSumLocal;

    bool hasWeight_ = false;
    bool hasGradW_ = false;
    bool hasLinearBeta_ = false;
    int64_t gateHalfOffset_ = 0;
    int64_t upHalfOffset_ = 0;

    int64_t tailRowFactorOfTailBlock_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t usedCoreNums_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    // Rows grad_weight owes a value for: bs, or realBs = min(groupRows, bs) when group_index is present.
    int64_t validRows_ = 0;
    // Zero-fill chunk rows for the d == 0 early exit; 0 when no zeros are owed.
    int64_t gradWZeroChunkRows_ = 0;
};

} // namespace SituGluGroupGradOp

#endif // SITU_GLU_GROUP_GRAD_KERNEL_H
