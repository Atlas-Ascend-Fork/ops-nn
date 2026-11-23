/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file apply_ftrl_dag.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_APPLY_FTRL_DAG_H
#define CANN_CUSTOM_OPS_APPLY_FTRL_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifdef __CCE_AICORE__
#include "kernel_operator.h"
#endif

namespace ApplyFtrlOp {
using namespace Ops::Base;

#ifdef __CCE_AICORE__
namespace PowerDetail {
namespace R = AscendC::Reg;
using F = R::RegTensor<float>;
using M = R::MaskReg;
constexpr R::DivSpecificMode DIV = {R::MaskMergeMode::ZEROING, true, AscendC::DivAlgo::DIFF_COMPENSATION};
// These algebraic identities use SDK arithmetic instead of a general power
// approximation. Keeping x^1 exact also preserves finite values near FLT_MAX.
__simd_vf__ inline void CanonicalPower(__ubuf__ float* dst, __ubuf__ float* base, __ubuf__ float* exponent,
                                       __ubuf__ float* sdkPower, uint32_t count)
{
    constexpr uint32_t LANES = AscendC::GetVecLen() / sizeof(float);
    const uint16_t repeats = static_cast<uint16_t>((count + LANES - 1) / LANES);
    for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
        const uint32_t offset = repeat * LANES;
        uint32_t remaining = count - offset;
        M mask = R::UpdateMask<float>(remaining);
        M identity, reciprocal;
        F a, p, power, one, inverse, result;
        R::LoadAlign(a, base + offset);
        R::LoadAlign(p, exponent + offset);
        R::LoadAlign(power, sdkPower + offset);
        R::Duplicate(one, 1.0f, mask);
        R::Div<float, &DIV>(inverse, one, a, mask);
        R::CompareScalar<float, AscendC::CMPMODE::EQ>(reciprocal, p, -1.0f, mask);
        R::Select(result, inverse, power, reciprocal);
        R::CompareScalar<float, AscendC::CMPMODE::EQ>(identity, p, 1.0f, mask);
        R::Select(result, a, result, identity);
        R::StoreAlign(dst + offset, result, mask);
    }
}

} // namespace PowerDetail

// Scalar widening that is bit-identical to Vec::Cast<T, U, CAST_MODE_NONE> applied to a
// broadcast tile: bf16 must go through ToFloat (bisheng rejects the direct static_cast),
// while half -> float and float -> float are exact.
template <typename T, typename U>
__aicore__ inline T ScalarToCompute(U value)
{
    if constexpr (AscendC::IsSameType<bfloat16_t, U>::value) {
        return AscendC::ToFloat(value);
    } else {
        return static_cast<T>(value);
    }
}
#endif

template <class T>
struct CanonicalPower : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline CanonicalPower(LocalTensor<T>& dst, LocalTensor<T>& base, LocalTensor<T>& exponent,
                                     LocalTensor<T>& sdkPower, uint32_t count)
    {
#ifdef __CCE_AICORE__
        static_assert(std::is_same<T, float>::value, "ApplyFtrl power uses FP32 intermediates");
        asc_vf_call<PowerDetail::CanonicalPower>((__ubuf__ float*)dst.GetPhyAddr(), (__ubuf__ float*)base.GetPhyAddr(),
                                                 (__ubuf__ float*)exponent.GetPhyAddr(),
                                                 (__ubuf__ float*)sdkPower.GetPhyAddr(), count);
#endif
    }
};

// ---------------------------------------------------------------------------
// Fused hand-written VF nodes (register-resident, no intermediate LocalTensor).
//
// Each node reproduces the *exact* instruction sequence of the framework node
// chain it replaces; only the intermediate UB materialization is removed. All
// arithmetic instructions are IEEE-exact single operations, so the per-element
// result is bit-identical (0-ULP) to the unfused DAG. The precision-sensitive
// ops (Vec::Power, Vec::DivHighPrecision, Vec::Cast) intentionally stay as
// regular DAG nodes.
//
// Formula order is preserved end to end:
//   accum_new = accum + grad * grad
//   sigma     = (accum_new^-lr_power - accum^-lr_power) / lr
//   linear_new = linear + (grad - sigma * var)      <- (sigma * var) formed first
//   x_res      = sign(linear_new) * l1 - linear_new
//   quadratic  = accum_new^-lr_power / lr + 2 * l2
//   var        = |linear_new| > l1 ? x_res / quadratic : 0
// ---------------------------------------------------------------------------

// accum_new = accum + grad * grad
// replaces: Mul(grad, grad) + Add(accum, ...)              [2 nodes -> 1 node]
template <typename T>
struct CalcAccumNew : public Vec::ElemwiseBinaryOP<T, T, T> {
    __aicore__ inline CalcAccumNew(LocalTensor<T>& accumNew, LocalTensor<T>& accum, LocalTensor<T>& grad, int32_t count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t LANES = AscendC::GetVecLen() / sizeof(T);
        const uint32_t totalLen = static_cast<uint32_t>(count);
        const uint16_t repeats = static_cast<uint16_t>((totalLen + LANES - 1) / LANES);
        static_assert(std::is_same<T, float>::value, "ApplyFtrl fused VF nodes run in FP32");
        __ubuf__ T* accumAddr = (__ubuf__ T*)accum.GetPhyAddr();
        __ubuf__ T* gradAddr = (__ubuf__ T*)grad.GetPhyAddr();
        __ubuf__ T* outAddr = (__ubuf__ T*)accumNew.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg preg;
            AscendC::Reg::RegTensor<T> accumReg, gradReg, gradSquareReg, accumNewReg;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t offset = repeat * LANES;
                uint32_t remaining = totalLen - offset;
                preg = AscendC::Reg::UpdateMask<T>(remaining);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(accumReg, accumAddr + offset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(gradReg, gradAddr + offset);
                AscendC::Reg::Mul(gradSquareReg, gradReg, gradReg, preg);
                AscendC::Reg::Add(accumNewReg, accumReg, gradSquareReg, preg);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(outAddr + offset, accumNewReg, preg);
            }
        }
#endif
    }
};

// linear_new = linear + (grad - sigma * var)
// replaces: Mul(sigma, var) + Sub(grad, ...) + Add(linear, ...)   [3 nodes -> 1 node]
template <typename T>
struct CalcLinearNew : public Vec::ElemwiseQuaternaryOP<T, T, T, T, T> {
    __aicore__ inline CalcLinearNew(LocalTensor<T>& linearNew, LocalTensor<T>& linear, LocalTensor<T>& grad,
                                    LocalTensor<T>& sigma, LocalTensor<T>& var, int32_t count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t LANES = AscendC::GetVecLen() / sizeof(T);
        const uint32_t totalLen = static_cast<uint32_t>(count);
        const uint16_t repeats = static_cast<uint16_t>((totalLen + LANES - 1) / LANES);
        static_assert(std::is_same<T, float>::value, "ApplyFtrl fused VF nodes run in FP32");
        __ubuf__ T* linAddr = (__ubuf__ T*)linear.GetPhyAddr();
        __ubuf__ T* gradAddr = (__ubuf__ T*)grad.GetPhyAddr();
        __ubuf__ T* sigmaAddr = (__ubuf__ T*)sigma.GetPhyAddr();
        __ubuf__ T* varAddr = (__ubuf__ T*)var.GetPhyAddr();
        __ubuf__ T* outAddr = (__ubuf__ T*)linearNew.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg preg;
            AscendC::Reg::RegTensor<T> linReg, gradReg, sigmaReg, varReg, sigmaMulVarReg, gradSubReg, linearNewReg;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t offset = repeat * LANES;
                uint32_t remaining = totalLen - offset;
                preg = AscendC::Reg::UpdateMask<T>(remaining);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(linReg, linAddr + offset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(gradReg, gradAddr + offset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(sigmaReg, sigmaAddr + offset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(varReg, varAddr + offset);
                AscendC::Reg::Mul(sigmaMulVarReg, sigmaReg, varReg, preg);
                AscendC::Reg::Sub(gradSubReg, gradReg, sigmaMulVarReg, preg);
                AscendC::Reg::Add(linearNewReg, linReg, gradSubReg, preg);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(outAddr + offset, linearNewReg, preg);
            }
        }
#endif
    }
};

// x_res = sign(linearNew) * l1 - linearNew
// replaces: Duplicate(0) + Compare(Pos) + Compare(Neg) + Select(Pos) + Select(Neg)
//           + Sub(sign) + Mul(sign*l1) + Sub(x_res)             [8 nodes -> 1 node]
template <typename T, typename U = float>
struct CalcXRes : public Vec::ElemwiseBinaryOP<T, T, U> {
    __aicore__ inline CalcXRes(LocalTensor<T>& xRes, LocalTensor<T>& linearNew, U l1, int32_t count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t LANES = AscendC::GetVecLen() / sizeof(T);
        const uint32_t totalLen = static_cast<uint32_t>(count);
        const uint16_t repeats = static_cast<uint16_t>((totalLen + LANES - 1) / LANES);
        static_assert(std::is_same<T, float>::value, "ApplyFtrl fused VF nodes run in FP32");
        const T l1Up = ScalarToCompute<T, U>(l1);
        __ubuf__ T* linAddr = (__ubuf__ T*)linearNew.GetPhyAddr();
        __ubuf__ T* resAddr = (__ubuf__ T*)xRes.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg preg, cmpPos, cmpNeg;
            AscendC::Reg::RegTensor<T> lin, one, zero, signPos, signNeg, sign, res;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t offset = repeat * LANES;
                uint32_t remaining = totalLen - offset;
                preg = AscendC::Reg::UpdateMask<T>(remaining);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(lin, linAddr + offset);
                AscendC::Reg::CompareScalar<T, AscendC::CMPMODE::GT>(cmpPos, lin, 0.0f, preg);
                AscendC::Reg::CompareScalar<T, AscendC::CMPMODE::LT>(cmpNeg, lin, 0.0f, preg);
                AscendC::Reg::Duplicate(one, static_cast<T>(1), preg);
                AscendC::Reg::Duplicate(zero, static_cast<T>(0), preg);
                AscendC::Reg::Select(signPos, one, zero, cmpPos);
                AscendC::Reg::Select(signNeg, one, zero, cmpNeg);
                AscendC::Reg::Sub(sign, signPos, signNeg, preg);
                AscendC::Reg::Muls(sign, sign, l1Up, preg);
                AscendC::Reg::Sub(res, sign, lin, preg);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(resAddr + offset, res, preg);
            }
        }
#endif
    }
};

// quadratic = powNewDivLr + 2 * l2
// replaces: Muls(l2, 2) + Add(...)                          [2 nodes -> 1 node]
template <typename T, typename U = float>
struct CalcQuad : public Vec::ElemwiseBinaryOP<T, T, U> {
    __aicore__ inline CalcQuad(LocalTensor<T>& quad, LocalTensor<T>& powNewDivLr, U l2, int32_t count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t LANES = AscendC::GetVecLen() / sizeof(T);
        const uint32_t totalLen = static_cast<uint32_t>(count);
        const uint16_t repeats = static_cast<uint16_t>((totalLen + LANES - 1) / LANES);
        static_assert(std::is_same<T, float>::value, "ApplyFtrl fused VF nodes run in FP32");
        const T l2Up = ScalarToCompute<T, U>(l2);
        __ubuf__ T* divAddr = (__ubuf__ T*)powNewDivLr.GetPhyAddr();
        __ubuf__ T* quadAddr = (__ubuf__ T*)quad.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg preg;
            AscendC::Reg::RegTensor<T> divReg, l2Reg, quadReg;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t offset = repeat * LANES;
                uint32_t remaining = totalLen - offset;
                preg = AscendC::Reg::UpdateMask<T>(remaining);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(divReg, divAddr + offset);
                AscendC::Reg::Duplicate(l2Reg, l2Up, preg);
                AscendC::Reg::Muls(l2Reg, l2Reg, static_cast<T>(2), preg);
                AscendC::Reg::Add(quadReg, divReg, l2Reg, preg);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(quadAddr + offset, quadReg, preg);
            }
        }
#endif
    }
};

// var = (|linearNew| > l1) ? varTrue : 0
// replaces: Abs + CompareScalar(GT) + Select                  [3 nodes -> 1 node]
template <typename T, typename U = float>
struct CalcVarGate : public Vec::ElemwiseTernaryOP<T, T, T, U> {
    __aicore__ inline CalcVarGate(LocalTensor<T>& varOut, LocalTensor<T>& varTrue, LocalTensor<T>& linearNew, U l1,
                                  int32_t count)
    {
#ifdef __CCE_AICORE__
        constexpr uint32_t LANES = AscendC::GetVecLen() / sizeof(T);
        const uint32_t totalLen = static_cast<uint32_t>(count);
        const uint16_t repeats = static_cast<uint16_t>((totalLen + LANES - 1) / LANES);
        static_assert(std::is_same<T, float>::value, "ApplyFtrl fused VF nodes run in FP32");
        const T l1Up = ScalarToCompute<T, U>(l1);
        __ubuf__ T* trueAddr = (__ubuf__ T*)varTrue.GetPhyAddr();
        __ubuf__ T* linAddr = (__ubuf__ T*)linearNew.GetPhyAddr();
        __ubuf__ T* outAddr = (__ubuf__ T*)varOut.GetPhyAddr();
        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg preg, gate;
            AscendC::Reg::RegTensor<T> trueReg, lin, absLin, zero, out;
            for (uint16_t repeat = 0; repeat < repeats; ++repeat) {
                const uint32_t offset = repeat * LANES;
                uint32_t remaining = totalLen - offset;
                preg = AscendC::Reg::UpdateMask<T>(remaining);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(trueReg, trueAddr + offset);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_NORM>(lin, linAddr + offset);
                AscendC::Reg::Abs(absLin, lin, preg);
                AscendC::Reg::CompareScalar<T, AscendC::CMPMODE::GT>(gate, absLin, l1Up, preg);
                AscendC::Reg::Duplicate(zero, static_cast<T>(0), preg);
                AscendC::Reg::Select(out, trueReg, zero, gate);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_NORM>(outAddr + offset, out, preg);
            }
        }
#endif
    }
};

template <typename U, typename T = float>
struct ApplyFtrlDag {
    using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpCopyInLinear = Bind<Vec::CopyIn<U>, Placeholder::In2<U>>;
    using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;
    using OpCopyInLr = Bind<Vec::Duplicate<U>, Placeholder::In4<U, Placeholder::ScalarAttr<true>>>;
    using OpCopyInLrPower = Bind<Vec::Duplicate<U>, Placeholder::In7<U, Placeholder::ScalarAttr<true>>>;
    // lr / lr_power keep the promoted tensor path: Vec::DivHighPrecision and Vec::Power
    // require tensor operands, and negating a NaN exponent through Muls must be preserved
    // for 0-ULP. l1 / l2 are consumed as raw scalars by the fused VF nodes below, so their
    // Duplicate + Cast tiles are dropped (bit-identical U -> FP32 widening in ScalarToCompute).

    using OpVarCast = Bind<Vec::Cast<T, U, 0>, OpCopyInVar>;
    using OpAccumCast = Bind<Vec::Cast<T, U, 0>, OpCopyInAccum>;
    using OpLinearCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLinear>;
    using OpGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyInGrad>;
    using OpLrCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
    using OpLrPowerCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLrPower>;

    using OpAccumNew = Bind<CalcAccumNew<T>, OpAccumCast, OpGradCast>;
    using OpAccumOutCast = Bind<Vec::Cast<U, T, 1>, OpAccumNew>;
    using OpCopyOutAccum = Bind<Vec::CopyOut<U>, Placeholder::Out1<U>, OpAccumOutCast>; // update input1: accum

    using OpNegOne = MAKE_CONST(T, -1);
    using OpIndex = Bind<Vec::Muls<T>, OpLrPowerCast, OpNegOne>;
    using OpAccNewSdkPower = Bind<Vec::Power<T>, OpAccumNew, OpIndex>;
    using OpAccSdkPower = Bind<Vec::Power<T>, OpAccumCast, OpIndex>;
    using OpAccNewPower = Bind<CanonicalPower<T>, OpAccumNew, OpIndex, OpAccNewSdkPower>;
    using OpAccPower = Bind<CanonicalPower<T>, OpAccumCast, OpIndex, OpAccSdkPower>;
    using OpAccSub = Bind<Vec::Sub<T>, OpAccNewPower, OpAccPower>;
    // Compute sigma before multiplying by var, as in the FTRL update formula.
    // This avoids forming the potentially overflowing product (power_new - power_old) * var.
    using OpSigma = Bind<Vec::DivHighPrecision<T>, OpAccSub, OpLrCast>;
    // linear + (grad - sigma * var): sigma * var is formed first, same as the framework order.
    using OpLinearNew = Bind<CalcLinearNew<T>, OpLinearCast, OpGradCast, OpSigma, OpVarCast>;
    using OpLinearOutCast = Bind<Vec::Cast<U, T, 1>, OpLinearNew>;
    using OpCopyOutLinear = Bind<Vec::CopyOut<U>, Placeholder::Out2<U>, OpLinearOutCast>; // update input2: linear

    // x_res = sign(linearNew) * l1 - linearNew  (was 8 separate DAG nodes)
    using OpXRes = Bind<CalcXRes<T, U>, OpLinearNew, Placeholder::In5<U, Placeholder::ScalarAttr<true>>>;
    using OpAccNewPowerDiv = Bind<Vec::DivHighPrecision<T>, OpAccNewPower, OpLrCast>;
    // quadratic = powNew / lr + 2 * l2  (was 2 separate DAG nodes)
    using OpQuad = Bind<CalcQuad<T, U>, OpAccNewPowerDiv, Placeholder::In6<U, Placeholder::ScalarAttr<true>>>;
    using OpVarSelectTrue = Bind<Vec::DivHighPrecision<T>, OpXRes, OpQuad>;
    // var = (|linearNew| > l1) ? varTrue : 0  (was 3 separate DAG nodes)
    using OpVarNew = Bind<CalcVarGate<T, U>, OpVarSelectTrue, OpLinearNew,
                          Placeholder::In5<U, Placeholder::ScalarAttr<true>>>;

    using OpVarOutCast = Bind<Vec::Cast<U, T, 1>, OpVarNew>;
    using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>; // output: var

    using Outputs = Elems<OpCopyOutVar, OpCopyOutAccum, OpCopyOutLinear>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ApplyFtrlOp

#endif // CANN_CUSTOM_OPS_APPLY_FTRL_DAG_H
