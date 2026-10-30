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

namespace ApplyFtrlOp {
using namespace Ops::Base;
const int COMPARE_MODE_LT = 0;
const int COMPARE_MODE_GT = 1;
const int SELECT_MODE_T_S = 1;

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

template <typename U, typename T = float>
struct ApplyFtrlDag {
    using OpCopyInVar = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCopyInAccum = Bind<Vec::CopyIn<U>, Placeholder::In1<U>>;
    using OpCopyInLinear = Bind<Vec::CopyIn<U>, Placeholder::In2<U>>;
    using OpCopyInGrad = Bind<Vec::CopyIn<U>, Placeholder::In3<U>>;
    using OpCopyInLr = Bind<Vec::Duplicate<U>, Placeholder::In4<U, Placeholder::ScalarAttr<true>>>;
    // Keep hyperparameters on the same promoted tensor path for every input dtype.
    using OpCopyInL1 = Bind<Vec::Duplicate<U>, Placeholder::In5<U, Placeholder::ScalarAttr<true>>>;
    using OpCopyInL2 = Bind<Vec::Duplicate<U>, Placeholder::In6<U, Placeholder::ScalarAttr<true>>>;
    using OpCopyInLrPower = Bind<Vec::Duplicate<U>, Placeholder::In7<U, Placeholder::ScalarAttr<true>>>;

    using OpVarCast = Bind<Vec::Cast<T, U, 0>, OpCopyInVar>;
    using OpAccumCast = Bind<Vec::Cast<T, U, 0>, OpCopyInAccum>;
    using OpLinearCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLinear>;
    using OpGradCast = Bind<Vec::Cast<T, U, 0>, OpCopyInGrad>;
    using OpLrCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
    using OpL1Cast = Bind<Vec::Cast<T, U, 0>, OpCopyInL1>;
    using OpL2Cast = Bind<Vec::Cast<T, U, 0>, OpCopyInL2>;
    using OpLrPowerCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLrPower>;

    using OpGradSquare = Bind<Vec::Mul<T>, OpGradCast, OpGradCast>;
    using OpAccumNew = Bind<Vec::Add<T>, OpAccumCast, OpGradSquare>;
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
    using OpSigmaMulVar = Bind<Vec::Mul<T>, OpSigma, OpVarCast>;
    using OpGradSub = Bind<Vec::Sub<T>, OpGradCast, OpSigmaMulVar>;
    using OpLinearNew = Bind<Vec::Add<T>, OpLinearCast, OpGradSub>;
    using OpLinearOutCast = Bind<Vec::Cast<U, T, 1>, OpLinearNew>;
    using OpCopyOutLinear = Bind<Vec::CopyOut<U>, Placeholder::Out2<U>, OpLinearOutCast>; // update input2: linear

    using OpOne = MAKE_CONST(T, 1);
    using OpZero = MAKE_CONST(T, 0);
    using OpZeroTensor = Bind<Vec::Duplicate<T>, OpZero>;
    using OpSignCmpPos = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_GT>, OpLinearNew, OpZero>;
    using OpSignCmpNeg = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_LT>, OpLinearNew, OpZero>;
    using OpSignSelectPos = Bind<Vec::Select<uint8_t, T, SELECT_MODE_T_S>, OpSignCmpPos, OpOne, OpZeroTensor>;
    using OpSignSelectNeg = Bind<Vec::Select<uint8_t, T, SELECT_MODE_T_S>, OpSignCmpNeg, OpOne, OpZeroTensor>;
    using OpSign = Bind<Vec::Sub<T>, OpSignSelectPos, OpSignSelectNeg>;
    using OpSignMulL1 = Bind<Vec::Mul<T>, OpSign, OpL1Cast>;
    using OpMulL1SubLinear = Bind<Vec::Sub<T>, OpSignMulL1, OpLinearNew>;
    using OpAccNewPowerDiv = Bind<Vec::DivHighPrecision<T>, OpAccNewPower, OpLrCast>;
    using OpTwo = MAKE_CONST(T, 2);
    using OpL2Mul2 = Bind<Vec::Muls<T>, OpL2Cast, OpTwo>;
    using OpAdd2L2 = Bind<Vec::Add<T>, OpAccNewPowerDiv, OpL2Mul2>;
    using OpVarSelectTrue = Bind<Vec::DivHighPrecision<T>, OpMulL1SubLinear, OpAdd2L2>;
    using OpLinearAbs = Bind<Vec::Abs<T>, OpLinearNew>;
    using OpAbsCmpL1 = Bind<Vec::Compare<uint8_t, T, COMPARE_MODE_GT>, OpLinearAbs, OpL1Cast>;
    using OpVarNew = Bind<Vec::Select<uint8_t, T, SELECT_MODE_T_S>, OpAbsCmpL1, OpVarSelectTrue, OpZero>;

    using OpVarOutCast = Bind<Vec::Cast<U, T, 1>, OpVarNew>;
    using OpCopyOutVar = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpVarOutCast>; // output: var

    using Outputs = Elems<OpCopyOutVar, OpCopyOutAccum, OpCopyOutLinear>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace ApplyFtrlOp

#endif // CANN_CUSTOM_OPS_APPLY_FTRL_DAG_H
