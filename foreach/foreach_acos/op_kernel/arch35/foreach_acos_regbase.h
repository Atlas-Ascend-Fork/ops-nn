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
 * \file foreach_acos_regbase.h
 * \brief Register-resident vector compute policy for foreach_acos on Ascend950.
 *
 * The whole element work is one vector scope with only the tile loop and compile-time
 * branches.  Transport (GM<->UB double buffering) is owned by
 * ForeachFlatRegbase::TypedFlowMover, so this policy allocates no extra UB.
 *
 * math: acos(x) = PI/2 - asin(|x|)*sign(x), with asin evaluated by a 5-term odd
 * polynomial over the reduced argument (mirrors simt_api acosf/asinf, without the
 * hardware Taylor path).  sign(x) is a compare+select (x>=0 ? +1 : -1), avoiding a
 * full vector divide.  half/bf16 are promoted to float in-register (unpack -> cast
 * -> compute -> cast -> pack); float computes directly.
 */

#ifndef FOREACH_ACOS_REGBASE_H
#define FOREACH_ACOS_REGBASE_H

#include "../../foreach_utils/arch35/foreach_flat_flow_regbase.h"

namespace NsForeachAcosRegbase {
using namespace AscendC;

constexpr uint32_t ACOS_VL = VECTOR_REG_WIDTH / sizeof(float); // 64 lanes per 256B register

constexpr float ACOS_THRESHOLD = 0.56000000238418579102f; // switch to sqrt((1-|x|)/2)
constexpr float ACOS_HALF_PI = 1.57079632679489661923f;

// Cast traits: promotion/demotion for half/bf16, and the int floor/rint pair used by
// the branchless threshold select below.
constexpr Reg::CastTrait ACOS_CAST_STORAGE_TO_FLOAT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr Reg::CastTrait ACOS_CAST_FLOAT_TO_STORAGE = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
constexpr Reg::CastTrait ACOS_CAST_FLOOR = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_FLOOR,
};
constexpr Reg::CastTrait ACOS_CAST_RINT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void AcosStore(Reg::RegTensor<float, Reg::RegTraitNumOne>& src,
                                 Reg::RegTensor<T, Reg::RegTraitNumOne>& storage, __ubuf__ T* dst, Reg::MaskReg& mask)
{
    if constexpr (IsSameType<T, float>::value) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(dst, src, mask);
    } else {
        Reg::Cast<T, float, ACOS_CAST_FLOAT_TO_STORAGE>(storage, src, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst, storage, mask);
    }
}

/*
 * Single vector body.  The tile loop has no runtime break/continue; the 16-bit staging is
 * if constexpr, so the emitted sequence is exactly the selected one.
 */
template <typename T>
__aicore__ inline void AcosVF(__ubuf__ T* inAddr, __ubuf__ T* outAddr, uint32_t dataCount)
{
    uint16_t repeatTimes = static_cast<uint16_t>((dataCount + ACOS_VL - 1) / ACOS_VL);
    uint32_t remain = dataCount;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float, Reg::RegTraitNumOne> src;
        Reg::RegTensor<float, Reg::RegTraitNumOne> absReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> tReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> sqrtReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> choiceReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> reducedReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> reduced2Reg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> polyReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> asinReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> endpReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> signReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> negOneReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> tmpReg;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> s32Reg;
        Reg::RegTensor<T, Reg::RegTraitNumOne> storage;
        Reg::MaskReg mask;
        Reg::MaskReg posMask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(remain);
            if constexpr (IsSameType<T, float>::value) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src, inAddr + i * ACOS_VL);
            } else {
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(storage, inAddr + i * ACOS_VL);
                Reg::Cast<float, T, ACOS_CAST_STORAGE_TO_FLOAT>(src, storage, mask);
            }

            // abs_x = |x|
            Reg::Abs(absReg, src, mask);
            // t = (1 - |x|) / 2 = 0.5 - 0.5*|x|
            Reg::Muls(tReg, absReg, -0.5f, mask);
            Reg::Adds(tReg, tReg, 0.5f, mask);
            // sqrt_t = sqrt((1-|x|)/2)
            Reg::Sqrt(sqrtReg, tReg, mask);
            // choice1 = (|x| <= threshold) ? 1 : 0  via  -floor(min(|x|,th)-th)
            Reg::Mins(choiceReg, absReg, ACOS_THRESHOLD, mask);
            Reg::Adds(choiceReg, choiceReg, -ACOS_THRESHOLD, mask);
            Reg::Cast<int32_t, float, ACOS_CAST_FLOOR>(s32Reg, choiceReg, mask);
            Reg::Cast<float, int32_t, ACOS_CAST_RINT>(choiceReg, s32Reg, mask);
            Reg::Muls(choiceReg, choiceReg, -1.0f, mask);
            // reduced = |x|*choice1 + sqrt_t*(1-choice1)
            Reg::Mul(reducedReg, absReg, choiceReg, mask);
            Reg::Muls(tmpReg, choiceReg, -1.0f, mask);
            Reg::Adds(tmpReg, tmpReg, 1.0f, mask);
            Reg::Mul(tmpReg, sqrtReg, tmpReg, mask);
            Reg::Add(reducedReg, reducedReg, tmpReg, mask);

            // reduced2 = reduced^2
            Reg::Mul(reduced2Reg, reducedReg, reducedReg, mask);
            // 5-term odd polynomial (asinf coefficients), Horner in z=reduced2
            Reg::Muls(polyReg, reduced2Reg, 0.05025001987814903259f, mask);
            Reg::Adds(polyReg, polyReg, 0.018773360177874565125f, mask);
            Reg::Mul(polyReg, polyReg, reduced2Reg, mask);
            Reg::Adds(polyReg, polyReg, 0.046769052743911743164f, mask);
            Reg::Mul(polyReg, polyReg, reduced2Reg, mask);
            Reg::Adds(polyReg, polyReg, 0.074823014438152313232f, mask);
            Reg::Mul(polyReg, polyReg, reduced2Reg, mask);
            Reg::Adds(polyReg, polyReg, 0.16667181253433227539f, mask);
            Reg::Mul(polyReg, polyReg, reduced2Reg, mask);
            // asin_reduced = reduced*poly + reduced  (= asin(reduced))
            Reg::Mul(tmpReg, reducedReg, polyReg, mask);
            Reg::Add(asinReg, tmpReg, reducedReg, mask);
            // endpoint (|x|>th): asin_abs = PI/2 - 2*asin_reduced, else keep
            Reg::Muls(endpReg, asinReg, -2.0f, mask);
            Reg::Adds(endpReg, endpReg, ACOS_HALF_PI, mask);
            Reg::Mul(asinReg, asinReg, choiceReg, mask);
            Reg::Muls(tmpReg, choiceReg, -1.0f, mask);
            Reg::Adds(tmpReg, tmpReg, 1.0f, mask);
            Reg::Mul(tmpReg, endpReg, tmpReg, mask);
            Reg::Add(asinReg, asinReg, tmpReg, mask);

            // sign(x) = (x >= 0) ? +1 : -1. NaN propagates through the asin chain and
            // acos(+/-0) is PI/2 regardless of sign, so compare+select is equivalent to
            // the previous division-based sign and avoids a full vector divide.
            Reg::Compares<float, CMPMODE::GE>(posMask, src, 0.0f, mask);
            Reg::Duplicate(signReg, 1.0f, mask);
            Reg::Duplicate(negOneReg, -1.0f, mask);
            Reg::Select<float>(signReg, signReg, negOneReg, posMask);
            // acos = PI/2 - asin(|x|)*sign(x)
            Reg::Mul(asinReg, asinReg, signReg, mask);
            Reg::Muls(asinReg, asinReg, -1.0f, mask);
            Reg::Adds(asinReg, asinReg, ACOS_HALF_PI, mask);

            AcosStore<T>(asinReg, storage, outAddr + i * ACOS_VL, mask);
        }
    }
}

template <typename T>
struct AcosPolicy {
    static constexpr bool kUsesExtraUb = false;
    static constexpr bool kScalarBatchEnabled = true;

    __aicore__ static inline void Run(LocalTensor<T>* inputs, LocalTensor<T> output, float scalar, uint32_t dataCount)
    {
        (void)scalar;
        if (dataCount == 0U) {
            return;
        }
        __ubuf__ T* inAddr = reinterpret_cast<__ubuf__ T*>(inputs[0].GetPhyAddr());
        __ubuf__ T* outAddr = reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr());
        AcosVF<T>(inAddr, outAddr, dataCount);
    }
};

} // namespace NsForeachAcosRegbase

#endif // FOREACH_ACOS_REGBASE_H
