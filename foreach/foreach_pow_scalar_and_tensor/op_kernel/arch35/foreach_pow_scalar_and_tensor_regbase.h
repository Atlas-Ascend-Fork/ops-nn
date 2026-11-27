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
 * \file foreach_pow_scalar_and_tensor_regbase.h
 * \brief RegBase compute policy for y = scalar ^ x (foreach_pow_scalar_and_tensor).
 *
 * The scalar base is classified once per launch; the tensor is the exponent.  The
 * transport (GM<->UB double buffering) is owned by ForeachFlatRegbase::TypedFlowMover,
 * so this policy allocates no extra UB.
 */

#ifndef FOREACH_POW_SCALAR_AND_TENSOR_REGBASE_H
#define FOREACH_POW_SCALAR_AND_TENSOR_REGBASE_H

#include "adv_api/math/power.h"
#include "../../foreach_utils/arch35/foreach_flat_flow_regbase.h"

namespace NsForeachPowScalarAndTensorRegbase {
using namespace AscendC;

constexpr uint32_t PSAT_VL = VECTOR_REG_WIDTH / sizeof(float);
// float32 can represent every integer exactly up to 2^24; beyond this every
// finite float32 is an even integer.  The int32 parity probe below would
// saturate at INT32_MAX (odd) for larger magnitudes and invert the sign of an
// overflowed pow(-b, x), so the parity result is masked out from this point on.
constexpr float PSAT_EXACT_INT_LIMIT = 16777216.0f;
// ln(FLT_MAX); exp(x) overflows to +Inf only above this.  The old 88.7 clamp
// turned finite bf16 near-max results (exp(88.71) = 3.36e38) into +Inf.
constexpr float PSAT_EXP_OVERFLOW = 88.72283935546875f;

constexpr Reg::CastTrait PSAT_CAST_STORAGE_TO_FLOAT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr Reg::CastTrait PSAT_CAST_FLOAT_TO_STORAGE = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr Reg::ExpSpecificMode PSAT_EXP_MODE = {Reg::MaskMergeMode::ZEROING, ExpAlgo::PRECISION_1ULP_FTZ_FALSE};

constexpr Reg::CastTrait PSAT_CAST_FLOAT_TO_INT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_TRUNC,
};

constexpr Reg::CastTrait PSAT_CAST_INT_TO_FLOAT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

enum PsatBaseMode : int32_t {
    PSAT_BASE_ZERO = 0,
    PSAT_BASE_ONE = 1,
    PSAT_BASE_NEG_ONE = 2,
    PSAT_BASE_POS = 3,
    PSAT_BASE_NEG = 4,
    PSAT_BASE_INF = 5,
    PSAT_BASE_NAN = 6,
    PSAT_BASE_INT_ONE = 10,
    PSAT_BASE_INT_NEG_ONE = 11,
    PSAT_BASE_INT_ZERO = 12,
    PSAT_BASE_INT_GENERAL = 13,
};

__aicore__ inline float PsatQNaN()
{
    uint32_t bits = 0x7FC00000u;
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

__aicore__ inline float PsatInfinity()
{
    uint32_t bits = 0x7F800000u;
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

template <typename T>
__aicore__ inline int32_t PsatClassifyBase(float b)
{
    if (b != b) {
        return PSAT_BASE_NAN;
    }
    if (b == 0.0f) {
        return PSAT_BASE_ZERO;
    }
    if (b == 1.0f) {
        return PSAT_BASE_ONE;
    }
    if (b == -1.0f) {
        return PSAT_BASE_NEG_ONE;
    }
    float inf = PsatInfinity();
    if (b == inf || b == -inf) {
        return PSAT_BASE_INF;
    }
    return b > 0.0f ? PSAT_BASE_POS : PSAT_BASE_NEG;
}

template <typename T>
__aicore__ inline void PsatLoadExp(Reg::RegTensor<float, Reg::RegTraitNumOne>& xf,
                                   Reg::RegTensor<T, Reg::RegTraitNumOne>& storage, __ubuf__ T* inAddr,
                                   Reg::MaskReg& mask)
{
    if constexpr (IsSameType<T, float>::value) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(xf, inAddr);
    } else {
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(storage, inAddr);
        Reg::Cast<float, T, PSAT_CAST_STORAGE_TO_FLOAT>(xf, storage, mask);
    }
}

template <typename T>
__aicore__ inline void PsatStore(Reg::RegTensor<float, Reg::RegTraitNumOne>& src,
                                 Reg::RegTensor<T, Reg::RegTraitNumOne>& storage, __ubuf__ T* outAddr,
                                 Reg::MaskReg& mask)
{
    if constexpr (IsSameType<T, float>::value) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(outAddr, src, mask);
    } else {
        Reg::Cast<T, float, PSAT_CAST_FLOAT_TO_STORAGE>(storage, src, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outAddr, storage, mask);
    }
}

/* Integer-exponent detection: truncate to int32 and compare back in float space. */
__aicore__ inline void PsatDetectInteger(Reg::RegTensor<float, Reg::RegTraitNumOne>& xf, Reg::MaskReg& mask,
                                         Reg::RegTensor<int32_t, Reg::RegTraitNumOne>& xi,
                                         Reg::RegTensor<float, Reg::RegTraitNumOne>& back, Reg::MaskReg& intMask)
{
    Reg::Cast<int32_t, float, PSAT_CAST_FLOAT_TO_INT>(xi, xf, mask);
    Reg::Cast<float, int32_t, PSAT_CAST_INT_TO_FLOAT>(back, xi, mask);
    Reg::Compare<float, CMPMODE::EQ>(intMask, back, xf, mask);
}

__aicore__ inline void PsatDetectLargeExponent(Reg::RegTensor<float, Reg::RegTraitNumOne>& xf, Reg::MaskReg& mask,
                                               Reg::RegTensor<float, Reg::RegTraitNumOne>& absReg,
                                               Reg::MaskReg& largeMask)
{
    Reg::Abs(absReg, xf, mask);
    Reg::Compares<float, CMPMODE::GE>(largeMask, absReg, PSAT_EXACT_INT_LIMIT, mask);
}

__aicore__ inline void PsatDetectOdd(Reg::RegTensor<int32_t, Reg::RegTraitNumOne>& xi,
                                     Reg::RegTensor<int32_t, Reg::RegTraitNumOne>& andReg,
                                     Reg::RegTensor<int32_t, Reg::RegTraitNumOne>& one, Reg::MaskReg& mask,
                                     Reg::MaskReg& oddMask)
{
    Reg::Duplicate(one, 1, mask);
    Reg::And(andReg, xi, one, mask);
    Reg::Compares<int32_t, CMPMODE::NE>(oddMask, andReg, 0, mask);
}

template <typename T, int32_t MODE, bool ODD>
__aicore__ inline void PsatFloatVF(__ubuf__ T* inAddr, __ubuf__ T* outAddr, float base, uint32_t dataCount)
{
    uint16_t repeatTimes = static_cast<uint16_t>((dataCount + PSAT_VL - 1) / PSAT_VL);
    uint32_t remain = dataCount;
    float infValue = PsatInfinity();
    float nanValue = PsatQNaN();
    __VEC_SCOPE__
    {
        Reg::RegTensor<float, Reg::RegTraitNumOne> xf;
        Reg::RegTensor<float, Reg::RegTraitNumOne> lnReg;
        Reg::RegTensor<float, Reg::RegTraitNumOne> yf;
        Reg::RegTensor<float, Reg::RegTraitNumOne> tmp;
        Reg::RegTensor<float, Reg::RegTraitNumOne> back;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> xi;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> andReg;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> one;
        Reg::RegTensor<T, Reg::RegTraitNumOne> storage;
        Reg::MaskReg mask;
        Reg::MaskReg intMask;
        Reg::MaskReg oddMask;
        Reg::MaskReg selMask;
        Reg::MaskReg negMask;
        Reg::MaskReg largeMask;
        if constexpr (MODE == PSAT_BASE_POS || MODE == PSAT_BASE_NEG) {
            uint32_t fullRemain = PSAT_VL;
            Reg::MaskReg fullMask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(fullRemain);
            Reg::Duplicate(lnReg, base > 0.0f ? base : -base, fullMask);
            Reg::Ln(lnReg, lnReg, fullMask);
        }
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(remain);
            PsatLoadExp<T>(xf, storage, inAddr + i * PSAT_VL, mask);

            if constexpr (MODE == PSAT_BASE_ONE) {
                Reg::Duplicate(yf, 1.0f, mask);
            } else if constexpr (MODE == PSAT_BASE_ZERO) {
                Reg::Duplicate(yf, nanValue, mask);
                Reg::Compares<float, CMPMODE::GT>(selMask, xf, 0.0f, mask);
                Reg::Duplicate(tmp, 0.0f, mask);
                Reg::Select(yf, tmp, yf, selMask);
                Reg::Compares<float, CMPMODE::EQ>(selMask, xf, 0.0f, mask);
                Reg::Duplicate(tmp, 1.0f, mask);
                Reg::Select(yf, tmp, yf, selMask);
                Reg::Compares<float, CMPMODE::LT>(selMask, xf, 0.0f, mask);
                Reg::Duplicate(tmp, infValue, mask);
                Reg::Select(yf, tmp, yf, selMask);
            } else if constexpr (MODE == PSAT_BASE_NEG_ONE) {
                PsatDetectInteger(xf, mask, xi, back, intMask);
                PsatDetectLargeExponent(xf, mask, back, largeMask);
                Reg::Or(intMask, intMask, largeMask, mask);
                PsatDetectOdd(xi, andReg, one, mask, oddMask);
                Reg::Compares<float, CMPMODE::LT>(selMask, back, PSAT_EXACT_INT_LIMIT, mask);
                Reg::And(oddMask, oddMask, selMask, mask);
                Reg::Duplicate(yf, nanValue, mask);
                Reg::Duplicate(tmp, 1.0f, mask);
                Reg::Select(yf, tmp, yf, intMask);
                Reg::And(selMask, oddMask, intMask, mask);
                Reg::Duplicate(tmp, -1.0f, mask);
                Reg::Select(yf, tmp, yf, selMask);
                // powf(-1, +-inf) == 1
                Reg::Abs(tmp, xf, mask);
                Reg::Compares<float, CMPMODE::EQ>(selMask, tmp, infValue, mask);
                Reg::Duplicate(tmp, 1.0f, mask);
                Reg::Select(yf, tmp, yf, selMask);
            } else if constexpr (MODE == PSAT_BASE_POS) {
                Reg::Mul(tmp, xf, lnReg, mask);
                Reg::Exp<float, &PSAT_EXP_MODE>(yf, tmp, mask);
                Reg::Compares<float, CMPMODE::LT>(selMask, tmp, -87.3f, mask);
                Reg::Duplicate(back, 0.0f, mask);
                Reg::Select(yf, back, yf, selMask);
                Reg::Compares<float, CMPMODE::GT>(selMask, tmp, PSAT_EXP_OVERFLOW, mask);
                Reg::Duplicate(back, infValue, mask);
                Reg::Select(yf, back, yf, selMask);
            } else if constexpr (MODE == PSAT_BASE_NEG) {
                PsatDetectInteger(xf, mask, xi, back, intMask);
                PsatDetectLargeExponent(xf, mask, back, largeMask);
                Reg::Or(intMask, intMask, largeMask, mask);
                PsatDetectOdd(xi, andReg, one, mask, oddMask);
                Reg::Compares<float, CMPMODE::LT>(selMask, back, PSAT_EXACT_INT_LIMIT, mask);
                Reg::And(oddMask, oddMask, selMask, mask);
                Reg::Mul(tmp, xf, lnReg, mask);
                Reg::Exp<float, &PSAT_EXP_MODE>(yf, tmp, mask);
                Reg::Compares<float, CMPMODE::LT>(selMask, tmp, -87.3f, mask);
                Reg::Duplicate(back, 0.0f, mask);
                Reg::Select(yf, back, yf, selMask);
                Reg::Compares<float, CMPMODE::GT>(selMask, tmp, PSAT_EXP_OVERFLOW, mask);
                Reg::Duplicate(back, infValue, mask);
                Reg::Select(yf, back, yf, selMask);
                // integer exponent: +/- |b|^x; non-integer: NaN.
                Reg::Duplicate(tmp, nanValue, mask);
                Reg::Select(tmp, yf, tmp, intMask);
                Reg::And(selMask, oddMask, intMask, mask);
                Reg::Neg(back, tmp, mask);
                Reg::Select(tmp, back, tmp, selMask);
                // |exponent| == inf: powf(-b, +-inf) follows |b|^x without a sign tie.
                Reg::Abs(back, xf, mask);
                Reg::Compares<float, CMPMODE::EQ>(selMask, back, infValue, mask);
                Reg::Select(tmp, yf, tmp, selMask);
                Reg::Move(yf, tmp);
            } else if constexpr (MODE == PSAT_BASE_INF) {
                Reg::Duplicate(yf, nanValue, mask);
                Reg::Compares<float, CMPMODE::GT>(selMask, xf, 0.0f, mask);
                Reg::Duplicate(tmp, base > 0.0f ? infValue : -infValue, mask);
                Reg::Select(yf, tmp, yf, selMask);
                Reg::Compares<float, CMPMODE::EQ>(selMask, xf, 0.0f, mask);
                Reg::Duplicate(tmp, 1.0f, mask);
                Reg::Select(yf, tmp, yf, selMask);
                Reg::Compares<float, CMPMODE::LT>(selMask, xf, 0.0f, mask);
                Reg::Duplicate(tmp, 0.0f, mask);
                Reg::Select(yf, tmp, yf, selMask);
            } else { // PSAT_BASE_NAN: torch returns 1 only for exponent 0
                Reg::Compares<float, CMPMODE::EQ>(selMask, xf, 0.0f, mask);
                Reg::Duplicate(yf, nanValue, mask);
                Reg::Duplicate(tmp, 1.0f, mask);
                Reg::Select(yf, tmp, yf, selMask);
            }
            PsatStore<T>(yf, storage, outAddr + i * PSAT_VL, mask);
        }
    }
}

template <typename T, int32_t MODE>
__aicore__ inline void PsatIntVF(__ubuf__ T* inAddr, __ubuf__ T* outAddr, int32_t base, uint32_t dataCount)
{
    uint16_t repeatTimes = static_cast<uint16_t>((dataCount + PSAT_VL - 1) / PSAT_VL);
    uint32_t remain = dataCount;

    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> xInt;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> acc;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> factor;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> one;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> shifted;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> andReg;
        Reg::RegTensor<int32_t, Reg::RegTraitNumOne> selected;
        Reg::MaskReg mask;
        Reg::MaskReg bitMask;
        Reg::MaskReg negMask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = Reg::UpdateMask<int32_t, Reg::RegTraitNumOne>(remain);
            Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>(xInt, inAddr + i * PSAT_VL);
            if constexpr (MODE == PSAT_BASE_INT_ONE) {
                Reg::Duplicate(acc, 1, mask);
            } else if constexpr (MODE == PSAT_BASE_INT_ZERO) {
                Reg::Compares<int32_t, CMPMODE::EQ>(bitMask, xInt, 0, mask);
                Reg::Duplicate(acc, 0, mask);
                Reg::Duplicate(one, 1, mask);
                Reg::Select(acc, one, acc, bitMask);
            } else if constexpr (MODE == PSAT_BASE_INT_NEG_ONE) {
                Reg::Duplicate(one, 1, mask);
                Reg::And(andReg, xInt, one, mask);
                Reg::Compares<int32_t, CMPMODE::EQ>(bitMask, andReg, 0, mask);
                Reg::Duplicate(acc, -1, mask);
                Reg::Select(acc, one, acc, bitMask);
            } else {
                Reg::Duplicate(one, 1, mask);
                Reg::Duplicate(acc, 1, mask);
                Reg::Duplicate(factor, base, mask);
                for (int32_t bit = 0; bit < 31; ++bit) {
                    Reg::ShiftRights(shifted, xInt, static_cast<int16_t>(bit), mask);
                    Reg::And(andReg, shifted, one, mask);
                    Reg::Compares<int32_t, CMPMODE::EQ>(bitMask, andReg, 1, mask);
                    Reg::Select(selected, factor, one, bitMask);
                    Reg::Mul(acc, acc, selected, mask);
                    Reg::Mul(factor, factor, factor, mask);
                }
                Reg::Compares<int32_t, CMPMODE::LT>(negMask, xInt, 0, mask);
                Reg::Duplicate(one, 0, mask);
                Reg::Select(acc, one, acc, negMask);
            }
            Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_NORM_B32>(outAddr + i * PSAT_VL, acc, mask);
        }
    }
}

constexpr AscendC::PowerConfig PSAT_POWER_CONFIG = {AscendC::PowerAlgo::DOUBLE_FLOAT_TECH};

template <typename T, typename ScalarT>
struct PowScalarAndTensorPolicy {
    static constexpr bool kUsesExtraUb = false;
    static constexpr bool kScalarBatchEnabled = true;

    __aicore__ static inline void Run(LocalTensor<T>* inputs, LocalTensor<T> output, ScalarT scalar, uint32_t dataCount)
    {
        if (dataCount == 0U) {
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            AscendC::Power<float, false, PSAT_POWER_CONFIG>(output, static_cast<float>(scalar), inputs[0], dataCount);
            return;
        }
        __ubuf__ T* inAddr = reinterpret_cast<__ubuf__ T*>(inputs[0].GetPhyAddr());
        __ubuf__ T* outAddr = reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr());
        if constexpr (IsSameType<T, int32_t>::value) {
            int64_t base = static_cast<int64_t>(scalar);
            if (base == 1) {
                PsatIntVF<T, PSAT_BASE_INT_ONE>(inAddr, outAddr, 1, dataCount);
            } else if (base == -1) {
                PsatIntVF<T, PSAT_BASE_INT_NEG_ONE>(inAddr, outAddr, -1, dataCount);
            } else if (base == 0) {
                PsatIntVF<T, PSAT_BASE_INT_ZERO>(inAddr, outAddr, 0, dataCount);
            } else {
                PsatIntVF<T, PSAT_BASE_INT_GENERAL>(inAddr, outAddr, static_cast<int32_t>(base), dataCount);
            }
        } else {
            float base = static_cast<float>(scalar);
            int32_t mode = PsatClassifyBase<T>(base);
            switch (mode) {
                case PSAT_BASE_ONE:
                    PsatFloatVF<T, PSAT_BASE_ONE, false>(inAddr, outAddr, base, dataCount);
                    break;
                case PSAT_BASE_ZERO:
                    PsatFloatVF<T, PSAT_BASE_ZERO, false>(inAddr, outAddr, base, dataCount);
                    break;
                case PSAT_BASE_NEG_ONE:
                    PsatFloatVF<T, PSAT_BASE_NEG_ONE, false>(inAddr, outAddr, base, dataCount);
                    break;
                case PSAT_BASE_POS:
                    PsatFloatVF<T, PSAT_BASE_POS, false>(inAddr, outAddr, base, dataCount);
                    break;
                case PSAT_BASE_NEG:
                    PsatFloatVF<T, PSAT_BASE_NEG, false>(inAddr, outAddr, base, dataCount);
                    break;
                case PSAT_BASE_INF:
                    PsatFloatVF<T, PSAT_BASE_INF, false>(inAddr, outAddr, base, dataCount);
                    break;
                default:
                    PsatFloatVF<T, PSAT_BASE_NAN, false>(inAddr, outAddr, base, dataCount);
                    break;
            }
        }
    }
};

} // namespace NsForeachPowScalarAndTensorRegbase

#endif // FOREACH_POW_SCALAR_AND_TENSOR_REGBASE_H
