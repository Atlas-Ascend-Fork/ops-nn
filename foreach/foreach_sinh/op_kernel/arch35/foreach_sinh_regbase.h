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
 * \file foreach_sinh_regbase.h
 * \brief Register-resident vector compute policy for foreach_sinh on Ascend950.
 *
 * The whole element work is one vector scope with only the tile loop and compile-time
 * branches.  Transport (GM<->UB double buffering) is owned by
 * ForeachFlatRegbase::TypedFlowMover, so this policy allocates no extra UB.
 *
 * math: sinh(x) = sign(x) * (e^(|x|-ln2) - e^(-|x|-ln2)), i.e. the operator golden's own
 * form.  The |x| form keeps the exponent below ln(FLT_MAX) for as long as the result itself
 * is finite, so the overflow boundary stays at asinh(FLT_MAX) = 89.415985107421875 instead of
 * the earlier ln(FLT_MAX), and it never divides by zero for very negative x.
 */

#ifndef FOREACH_SINH_REGBASE_H
#define FOREACH_SINH_REGBASE_H

#include "../../foreach_utils/arch35/foreach_flat_flow_regbase.h"

namespace NsForeachSinhRegbase {
using namespace AscendC;

constexpr uint32_t SINH_VL = VECTOR_REG_WIDTH / sizeof(float); // 64 lanes per 256B register
constexpr float SINH_LN2 = 0.6931471805599453f;
constexpr float SINH_QUARTER = 0.25f;
// asinh(FLT_MAX); beyond it sinh(x) is +-inf in float32 and in the torch golden.
constexpr float SINH_OVERFLOW_THRESHOLD = 89.415985107421875f;

// Two mutually exclusive compute forms, both compile-time (probe-1 on row-29/row-30 selects one):
//   RECIPROCAL  e - 0.25/e        : one vexp + one vdiv
//   TWO_EXP     e - e^(-|x|-ln2)  : two vexp, no vdiv (exactly the golden third_party form)
enum SinhForm : int32_t {
    SINH_FORM_RECIPROCAL = 0,
    SINH_FORM_TWO_EXP = 1,
};

// probe-1 result (round 1, row-29/row-30): both forms are inside the jitter band, so the
// two-Exp form is kept -- it matches the golden operation for operation and needs no
// approximate hardware divide.  Set to SINH_FORM_RECIPROCAL to build the other form.
constexpr SinhForm SINH_ACTIVE_FORM = SINH_FORM_TWO_EXP;
// probe-2 result: device vexp returns +-inf on overflow (it does not saturate to FLT_MAX),
// so no explicit overflow Select is emitted.
constexpr bool SINH_EMIT_OVERFLOW_SELECT = false;

static constexpr Reg::ExpSpecificMode SINH_EXP_MODE = {Reg::MaskMergeMode::ZEROING, ExpAlgo::PRECISION_1ULP_FTZ_TRUE};
static constexpr Reg::DivSpecificMode SINH_DIV_MODE = {Reg::MaskMergeMode::ZEROING, false,
                                                       DivAlgo::PRECISION_1ULP_FTZ_TRUE};

constexpr Reg::CastTrait SINH_CAST_STORAGE_TO_FLOAT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr Reg::CastTrait SINH_CAST_FLOAT_TO_STORAGE = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void SinhStore(Reg::RegTensor<float, Reg::RegTraitNumOne>& src,
                                 Reg::RegTensor<T, Reg::RegTraitNumOne>& storage, __ubuf__ T* dst, Reg::MaskReg& mask)
{
    if constexpr (IsSameType<T, float>::value) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(dst, src, mask);
    } else {
        Reg::Cast<T, float, SINH_CAST_FLOAT_TO_STORAGE>(storage, src, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst, storage, mask);
    }
}

/*
 * Single vector body.  The tile loop has no runtime break/continue; the form and the 16-bit
 * staging are if constexpr, so the emitted sequence is exactly the selected one.
 */
template <typename T>
__aicore__ inline void SinhVF(__ubuf__ T* inAddr, __ubuf__ T* outAddr, uint32_t dataCount)
{
    uint16_t repeatTimes = static_cast<uint16_t>((dataCount + SINH_VL - 1) / SINH_VL);
    uint32_t remain = dataCount;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float, Reg::RegTraitNumOne> xf;
        Reg::RegTensor<float, Reg::RegTraitNumOne> axf;
        Reg::RegTensor<float, Reg::RegTraitNumOne> ex;
        Reg::RegTensor<float, Reg::RegTraitNumOne> yf;
        Reg::RegTensor<float, Reg::RegTraitNumOne> tmp;
        Reg::RegTensor<T, Reg::RegTraitNumOne> storage;
        Reg::MaskReg mask;
        Reg::MaskReg negMask;
        Reg::MaskReg ovfMask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(remain);
            if constexpr (IsSameType<T, float>::value) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(xf, inAddr + i * SINH_VL);
            } else {
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(storage, inAddr + i * SINH_VL);
                Reg::Cast<float, T, SINH_CAST_STORAGE_TO_FLOAT>(xf, storage, mask);
            }
            Reg::Abs(axf, xf, mask);
            Reg::Adds(tmp, axf, -SINH_LN2, mask);
            Reg::Exp<float, &SINH_EXP_MODE>(ex, tmp, mask);
            if constexpr (SINH_ACTIVE_FORM == SINH_FORM_RECIPROCAL) {
                // 0.25 / e^(|x|-ln2) == e^(-|x|-ln2): the second term of the golden form.
                Reg::Duplicate(yf, SINH_QUARTER, mask);
                Reg::Div<float, &SINH_DIV_MODE>(yf, yf, ex, mask);
                Reg::Sub(yf, ex, yf, mask);
            } else {
                // e^(-|x|-ln2) = exp(-|x| - ln2); -|x| - ln2 is NOT -(|x| - ln2).
                Reg::Neg(tmp, axf, mask);
                Reg::Adds(tmp, tmp, -SINH_LN2, mask);
                Reg::Exp<float, &SINH_EXP_MODE>(yf, tmp, mask);
                Reg::Sub(yf, ex, yf, mask);
            }
            Reg::Compares<float, CMPMODE::LT>(negMask, xf, 0.0f, mask);
            Reg::Neg(tmp, yf, mask);
            Reg::Select(yf, tmp, yf, negMask);
            if constexpr (SINH_EMIT_OVERFLOW_SELECT) {
                // Only emitted when probe-2 shows vexp saturating instead of producing inf.
                Reg::Duplicate(ex, __builtin_inff(), mask);
                Reg::Compares<float, CMPMODE::GE>(ovfMask, axf, SINH_OVERFLOW_THRESHOLD, mask);
                Reg::Neg(tmp, ex, mask);
                Reg::Select(ex, tmp, ex, negMask);
                Reg::Select(yf, ex, yf, ovfMask);
            }
            SinhStore<T>(yf, storage, outAddr + i * SINH_VL, mask);
        }
    }
}

template <typename T>
struct SinhPolicy {
    static constexpr bool kUsesExtraUb = false;

    __aicore__ static inline void Run(LocalTensor<T>* inputs, LocalTensor<T> output, float scalar, uint32_t dataCount)
    {
        (void)scalar;
        if (dataCount == 0U) {
            return;
        }
        __ubuf__ T* inAddr = reinterpret_cast<__ubuf__ T*>(inputs[0].GetPhyAddr());
        __ubuf__ T* outAddr = reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr());
        SinhVF<T>(inAddr, outAddr, dataCount);
    }
};

} // namespace NsForeachSinhRegbase

#endif // FOREACH_SINH_REGBASE_H
