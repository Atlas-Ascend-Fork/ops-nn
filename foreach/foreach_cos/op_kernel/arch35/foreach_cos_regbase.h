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
 * \file foreach_cos_regbase.h
 * \brief Register-resident vector compute policy for foreach_cos on Ascend950.
 *
 * The whole element work is one vector scope with only the tile loop and compile-time
 * branches.  Transport (GM<->UB double buffering) is owned by
 * ForeachFlatRegbase::TypedFlowMover, so this policy allocates no extra UB.
 *
 * math: cos(x) = (-1)^k * sin(x - k*pi + pi/2) with k = round(x/pi + 1/2), i.e. the same
 * polynomial range-reduction and sin series the AscendC library Cos uses on DAV_3510, kept
 * register resident so no shared tmp buffer is required.
 */

#ifndef FOREACH_COS_REGBASE_H
#define FOREACH_COS_REGBASE_H

#include "../../foreach_utils/arch35/foreach_flat_flow_regbase.h"

namespace NsForeachCosRegbase {
using namespace AscendC;

constexpr uint32_t COS_VL = VECTOR_REG_WIDTH / sizeof(float); // 64 lanes per 256B register
// define the number of x div pi
constexpr float COS_PI_FOR_X_TODIV = 0.3183098733425140380859375f;
// define the PI for compute
constexpr float COS_PI_0 = 3.140625f;
constexpr float COS_KPI_FIRS_PI_MULS = 0.0009670257568359375f;
constexpr float COS_KPI_TWI_PI_MULS = 6.2771141529083251953125e-7f;
constexpr float COS_KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10f;
constexpr float COS_KPI_FOR_PI_MULS = -1.0290623200529979163359041220560e-13f;
// define the number of down of pi_div
constexpr float COS_PI_DOWN = 1.57079637050628662109375f;
// kpi_2
constexpr float COS_PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375f;
// res_up polynomial coefficients
constexpr float COS_RES_MULTI_SCA = 2.604926501e-6f;
constexpr float COS_RES_ADDICT_UP = -0.0001980894471f;
constexpr float COS_2ADDS = 0.008333049340f;
constexpr float COS_3ADDS = -0.1666665792f;
constexpr float COS_POINT_FIVE = 0.5f;
constexpr float COS_M4_SCA = 4.0f;
constexpr float COS_K2_SCA = -2.0f;
constexpr float COS_SCALAR_ONE = 1.0f;

constexpr Reg::CastTrait COS_CAST_STORAGE_TO_FLOAT = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::UNKNOWN,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr Reg::CastTrait COS_CAST_FLOAT_TO_STORAGE = {
    Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT,
    Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline void CosStore(Reg::RegTensor<float, Reg::RegTraitNumOne>& src,
                                Reg::RegTensor<T, Reg::RegTraitNumOne>& storage, __ubuf__ T* dst, Reg::MaskReg& mask)
{
    if constexpr (IsSameType<T, float>::value) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(dst, src, mask);
    } else {
        Reg::Cast<T, float, COS_CAST_FLOAT_TO_STORAGE>(storage, src, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(dst, storage, mask);
    }
}

/*
 * Single vector body.  The tile loop has no runtime break/continue; the 16-bit staging is
 * if constexpr, so the emitted sequence is exactly the selected one.
 */
template <typename T>
__aicore__ inline void CosVF(__ubuf__ T* inAddr, __ubuf__ T* outAddr, uint32_t dataCount)
{
    uint16_t repeatTimes = static_cast<uint16_t>((dataCount + COS_VL - 1) / COS_VL);
    uint32_t remain = dataCount;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float, Reg::RegTraitNumOne> src;
        Reg::RegTensor<float, Reg::RegTraitNumOne> x;
        Reg::RegTensor<float, Reg::RegTraitNumOne> round;
        Reg::RegTensor<float, Reg::RegTraitNumOne> kpi;
        Reg::RegTensor<float, Reg::RegTraitNumOne> yf;
        Reg::RegTensor<T, Reg::RegTraitNumOne> storage;
        Reg::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(remain);
            if constexpr (IsSameType<T, float>::value) {
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(src, inAddr + i * COS_VL);
            } else {
                Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(storage, inAddr + i * COS_VL);
                Reg::Cast<float, T, COS_CAST_STORAGE_TO_FLOAT>(src, storage, mask);
            }

            // k = round(x / pi + 1/2), tie to even
            Reg::Muls(round, src, COS_PI_FOR_X_TODIV, mask);
            Reg::Adds(round, round, COS_POINT_FIVE, mask);
            Reg::Truncate<float, RoundMode::CAST_RINT, Reg::MaskMergeMode::ZEROING>(round, round, mask);
            // x -= k*pi_0
            Reg::Muls(kpi, round, COS_PI_0, mask);
            Reg::Sub(x, src, kpi, mask);
            // x -= k*pi_1
            Reg::Muls(kpi, round, COS_KPI_FIRS_PI_MULS, mask);
            Reg::Sub(x, x, kpi, mask);
            // x += pi/2
            Reg::Adds(x, x, COS_PI_DOWN, mask);
            // x -= k*pi_2
            Reg::Muls(kpi, round, COS_KPI_TWI_PI_MULS, mask);
            Reg::Sub(x, x, kpi, mask);
            // x -= k*pi_3
            Reg::Muls(kpi, round, COS_KPI_THIR_PI_MULS, mask);
            Reg::Sub(x, x, kpi, mask);
            // x -= k*pi_4
            Reg::Muls(kpi, round, COS_KPI_FOR_PI_MULS, mask);
            Reg::Sub(x, x, kpi, mask);
            // x += kpi_2
            Reg::Adds(x, x, COS_PI_RESDOWN_ADDS_NEG, mask);

            // kpi = x^2
            Reg::Mul(kpi, x, x, mask);
            // sign: kover2floorm4 + k2 + 1 = 1 - 2*(k mod 2)
            Reg::Muls(yf, round, COS_POINT_FIVE, mask);
            Reg::Truncate<float, RoundMode::CAST_FLOOR, Reg::MaskMergeMode::ZEROING>(yf, yf, mask);
            Reg::Muls(yf, yf, COS_M4_SCA, mask);
            Reg::Muls(round, round, COS_K2_SCA, mask);
            Reg::Add(yf, yf, round, mask);
            Reg::Adds(yf, yf, COS_SCALAR_ONE, mask);
            // res_up = sin(x)/x via odd polynomial in x^2
            Reg::Muls(round, kpi, COS_RES_MULTI_SCA, mask);
            Reg::Adds(round, round, COS_RES_ADDICT_UP, mask);
            Reg::Mul(round, round, kpi, mask);
            Reg::Adds(round, round, COS_2ADDS, mask);
            Reg::Mul(round, round, kpi, mask);
            Reg::Adds(round, round, COS_3ADDS, mask);
            Reg::Mul(round, round, kpi, mask);
            Reg::Adds(round, round, COS_SCALAR_ONE, mask);
            // sin(x) = x*P(x); cos(x) = sign * sin(x)
            Reg::Mul(round, round, x, mask);
            Reg::Mul(yf, round, yf, mask);
            Reg::Mins(yf, yf, COS_SCALAR_ONE, mask);
            Reg::Maxs(yf, yf, -COS_SCALAR_ONE, mask);

            CosStore<T>(yf, storage, outAddr + i * COS_VL, mask);
        }
    }
}

template <typename T>
struct CosPolicy {
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
        CosVF<T>(inAddr, outAddr, dataCount);
    }
};

} // namespace NsForeachCosRegbase

#endif // FOREACH_COS_REGBASE_H
