/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FOREACH_SUB_SCALAR_REGBASE_H
#define FOREACH_SUB_SCALAR_REGBASE_H
#include "../../foreach_utils/arch35/foreach_flat_flow_regbase.h"
#include "../../foreach_utils/arch35/foreach_unary_regbase_common.h"
namespace ForeachSubScalar {
using namespace AscendC;

template <typename T, typename ScalarT>
__simd_vf__ inline void SubVF(__ubuf__ T* input, __ubuf__ T* output, ScalarT scalar, uint32_t count)
{
    constexpr bool PROMOTE = IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value;
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    const uint16_t repeats = static_cast<uint16_t>(CeilDivision(count, VL));
    uint32_t remaining = count;
    for (uint16_t r = 0; r < repeats; ++r) {
        if constexpr (PROMOTE) {
            auto mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(remaining);
            Reg::RegTensor<T, Reg::RegTraitNumOne> stored, result;
            Reg::RegTensor<float, Reg::RegTraitNumOne> value, scalarReg;
            Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(stored, input + r * VL);
            Reg::Cast<float, T, ForeachUnaryRegbase::UNARY_CAST_TO_FLOAT>(value, stored, mask);
            Reg::Duplicate(scalarReg, static_cast<float>(scalar), mask);
            Reg::Sub(value, value, scalarReg, mask);
            Reg::Cast<T, float, ForeachUnaryRegbase::UNARY_CAST_TO_STORAGE>(result, value, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(output + r * VL, result, mask);
        } else {
            auto mask = Reg::UpdateMask<T, Reg::RegTraitNumOne>(remaining);
            Reg::RegTensor<T, Reg::RegTraitNumOne> value, scalarReg;
            Reg::LoadAlign<T, Reg::LoadDist::DIST_NORM>(value, input + r * VL);
            Reg::Duplicate(scalarReg, static_cast<T>(scalar), mask);
            Reg::Sub(value, value, scalarReg, mask);
            Reg::StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(output + r * VL, value, mask);
        }
    }
}
template <typename T, typename ScalarT>
struct Policy {
    static constexpr bool kUsesExtraUb = false;
    __aicore__ static inline void Run(LocalTensor<T>* input, LocalTensor<T> output, ScalarT scalar, uint32_t count)
    {
        asc_vf_call<SubVF<T, ScalarT>>(reinterpret_cast<__ubuf__ T*>(input[0].GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ T*>(output.GetPhyAddr()), scalar, count);
    }
};
template <typename T, typename ScalarT, typename Tiling>
using FlatKernel = ForeachFlatRegbase::FlatFlowKernel<T, ScalarT, Tiling, 1, Policy<T, ScalarT>>;
} // namespace ForeachSubScalar
#endif
