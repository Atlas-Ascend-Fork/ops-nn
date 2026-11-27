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
 * \file foreach_neg_regbase.h
 * \brief Register-only compute policy for ForeachNeg on Ascend950.
 */

#ifndef FOREACH_NEG_REGBASE_H
#define FOREACH_NEG_REGBASE_H

#include "../../foreach_utils/arch35/foreach_unary_regbase_common.h"

namespace ForeachNeg {
using namespace AscendC;

struct NegOperation {
    static constexpr bool kPromoteHalf = false;

    template <typename T>
    __simd_callee__ static inline void Apply(Reg::RegTensor<T, Reg::RegTraitNumOne>& dst,
                                             Reg::RegTensor<T, Reg::RegTraitNumOne>& src, Reg::MaskReg& mask)
    {
        Reg::Neg(dst, src, mask);
    }
};

template <typename T, typename Tiling>
using ForeachNegRegbase = ForeachUnaryRegbase::UnaryApplyKernel<T, Tiling, NegOperation>;
} // namespace ForeachNeg

#endif // FOREACH_NEG_REGBASE_H
