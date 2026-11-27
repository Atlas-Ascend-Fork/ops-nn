/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FOREACH_SIGMOID_REGBASE_H
#define FOREACH_SIGMOID_REGBASE_H
#include "../../foreach_utils/arch35/foreach_unary_regbase_common.h"
namespace ForeachSigmoid {
using namespace AscendC;
struct Operation {
    static constexpr bool kPromoteHalf = true;
    template <typename T>
    __simd_callee__ static inline void Apply(Reg::RegTensor<T, Reg::RegTraitNumOne>& output,
                                             Reg::RegTensor<T, Reg::RegTraitNumOne>& input, Reg::MaskReg& mask)
    {
        Reg::RegTensor<T, Reg::RegTraitNumOne> one;
        Reg::Muls(output, input, T(-1.0), mask);
        Reg::Exp(output, output, mask);
        Reg::Adds(output, output, T(1.0), mask);
        Reg::Duplicate(one, T(1.0), mask);
        Reg::Div(output, one, output, mask);
    }
};
template <typename T, typename Tiling>
using FlatKernel = ForeachUnaryRegbase::UnaryApplyKernel<T, Tiling, Operation>;
} // namespace ForeachSigmoid
#endif
