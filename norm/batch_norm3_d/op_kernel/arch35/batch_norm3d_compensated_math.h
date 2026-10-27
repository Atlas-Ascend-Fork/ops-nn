/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef NORM_BATCH_NORM3D_COMPENSATED_MATH_H
#define NORM_BATCH_NORM3D_COMPENSATED_MATH_H

#include "kernel_operator.h"

namespace BatchNorm3DOps {
namespace CompensatedBN {
using namespace AscendC::Reg;

// Only the residual is sanitized. The leading value keeps IEEE NaN/Inf
// propagation, including an infinite mean and a NaN centered variance.
__aicore__ inline void FiniteResidual(RegTensor<float>& value, MaskReg& mask)
{
    RegTensor<float> magnitude, zeros;
    MaskReg finite;
    Abs(magnitude, value, mask);
    CompareScalar<float, AscendC::CMPMODE::LE>(finite, magnitude, 0x1.fffffep127f, mask);
    Duplicate(zeros, 0.0f, mask);
    Select(value, value, zeros, finite);
}

__aicore__ inline void TwoSum(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& a, RegTensor<float>& b,
                              MaskReg& mask)
{
    RegTensor<float> sum, v, error, temp;
    Add(sum, a, b, mask);
    Sub(v, sum, a, mask);
    Sub(temp, sum, v, mask);
    Sub(error, a, temp, mask);
    Sub(temp, b, v, mask);
    Add(error, error, temp, mask);
    FiniteResidual(error, mask);
    Adds(hi, sum, 0.0f, mask);
    Adds(lo, error, 0.0f, mask);
}

__aicore__ inline void AddPair(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& ah, RegTensor<float>& al,
                               RegTensor<float>& bh, RegTensor<float>& bl, MaskReg& mask)
{
    RegTensor<float> sum, error, low;
    TwoSum(sum, error, ah, bh, mask);
    Add(low, al, bl, mask);
    Add(error, error, low, mask);
    TwoSum(hi, lo, sum, error, mask);
}

__aicore__ inline void MulPair(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& ah, RegTensor<float>& al,
                               RegTensor<float>& bh, RegTensor<float>& bl, MaskReg& mask)
{
    RegTensor<float> product, error, low;
    Mul(product, ah, bh, mask);
    Muls(error, product, -1.0f, mask);
    MulAddDst(error, ah, bh, mask);
    MulAddDst(error, ah, bl, mask);
    MulAddDst(error, al, bh, mask);
    MulAddDst(error, al, bl, mask);
    FiniteResidual(error, mask);
    TwoSum(hi, lo, product, error, mask);
}

__aicore__ inline void ComputeInverseRoot(AscendC::Reg::RegTensor<float>& rh, AscendC::Reg::RegTensor<float>& rl,
                                          AscendC::Reg::RegTensor<float>& hi, AscendC::Reg::RegTensor<float>& lo,
                                          AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    static constexpr DivSpecificMode divMode{MaskMergeMode::ZEROING, false, AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
    RegTensor<float> root, one, zero, ph, pl, error;
    Duplicate(one, 1.0f, mask);
    Duplicate(zero, 0.0f, mask);
    // The compensated Newton step below refines the intrinsic seed and
    // also supports epsilon above the fast-sqrt algorithm's 2^126 limit.
    Sqrt(root, hi, mask);
    Div<float, &divMode>(rh, one, root, mask);
    MulPair(ph, pl, hi, lo, rh, zero, mask);
    MulPair(ph, pl, ph, pl, rh, zero, mask);
    Sub(error, one, ph, mask);
    Sub(error, error, pl, mask);
    Mul(rl, rh, error, mask);
    Muls(rl, rl, 0.5f, mask);
    FiniteResidual(rl, mask);
    TwoSum(rh, rl, rh, rl, mask);
}

} // namespace CompensatedBN
} // namespace BatchNorm3DOps
#endif // NORM_BATCH_NORM3D_COMPENSATED_MATH_H
