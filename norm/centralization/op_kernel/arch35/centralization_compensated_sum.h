/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_COMPENSATED_SUM_H_
#define OPS_NORM_CENTRALIZATION_COMPENSATED_SUM_H_

#include "kernel_operator.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace Centralization {
__aicore__ inline void SplitMeanProductOperand(float value, float& high, float& low)
{
    volatile float split = 4097.0f * value;
    high = split - (split - value);
    low = value - high;
}

// Compute sum - mean * count. Keep both the split and the high-part product in range
// by scaling sum and mean together; restore the scale only after cancellation.
__aicore__ inline float MeanProductResidual(float mean, float count, float sum)
{
    float restoreScale = 1.0f;
    if (sum > 0x1p110f || sum < -0x1p110f) {
        sum *= 0x1p-16f;
        mean *= 0x1p-16f;
        restoreScale = 0x1p16f;
    }
    float meanHi, meanLo, countHi, countLo;
    SplitMeanProductOperand(mean, meanHi, meanLo);
    SplitMeanProductOperand(count, countHi, countLo);
    volatile float product = mean * count;
    volatile float highError = meanHi * countHi - product;
    volatile float crossHi = meanHi * countLo;
    volatile float crossLo = meanLo * countHi;
    volatile float lowProduct = meanLo * countLo;
    const float productError = ((highError + crossHi) + crossLo) + lowProduct;
    return ((sum - product) - productError) * restoreScale;
}

// Knuth's TwoSum keeps the rounding error of each vector addition.
__simd_callee__ inline void AddWithResidual(AscendC::Reg::RegTensor<float>& sum,
                                            AscendC::Reg::RegTensor<float>& correction,
                                            AscendC::Reg::RegTensor<float>& value, AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<float> next;
    RegTensor<float> rhs;
    RegTensor<float> lhs;
    Add(next, sum, value, mask);
    Sub(rhs, next, sum, mask);
    Sub(lhs, next, rhs, mask);
    Sub(lhs, sum, lhs, mask);
    Sub(rhs, value, rhs, mask);
    Add(lhs, lhs, rhs, mask);
    Add(correction, correction, lhs, mask);
    sum = next;
}
constexpr uint32_t kResidualOffset = 32 / sizeof(float);

__aicore__ inline void ReduceTileWithResidual(AscendC::LocalTensor<float>& calc, AscendC::LocalTensor<float>& result,
                                              uint32_t validElements)
{
    using namespace AscendC::Reg;
    constexpr uint32_t lanes = NormCommon::V_LENGTH;
    constexpr uint16_t foldStages = 6;
    static_assert(lanes == (1U << foldStages));
    __local_mem__ float* inputAddr = (__local_mem__ float*)calc.GetPhyAddr();
    __local_mem__ float* resultAddr = (__local_mem__ float*)result.GetPhyAddr();
    const uint16_t loops = static_cast<uint16_t>((validElements + lanes - 1) / lanes);
    __VEC_SCOPE__
    {
        RegTensor<float> sum;
        RegTensor<float> correction;
        RegTensor<float> value;
        RegTensor<float> zero;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        Duplicate(zero, 0.0f, full);
        Duplicate(sum, 0.0f, full);
        Duplicate(correction, 0.0f, full);
        for (uint16_t block = 0; block < loops; ++block) {
            uint32_t remaining = validElements - block * lanes;
            MaskReg active = UpdateMask<float>(remaining);
            LoadAlign(value, inputAddr + block * lanes);
            Select(value, value, zero, active);
            AddWithResidual(sum, correction, value, full);
        }
        // Deinterleave each vector to put the even and odd members of each pair
        // in the low lanes. Carry both sums and their accumulated residuals.
        for (uint16_t stage = 1; stage <= foldStages; ++stage) {
            uint32_t activeCount = lanes >> stage;
            MaskReg active = UpdateMask<float>(activeCount);
            RegTensor<float> even;
            RegTensor<float> odd;
            RegTensor<float> evenCorrection;
            RegTensor<float> oddCorrection;
            DeInterleave(even, odd, sum, sum);
            DeInterleave(evenCorrection, oddCorrection, correction, correction);
            Add(correction, evenCorrection, oddCorrection, active);
            AddWithResidual(even, correction, odd, active);
            sum = even;
        }
        MaskReg one = CreateMask<float, MaskPattern::VL1>();
        StoreAlign(resultAddr, sum, one);
        StoreAlign(resultAddr + kResidualOffset, correction, one);
    }
}

} // namespace Centralization
#endif
