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

// Retain the rounding residual of x - meanHi until the low part of the mean is subtracted.
__simd_callee__ inline void SubtractMeanWithResidual(AscendC::Reg::RegTensor<float>& value,
                                                     AscendC::Reg::RegTensor<float>& negativeHi,
                                                     AscendC::Reg::RegTensor<float>& low, AscendC::Reg::MaskReg& active)
{
    using namespace AscendC::Reg;
    RegTensor<float> residual;
    Duplicate(residual, 0.0f, active);
    AddWithResidual(value, residual, negativeHi, active);
    Sub(residual, residual, low, active);
    Add(residual, value, residual, active);
    RegTensor<float> magnitude;
    Abs(magnitude, value, active);
    MaskReg finite;
    Compares<float, CMPMODE::LE>(finite, magnitude, 0x1.fffffep127f, active);
    Select(value, residual, value, finite);
}

// Leave one additional bit of headroom for the compensated sum.
__aicore__ inline float ShortRowRestoreFactor(int64_t count)
{
    uint32_t power = 1;
    while (power < count) {
        power <<= 1;
    }
    return static_cast<float>(2 * power);
}

// One lane per row. Gather a column at a time so each TwoSum correction
// is carried through the entire reduction, then scatter the centered values.
__aicore__ inline void CenterPackedRows(AscendC::LocalTensor<float>& input, AscendC::LocalTensor<float>& output,
                                        uint32_t rowCount, uint16_t cols, uint32_t columnStride = 1,
                                        uint32_t firstColumn = 0)
{
    using namespace AscendC::Reg;
    using NormCommon::V_LENGTH;
    __local_mem__ float* inputAddr = (__local_mem__ float*)input.GetPhyAddr();
    __local_mem__ float* outputAddr = (__local_mem__ float*)output.GetPhyAddr();
    const uint32_t stride = cols * columnStride;
    const uint16_t blocks = static_cast<uint16_t>((rowCount + V_LENGTH - 1) / V_LENGTH);
    // Keep headroom for the sum and the residual calculation even at FLT_MAX.
    const float restoreFactor = ShortRowRestoreFactor(cols);
    const float scaleFactor = 1.0f / restoreFactor;
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> rowOffsets;
        RegTensor<uint32_t> offsets;
        RegTensor<float> count;
        RegTensor<float> zero;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        AscendC::Reg::Arange(reinterpret_cast<RegTensor<int32_t>&>(rowOffsets), 0);
        Muls(rowOffsets, rowOffsets, stride, full);
        Duplicate(count, static_cast<float>(cols), full);
        Duplicate(zero, 0.0f, full);
        uint32_t remaining = rowCount;
        for (uint16_t block = 0; block < blocks; ++block) {
            MaskReg active = UpdateMask<float>(remaining);
            const uint32_t base = block * V_LENGTH * stride;
            RegTensor<float> sum;
            RegTensor<float> correction;
            RegTensor<float> value;
            RegTensor<float> meanHi;
            RegTensor<float> meanLo;
            RegTensor<float> scale;
            RegTensor<float> divisor;
            RegTensor<float> one;
            // Probe in the same column order as the compensated sum, one lane per row.
            Duplicate(sum, 0.0f, active);
            for (uint16_t col = 0; col < cols; ++col) {
                Adds(offsets, rowOffsets, col * columnStride + firstColumn, active);
                AscendC::Reg::Gather(value, inputAddr + base, offsets, active);
                Add(sum, sum, value, active);
            }
            Abs(value, sum, active);
            MaskReg unscaled;
            Compares<float, CMPMODE::LE>(unscaled, value, 0x1.fffffep127f, active);
            Duplicate(one, 1.0f, active);
            Duplicate(scale, scaleFactor, active);
            Select(scale, one, scale, unscaled);
            Mul(divisor, count, scale, active);
            Duplicate(sum, 0.0f, active);
            Duplicate(correction, 0.0f, active);
            for (uint16_t col = 0; col < cols; ++col) {
                Adds(offsets, rowOffsets, col * columnStride + firstColumn, active);
                AscendC::Reg::Gather(value, inputAddr + base, offsets, active);
                Mul(value, value, scale, active);
                AddWithResidual(sum, correction, value, active);
            }
            Div(meanHi, sum, divisor, active);
            Muls(meanLo, meanHi, -1.0f, active);
            MulDstAdd<float, AscendC::Reg::MaskMergeMode::ZEROING>(meanLo, divisor, sum, active);
            Add(meanLo, meanLo, correction, active);
            Div(meanLo, meanLo, divisor, active);
            Abs(value, meanHi, active);
            MaskReg finite;
            Compares<float, CMPMODE::LE>(finite, value, 3.4028235e38f, active);
            Select(meanLo, meanLo, zero, finite);
            RegTensor<float> negMean;
            Muls(negMean, meanHi, -1.0f, active);
            for (uint16_t col = 0; col < cols; ++col) {
                Adds(offsets, rowOffsets, col * columnStride + firstColumn, active);
                AscendC::Reg::Gather(value, inputAddr + base, offsets, active);
                SubtractMeanWithResidual(value, negMean, meanLo, active);
                AscendC::Reg::Scatter(outputAddr + base, value, offsets, active);
            }
        }
    }
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
