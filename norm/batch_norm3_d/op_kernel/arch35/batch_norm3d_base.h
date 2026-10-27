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
 * \file batch_norm3d_base.h
 * \brief
 */
#ifndef NORM_BATCH_NORM3D_BASE_H
#define NORM_BATCH_NORM3D_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "batch_norm3_d_tiling_data.h"
#include "batch_norm3d_compensated_math.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace BatchNorm3DOps {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int32_t BUFFER_DEPTH = 1;
static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
static constexpr uint16_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static uint32_t FLOAT_BYTES = 4;
constexpr static float POS_INF = 3.40282366920938E+38;
constexpr static float zero = 0.0f;
constexpr static int64_t NDDMA_THRESHOLD = 32;
constexpr static int64_t NDDMA_SECOND_DIM = 1;
constexpr static int64_t NDDMA_THIRD_DIM = 2;
constexpr static int64_t NDDMA_DIM_NUM = 3;
constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <bool NEED_MAX = true>
__aicore__ inline void ComputeRstd(RegTensor<float>& var, RegTensor<float>& rstd, MaskReg& mask, float epsilon)
{
    NormCommon::ComputeRstdNewtonRaphsonReg<NEED_MAX>(var, rstd, mask, epsilon);
    // The common refinement starts from 1 / (variance + epsilon). Above
    // 2^126 that reciprocal is subnormal even though rstd is normal.
    // Take the square root first in this range to avoid losing the reciprocal.
    constexpr float MAX_NORMAL_RECIPROCAL_INPUT = 0x1p126f;
    RegTensor<float> sqrtVar;
    RegTensor<float> one;
    RegTensor<float> largeVarRstd;
    MaskReg largeVariance;
    MaskReg finiteLargeVariance;
    CompareScalar<float, CMPMODE::GT>(largeVariance, var, MAX_NORMAL_RECIPROCAL_INPUT, mask);
    CompareScalar<float, CMPMODE::LT>(finiteLargeVariance, var, POS_INF, largeVariance);
    Sqrt(sqrtVar, var, finiteLargeVariance);
    Duplicate(one, 1.0f, finiteLargeVariance);
    Div(largeVarRstd, one, sqrtVar, finiteLargeVariance);
    Select(rstd, largeVarRstd, rstd, finiteLargeVariance);
}

__aicore__ inline void NormalizeWithScaleBias(RegTensor<float>& x, RegTensor<float>& scale, RegTensor<float>& bias,
                                              RegTensor<float>& mean, RegTensor<float>& rstd, RegTensor<float>& y,
                                              MaskReg& mask)
{
    RegTensor<float> scaled;
    RegTensor<float> absValue;
    RegTensor<float> alternate;
    MaskReg overflow;
    MaskReg finiteScale;
    MaskReg finiteInput;
    Sub(x, x, mean, mask);
    Mul(scaled, x, scale, mask);
    Abs(absValue, scaled, mask);
    CompareScalar<float, CMPMODE::EQ>(overflow, absValue, POS_INF, mask);
    Abs(absValue, scale, mask);
    CompareScalar<float, CMPMODE::LT>(finiteScale, absValue, POS_INF, overflow);
    Abs(absValue, x, mask);
    CompareScalar<float, CMPMODE::LT>(finiteInput, absValue, POS_INF, finiteScale);
    // Preserve the usual multiplication order. Only when finite operands
    // overflow in x * scale, try normalization first. Unconditionally doing
    // so can instead overflow x * rstd when a small scale would avoid it.
    Mul(alternate, x, rstd, finiteInput);
    Mul(alternate, alternate, scale, finiteInput);
    Mul(scaled, scaled, rstd, mask);
    Select(scaled, alternate, scaled, finiteInput);
    Add(y, scaled, bias, mask);
}

template <typename T>
__aicore__ inline void ComputeInferenceRstd(RegTensor<float>& var, RegTensor<float>& rstd, RegTensor<float>& rstdLow,
                                            MaskReg& mask, float epsilon)
{
    if constexpr (!IsSameType<T, float>::value) {
        ComputeRstd(var, rstd, mask, epsilon);
        Duplicate(rstdLow, 0.0f, mask);
    } else {
        using namespace CompensatedBN;
        RegTensor<float> eps, magnitude, varScaled, epsScaled, factor, hi, lo;
        MaskReg large;
        Duplicate(eps, epsilon, mask);
        Abs(magnitude, var, mask);
        Abs(factor, eps, mask);
        Max(magnitude, magnitude, factor, mask);
        CompareScalar<float, CMPMODE::GT>(large, magnitude, 0x1p124f, mask);
        // Scale before adding so two finite FP32 inputs can have a finite
        // inverse root even when their unscaled sum exceeds FLT_MAX.
        Duplicate(factor, 1.0f, mask);
        Duplicate(hi, 0.25f, mask);
        Select(factor, hi, factor, large);
        Mul(varScaled, var, factor, mask);
        Mul(epsScaled, eps, factor, mask);
        TwoSum(hi, lo, varScaled, epsScaled, mask);
        ComputeInverseRoot(rstd, rstdLow, hi, lo, mask);
        Duplicate(factor, 1.0f, mask);
        Duplicate(hi, 0.5f, mask);
        Select(factor, hi, factor, large);
        Mul(rstd, rstd, factor, mask);
        Mul(rstdLow, rstdLow, factor, mask);
    }
}

template <typename T>
__aicore__ inline void NormalizeInference(RegTensor<float>& x, RegTensor<float>& scale, RegTensor<float>& bias,
                                          RegTensor<float>& mean, RegTensor<float>& rstd, RegTensor<float>& rstdLow,
                                          RegTensor<float>& y, MaskReg& mask)
{
    RegTensor<float> magnitude;
    MaskReg infiniteBias, finiteX, finiteMean, finiteScale, finiteAffine;
    Abs(magnitude, bias, mask);
    CompareScalar<float, CMPMODE::EQ>(infiniteBias, magnitude, POS_INF, mask);
    Abs(magnitude, x, mask);
    CompareScalar<float, CMPMODE::LE>(finiteX, magnitude, 0x1.fffffep127f, infiniteBias);
    Abs(magnitude, mean, mask);
    CompareScalar<float, CMPMODE::LE>(finiteMean, magnitude, 0x1.fffffep127f, finiteX);
    Abs(magnitude, scale, mask);
    CompareScalar<float, CMPMODE::LE>(finiteScale, magnitude, 0x1.fffffep127f, finiteMean);
    Abs(magnitude, rstd, mask);
    CompareScalar<float, CMPMODE::LE>(finiteAffine, magnitude, 0x1.fffffep127f, finiteScale);
    if constexpr (!IsSameType<T, float>::value) {
        NormalizeWithScaleBias(x, scale, bias, mean, rstd, y, mask);
    } else {
        using namespace CompensatedBN;
        RegTensor<float> negativeMean, zeros, centered, centeredLow, scaled, scaledLow;
        RegTensor<float> magnitude, alternate, alternateLow;
        MaskReg overflow, finiteScale, finiteInput;
        Duplicate(zeros, 0.0f, mask);
        Muls(negativeMean, mean, -1.0f, mask);
        TwoSum(centered, centeredLow, x, negativeMean, mask);
        MulPair(scaled, scaledLow, centered, centeredLow, scale, zeros, mask);
        Abs(magnitude, scaled, mask);
        CompareScalar<float, CMPMODE::EQ>(overflow, magnitude, POS_INF, mask);
        Abs(magnitude, scale, mask);
        CompareScalar<float, CMPMODE::LT>(finiteScale, magnitude, POS_INF, overflow);
        Abs(magnitude, centered, mask);
        CompareScalar<float, CMPMODE::LT>(finiteInput, magnitude, POS_INF, finiteScale);
        MulPair(alternate, alternateLow, centered, centeredLow, rstd, rstdLow, finiteInput);
        MulPair(alternate, alternateLow, alternate, alternateLow, scale, zeros, finiteInput);
        MulPair(scaled, scaledLow, scaled, scaledLow, rstd, rstdLow, mask);
        Select(scaled, alternate, scaled, finiteInput);
        Select(scaledLow, alternateLow, scaledLow, finiteInput);
        // Preserve the product's residual until after the bias addition;
        // rounding it earlier loses relative accuracy near cancellation.
        AddPair(scaled, scaledLow, scaled, scaledLow, bias, zeros, mask);
        Add(y, scaled, scaledLow, mask);
    }
    // A finite affine term plus infinite bias is that bias. Do not let an
    // intermediate FP32 overflow turn this into opposite Inf + Inf -> NaN.
    Select(y, bias, y, finiteAffine);
}

__aicore__ inline void UpdateWelfordMean(RegTensor<float>& mean, RegTensor<float>& value, RegTensor<float>& scaledDelta,
                                         MaskReg& mask)
{
    RegTensor<float> absMean;
    RegTensor<float> absValue;
    RegTensor<float> nonFiniteMean;
    MaskReg meanIsInf;
    MaskReg valueIsInf;
    MaskReg eitherIsInf;
    Abs(absMean, mean, mask);
    Abs(absValue, value, mask);
    CompareScalar<float, CMPMODE::EQ>(meanIsInf, absMean, POS_INF, mask);
    CompareScalar<float, CMPMODE::EQ>(valueIsInf, absValue, POS_INF, mask);
    AscendC::MicroAPI::Or(eitherIsInf, meanIsInf, valueIsInf, mask);
    Add(nonFiniteMean, mean, value, mask);
    // Keep the finite Welford recurrence. For infinite inputs, subtracting
    // the old mean can turn a same-sign infinite mean into NaN. Addition
    // preserves that mean and still propagates NaN or opposite infinities.
    Add(mean, mean, scaledDelta, mask);
    Select(mean, nonFiniteMean, mean, eitherIsInf);
}

template <typename T>
__aicore__ inline void LoadTensorForDtypeT(AscendC::MicroAPI::RegTensor<float>& dst, __local_mem__ T* src,
                                           AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
    } else { // fp16、bf16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void LoadTensorForDtypeTBrc(RegTensor<float>& dst, __local_mem__ T* src, MaskReg& preg,
                                              uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst, (__local_mem__ float*)src + offset);
    } else { // fp16、bf16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_BRC_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitB162B32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeT(RegTensor<float>& dst1, RegTensor<float>& dst2, __local_mem__ T* src1,
                                              __local_mem__ T* src2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                              uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16R;
        RegTensor<half> xFp16Q;
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half*)(src1) + (src1Offset)));
        DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16R;
        RegTensor<bfloat16_t> xFp16Q;
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ bfloat16_t*)(src1) + (src1Offset)));
        DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        DataCopy(dst1, ((__local_mem__ float*)(src1) + (src1Offset)));
        DataCopy(dst2, ((__local_mem__ float*)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void LoadTwoTensorForDtypeTBrc(RegTensor<float>& dst1, RegTensor<float>& dst2, __local_mem__ T* src1,
                                                 __local_mem__ T* src2, MaskReg& dst1Preg, MaskReg& dst2Preg,
                                                 uint32_t src1Offset, uint32_t src2Offset)
{
    if constexpr (IsSameType<T, half>::value) {
        RegTensor<half> xFp16Q;
        RegTensor<half> xFp16R;
        DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16Q, ((__local_mem__ half*)(src1) + (src1Offset)));
        DataCopy<half, LoadDist::DIST_BRC_B16>(xFp16R, ((__local_mem__ half*)(src2) + (src2Offset)));
        Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        RegTensor<bfloat16_t> xFp16Q;
        RegTensor<bfloat16_t> xFp16R;
        DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16Q, ((__local_mem__ bfloat16_t*)(src1) + (src1Offset)));
        DataCopy<bfloat16_t, LoadDist::DIST_BRC_B16>(xFp16R, ((__local_mem__ bfloat16_t*)(src2) + (src2Offset)));
        Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
        Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
    } else {
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst1, ((__local_mem__ float*)(src1) + (src1Offset)));
        DataCopy<float, LoadDist::DIST_BRC_B32>(dst2, ((__local_mem__ float*)(src2) + (src2Offset)));
    }
}

template <typename T>
__aicore__ inline void StoreTensorForDtypeT(__local_mem__ T* dst, AscendC::MicroAPI::RegTensor<float>& src,
                                            AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T> xFp16;
        Cast<T, float, castTraitB322B16>(xFp16, src, preg);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

} // namespace BatchNorm3DOps

#endif // NORM_BATCH_NORM3D_BASE_H
