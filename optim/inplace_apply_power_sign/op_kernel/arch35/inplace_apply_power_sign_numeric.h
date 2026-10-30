/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INPLACE_APPLY_POWER_SIGN_NUMERIC_H
#define INPLACE_APPLY_POWER_SIGN_NUMERIC_H

#include "kernel_operator.h"

namespace NsInplaceApplyPowerSign {

// The CPU TensorFlow worker sets MXCSR DAZ and FTZ (ScopedFlushDenormal).
// Match that arithmetic environment without altering the external tensors,
// dtype conversion, normal values, NaN or Inf. Preserve the sign of zero.
// VF loads use native masked DATA_BLOCK_COPY with stride 1 (contiguous 32B
// blocks), including when the allocation or last tile is shorter than a VL.
__simd_vf__ inline void PowerSignFlushFp32Vf(__ubuf__ uint32_t* data, uint32_t count)
{
    using namespace AscendC;
    constexpr uint32_t width = GetVecLen() / sizeof(uint32_t);
    const uint16_t repeats = static_cast<uint16_t>((count + width - 1) / width);
    Reg::RegTensor<uint32_t> input, magnitude, sign, absMask, signMask, result;
    Reg::MaskReg active, tiny;
    Reg::Duplicate(absMask, 0x7fffffffU);
    Reg::Duplicate(signMask, 0x80000000U);
    for (uint16_t i = 0; i < repeats; ++i) {
        active = Reg::UpdateMask<uint32_t>(count);
        Reg::LoadAlign<uint32_t, Reg::DataCopyMode::DATA_BLOCK_COPY>(input, data + i * width, 1, active);
        Reg::And(magnitude, input, absMask, active);
        Reg::And(sign, input, signMask, active);
        Reg::CompareScalar<uint32_t, CMPMODE::LT>(tiny, magnitude, 0x00800000U, active);
        Reg::Select(result, sign, input, tiny);
        Reg::StoreAlign(data + i * width, result, active);
    }
}

__aicore__ inline void PowerSignFlushFp32(const AscendC::LocalTensor<float>& value, uint32_t count)
{
    AscendC::PipeBarrier<PIPE_V>();
    PowerSignFlushFp32Vf(reinterpret_cast<__ubuf__ uint32_t*>(value.GetPhyAddr()), count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline float PowerSignFlushScalarFp32(float value)
{
    union Bits {
        float value;
        uint32_t bits;
    } data;
    data.value = value;
    if ((data.bits & 0x7fffffffU) < 0x00800000U) {
        data.bits &= 0x80000000U;
    }
    return data.value;
}

__aicore__ inline void PowerSignMuls(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
                                     float scalar, uint32_t count)
{
    AscendC::Muls(dst, src, PowerSignFlushScalarFp32(scalar), count);
    PowerSignFlushFp32(dst, count);
}

__aicore__ inline void PowerSignMul(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& lhs,
                                    const AscendC::LocalTensor<float>& rhs, uint32_t count)
{
    AscendC::Mul(dst, lhs, rhs, count);
    PowerSignFlushFp32(dst, count);
}

__aicore__ inline void PowerSignAdd(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& lhs,
                                    const AscendC::LocalTensor<float>& rhs, uint32_t count)
{
    AscendC::Add(dst, lhs, rhs, count);
    PowerSignFlushFp32(dst, count);
}

__aicore__ inline void PowerSignSub(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& lhs,
                                    const AscendC::LocalTensor<float>& rhs, uint32_t count)
{
    AscendC::Sub(dst, lhs, rhs, count);
    PowerSignFlushFp32(dst, count);
}

__simd_callee__ inline void PowerSignFma(AscendC::Reg::RegTensor<float>& dst, AscendC::Reg::RegTensor<float>& lhs,
                                         AscendC::Reg::RegTensor<float>& rhs, AscendC::Reg::RegTensor<float>& addend,
                                         AscendC::Reg::MaskReg& active)
{
    // MulDstAdd is the native fused dst = dst * src0 + src1 instruction.
    // A temporary also makes this helper safe when dst aliases an operand.
    AscendC::Reg::RegTensor<float> value;
    AscendC::Reg::Move(value, lhs, active);
    AscendC::Reg::MulDstAdd(value, rhs, addend, active);
    AscendC::Reg::Move(dst, value, active);
}

__simd_callee__ inline void PowerSignFmaConstants(AscendC::Reg::RegTensor<float>& dst,
                                                  AscendC::Reg::RegTensor<float>& lhs, float multiplier, float addend,
                                                  AscendC::Reg::MaskReg& active)
{
    AscendC::Reg::RegTensor<float> rhs, offset;
    AscendC::Reg::Duplicate(rhs, multiplier);
    AscendC::Reg::Duplicate(offset, addend);
    PowerSignFma(dst, lhs, rhs, offset, active);
}

__simd_callee__ inline void PowerSignSmallExpPolynomial(AscendC::Reg::RegTensor<float>& result,
                                                        AscendC::Reg::RegTensor<float>& x,
                                                        AscendC::Reg::MaskReg& active)
{
    using namespace AscendC;
    Reg::RegTensor<float> x2, even, odd, low, constant;
    // Eigen GenericPacketMathFunctions.h:509-559. In |x|<=0.25 the range
    // reduction has m=0, r=x and 2^m=1, so these are the complete operations.
    Reg::Mul(x2, x, x, active);
    PowerSignFmaConstants(even, x2, 1.37449637986719608306884765625e-3f, 4.166965186595916748046875e-2f, active);
    PowerSignFmaConstants(odd, x2, 8.36894474923610687255859375e-3f, 0.16666518151760101318359375f, active);
    Reg::Duplicate(constant, 0.49999988079071044921875f);
    PowerSignFma(even, x2, even, constant, active);
    Reg::Adds(low, x, 1.0f, active);
    PowerSignFma(result, x, odd, even, active);
    PowerSignFma(result, x2, result, low, active);
}

__simd_vf__ inline void PowerSignSmallExpVf(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
{
    using namespace AscendC;
    constexpr uint32_t width = GetVecLen() / sizeof(float);
    const uint16_t repeats = static_cast<uint16_t>((count + width - 1) / width);
    Reg::RegTensor<float> input, magnitude, reduced, zero, polynomial, fallback, result;
    Reg::MaskReg active, small;
    Reg::Duplicate(zero, 0.0f);
    for (uint16_t i = 0; i < repeats; ++i) {
        active = Reg::UpdateMask<float>(count);
        Reg::LoadAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(input, src + i * width, 1, active);
        Reg::LoadAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(fallback, dst + i * width, 1, active);
        Reg::Abs(magnitude, input, active);
        Reg::CompareScalar<float, CMPMODE::LE>(small, magnitude, 0.25f, active);
        // Keep large/NaN/Inf lanes out of the polynomial's arithmetic.
        Reg::Select(reduced, input, zero, small);
        PowerSignSmallExpPolynomial(polynomial, reduced, active);
        Reg::Select(result, polynomial, fallback, small);
        Reg::StoreAlign(dst + i * width, result, active);
    }
}

__aicore__ inline void PowerSignExp(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
                                    uint32_t count)
{
    // Preserve the existing native Exp handling outside the proven small range.
    AscendC::Exp(dst, src, static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();
    PowerSignSmallExpVf(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()), count);
    AscendC::PipeBarrier<PIPE_V>();
}

// Eigen's floating-point sign preserves NaN. AscendC::Sign selects zero
// when both comparisons are false, which would hide NaN in the momentum.
__simd_vf__ inline void PowerSignSignFp32(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
{
    using namespace AscendC;
    constexpr uint32_t width = GetVecLen() / sizeof(float);
    const uint16_t repeats = static_cast<uint16_t>((count + width - 1) / width);
    Reg::RegTensor<float> input, result, zero, one, minusOne;
    Reg::MaskReg active, negative, positive, nan;
    Reg::Duplicate(zero, 0.0f);
    Reg::Duplicate(one, 1.0f);
    Reg::Duplicate(minusOne, -1.0f);
    for (uint16_t i = 0; i < repeats; ++i) {
        active = Reg::UpdateMask<float>(count);
        Reg::LoadAlign<float, Reg::DataCopyMode::DATA_BLOCK_COPY>(input, src + i * width, 1, active);
        Reg::CompareScalar<float, CMPMODE::LT>(negative, input, 0.0f, active);
        Reg::CompareScalar<float, CMPMODE::GT>(positive, input, 0.0f, active);
        Reg::Compare<float, CMPMODE::NE>(nan, input, input, active);
        Reg::Select(result, minusOne, zero, negative);
        Reg::Select(result, one, result, positive);
        Reg::Select(result, input, result, nan);
        Reg::StoreAlign(dst + i * width, result, active);
    }
}

__aicore__ inline void PowerSignSign(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src,
                                     uint32_t count)
{
    PowerSignSignFp32(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()), count);
}

} // namespace NsInplaceApplyPowerSign
#endif
