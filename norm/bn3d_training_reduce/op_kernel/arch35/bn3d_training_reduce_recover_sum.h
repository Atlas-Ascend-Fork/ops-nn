/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN3D_TRAINING_REDUCE_RECOVER_SUM_H_
#define BN3D_TRAINING_REDUCE_RECOVER_SUM_H_

#include "kernel_operator.h"
#include "simt_api/common_functions.h"
#include "simt_api/device_functions.h"

namespace BN3DTrainingReduceOps {
constexpr uint32_t RECOVER_SUM_THREADS = 128;
constexpr uint32_t SUM_WORDS = 12;
constexpr uint32_t FLOAT_INF_BITS = 0x7f800000U;

// Finite FP32 inputs are integer multiples of 2^-149. The largest input needs
// 277 bits, and an int64-sized reduction needs at most 340 magnitude bits.
// A signed 384-bit accumulator therefore cannot overflow. It is used only
// after the ordinary floating-point sum has become non-finite.
struct WideSum {
    uint32_t words[SUM_WORDS]{};
    bool positiveInf{false};
    bool negativeInf{false};
    bool nan{false};

    __simt_callee__ inline void AddWord(uint32_t index, uint32_t value, bool negative)
    {
        for (; index < SUM_WORDS && value != 0; ++index) {
            const uint32_t previous = words[index];
            words[index] = negative ? previous - value : previous + value;
            value = negative ? static_cast<uint32_t>(previous < value) : static_cast<uint32_t>(words[index] < previous);
        }
    }

    __simt_callee__ inline void AddBits(uint32_t bits)
    {
        const uint32_t exponent = (bits >> 23U) & 255U;
        const uint32_t fraction = bits & 0x7fffffU;
        const bool negative = (bits >> 31U) != 0;
        if (exponent == 255U) {
            nan |= fraction != 0;
            positiveInf |= fraction == 0 && !negative;
            negativeInf |= fraction == 0 && negative;
            return;
        }
        const uint32_t mantissa = fraction | (exponent == 0 ? 0U : 0x800000U);
        const uint32_t shift = exponent == 0 ? 0U : exponent - 1U;
        const uint32_t index = shift / 32U;
        const uint64_t shifted = static_cast<uint64_t>(mantissa) << (shift % 32U);
        AddWord(index, static_cast<uint32_t>(shifted), negative);
        AddWord(index + 1U, static_cast<uint32_t>(shifted >> 32U), negative);
    }

    __simt_callee__ inline uint32_t Bit(uint32_t position) const
    {
        return (words[position / 32U] >> (position % 32U)) & 1U;
    }

    __simt_callee__ inline uint32_t RoundBits()
    {
        if (nan || (positiveInf && negativeInf)) {
            return 0x7fc00000U;
        }
        if (positiveInf || negativeInf) {
            return FLOAT_INF_BITS | (negativeInf ? 0x80000000U : 0U);
        }
        const uint32_t sign = words[SUM_WORDS - 1U] & 0x80000000U;
        if (sign != 0) {
            uint64_t carry = 1;
            for (uint32_t i = 0; i < SUM_WORDS; ++i) {
                const uint64_t value = static_cast<uint64_t>(~words[i]) + carry;
                words[i] = static_cast<uint32_t>(value);
                carry = value >> 32U;
            }
        }
        int32_t top = static_cast<int32_t>(SUM_WORDS * 32U) - 1;
        while (top >= 0 && Bit(static_cast<uint32_t>(top)) == 0) {
            --top;
        }
        if (top < 0) {
            return 0;
        }
        if (top < 23) {
            return sign | words[0]; // Exact subnormal, including its sign.
        }
        if (top > 276) {
            return sign | FLOAT_INF_BITS;
        }
        uint32_t shift = static_cast<uint32_t>(top) - 23U;
        const uint32_t word = shift / 32U;
        const uint64_t pair = static_cast<uint64_t>(words[word]) | (static_cast<uint64_t>(words[word + 1U]) << 32U);
        uint32_t mantissa = static_cast<uint32_t>(pair >> (shift % 32U));
        if (shift != 0 && Bit(shift - 1U) != 0) {
            bool sticky = false;
            for (uint32_t bit = 0; bit + 1U < shift; ++bit) {
                sticky |= Bit(bit) != 0;
            }
            mantissa += static_cast<uint32_t>(sticky || (mantissa & 1U) != 0);
        }
        if (mantissa == 0x1000000U) {
            mantissa >>= 1U;
            ++shift;
        }
        const uint32_t exponent = shift + 1U;
        return sign | (exponent >= 255U ? FLOAT_INF_BITS : ((exponent << 23U) | (mantissa & 0x7fffffU)));
    }
};

template <typename T_X>
__simt_vf__ __aicore__ LAUNCH_BOUND(RECOVER_SUM_THREADS) inline void RecoverNonFiniteSum(__gm__ T_X* x,
                                                                                         __ubuf__ float* sums,
                                                                                         uint64_t numN, uint64_t numC,
                                                                                         uint64_t numR0, uint64_t cBase,
                                                                                         uint32_t outputs, uint32_t c0)
{
    for (uint32_t out = AscendC::Simt::GetThreadIdx(); out < outputs; out += RECOVER_SUM_THREADS) {
        if ((__float_as_uint(sums[out]) & FLOAT_INF_BITS) != FLOAT_INF_BITS) {
            continue;
        }
        WideSum accumulator{};
        const uint64_t channel = cBase + out / c0;
        for (uint64_t n = 0; n < numN; ++n) {
            const uint64_t base = (n * numC + channel) * numR0;
            for (uint64_t r = out % c0; r < numR0; r += c0) {
                if constexpr (AscendC::IsSameType<T_X, float>::value) {
                    accumulator.AddBits(reinterpret_cast<__gm__ uint32_t*>(x)[base + r]);
                } else {
                    // BF16 -> FP32 is an exact bit expansion, including subnormals.
                    const uint32_t bits = reinterpret_cast<__gm__ uint16_t*>(x)[base + r];
                    accumulator.AddBits(bits << 16U);
                }
            }
        }
        sums[out] = __uint_as_float(accumulator.RoundBits());
    }
}
} // namespace BN3DTrainingReduceOps
#endif // BN3D_TRAINING_REDUCE_RECOVER_SUM_H_
