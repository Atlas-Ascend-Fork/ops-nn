/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "arch35/swiglu_mx_quant_perf.h"
#include "arch35/swiglu_mxfp4_quant_perf.h"

namespace {
template <typename X, typename Y, bool outputOrigin>
__global__ __aicore__ void MxTailKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale,
                                        GM_ADDR yOrigin, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    auto* data = reinterpret_cast<SwigluGroupQuantTilingData*>(tiling);
    if constexpr (std::is_same_v<Y, fp4x2_e2m1_t> || std::is_same_v<Y, fp4x2_e1m2_t>) {
        SwigluGroupQuant::SwigluMxFp4QuantPerf<X, Y, uint8_t, outputOrigin> op;
        op.Init(x, weight, groupIndex, y, yScale, yOrigin, nullptr, data, &pipe);
        op.Process();
    } else {
        SwigluGroupQuant::SwigluMxQuantPerf<X, Y, uint8_t, outputOrigin> op;
        op.Init(x, weight, groupIndex, y, yScale, yOrigin, nullptr, data, &pipe);
        op.Process();
    }
}

float DecodeFp8(uint8_t bits, bool e5m2)
{
    const int mantissaBits = e5m2 ? 2 : 3;
    const int bias = e5m2 ? 15 : 7;
    const int exponent = (bits & 0x7f) >> mantissaBits;
    const int mantissa = bits & ((1 << mantissaBits) - 1);
    const float value = std::ldexp(
        exponent == 0 ? mantissa / float(1 << mantissaBits) : 1.0f + mantissa / float(1 << mantissaBits),
        (exponent == 0 ? 1 : exponent) - bias);
    return bits & 0x80 ? -value : value;
}

float DecodeFp4(uint8_t bits, bool e2m1)
{
    constexpr float e2[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    constexpr float e1[] = {0, 0.25f, 0.5f, 0.75f, 1, 1.25f, 1.5f, 1.75f};
    const float value = e2m1 ? e2[bits & 7] : e1[bits & 7];
    return bits & 8 ? -value : value;
}

// Allocate guard regions in GM as well as checking every valid output. Different groups/rows
// use different powers of two so row pitches, chunk offsets and padding cannot alias silently.
template <typename X, typename Y, bool outputOrigin = true>
void CheckMxTail(int64_t d, int64_t dFactor = 0, bool weighted = true, bool clamp = false, int64_t groupListType = 1)
{
    constexpr bool fp4 = std::is_same_v<Y, fp4x2_e2m1_t> || std::is_same_v<Y, fp4x2_e1m2_t>;
    constexpr bool e2m1 = std::is_same_v<Y, fp4x2_e2m1_t>;
    constexpr bool e5m2 = std::is_same_v<Y, fp8_e5m2_t>;
    constexpr int64_t rows = 7;
    // CANN 9.2.1's CPU size checker treats compact FP8 scale rows as 32B padded
    // rows. Use one row per tile here; FP4 has a larger scale allocation and tests
    // compact multi-row DMA below with rowFactor=2.
    constexpr int64_t rowFactor = fp4 ? 2 : 1;
    constexpr int64_t activeRows = 6;
    constexpr uint8_t sentinel = 0xa5;
    const int64_t h = d / 2;
    const int64_t groups = h / 32;
    const int64_t scalePitch = (groups + 1) / 2 * 2;
    const size_t yBytes = rows * h / (fp4 ? 2 : 1);
    const size_t originBytes = rows * h * sizeof(X);
    const size_t scaleBytes = rows * scalePitch;
    dFactor = dFactor == 0 ? h : dFactor;
    auto alloc = [](size_t bytes) { return reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes)); };
    uint8_t* x = alloc(rows * d * sizeof(X));
    uint8_t* y = alloc(yBytes + 64);
    uint8_t* origin = alloc(originBytes + 64);
    uint8_t* scales = alloc(scaleBytes + 64);
    uint8_t* weight = alloc(rows * sizeof(float));
    uint8_t* groupIndex = alloc(2 * groupListType * sizeof(int64_t));
    uint8_t* tiling = alloc(sizeof(SwigluGroupQuantTilingData));
    std::memset(y, sentinel, yBytes + 64);
    std::memset(origin, sentinel, originBytes + 64);
    std::memset(scales, sentinel, scaleBytes + 64);
    auto* counts = reinterpret_cast<int64_t*>(groupIndex);
    if (groupListType == 1) {
        counts[0] = 2;
        counts[1] = activeRows - 2;
    } else {
        counts[0] = 97;
        counts[1] = 2;
        counts[2] = -17;
        counts[3] = activeRows - 2;
    }
    std::vector<float> expected(rows * h);
    for (int64_t row = 0; row < rows; ++row) {
        reinterpret_cast<float*>(weight)[row] = std::ldexp(1.0f, row % 3 - 1);
        for (int64_t col = 0; col < h; ++col) {
            const float up = std::ldexp(col % 2 ? -1.0f : 1.0f, (row + col / 32) % 6 - 2);
            reinterpret_cast<X*>(x)[row * d + col] = static_cast<X>(1.0f);
            reinterpret_cast<X*>(x)[row * d + h + col] = static_cast<X>(up);
            const float boundedUp = clamp ? std::max(-1.5f, std::min(up, 1.5f)) : up;
            expected[row * h + col] = boundedUp / (1.0f + std::exp(-1.0f));
        }
    }
    auto* td = reinterpret_cast<SwigluGroupQuantTilingData*>(tiling);
    *td = {};
    td->bs = rows;
    td->d = d;
    td->splitD = h;
    td->scaleCol = groups;
    td->rowOfFormerBlock = 4;
    td->rowOfTailBlock = 3;
    td->rowLoopOfFormerBlock = (4 + rowFactor - 1) / rowFactor;
    td->rowLoopOfTailBlock = (3 + rowFactor - 1) / rowFactor;
    td->rowFactor = rowFactor;
    td->tailRowFactorOfFormerBlock = rowFactor;
    td->tailRowFactorOfTailBlock = 1;
    td->dFactor = dFactor;
    td->dLoop = (h + dFactor - 1) / dFactor;
    td->tailDFactor = h - (td->dLoop - 1) * dFactor;
    td->roundScale = 1;
    td->outputOrigin = outputOrigin;
    td->hasClampLimit = clamp;
    td->clampLimit = 1.5f;
    td->ubSize = 253952;
    td->coreNum = 2;
    td->groupListType = groupListType;
    td->g = 2 * groupListType;
    td->gFactor = td->g;
    td->gLoop = 1;
    td->tailGFactor = td->g;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kernel = MxTailKernel<X, Y, outputOrigin>;
    ICPU_RUN_KF(kernel, 2, x, weighted ? weight : nullptr, groupIndex, y, scales, outputOrigin ? origin : nullptr,
                tiling);
    for (int64_t row = 0; row < activeRows; ++row) {
        for (int64_t group = 0; group < groups; ++group) {
            const float w = weighted ? reinterpret_cast<float*>(weight)[row] : 1.0f;
            float value = expected[row * h + group * 32] * w;
            if constexpr (fp4) {
                value = static_cast<float>(static_cast<X>(value));
            }
            const int exponent = fp4 ? int(std::floor(std::log2(std::fabs(value)))) - (e2m1 ? 2 : 0) :
                                       int(std::ceil(std::log2(std::max(std::fabs(value), 0.0001f) /
                                                               (e5m2 ? 57344.0f : 448.0f))));
            ASSERT_EQ(scales[row * scalePitch + group], exponent + 127)
                << "D=" << d << " row=" << row << " group=" << group;
            for (int64_t col = group * 32; col < (group + 1) * 32; ++col) {
                const int64_t index = row * h + col;
                const float quantized = fp4 ? DecodeFp4((y[index / 2] >> ((index % 2) * 4)) & 15, e2m1) :
                                              DecodeFp8(y[index], e5m2);
                const float restored = std::ldexp(quantized, exponent);
                // Half of the largest relative quantization step, plus input rounding allowance.
                const float tolerance = std::fabs(expected[index] * w) * (fp4 ? 0.19f : (e5m2 ? 0.13f : 0.065f));
                ASSERT_NEAR(restored, expected[index] * w, tolerance) << "D=" << d << " index=" << index;
                if constexpr (outputOrigin) {
                    ASSERT_NEAR(static_cast<float>(reinterpret_cast<X*>(origin)[index]), expected[index],
                                std::fabs(expected[index]) * 0.005f)
                        << "D=" << d << " index=" << index;
                }
            }
        }
        if (scalePitch != groups) {
            EXPECT_EQ(scales[row * scalePitch + groups], 0) << "row=" << row;
        }
    }
    for (size_t i = activeRows * h / (fp4 ? 2 : 1); i < yBytes + 64; ++i)
        EXPECT_EQ(y[i], sentinel);
    for (size_t i = activeRows * scalePitch; i < scaleBytes + 64; ++i)
        EXPECT_EQ(scales[i], sentinel);
    for (size_t i = outputOrigin ? activeRows * h * sizeof(X) : 0; i < originBytes + 64; ++i)
        EXPECT_EQ(origin[i], sentinel);
    for (auto* ptr : {x, y, origin, scales, weight, groupIndex, tiling})
        AscendC::GmFree(ptr);
}

TEST(SwigluGroupQuantMxTailTest, fp8_e4m3_fp16)
{
    for (int64_t d : {64, 128, 192, 256, 320, 448, 512, 576})
        CheckMxTail<half, fp8_e4m3fn_t>(d);
    CheckMxTail<half, fp8_e4m3fn_t>(832, 128, true, true, 2);
    CheckMxTail<half, fp8_e4m3fn_t, false>(320, 0, false);
}

TEST(SwigluGroupQuantMxTailTest, fp8_e5m2_bf16)
{
    for (int64_t d : {64, 128, 192, 256, 320, 448, 512, 576})
        CheckMxTail<bfloat16_t, fp8_e5m2_t>(d, 0, true, true, 2);
    CheckMxTail<bfloat16_t, fp8_e5m2_t>(832, 128, false);
}

TEST(SwigluGroupQuantMxTailTest, fp4_e2m1_fp16)
{
    for (int64_t d : {64, 128, 192, 256, 320, 448, 512, 576})
        CheckMxTail<half, fp4x2_e2m1_t>(d);
    CheckMxTail<half, fp4x2_e2m1_t>(832, 160, true, true, 2);
    CheckMxTail<half, fp4x2_e2m1_t, false>(320, 96, false);
}

TEST(SwigluGroupQuantMxTailTest, fp4_e1m2_bf16)
{
    for (int64_t d : {64, 128, 192, 256, 320, 448, 512, 576})
        CheckMxTail<bfloat16_t, fp4x2_e1m2_t>(d, 0, true, true, 2);
    CheckMxTail<bfloat16_t, fp4x2_e1m2_t>(832, 160, false);
}
} // namespace
