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
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../test_add_rms_norm_cast_regbase.h"
#include "add_rms_norm_cast_apt.cpp"

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
#error "Recompute tests must execute the Ascend 950 kernel branch"
#endif

namespace {
struct RecomputeGmDeleter {
    void operator()(uint8_t* pointer) const { AscendC::GmFree(pointer); }
};
using RecomputeGmBuffer = std::unique_ptr<uint8_t, RecomputeGmDeleter>;
RecomputeGmBuffer RecomputeAllocate(size_t bytes)
{
    return RecomputeGmBuffer(reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes)));
}

void RunRecomputeOutputs(uint64_t n, uint64_t tile, uint32_t aliasInput = 0)
{
    // Multiple rows per core and an incomplete last core. Shapes exercise both tail-fold branches.
    using Input = DTYPE_X1;
    constexpr uint64_t ROWS = 5;
    constexpr uint32_t CORES = 2;
    constexpr size_t GUARD = 16;
    constexpr float SENTINEL = 123.0f;
    const size_t elements = ROWS * n;
    auto x1 = RecomputeAllocate((elements + 2 * GUARD) * sizeof(Input));
    auto x2 = RecomputeAllocate((elements + 2 * GUARD) * sizeof(Input));
    auto gamma = RecomputeAllocate(n * sizeof(Input));
    auto x = RecomputeAllocate((elements + 2 * GUARD) * sizeof(Input));
    auto y1 = RecomputeAllocate((elements + 2 * GUARD) * sizeof(float));
    auto y2 = RecomputeAllocate((elements + 2 * GUARD) * sizeof(Input));
    auto rstd = RecomputeAllocate((ROWS + 2 * GUARD) * sizeof(float));
    auto workspace = RecomputeAllocate(4096);
    auto tiling = RecomputeAllocate(sizeof(AddRmsNormCastRegbaseTilingData));
    auto* a = reinterpret_cast<Input*>(x1.get()) + GUARD;
    auto* b = reinterpret_cast<Input*>(x2.get()) + GUARD;
    auto* g = reinterpret_cast<Input*>(gamma.get());
    auto* outX = reinterpret_cast<Input*>(x.get());
    auto* outY1 = reinterpret_cast<float*>(y1.get());
    auto* outY2 = reinterpret_cast<Input*>(y2.get());
    auto* outRstd = reinterpret_cast<float*>(rstd.get());
    std::fill_n(reinterpret_cast<Input*>(x1.get()), elements + 2 * GUARD, static_cast<Input>(SENTINEL));
    std::fill_n(reinterpret_cast<Input*>(x2.get()), elements + 2 * GUARD, static_cast<Input>(SENTINEL));
    // Save the fp32 sums before the kernel overwrites an aliased input in ICPU GM.
    std::vector<float> originalSums(elements);
    for (size_t i = 0; i < elements; ++i) {
        a[i] = static_cast<Input>((static_cast<int>(i % 31) - 15) * 0.125f);
        b[i] = static_cast<Input>(aliasInput == 0 ? (static_cast<int>(i % 19) - 9) * 0.0625f :
                                                    (1 + i % 3) * 0.000244140625f);
        originalSums[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
    }
    for (size_t j = 0; j < n; ++j) {
        g[j] = static_cast<Input>(0.5f + (j % 7) * 0.125f);
    }
    std::fill_n(outX, elements + 2 * GUARD, static_cast<Input>(SENTINEL));
    std::fill_n(outY1, elements + 2 * GUARD, SENTINEL);
    std::fill_n(outY2, elements + 2 * GUARD, static_cast<Input>(SENTINEL));
    std::fill_n(outRstd, ROWS + 2 * GUARD, SENTINEL);
    std::fill_n(workspace.get(), 4096, uint8_t{0xA5});
    if (aliasInput != 0) {
        outX = aliasInput == 1 ? a - GUARD : b - GUARD;
        ASSERT_TRUE(std::any_of(originalSums.begin(), originalSums.end(),
                                [](float sum) { return sum != static_cast<float>(static_cast<Input>(sum)); }));
    }
    auto* data = reinterpret_cast<AddRmsNormCastRegbaseTilingData*>(tiling.get());
    *data = {};
    data->numM = ROWS;
    data->numN = n;
    data->baseM = 1;
    data->baseN = tile;
    data->baseNDtypeAlign = tile;
    data->baseNReduceAlign = tile;
    data->powerSplit = tile;
    data->powerLoop = 1;
    while (data->powerLoop * 2 * tile <= n) {
        data->powerLoop *= 2;
    }
    data->mPerCore = 3;
    data->mLastCore = 2;
    data->epsilon = 1e-6f;
    data->avgFactor = 1.0f / n;
    ICPU_SET_TILING_KEY(101);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm_cast, CORES, reinterpret_cast<uint8_t*>(a), reinterpret_cast<uint8_t*>(b), gamma.get(),
                reinterpret_cast<uint8_t*>(outY1 + GUARD), reinterpret_cast<uint8_t*>(outY2 + GUARD),
                reinterpret_cast<uint8_t*>(outRstd + GUARD), reinterpret_cast<uint8_t*>(outX + GUARD), workspace.get(),
                tiling.get());
    for (size_t row = 0; row < ROWS; ++row) {
        double squares = 0;
        for (size_t j = 0; j < n; ++j) {
            const size_t k = row * n + j;
            const float sum = originalSums[k];
            squares += static_cast<double>(sum) * sum;
        }
        const float expectedRstd = 1.0 / std::sqrt(squares / n + data->epsilon);
        ASSERT_NEAR(outRstd[GUARD + row], expectedRstd, 1e-5f);
        for (size_t j = 0; j < n; ++j) {
            const size_t k = row * n + j;
            const float sum = originalSums[k];
            const float expectedY = (sum * expectedRstd) * static_cast<float>(g[j]);
            ASSERT_EQ(static_cast<float>(outX[GUARD + k]), static_cast<float>(static_cast<Input>(sum)));
            ASSERT_NEAR(outY1[GUARD + k], expectedY, 1e-4f + std::abs(expectedY) * 1e-4f);
            ASSERT_NEAR(static_cast<float>(outY2[GUARD + k]), expectedY, 0.001f + std::abs(expectedY) * 0.008f);
        }
    }
    for (size_t i = 0; i < GUARD; ++i) {
        ASSERT_EQ(static_cast<float>((a - GUARD)[i]), SENTINEL);
        ASSERT_EQ(static_cast<float>(a[elements + i]), SENTINEL);
        ASSERT_EQ(static_cast<float>((b - GUARD)[i]), SENTINEL);
        ASSERT_EQ(static_cast<float>(b[elements + i]), SENTINEL);
        ASSERT_EQ(static_cast<float>(outX[i]), SENTINEL);
        ASSERT_EQ(static_cast<float>(outX[GUARD + elements + i]), SENTINEL);
        ASSERT_EQ(outY1[i], SENTINEL);
        ASSERT_EQ(outY1[GUARD + elements + i], SENTINEL);
        ASSERT_EQ(static_cast<float>(outY2[i]), SENTINEL);
        ASSERT_EQ(static_cast<float>(outY2[GUARD + elements + i]), SENTINEL);
        ASSERT_EQ(outRstd[i], SENTINEL);
        ASSERT_EQ(outRstd[GUARD + ROWS + i], SENTINEL);
    }
    for (size_t i = 0; i < 4096; ++i) {
        ASSERT_EQ(workspace.get()[i], uint8_t{0xA5});
    }
}

class ADD_RMS_RECOMPUTE_SUITE : public testing::TestWithParam<uint64_t> {};

TEST_P(ADD_RMS_RECOMPUTE_SUITE, matches_all_outputs_and_preserves_guards_and_workspace)
{
    RunRecomputeOutputs(GetParam(), 8192);
}

INSTANTIATE_TEST_SUITE_P(TailBoundaries, ADD_RMS_RECOMPUTE_SUITE,
                         testing::Values(8192ULL, 10000ULL, 12289ULL, 16000ULL, 16384ULL, 50001ULL, 229376ULL));

class ADD_RMS_ALIAS_SUITE : public testing::TestWithParam<std::pair<uint64_t, uint32_t>> {};

TEST_P(ADD_RMS_ALIAS_SUITE, matches_all_outputs_and_preserves_guards_and_workspace)
{
    const auto& param = GetParam();
    RunRecomputeOutputs(param.first, 8192, param.second);
}

INSTANTIATE_TEST_SUITE_P(ExactInputAliases, ADD_RMS_ALIAS_SUITE,
                         testing::Values(std::make_pair(16000ULL, 1U), std::make_pair(16000ULL, 2U),
                                         std::make_pair(16385ULL, 1U), std::make_pair(16385ULL, 2U)));
} // namespace
