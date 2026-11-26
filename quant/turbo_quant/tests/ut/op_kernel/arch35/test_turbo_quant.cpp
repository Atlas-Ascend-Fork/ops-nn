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
 * \file test_turbo_quant.cpp
 * \brief ascend950 (arch35/regbase) kernel UT for TurboQuant.
 *        Branch-coverage cases are run-only with an informational golden compare (older c310
 *        simulators skip unknown regbase instructions). The maximum-batch paths -- 24 tokens per
 *        Compute, alone and followed by a tail batch -- assert the golden compare: the flat 910b
 *        suite already asserts against this kernel on the same target, so a simulator able to run
 *        it is required anyway. The flat suite cannot cover batch 24 itself (its compile-time
 *        maximum is 12 on 910b).
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

extern "C" void turbo_quant(uint8_t* latent, uint8_t* centroids, uint8_t* y, uint8_t* scale, uint8_t* workspace,
                            uint8_t* tiling);

namespace {
constexpr uint32_t HEAD_DIM = 512;
constexpr uint32_t PACKED_BYTES = 256;
constexpr uint32_t SCALE_BYTES = 2;
constexpr uint32_t N_CENT = 16;
constexpr size_t WORKSPACE_SIZE = 32;
constexpr char DATA_DIR[] = "./turbo_quant_data";

// The golden script is shared with the flat suite; `asserted` turns its exit code into a test
// verdict instead of a log line, for the cases whose numerical coverage this file claims.
void RunGoldenCompare(uint32_t numTokens, bool asserted)
{
    std::string cmd = std::string("cd ") + DATA_DIR + " && python3 compare_data.py " + std::to_string(numTokens);
    int ret = system(cmd.c_str());
    if (asserted) {
        EXPECT_EQ(ret, 0) << "golden compare failed, numTokens=" << numTokens;
    } else {
        std::cout << "[INFO] golden compare numTokens=" << numTokens << " ret=" << ret
                  << " (informational on simulator)" << std::endl;
    }
}

// The control-flow cases mirror the 910b suite's shape matrix and stay run-only: every batch/tail
// branch of the arch35 kernel gets executed and must not fault; outputs go through the golden
// script informationally so simulator runs can read the residual.
// tokensPerCore is what decides the per-core token count, and the kernel runs
// count = min(remaining, tokensPerBatch) per Compute: a case only exercises a full batch of B when
// the core owns at least B tokens.
void RunSmokeCase(uint32_t numTokens, uint32_t tokensPerCore, uint32_t tokensPerBatch, const char* dist = "gauss_lat",
                  bool asserted = false)
{
    const uint32_t blockDim = numTokens == 0 ? 1 : (numTokens + tokensPerCore - 1) / tokensPerCore;

    system("cp -rf "
           "../../../../quant/turbo_quant/tests/ut/op_kernel/"
           "turbo_quant_data ./");
    system("chmod -R 755 ./turbo_quant_data/");
    std::string genCmd = std::string("cd ") + DATA_DIR + " && python3 gen_data.py " + std::to_string(numTokens) + " " +
                         dist;
    ASSERT_EQ(system(genCmd.c_str()), 0);

    size_t latentByteSize = static_cast<size_t>(numTokens) * HEAD_DIM * sizeof(float);
    size_t centByteSize = N_CENT * sizeof(float);
    size_t yByteSize = static_cast<size_t>(numTokens) * PACKED_BYTES;
    size_t scaleByteSize = static_cast<size_t>(numTokens) * SCALE_BYTES;

    // GmAlloc(0) is not meaningful; the empty-tensor case still needs valid pointers to hand the kernel
    uint8_t* latent = (uint8_t*)AscendC::GmAlloc(latentByteSize == 0 ? 1 : latentByteSize);
    uint8_t* centroids = (uint8_t*)AscendC::GmAlloc(centByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize == 0 ? 1 : yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize == 0 ? 1 : scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(TurboQuantTilingData));

    if (latentByteSize > 0) {
        ReadFile(std::string(DATA_DIR) + "/input_latent.bin", latentByteSize, latent, latentByteSize);
    }
    ReadFile(std::string(DATA_DIR) + "/input_centroids.bin", centByteSize, centroids, centByteSize);

    TurboQuantTilingData* tilingData = reinterpret_cast<TurboQuantTilingData*>(tiling);
    tilingData->numTokens = numTokens;
    tilingData->tokensPerCore = tokensPerCore;
    tilingData->headDim = HEAD_DIM;
    tilingData->packedBytes = PACKED_BYTES;
    tilingData->tokensPerBatch = tokensPerBatch;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(turbo_quant, blockDim, latent, centroids, y, scale, workspace, (uint8_t*)(tilingData));

    WriteFile(std::string(DATA_DIR) + "/output_y.bin", y, yByteSize == 0 ? 1 : yByteSize);
    WriteFile(std::string(DATA_DIR) + "/output_scale.bin", scale, scaleByteSize == 0 ? 1 : scaleByteSize);

    AscendC::GmFree((void*)latent);
    AscendC::GmFree((void*)centroids);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)scale);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    // ICPU_RUN_KF runs on the CPU simulator; see the flat suite for the env marker rationale.
    setenv("TQ_UT_SIMULATOR", "1", 1);
    RunGoldenCompare(numTokens, asserted);
}
} // namespace

class turbo_quant_arch35_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "turbo_quant_arch35_test SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "turbo_quant_arch35_test TearDown\n" << std::endl; }
};

TEST_F(turbo_quant_arch35_test, single_token_single_core) { RunSmokeCase(1, 1, 1); }

// Ragged split: the last core owns fewer tokens than tokensPerCore, in ragged batches (3+3+2).
TEST_F(turbo_quant_arch35_test, batched_ragged_tail) { RunSmokeCase(33, 8, 3); }

// Batch at the arch35 compile-time maximum of 24, with every core owning a whole batch:
// 48 tokens / 24 per core -> two cores, each running exactly one Compute(24), the second one
// at a nonzero token offset. The golden compare is asserted (norm/scale slots 13..24, nibble
// staging and packed output offsets of the upper half of the batch are all checked).
TEST_F(turbo_quant_arch35_test, regbase_batched_max) { RunSmokeCase(48, 24, 24, "gauss_lat", true); }

// A full 24-token batch followed by a tail batch inside one core (30 tokens on one core: 24 + 6):
// the tail batch reuses the queues and staging areas right after a maximum-size batch.
TEST_F(turbo_quant_arch35_test, regbase_full_batch_then_tail) { RunSmokeCase(30, 30, 24, "gauss_lat", true); }

// Empty tensor must write nothing and not fault.
TEST_F(turbo_quant_arch35_test, empty_tensor) { RunSmokeCase(0, 1, 1); }

// Wide-range inputs saturate into the outermost buckets after normalization.
TEST_F(turbo_quant_arch35_test, uniform_wide_range) { RunSmokeCase(33, 8, 8, "uniform"); }

// Non-finite inputs: NaN collapses every comparison, +-INF drives 1/norm to 0 / NaN.
TEST_F(turbo_quant_arch35_test, input_nan) { RunSmokeCase(9, 4, 4, "nan"); }

TEST_F(turbo_quant_arch35_test, input_posinf) { RunSmokeCase(9, 4, 4, "posinf"); }
