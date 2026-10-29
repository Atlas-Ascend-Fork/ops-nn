/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_kernel/arch92/swiglu_hi_f4_quant_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void swiglu_hi_f4_quant(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale,
                                                         GM_ADDR workspace, GM_ADDR tiling);

class SwigluHiF4QuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluHiF4QuantKernelTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SwigluHiF4QuantKernelTest TearDown" << std::endl; }
};

static void RunKernelTest(size_t batchSize, size_t seqLen, size_t hiddenDim, uint64_t tilingKey)
{
    size_t inputSize = batchSize * seqLen * hiddenDim * 2 * sizeof(uint16_t);
    size_t outputSize = batchSize * seqLen * hiddenDim * sizeof(uint8_t) / 2;
    size_t scaleSize = batchSize * seqLen * (hiddenDim / 64) * sizeof(uint32_t);
    size_t tilingDataSize = sizeof(SwigluHiF4QuantTilingData);
    size_t workspaceSize = 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* group_index = nullptr;
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    SwigluHiF4QuantTilingData* tilingData = reinterpret_cast<SwigluHiF4QuantTilingData*>(tiling);
    tilingData->usedCoreNum = 1;
    // Data shape
    tilingData->inputDimNeg1 = 128;
    tilingData->inputDimNeg2 = 1;
    tilingData->outputYDimNeg1 = 64;
    tilingData->groupIndexNum = 0;
    // Memory allocation
    tilingData->basicDimNeg1 = 512;
    tilingData->basicDimNeg2 = 1;
    tilingData->maxBasicNumUbDimNeg1 = 1;
    tilingData->maxBasicNumUbDimNeg2 = 42;
    tilingData->ubLoopPerRow = 1;
    tilingData->ubTailPerRow = 64;
    // Inter-core split
    tilingData->frontCoreNum = 0;
    tilingData->frontCoreBasicNumDimNeg2 = 2;
    tilingData->frontCoreLoopTimes = 1;
    tilingData->frontCoreLastLoopBasicNum = 2;
    tilingData->tailCoreBasicNumDimNeg2 = 1;
    tilingData->tailCoreLoopTimes = 1;
    tilingData->tailCoreLastLoopBasicNum = 1;
    // SwiGLU attributes
    tilingData->swigluMode = 0;
    tilingData->activateLeft = 1;
    tilingData->clampLimit = 7.000000;
    tilingData->gluAlpha = 1.702000;
    tilingData->gluBias = 1.000000;
    tilingData->scaleAlg = 0;

    uint32_t blockDim = 1;
    ICPU_SET_TILING_KEY(1000);
    ICPU_RUN_KF(swiglu_hi_f4_quant, blockDim, x, group_index, y, scale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(SwigluHiF4QuantKernelTest, basic_test)
{
    size_t batchSize = 8;
    size_t seqLen = 128;
    size_t hiddenDim = 4096;
    uint64_t tilingKey = 1000;
    RunKernelTest(batchSize, seqLen, hiddenDim, tilingKey);
}
