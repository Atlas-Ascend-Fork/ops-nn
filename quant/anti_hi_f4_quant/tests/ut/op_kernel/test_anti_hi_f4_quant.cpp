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
#include "../../../op_kernel/anti_hi_f4_quant.cpp"

using namespace std;

extern "C" __global__ __aicore__ void anti_hi_f4_quant(GM_ADDR x, GM_ADDR scale, GM_ADDR y, GM_ADDR workspace,
                                                       GM_ADDR tiling);

class anti_hi_f4_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "anti_hi_f4_quant_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "anti_hi_f4_quant_test TearDown" << endl; }
};

TEST_F(anti_hi_f4_quant_test, test_case_per_head_bf16)
{
    int64_t rowNum = 64;
    int64_t colNum = 128;

    size_t inputXSize = rowNum * colNum / 2;
    size_t outputYSize = rowNum * colNum * sizeof(bfloat16_t);
    size_t inputScaleSize = (rowNum / 64) * colNum * sizeof(uint16_t) * 2;
    size_t tilingDataSize = sizeof(AntiHiF4QuantNotTailTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(inputScaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    AntiHiF4QuantNotTailTilingData* tilingData = reinterpret_cast<AntiHiF4QuantNotTailTilingData*>(tiling);

    tilingData->tilingKey = 1;
    tilingData->totalCoreNum = 64;
    tilingData->usedCoreNum = 1;
    tilingData->ubSize = 253952;
    tilingData->vfLen = 256;
    tilingData->workspaceSize = 0;
    tilingData->dstType = 27;
    tilingData->xDimNeg1 = colNum;
    tilingData->xDimNeg2 = rowNum;
    tilingData->xDimNeg3 = 1;
    tilingData->splitBlockColSize = rowNum;
    tilingData->splitBlockRowSize = colNum;
    tilingData->tailBlockRowSize = colNum;
    tilingData->tailBlockColSize = rowNum;
    tilingData->splitBlockColNum = 1;
    tilingData->splitBlockRowNum = 1;
    tilingData->headCoreLoopNum = 1;
    tilingData->tailCoreLoopNum = 1;
    tilingData->headCoreNum = 1;
    tilingData->totalTaskNum = 1;
    tilingData->ubBlockNum = 1;
    tilingData->taskNumPerBatch = 1;

    ICPU_SET_TILING_KEY(1);

    auto kernel = [](GM_ADDR x, GM_ADDR scale, GM_ADDR y, GM_ADDR ws, GM_ADDR tiling) {
        ::anti_hi_f4_quant<1>(x, scale, y, ws, tiling);
    };
    ICPU_RUN_KF(kernel, blockDim, x, scale, y, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

TEST_F(anti_hi_f4_quant_test, test_case_per_channel_bf16)
{
    int64_t rowNum = 64;
    int64_t colNum = 128;
    int64_t groupNum = rowNum * colNum / 64;

    size_t inputXSize = rowNum * colNum / 2;
    size_t outputYSize = rowNum * colNum * sizeof(bfloat16_t);
    size_t inputScaleSize = groupNum * sizeof(float);
    size_t tilingDataSize = sizeof(AntiHiF4QuantTailTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(inputScaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    AntiHiF4QuantTailTilingData* tilingData = reinterpret_cast<AntiHiF4QuantTailTilingData*>(tiling);

    tilingData->totalCoreNum = 64;
    tilingData->usedCoreNum = 2;
    tilingData->tilingKey = 0;
    tilingData->groupNum = 64;
    tilingData->alignGroupNum = 8;
    tilingData->rowPerHeadCore = 1;
    tilingData->rowPerTailCore = 1;
    tilingData->multiRowNumHeadCore = 64;
    tilingData->multiRowNumTailCore = 64;
    tilingData->scaleLoopEle = 1;
    tilingData->scaleLoopTimes = 8;
    tilingData->scaleLoopTail = 8;
    tilingData->outLoopEle = 4;
    tilingData->outLoopTimes = 32;
    tilingData->outLoopTail = 32;
    tilingData->dstType = 27;

    ICPU_SET_TILING_KEY(0);

    auto kernel = [](GM_ADDR x, GM_ADDR scale, GM_ADDR y, GM_ADDR ws, GM_ADDR tiling) {
        ::anti_hi_f4_quant<0>(x, scale, y, ws, tiling);
    };
    ICPU_RUN_KF(kernel, blockDim, x, scale, y, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(y);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}
