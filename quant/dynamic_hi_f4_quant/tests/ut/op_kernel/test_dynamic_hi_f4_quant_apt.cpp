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
#include "dynamic_hi_f4_quant_data_utils.h"
#include "../../../op_kernel/arch35/dynamic_hi_f4_quant_tilingdata.h"

using namespace std;

extern "C" __global__ __aicore__ void dynamic_hi_f4_quant(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace,
                                                          GM_ADDR tiling);

class DynamicHiF4QuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "DynamicHiF4QuantKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "DynamicHiF4QuantKernelTest TearDown" << endl; }
};

static void RunKernelTestAxisNeg1(int64_t dim0, int64_t dim1, int64_t totalCoreNum, uint64_t tilingKey)
{
    int64_t groupNum = dim0 * dim1 / 64;
    int64_t alignGroupNum = 8;
    int64_t usedCoreNum = (groupNum + 31) / 32;
    if (usedCoreNum > totalCoreNum)
        usedCoreNum = totalCoreNum;

    int64_t blockFormer = (groupNum + usedCoreNum - 1) / usedCoreNum;
    usedCoreNum = (groupNum + blockFormer - 1) / blockFormer;
    int64_t blockTail = groupNum - (usedCoreNum - 1) * blockFormer;

    int64_t ubLoopOfFormerBlock = (blockFormer + alignGroupNum - 1) / alignGroupNum;
    int64_t ubLoopOfTailBlock = (blockTail + alignGroupNum - 1) / alignGroupNum;
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * alignGroupNum;
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * alignGroupNum;

    size_t inputSize = dim0 * dim1 * sizeof(uint16_t);
    size_t outputSize = dim0 * dim1 / 2;
    size_t scaleSize = dim0 * (dim1 / 64) * sizeof(float);
    size_t tilingDataSize = sizeof(DynamicHiF4QuantTilingData);
    size_t workspaceSize = 32;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    DynamicHiF4QuantTilingData* tilingData = reinterpret_cast<DynamicHiF4QuantTilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->groupNum = blockFormer;
    tilingData->alignGroupNum = alignGroupNum;
    tilingData->rowPerHeadCore = ubLoopOfFormerBlock;
    tilingData->rowPerTailCore = ubLoopOfTailBlock;
    tilingData->multiRowNumHeadCore = ubTailOfFormerBlock;
    tilingData->multiRowNumTailCore = ubTailOfTailBlock;
    tilingData->innerLoopEle = (alignGroupNum + 7) / 8;
    tilingData->innerLoopTimes = (ubTailOfFormerBlock + 7) / 8;
    tilingData->innerLoopTail = (ubTailOfTailBlock + 7) / 8;
    tilingData->max8LoopEle = (alignGroupNum + 15) / 16;
    tilingData->max8LoopTimes = (ubTailOfFormerBlock + 15) / 16;
    tilingData->max8LoopTail = (ubTailOfTailBlock + 15) / 16;
    tilingData->scaleLoopEle = (alignGroupNum + 255) / 256;
    tilingData->scaleLoopTimes = (ubTailOfFormerBlock + 255) / 256;
    tilingData->scaleLoopTail = (ubTailOfTailBlock + 255) / 256;
    tilingData->outLoopEle = (alignGroupNum + 1) / 2;
    tilingData->outLoopTimes = (ubTailOfFormerBlock + 1) / 2;
    tilingData->outLoopTail = (ubTailOfTailBlock + 1) / 2;

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dynamic_hi_f4_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

static void RunKernelTestAxisNeg2(int64_t dim0, int64_t dimNeg2, int64_t dimNeg1, int64_t totalCoreNum,
                                  uint64_t tilingKey)
{
    int64_t blockW = 128;
    int64_t blockSize = 64;
    int64_t splitBlockH = 64;
    int64_t groupPerUb = 1;

    int64_t dimNeg2SplitBlockNum = (dimNeg2 + splitBlockH - 1) / splitBlockH;
    int64_t dimNeg1BlockNum = (dimNeg1 + blockW - 1) / blockW;
    int64_t totalTaskNum = dim0 * dimNeg2SplitBlockNum * dimNeg1BlockNum;

    int64_t usedCoreNum = std::min(totalCoreNum, totalTaskNum);
    int64_t blockPerHeadCore = (totalTaskNum + totalCoreNum - 1) / totalCoreNum;
    int64_t blockPerTailCore = totalTaskNum / totalCoreNum;
    int64_t headCoreNum = totalTaskNum % totalCoreNum;

    size_t inputSize = dim0 * dimNeg2 * dimNeg1 * sizeof(uint16_t);
    size_t outputSize = dim0 * dimNeg2 * dimNeg1 / 2;
    size_t scaleSize = dim0 * (dimNeg2 / 64) * dimNeg1 * 2 * sizeof(uint16_t);
    size_t tilingDataSize = sizeof(DynamicHiF4QuantTilingData);
    size_t workspaceSize = 32;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* scaleBuf = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    DynamicHiF4QuantTilingData* tilingData = reinterpret_cast<DynamicHiF4QuantTilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->dim0 = dim0;
    tilingData->dimNeg2 = dimNeg2;
    tilingData->dimNeg1 = dimNeg1;
    tilingData->blockW = blockW;
    tilingData->tilingKey = tilingKey;
    tilingData->splitBlockH = splitBlockH;
    tilingData->dimNeg2Tail = dimNeg2 % splitBlockH == 0 ? splitBlockH : dimNeg2 % splitBlockH;
    tilingData->dimNeg1Tail = dimNeg1 % blockW == 0 ? blockW : dimNeg1 % blockW;
    tilingData->dimNeg2SplitBlockNum = dimNeg2SplitBlockNum;
    tilingData->dimNeg1BlockNum = dimNeg1BlockNum;
    tilingData->blockPerHeadCore = blockPerHeadCore;
    tilingData->blockPerTailCore = blockPerTailCore;
    tilingData->headCoreNum = headCoreNum;
    tilingData->dimNeg2IsOdd = (dimNeg2 / 64) % 2;
    tilingData->dimNeg1IsOdd = (dimNeg1 / 64) % 2;
    tilingData->dimNeg1IsPad = dimNeg1 % 64 == 0 ? 0 : 1;
    tilingData->blockCountPerBatch = dimNeg2SplitBlockNum * dimNeg1BlockNum;
    tilingData->scale1ColCountPerBatch = (dimNeg1 + 63) / 64;
    tilingData->scale2RowCountPerBatch = (dimNeg2 + 63) / 64;

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dynamic_hi_f4_quant, blockDim, x, y, scaleBuf, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(scaleBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg1_bf16_basic)
{
    int64_t dim0 = 2;
    int64_t dim1 = 128;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 0;
    RunKernelTestAxisNeg1(dim0, dim1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg1_bf16_large)
{
    int64_t dim0 = 16;
    int64_t dim1 = 1024;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 0;
    RunKernelTestAxisNeg1(dim0, dim1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg1_small)
{
    int64_t dim0 = 1;
    int64_t dim1 = 64;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 0;
    RunKernelTestAxisNeg1(dim0, dim1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_bf16_basic)
{
    int64_t dim0 = 1;
    int64_t dimNeg2 = 128;
    int64_t dimNeg1 = 128;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_bf16_large)
{
    int64_t dim0 = 4;
    int64_t dimNeg2 = 512;
    int64_t dimNeg1 = 256;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey);
}
