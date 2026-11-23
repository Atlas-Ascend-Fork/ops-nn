/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
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
#include "situ_glu_group_quant_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

// The kernel entry is now a TPL template (optional-input presence dimensions,
// swiglu_mx_quant caliber): include the entry source under __CCE_KT_TEST__ and
// call it with explicit template arguments.
#ifdef __CCE_KT_TEST__
#include "../../../op_kernel/situ_glu_group_quant.cpp"
#endif

using namespace std;
using namespace SituGluGroupQuantOp;

class SituGluGroupQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SituGluGroupQuantKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "SituGluGroupQuantKernelTest TearDown" << endl; }
};

static void InitTilingData(SituGluGroupQuantTilingData* tilingData, int64_t totalRows, int64_t epr,
                           int64_t groupIndexCount, float beta, float linearBeta)
{
    tilingData->totalElements = totalRows * epr;
    tilingData->outputElementsPerRow = epr;
    tilingData->groupIndexCount = groupIndexCount;
    tilingData->chunkElements = epr < 3072 ? epr : 3072;
    tilingData->beta = beta;
    tilingData->linearBeta = linearBeta;
    tilingData->blockSize = 32;
    int64_t groupCount = (epr + 31) / 32;
    tilingData->scaleColNum = (groupCount + 1) / 2 * 2;
    tilingData->rowsPerBatch = 1; // UT keeps the legacy per-row cycle (K=1)
}

// Test BF16 basic: 2 experts x 4 rows each = 8 valid rows, EPR=128
TEST_F(SituGluGroupQuantKernelTest, test_situ_glu_group_quant_bf16_basic)
{
    const int64_t totalRows = 8;
    const int64_t epr = 128;
    const int64_t groupIndexCount = 2;

    size_t inputSize = totalRows * epr * 2 * sizeof(uint16_t);
    size_t outputSize = totalRows * epr * sizeof(uint8_t);
    int64_t groupCount = (epr + 31) / 32;
    int64_t scaleColNum = (groupCount + 1) / 2 * 2;
    size_t scaleSize = totalRows * scaleColNum * sizeof(uint8_t);
    size_t tokensSize = groupIndexCount * sizeof(int64_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(tokensSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* y_scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(SituGluGroupQuantTilingData));

    for (size_t i = 0; i < inputSize; i++) {
        x[i] = static_cast<uint8_t>(rand() % 256);
    }
    int64_t* tokens = reinterpret_cast<int64_t*>(groupIndex);
    tokens[0] = 4;
    tokens[1] = 4;

    SituGluGroupQuantTilingData* tilingData = reinterpret_cast<SituGluGroupQuantTilingData*>(tiling);
    InitTilingData(tilingData, totalRows, epr, groupIndexCount, 1.0f, 2.0f);

#undef DTYPE_X
#undef DTYPE_Y
#define DTYPE_X bfloat16_t
#define DTYPE_Y fp8_e4m3fn_t

    auto KernelFunc = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR y_scale, GM_ADDR workspace,
                         GM_ADDR tiling) {
        ::situ_glu_group_quant<TPL_HAS_GROUP_INDEX, TPL_NO_WEIGHT, TPL_ACTIVATE_NOT_LEFT, TPL_GROUP_LIST_COUNT,
                               TPL_LINEAR_ENABLED>(x, weight, groupIndex, y, y_scale, workspace, tiling);
    };

    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, nullptr, groupIndex, y, y_scale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(y);
    AscendC::GmFree(y_scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test sparse semantics: group_index sums to 5 < totalRows=8 (padding skipped),
// zero-token expert included, high_precision on
TEST_F(SituGluGroupQuantKernelTest, test_situ_glu_group_quant_padding_rows)
{
    const int64_t totalRows = 8;
    const int64_t epr = 128;
    const int64_t groupIndexCount = 3;

    size_t inputSize = totalRows * epr * 2 * sizeof(uint16_t);
    size_t outputSize = totalRows * epr * sizeof(uint8_t);
    int64_t groupCount = (epr + 31) / 32;
    int64_t scaleColNum = (groupCount + 1) / 2 * 2;
    size_t scaleSize = totalRows * scaleColNum * sizeof(uint8_t);
    size_t tokensSize = groupIndexCount * sizeof(int64_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(tokensSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* y_scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(SituGluGroupQuantTilingData));

    for (size_t i = 0; i < inputSize; i++) {
        x[i] = static_cast<uint8_t>(rand() % 256);
    }
    int64_t* tokens = reinterpret_cast<int64_t*>(groupIndex);
    tokens[0] = 2;
    tokens[1] = 0; // zero-token expert contributes no rows
    tokens[2] = 3;

    SituGluGroupQuantTilingData* tilingData = reinterpret_cast<SituGluGroupQuantTilingData*>(tiling);
    InitTilingData(tilingData, totalRows, epr, groupIndexCount, 1.0f, 2.0f);

#undef DTYPE_X
#undef DTYPE_Y
#define DTYPE_X bfloat16_t
#define DTYPE_Y fp8_e4m3fn_t

    auto KernelFunc = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR y_scale, GM_ADDR workspace,
                         GM_ADDR tiling) {
        ::situ_glu_group_quant<TPL_HAS_GROUP_INDEX, TPL_NO_WEIGHT, TPL_ACTIVATE_NOT_LEFT, TPL_GROUP_LIST_COUNT,
                               TPL_LINEAR_ENABLED>(x, weight, groupIndex, y, y_scale, workspace, tiling);
    };

    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, nullptr, groupIndex, y, y_scale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(y);
    AscendC::GmFree(y_scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Test the recipes chunk boundary EPR=3072 (linked kernel copy is bf16-flavored:
// the tbe source is compiled once with the injected DTYPE macros)
TEST_F(SituGluGroupQuantKernelTest, test_situ_glu_group_quant_epr_3072)
{
    const int64_t totalRows = 2;
    const int64_t epr = 3072;
    const int64_t groupIndexCount = 1;

    size_t inputSize = totalRows * epr * 2 * sizeof(uint16_t);
    size_t outputSize = totalRows * epr * sizeof(uint8_t);
    int64_t groupCount = (epr + 31) / 32;
    int64_t scaleColNum = (groupCount + 1) / 2 * 2;
    size_t scaleSize = totalRows * scaleColNum * sizeof(uint8_t);
    size_t tokensSize = groupIndexCount * sizeof(int64_t);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(tokensSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* y_scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(SituGluGroupQuantTilingData));

    for (size_t i = 0; i < inputSize; i++) {
        x[i] = static_cast<uint8_t>(rand() % 256);
    }
    int64_t* tokens = reinterpret_cast<int64_t*>(groupIndex);
    tokens[0] = 2;

    SituGluGroupQuantTilingData* tilingData = reinterpret_cast<SituGluGroupQuantTilingData*>(tiling);
    InitTilingData(tilingData, totalRows, epr, groupIndexCount, 1.0f, 1.0f);

    auto KernelFunc = [](GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR y_scale, GM_ADDR workspace,
                         GM_ADDR tiling) {
        ::situ_glu_group_quant<TPL_HAS_GROUP_INDEX, TPL_NO_WEIGHT, TPL_ACTIVATE_NOT_LEFT, TPL_GROUP_LIST_COUNT,
                               TPL_LINEAR_ENABLED>(x, weight, groupIndex, y, y_scale, workspace, tiling);
    };

    uint32_t blockDim = 24;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelFunc, blockDim, x, nullptr, groupIndex, y, y_scale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(y);
    AscendC::GmFree(y_scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
