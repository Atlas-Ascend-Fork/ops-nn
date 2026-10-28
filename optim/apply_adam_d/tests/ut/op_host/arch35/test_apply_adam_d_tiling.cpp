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
 * \file test_apply_adam_d_tiling.cpp
 * \brief ApplyAdamD Tiling UT
 *
 * 算子接口：10 输入(var,m,v,beta1_power,beta2_power,lr,beta1,beta2,epsilon,grad) /
 *           3 输出(var,m,v) / 2 属性(use_locking,use_nesterov)。
 * TilingKey 由 var 的数据类型决定：FLOAT16 -> 101，BF16 -> 102，FLOAT -> 103。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/apply_adam_d_tiling.h"

using namespace std;
using namespace ge;

class ApplyAdamDTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdamDTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyAdamDTilingTest TearDown" << std::endl; }
};

// 构造 10 输入 + 3 输出 + 2 属性 的 Tiling 上下文（var/m/v/grad 同形，6 个标量为 [1]）。
static gert::TilingContextPara MakeTilingPara(ge::DataType varDtype, bool useNesterov,
                                              optiling::ApplyAdamDCompileInfo* compileInfo)
{
    gert::StorageShape tensorShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape scalarShape = {{1}, {1}};
    std::vector<gert::TilingContextPara::TensorDescription> inputs({
        {tensorShape, varDtype, ge::FORMAT_ND}, // var
        {tensorShape, varDtype, ge::FORMAT_ND}, // m
        {tensorShape, varDtype, ge::FORMAT_ND}, // v
        {scalarShape, varDtype, ge::FORMAT_ND}, // beta1_power
        {scalarShape, varDtype, ge::FORMAT_ND}, // beta2_power
        {scalarShape, varDtype, ge::FORMAT_ND}, // lr
        {scalarShape, varDtype, ge::FORMAT_ND}, // beta1
        {scalarShape, varDtype, ge::FORMAT_ND}, // beta2
        {scalarShape, varDtype, ge::FORMAT_ND}, // epsilon
        {tensorShape, varDtype, ge::FORMAT_ND}, // grad
    });
    std::vector<gert::TilingContextPara::TensorDescription> outputs({
        {tensorShape, varDtype, ge::FORMAT_ND}, // var
        {tensorShape, varDtype, ge::FORMAT_ND}, // m
        {tensorShape, varDtype, ge::FORMAT_ND}, // v
    });
    std::vector<gert::TilingContextPara::OpAttr> attrs({
        {"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)},
        {"use_nesterov", Ops::NN::AnyValue::CreateFrom<bool>(useNesterov)},
    });
    return gert::TilingContextPara("ApplyAdamD", inputs, outputs, attrs, compileInfo, 64, 245760, 4096);
}

static void RunApplyAdamDTilingCase(ge::DataType varDtype, bool useNesterov, int64_t expectTilingKey,
                                    const std::string& expectTilingData)
{
    optiling::ApplyAdamDCompileInfo compileInfo = {64, 245760};
    auto para = MakeTilingPara(varDtype, useNesterov, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, expectTilingKey);
    EXPECT_EQ(tilingInfo.blockNum, 58);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1u);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 32u);
    std::string tilingDataStr;
    for (size_t i = 0; i < tilingInfo.tilingDataSize / sizeof(int64_t); ++i) {
        tilingDataStr += std::to_string(reinterpret_cast<const int64_t*>(tilingInfo.tilingData.get())[i]);
        tilingDataStr += " ";
    }
    EXPECT_EQ(tilingDataStr, expectTilingData);
}

TEST_F(ApplyAdamDTilingTest, apply_adam_d_tiling_float16)
{
    RunApplyAdamDTilingCase(ge::DT_FLOAT16, false, 101, "204800 13743895347264 3584 58 2 1 384 512 3200 1 0 0 ");
}

TEST_F(ApplyAdamDTilingTest, apply_adam_d_tiling_bf16)
{
    RunApplyAdamDTilingCase(ge::DT_BF16, false, 102, "204800 13743895347264 3584 58 2 1 384 512 3200 1 0 0 ");
}

TEST_F(ApplyAdamDTilingTest, apply_adam_d_tiling_float)
{
    RunApplyAdamDTilingCase(ge::DT_FLOAT, false, 103, "204800 18691697672256 3584 58 1 1 3584 512 4352 1 0 0 ");
}

TEST_F(ApplyAdamDTilingTest, apply_adam_d_tiling_float16_nesterov)
{
    RunApplyAdamDTilingCase(ge::DT_FLOAT16, true, 101, "204800 13743895347264 3584 58 2 1 384 512 3200 1 0 1 ");
}

TEST_F(ApplyAdamDTilingTest, apply_adam_d_tiling_float_nesterov)
{
    RunApplyAdamDTilingCase(ge::DT_FLOAT, true, 103, "204800 18691697672256 3584 58 1 1 3584 512 4352 1 0 1 ");
}
