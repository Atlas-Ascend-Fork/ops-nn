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
 * \file test_apply_adam_d_infershape.cpp
 * \brief ApplyAdamD InferShape UT
 *
 * 算子接口：10 输入(var,m,v,beta1_power,beta2_power,lr,beta1,beta2,epsilon,grad) / 3 输出(var,m,v)。
 * 推导规则：输出 var/m/v 分别跟随同名输入的 shape（InferShape4ApplyAdamD 逐个拷贝）。
 */

#include <gtest/gtest.h>
#include <iostream>
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"

using namespace ge;

class ApplyAdamDInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdamDInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyAdamDInfershapeTest TearDown" << std::endl; }
};

static void RunApplyAdamDInferShapeCase(gert::Shape& varShape, gert::Shape& mShape, gert::Shape& vShape,
                                        gert::Shape& scalarShape, gert::Shape& gradShape,
                                        const std::vector<std::vector<int64_t>>& expect)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamD"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyAdamD")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape outVarShape = {};
    gert::Shape outMShape = {};
    gert::Shape outVShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &scalarShape, &scalarShape, &scalarShape, &scalarShape,
                                    &scalarShape, &scalarShape, &gradShape})
                      .OutputShapes({&outVarShape, &outMShape, &outVShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"use_nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    ASSERT_EQ(expect.size(), 3u);
    for (size_t idx = 0; idx < expect.size(); ++idx) {
        auto* outShape = context->GetOutputShape(idx);
        ASSERT_NE(outShape, nullptr);
        ASSERT_EQ(outShape->GetDimNum(), expect[idx].size());
        for (size_t d = 0; d < expect[idx].size(); ++d) {
            EXPECT_EQ(outShape->GetDim(d), expect[idx][d]);
        }
    }
}

// 输出 var/m/v 分别跟随同名输入（静态 shape）
TEST_F(ApplyAdamDInfershapeTest, apply_adam_d_infershape_test_01)
{
    gert::Shape varShape = {16, 48, 16, 32};
    gert::Shape mShape = {16, 48, 16, 32};
    gert::Shape vShape = {16, 48, 16, 32};
    gert::Shape scalarShape = {1};
    gert::Shape gradShape = {16, 48, 16, 32};
    RunApplyAdamDInferShapeCase(varShape, mShape, vShape, scalarShape, gradShape,
                                {{16, 48, 16, 32}, {16, 48, 16, 32}, {16, 48, 16, 32}});
}

// 动态 shape
TEST_F(ApplyAdamDInfershapeTest, apply_adam_d_infershape_test_dynamic)
{
    gert::Shape varShape = {-1, 32};
    gert::Shape mShape = {-1, 32};
    gert::Shape vShape = {-1, 32};
    gert::Shape scalarShape = {1};
    gert::Shape gradShape = {-1, 32};
    RunApplyAdamDInferShapeCase(varShape, mShape, vShape, scalarShape, gradShape, {{-1, 32}, {-1, 32}, {-1, 32}});
}

// 逐输入独立拷贝：var/m/v 取不同 shape，三输出分别匹配对应输入
TEST_F(ApplyAdamDInfershapeTest, apply_adam_d_infershape_test_per_input_copy)
{
    gert::Shape varShape = {2, 3};
    gert::Shape mShape = {4};
    gert::Shape vShape = {5, 6};
    gert::Shape scalarShape = {1};
    gert::Shape gradShape = {2, 3};
    RunApplyAdamDInferShapeCase(varShape, mShape, vShape, scalarShape, gradShape, {{2, 3}, {4}, {5, 6}});
}
