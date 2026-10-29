/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "../../../op_graph/swiglu_hi_f4_quant_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

namespace {
class SwigluHiF4QuantTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SwigluHiF4QuantTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SwigluHiF4QuantTest TearDown" << std::endl; }
};

TEST_F(SwigluHiF4QuantTest, infershape_case_fp16)
{
    ge::op::SwigluHiF4Quant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({8, 128, 8192});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{
        {"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "axis", "scale_alg"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {8, 128, 4096};
    std::vector<int64_t> expectedScaleShape = {8, 128, 64, 1};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(SwigluHiF4QuantTest, infershape_case_bf16)
{
    ge::op::SwigluHiF4Quant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({4, 64, 2048});
    xDesc.SetDataType(ge::DT_BF16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{
        {"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "axis", "scale_alg"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {4, 64, 1024};
    std::vector<int64_t> expectedScaleShape = {4, 64, 16, 1};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(SwigluHiF4QuantTest, infershape_case_2d_shape)
{
    ge::op::SwigluHiF4Quant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({256, 4096});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{
        {"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "axis", "scale_alg"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {256, 2048};
    std::vector<int64_t> expectedScaleShape = {256, 32, 1};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(SwigluHiF4QuantTest, infershape_case_4d_shape)
{
    ge::op::SwigluHiF4Quant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({2, 4, 16, 2048});
    xDesc.SetDataType(ge::DT_BF16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{
        {"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "axis", "scale_alg"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {2, 4, 16, 1024};
    std::vector<int64_t> expectedScaleShape = {2, 4, 16, 16, 1};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(SwigluHiF4QuantTest, infershape_case_dynamic_shape)
{
    ge::op::SwigluHiF4Quant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({-2});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{
        {"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "axis", "scale_alg"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {-2};
    std::vector<int64_t> expectedScaleShape = {-2};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(SwigluHiF4QuantTest, infershape_error_invalid_dim_not_divisible_by_2)
{
    ge::op::SwigluHiF4Quant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({4, 64, 1023});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{
        {"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "axis", "scale_alg"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(SwigluHiF4QuantTest, infershape_error_rank_1)
{
    ge::op::SwigluHiF4Quant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({8192});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{
        {"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "axis", "scale_alg"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

TEST_F(SwigluHiF4QuantTest, infershape_error_rank_exceed_max)
{
    ge::op::SwigluHiF4Quant op;
    ge::TensorDesc xDesc;
    ge::Shape xShape({1, 2, 3, 4, 5, 6, 7, 128});
    xDesc.SetDataType(ge::DT_FLOAT16);
    xDesc.SetShape(xShape);
    xDesc.SetOriginShape(xShape);
    op.UpdateInputDesc("x", xDesc);

    Runtime2TestParam param{
        {"activate_dim", "activate_left", "swiglu_mode", "clamp_limit", "glu_alpha", "glu_bias", "axis", "scale_alg"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

} // namespace
