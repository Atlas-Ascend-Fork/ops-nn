/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/fused_bias_leaky_relu_grad_proto.h"
#include "platform/platform_info.h"

class FusedBiasLeakyReluGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedBiasLeakyReluGrad InferShape test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FusedBiasLeakyReluGrad InferShape test TearDown" << std::endl; }
};

// 两输入同 shape：输出与输入一致
TEST_F(FusedBiasLeakyReluGradInferShapeTest, infershape_test_case_1d)
{
    ge::op::FusedBiasLeakyReluGrad op;
    ge::TensorDesc tensorYGrad;
    ge::Shape shape({1024});
    tensorYGrad.SetDataType(ge::DT_FLOAT);
    tensorYGrad.SetShape(shape);
    tensorYGrad.SetOriginShape(shape);

    ge::TensorDesc tensorFeatures;
    tensorFeatures.SetDataType(ge::DT_FLOAT);
    tensorFeatures.SetShape(shape);
    tensorFeatures.SetOriginShape(shape);

    op.UpdateInputDesc("y_grad", tensorYGrad);
    op.UpdateInputDesc("features", tensorFeatures);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto outputDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {1024};
    EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedShape);
}

TEST_F(FusedBiasLeakyReluGradInferShapeTest, infershape_test_case_2d)
{
    ge::op::FusedBiasLeakyReluGrad op;
    ge::TensorDesc tensorYGrad;
    ge::Shape shape({2, 1024});
    tensorYGrad.SetDataType(ge::DT_FLOAT);
    tensorYGrad.SetShape(shape);
    tensorYGrad.SetOriginShape(shape);

    ge::TensorDesc tensorFeatures;
    tensorFeatures.SetDataType(ge::DT_FLOAT);
    tensorFeatures.SetShape(shape);
    tensorFeatures.SetOriginShape(shape);

    op.UpdateInputDesc("y_grad", tensorYGrad);
    op.UpdateInputDesc("features", tensorFeatures);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto outputDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {2, 1024};
    EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedShape);
}

TEST_F(FusedBiasLeakyReluGradInferShapeTest, infershape_test_case_4d)
{
    ge::op::FusedBiasLeakyReluGrad op;
    ge::TensorDesc tensorYGrad;
    ge::Shape shape({2, 4, 8, 16});
    tensorYGrad.SetDataType(ge::DT_FLOAT);
    tensorYGrad.SetShape(shape);
    tensorYGrad.SetOriginShape(shape);

    ge::TensorDesc tensorFeatures;
    tensorFeatures.SetDataType(ge::DT_FLOAT);
    tensorFeatures.SetShape(shape);
    tensorFeatures.SetOriginShape(shape);

    op.UpdateInputDesc("y_grad", tensorYGrad);
    op.UpdateInputDesc("features", tensorFeatures);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto outputDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {2, 4, 8, 16};
    EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedShape);
}

// float16
TEST_F(FusedBiasLeakyReluGradInferShapeTest, infershape_test_case_float16)
{
    ge::op::FusedBiasLeakyReluGrad op;
    ge::TensorDesc tensorYGrad;
    ge::Shape shape({8192});
    tensorYGrad.SetDataType(ge::DT_FLOAT16);
    tensorYGrad.SetShape(shape);
    tensorYGrad.SetOriginShape(shape);

    ge::TensorDesc tensorFeatures;
    tensorFeatures.SetDataType(ge::DT_FLOAT16);
    tensorFeatures.SetShape(shape);
    tensorFeatures.SetOriginShape(shape);

    op.UpdateInputDesc("y_grad", tensorYGrad);
    op.UpdateInputDesc("features", tensorFeatures);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto outputDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {8192};
    EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedShape);
}

// 广播：features({8,1}) 与 y_grad({8,1024}) -> 输出 {8,1024}
TEST_F(FusedBiasLeakyReluGradInferShapeTest, infershape_test_case_broadcast)
{
    ge::op::FusedBiasLeakyReluGrad op;
    ge::TensorDesc tensorYGrad;
    ge::Shape yGradShape({8, 1024});
    tensorYGrad.SetDataType(ge::DT_FLOAT);
    tensorYGrad.SetShape(yGradShape);
    tensorYGrad.SetOriginShape(yGradShape);

    ge::TensorDesc tensorFeatures;
    ge::Shape featuresShape({8, 1});
    tensorFeatures.SetDataType(ge::DT_FLOAT);
    tensorFeatures.SetShape(featuresShape);
    tensorFeatures.SetOriginShape(featuresShape);

    op.UpdateInputDesc("y_grad", tensorYGrad);
    op.UpdateInputDesc("features", tensorFeatures);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto outputDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {8, 1024};
    EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedShape);
}

// 广播：低 rank 输入
TEST_F(FusedBiasLeakyReluGradInferShapeTest, infershape_test_case_broadcast_low_rank)
{
    ge::op::FusedBiasLeakyReluGrad op;
    ge::TensorDesc tensorYGrad;
    ge::Shape yGradShape({1024});
    tensorYGrad.SetDataType(ge::DT_FLOAT);
    tensorYGrad.SetShape(yGradShape);
    tensorYGrad.SetOriginShape(yGradShape);

    ge::TensorDesc tensorFeatures;
    ge::Shape featuresShape({8, 1024});
    tensorFeatures.SetDataType(ge::DT_FLOAT);
    tensorFeatures.SetShape(featuresShape);
    tensorFeatures.SetOriginShape(featuresShape);

    op.UpdateInputDesc("y_grad", tensorYGrad);
    op.UpdateInputDesc("features", tensorFeatures);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto outputDesc = op.GetOutputDesc(0);
    std::vector<int64_t> expectedShape = {8, 1024};
    EXPECT_EQ(outputDesc.GetShape().GetDims(), expectedShape);
}

// 广播不兼容：应失败
TEST_F(FusedBiasLeakyReluGradInferShapeTest, infershape_test_case_broadcast_incompatible)
{
    ge::op::FusedBiasLeakyReluGrad op;
    ge::TensorDesc tensorYGrad;
    ge::Shape yGradShape({2, 6});
    tensorYGrad.SetDataType(ge::DT_FLOAT);
    tensorYGrad.SetShape(yGradShape);
    tensorYGrad.SetOriginShape(yGradShape);

    ge::TensorDesc tensorFeatures;
    ge::Shape featuresShape({2, 8});
    tensorFeatures.SetDataType(ge::DT_FLOAT);
    tensorFeatures.SetShape(featuresShape);
    tensorFeatures.SetOriginShape(featuresShape);

    op.UpdateInputDesc("y_grad", tensorYGrad);
    op.UpdateInputDesc("features", tensorFeatures);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

// 空张量
TEST_F(FusedBiasLeakyReluGradInferShapeTest, infershape_test_case_empty)
{
    ge::op::FusedBiasLeakyReluGrad op;
    ge::TensorDesc tensorYGrad;
    ge::Shape shape({0});
    tensorYGrad.SetDataType(ge::DT_FLOAT);
    tensorYGrad.SetShape(shape);
    tensorYGrad.SetOriginShape(shape);

    ge::TensorDesc tensorFeatures;
    tensorFeatures.SetDataType(ge::DT_FLOAT);
    tensorFeatures.SetShape(shape);
    tensorFeatures.SetOriginShape(shape);

    op.UpdateInputDesc("y_grad", tensorYGrad);
    op.UpdateInputDesc("features", tensorFeatures);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

// Ascend950 平台
TEST_F(FusedBiasLeakyReluGradInferShapeTest, ascend950_test)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    ge::op::FusedBiasLeakyReluGrad op;
    ge::TensorDesc tensorYGrad;
    ge::Shape shape({1, 64, 128});
    tensorYGrad.SetDataType(ge::DT_FLOAT);
    tensorYGrad.SetShape(shape);
    tensorYGrad.SetOriginShape(shape);

    ge::TensorDesc tensorFeatures;
    tensorFeatures.SetDataType(ge::DT_FLOAT);
    tensorFeatures.SetShape(shape);
    tensorFeatures.SetOriginShape(shape);

    op.UpdateInputDesc("y_grad", tensorYGrad);
    op.UpdateInputDesc("features", tensorFeatures);
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}
