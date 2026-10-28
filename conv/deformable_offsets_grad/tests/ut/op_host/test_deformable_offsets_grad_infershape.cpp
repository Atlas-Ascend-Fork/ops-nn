/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h"      // NOLINT
#include "kernel_run_context_facker.h" // NOLINT

class DeformableOffsetsGradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DeformableOffsetsGradInfershapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DeformableOffsetsGradInfershapeTest TearDown" << std::endl; }
};

// 输出 shape 推导：grad_x 跟随 x，grad_offsets 跟随 offsets
TEST_F(DeformableOffsetsGradInfershapeTest, deformable_offsets_grad_infershape_test_01)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DeformableOffsetsGrad"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DeformableOffsetsGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {2, 8, 16, 16};
    gert::Shape xShape = {2, 4, 16, 16};
    gert::Shape offsetsShape = {2, 27, 16, 16};
    gert::Shape gradXOutShape = {};
    gert::Shape gradOffsetsOutShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .InputShapes({&gradShape, &xShape, &offsetsShape})
                      .OutputShapes({&gradXOutShape, &gradOffsetsOutShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(2), 16);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)->GetDim(3), 16);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(0), 2);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(1), 27);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(2), 16);
    EXPECT_EQ(holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1)->GetDim(3), 16);
}

// 动态 shape 推导：grad_x 跟随 x（含 -1），grad_offsets 跟随 offsets（含 -1）
TEST_F(DeformableOffsetsGradInfershapeTest, deformable_offsets_grad_infershape_test_dynamic)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DeformableOffsetsGrad"), nullptr);
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DeformableOffsetsGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape gradShape = {-1, -1, -1, -1};
    gert::Shape xShape = {-1, 4, -1, -1};
    gert::Shape offsetsShape = {-1, 27, -1, -1};
    gert::Shape gradXOutShape = {};
    gert::Shape gradOffsetsOutShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 2)
                      .InputShapes({&gradShape, &xShape, &offsetsShape})
                      .OutputShapes({&gradXOutShape, &gradOffsetsOutShape})
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), -1);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 4);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(2), -1);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(3), -1);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), -1);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(1), 27);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(2), -1);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(3), -1);
}
