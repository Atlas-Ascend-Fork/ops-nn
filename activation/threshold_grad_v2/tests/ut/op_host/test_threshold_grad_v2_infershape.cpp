/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "array_ops.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
using namespace ge;

class ThresholdGradV2InferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ThresholdGradV2InferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ThresholdGradV2InferShapeTest TearDown" << std::endl; }
};

static void RunInferShapeCase(gert::StorageShape& gradientsShape, gert::StorageShape& featuresShape,
                              gert::StorageShape& thresholdShape, gert::Shape& expectedShape, ge::DataType dtype)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("ThresholdGradV2")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&gradientsShape, &featuresShape, &thresholdShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*outputDesc), Ops::Base::ToString(expectedShape));
}

TEST_F(ThresholdGradV2InferShapeTest, test_same_shape_fp32)
{
    gert::StorageShape gradientsShape = {{16, 16}, {16, 16}};
    gert::StorageShape featuresShape = {{16, 16}, {16, 16}};
    gert::StorageShape thresholdShape = {{16, 16}, {16, 16}};
    gert::Shape expectedShape = {16, 16};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_FLOAT);
}

TEST_F(ThresholdGradV2InferShapeTest, test_broadcast_shape)
{
    gert::StorageShape gradientsShape = {{4, 1}, {4, 1}};
    gert::StorageShape featuresShape = {{1, 8}, {1, 8}};
    gert::StorageShape thresholdShape = {{4, 8}, {4, 8}};
    gert::Shape expectedShape = {4, 8};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_FLOAT16);
}

TEST_F(ThresholdGradV2InferShapeTest, test_threshold_scalar_shape)
{
    gert::StorageShape gradientsShape = {{2, 64}, {2, 64}};
    gert::StorageShape featuresShape = {{2, 64}, {2, 64}};
    gert::StorageShape thresholdShape = {{1}, {1}};
    gert::Shape expectedShape = {2, 64};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_BF16);
}

TEST_F(ThresholdGradV2InferShapeTest, test_threshold_last_dim_broadcast)
{
    gert::StorageShape gradientsShape = {{8, 8}, {8, 8}};
    gert::StorageShape featuresShape = {{8, 8}, {8, 8}};
    gert::StorageShape thresholdShape = {{8, 1}, {8, 1}};
    gert::Shape expectedShape = {8, 8};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_INT32);
}

TEST_F(ThresholdGradV2InferShapeTest, test_int8_same_shape)
{
    gert::StorageShape gradientsShape = {{8, 8}, {8, 8}};
    gert::StorageShape featuresShape = {{8, 8}, {8, 8}};
    gert::StorageShape thresholdShape = {{8, 8}, {8, 8}};
    gert::Shape expectedShape = {8, 8};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_INT8);
}

TEST_F(ThresholdGradV2InferShapeTest, test_uint8_same_shape)
{
    gert::StorageShape gradientsShape = {{16}, {16}};
    gert::StorageShape featuresShape = {{16}, {16}};
    gert::StorageShape thresholdShape = {{16}, {16}};
    gert::Shape expectedShape = {16};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_UINT8);
}

TEST_F(ThresholdGradV2InferShapeTest, test_8d_shape)
{
    gert::StorageShape gradientsShape = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}};
    gert::StorageShape featuresShape = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}};
    gert::StorageShape thresholdShape = {{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}};
    gert::Shape expectedShape = {1, 2, 3, 4, 5, 6, 7, 8};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_FLOAT);
}

TEST_F(ThresholdGradV2InferShapeTest, test_empty_tensor)
{
    gert::StorageShape gradientsShape = {{0, 0}, {0, 0}};
    gert::StorageShape featuresShape = {{0, 0}, {0, 0}};
    gert::StorageShape thresholdShape = {{0, 0}, {0, 0}};
    gert::Shape expectedShape = {0, 0};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_FLOAT);
}

TEST_F(ThresholdGradV2InferShapeTest, test_scalar_input)
{
    gert::StorageShape gradientsShape = {{}, {}};
    gert::StorageShape featuresShape = {{}, {}};
    gert::StorageShape thresholdShape = {{}, {}};
    gert::Shape expectedShape = {};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_FLOAT);
}

TEST_F(ThresholdGradV2InferShapeTest, test_broadcast_different_rank)
{
    gert::StorageShape gradientsShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape featuresShape = {{3, 4}, {3, 4}};
    gert::StorageShape thresholdShape = {{4}, {4}};
    gert::Shape expectedShape = {2, 3, 4};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_FLOAT);
}

TEST_F(ThresholdGradV2InferShapeTest, test_large_shape)
{
    gert::StorageShape gradientsShape = {{1024, 1024}, {1024, 1024}};
    gert::StorageShape featuresShape = {{1024, 1024}, {1024, 1024}};
    gert::StorageShape thresholdShape = {{1, 1024}, {1, 1024}};
    gert::Shape expectedShape = {1024, 1024};
    RunInferShapeCase(gradientsShape, featuresShape, thresholdShape, expectedShape, ge::DT_FLOAT16);
}
