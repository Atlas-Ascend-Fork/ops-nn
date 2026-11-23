/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "infer_shape_context_faker.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/situ_glu_group_quant_proto.h"

namespace {
class SituGluGroupQuantTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SituGluGroupQuantTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SituGluGroupQuantTest TearDown" << std::endl; }
};

// Both inputs are required, so the generic InferShapeTest driver works here
// (attrs follow the def registration order: beta, linearBeta, high_precision).
static ge::graphStatus ExecuteInferShape(gert::StorageShape& xShape, gert::StorageShape& tokensShape,
                                         std::vector<int64_t>& yDims, std::vector<int64_t>& y_scaleDims)
{
    std::string opType("SituGluGroupQuant");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    EXPECT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    EXPECT_NE(inferShapeFunc, nullptr);

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1}, {1, 1})
                      .InputShapes({&xShape, &tokensShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"beta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"linearBeta", Ops::NN::AnyValue::CreateFrom<float>(2.0f)},
                                  {"high_precision", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_NE(context, nullptr);
    ge::graphStatus status = inferShapeFunc(context);
    if (status == ge::GRAPH_SUCCESS) {
        const gert::Shape* yOut = context->GetOutputShape(0);
        const gert::Shape* scaleOut = context->GetOutputShape(1);
        for (size_t i = 0; i < yOut->GetDimNum(); ++i) {
            yDims.push_back(yOut->GetDim(i));
        }
        for (size_t i = 0; i < scaleOut->GetDimNum(); ++i) {
            y_scaleDims.push_back(scaleOut->GetDim(i));
        }
    }
    return status;
}

TEST_F(SituGluGroupQuantTest, SituGluGroupQuant_infershape_3d_basic)
{
    gert::StorageShape xShape = {{8, 128, 8192}, {8, 128, 8192}};
    gert::StorageShape tokensShape = {{4}, {4}};
    std::vector<int64_t> yDims;
    std::vector<int64_t> y_scaleDims;
    ASSERT_EQ(ExecuteInferShape(xShape, tokensShape, yDims, y_scaleDims), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yDims, (std::vector<int64_t>{8, 128, 4096}));
    EXPECT_EQ(y_scaleDims, (std::vector<int64_t>{8, 128, 64, 2}));
}

TEST_F(SituGluGroupQuantTest, SituGluGroupQuant_infershape_2d)
{
    gert::StorageShape xShape = {{4096, 6144}, {4096, 6144}};
    gert::StorageShape tokensShape = {{8}, {8}};
    std::vector<int64_t> yDims;
    std::vector<int64_t> y_scaleDims;
    ASSERT_EQ(ExecuteInferShape(xShape, tokensShape, yDims, y_scaleDims), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yDims, (std::vector<int64_t>{4096, 3072}));
    EXPECT_EQ(y_scaleDims, (std::vector<int64_t>{4096, 48, 2}));
}

TEST_F(SituGluGroupQuantTest, SituGluGroupQuant_infershape_1d)
{
    gert::StorageShape xShape = {{256}, {256}};
    gert::StorageShape tokensShape = {{1}, {1}};
    std::vector<int64_t> yDims;
    std::vector<int64_t> y_scaleDims;
    ASSERT_EQ(ExecuteInferShape(xShape, tokensShape, yDims, y_scaleDims), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yDims, (std::vector<int64_t>{128}));
    EXPECT_EQ(y_scaleDims, (std::vector<int64_t>{2, 2}));
}

TEST_F(SituGluGroupQuantTest, SituGluGroupQuant_infershape_rank_above_8_allowed)
{
    // recipes caliber: rank >= 1 without the 8-dim cap
    gert::StorageShape xShape = {{2, 2, 2, 2, 2, 2, 2, 2, 64}, {2, 2, 2, 2, 2, 2, 2, 2, 64}};
    gert::StorageShape tokensShape = {{2}, {2}};
    std::vector<int64_t> yDims;
    std::vector<int64_t> y_scaleDims;
    ASSERT_EQ(ExecuteInferShape(xShape, tokensShape, yDims, y_scaleDims), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yDims, (std::vector<int64_t>{2, 2, 2, 2, 2, 2, 2, 2, 32}));
    EXPECT_EQ(y_scaleDims, (std::vector<int64_t>{2, 2, 2, 2, 2, 2, 2, 2, 1, 2}));
}

TEST_F(SituGluGroupQuantTest, SituGluGroupQuant_infershape_dynamic_last_dim)
{
    // last dim -1 stays dynamic; scale tail computed as -1 as well
    gert::StorageShape xShape = {{4, -1}, {4, -1}};
    gert::StorageShape tokensShape = {{4}, {4}};
    std::vector<int64_t> yDims;
    std::vector<int64_t> y_scaleDims;
    ASSERT_EQ(ExecuteInferShape(xShape, tokensShape, yDims, y_scaleDims), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yDims, (std::vector<int64_t>{4, -1}));
    EXPECT_EQ(y_scaleDims, (std::vector<int64_t>{4, -1, 2}));
}

TEST_F(SituGluGroupQuantTest, SituGluGroupQuant_infershape_error_odd_last_dim)
{
    gert::StorageShape xShape = {{4, 64, 1023}, {4, 64, 1023}};
    gert::StorageShape tokensShape = {{4}, {4}};
    std::vector<int64_t> yDims;
    std::vector<int64_t> y_scaleDims;
    EXPECT_NE(ExecuteInferShape(xShape, tokensShape, yDims, y_scaleDims), ge::GRAPH_SUCCESS);
}

TEST_F(SituGluGroupQuantTest, SituGluGroupQuant_infershape_error_zero_last_dim)
{
    gert::StorageShape xShape = {{4, 0}, {4, 0}};
    gert::StorageShape tokensShape = {{4}, {4}};
    std::vector<int64_t> yDims;
    std::vector<int64_t> y_scaleDims;
    EXPECT_NE(ExecuteInferShape(xShape, tokensShape, yDims, y_scaleDims), ge::GRAPH_SUCCESS);
}

TEST_F(SituGluGroupQuantTest, SituGluGroupQuant_infershape_unknown_rank_propagates)
{
    // unknown rank (-2): both outputs propagate the [-2] rank; concrete shapes
    // are validated later by the host tiling
    gert::StorageShape xShape = {{-2}, {-2}};
    gert::StorageShape tokensShape = {{4}, {4}};
    std::vector<int64_t> yDims;
    std::vector<int64_t> y_scaleDims;
    ASSERT_EQ(ExecuteInferShape(xShape, tokensShape, yDims, y_scaleDims), ge::GRAPH_SUCCESS);
    EXPECT_EQ(yDims, (std::vector<int64_t>{-2}));
    EXPECT_EQ(y_scaleDims, (std::vector<int64_t>{-2}));
}

TEST_F(SituGluGroupQuantTest, SituGluGroupQuant_infershape_error_scalar_x)
{
    // rank 0 is rejected
    gert::StorageShape xShape = {{}, {}};
    gert::StorageShape tokensShape = {{4}, {4}};
    std::vector<int64_t> yDims;
    std::vector<int64_t> y_scaleDims;
    EXPECT_NE(ExecuteInferShape(xShape, tokensShape, yDims, y_scaleDims), ge::GRAPH_SUCCESS);
}
} // namespace
