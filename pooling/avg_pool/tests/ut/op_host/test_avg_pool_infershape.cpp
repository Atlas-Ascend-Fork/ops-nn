/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;

class AvgPoolInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AvgPoolInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AvgPoolInfershape TearDown" << std::endl; }
};

static void RunAvgPoolInferShape(gert::Shape& xShape, const gert::Shape& expectedShape,
                                 const std::vector<int64_t>& ksize, const std::vector<int64_t>& strides,
                                 const std::string& padding)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AvgPool")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    gert::Shape yShape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expectedShape));
}

// 常规场景：NCHW，SAME模式
TEST_F(AvgPoolInfershape, avg_pool_infershape_test_1)
{
    gert::Shape xShape = {1, 3, 6, 6};
    gert::Shape expectedShape = {1, 3, 6, 6};
    RunAvgPoolInferShape(xShape, expectedShape, {1, 1, 2, 2}, {1, 1, 1, 1}, "SAME");
}

// 常规场景：NCHW，VALID模式
TEST_F(AvgPoolInfershape, avg_pool_infershape_test_2)
{
    gert::Shape xShape = {1, 3, 6, 6};
    gert::Shape expectedShape = {1, 3, 5, 5};
    RunAvgPoolInferShape(xShape, expectedShape, {1, 1, 2, 2}, {1, 1, 1, 1}, "VALID");
}

// H/W维未知（-1）场景：输出H/W置为未知维度，N/C透传
TEST_F(AvgPoolInfershape, avg_pool_infershape_unknown_hw_dim)
{
    gert::Shape xShape = {2, 3, -1, -1};
    gert::Shape expectedShape = {2, 3, -1, -1};
    RunAvgPoolInferShape(xShape, expectedShape, {1, 1, 2, 2}, {1, 1, 1, 1}, "SAME");
}

// H/W维未知（-1）场景：VALID模式同样透传未知
TEST_F(AvgPoolInfershape, avg_pool_infershape_unknown_hw_dim_valid)
{
    gert::Shape xShape = {2, 3, -1, -1};
    gert::Shape expectedShape = {2, 3, -1, -1};
    RunAvgPoolInferShape(xShape, expectedShape, {1, 1, 2, 2}, {1, 1, 1, 1}, "VALID");
}

// C维未知（-1）场景：C维透传，已知维度正常计算
TEST_F(AvgPoolInfershape, avg_pool_infershape_unknown_channel_dim)
{
    gert::Shape xShape = {2, -1, 6, 6};
    gert::Shape expectedShape = {2, -1, 6, 6};
    RunAvgPoolInferShape(xShape, expectedShape, {1, 1, 2, 2}, {1, 1, 1, 1}, "SAME");
}

// 全未知维度（-1）场景
TEST_F(AvgPoolInfershape, avg_pool_infershape_all_unknown_dim)
{
    gert::Shape xShape = {-1, -1, -1, -1};
    gert::Shape expectedShape = {-1, -1, -1, -1};
    RunAvgPoolInferShape(xShape, expectedShape, {1, 1, 2, 2}, {1, 1, 1, 1}, "SAME");
}

// 未知rank（-2）场景：输出置为未知rank
TEST_F(AvgPoolInfershape, avg_pool_infershape_unknown_rank)
{
    gert::Shape xShape = {-2};
    gert::Shape expectedShape = {-2};
    RunAvgPoolInferShape(xShape, expectedShape, {1, 1, 2, 2}, {1, 1, 1, 1}, "SAME");
}
