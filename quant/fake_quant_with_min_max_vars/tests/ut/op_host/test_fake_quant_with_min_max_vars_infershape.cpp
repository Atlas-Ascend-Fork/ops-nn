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
#include <vector>

#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/fake_quant_with_min_max_vars_proto.h"

class FakeQuantWithMinMaxVarsInferShapeTest : public testing::TestWithParam<std::vector<int64_t>> {};

TEST_P(FakeQuantWithMinMaxVarsInferShapeTest, output_shape_matches_input)
{
    const auto& shape = GetParam();
    ge::op::FakeQuantWithMinMaxVars op;
    ge::TensorDesc xDesc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
    xDesc.SetOriginShape(ge::Shape(shape));
    op.UpdateInputDesc("x", xDesc);
    op.UpdateInputDesc("min", create_desc({1}, ge::DT_FLOAT));
    op.UpdateInputDesc("max", create_desc({1}, ge::DT_FLOAT));

    Runtime2TestParam param;
    ASSERT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDescByName("y").GetShape().GetDims(), shape);
}

INSTANTIATE_TEST_SUITE_P(InputShapes, FakeQuantWithMinMaxVarsInferShapeTest,
                         testing::Values(std::vector<int64_t>{1}, std::vector<int64_t>{1024},
                                         std::vector<int64_t>{4, 256}, std::vector<int64_t>{2, 3, 4, 8},
                                         std::vector<int64_t>{2, 1, 3, 1, 4, 1, 5, 8}, std::vector<int64_t>{0, 4},
                                         std::vector<int64_t>{-1, 256}));

TEST(FakeQuantWithMinMaxVarsInferDataTypeTest, infer_dtype_float)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxVars");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);

    ge::DataType xType = ge::DT_FLOAT;
    ge::DataType minType = ge::DT_FLOAT;
    ge::DataType maxType = ge::DT_FLOAT;
    ge::DataType yType = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&xType, &minType, &maxType})
                      .OutputDataTypes({&yType})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}
