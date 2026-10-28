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
#include <cstdint>
#include <vector>
#include "infer_shape_context_faker.h"
#include "register/op_impl_registry.h"
#include "op_impl_registry.h"

namespace {
struct InferCase {
    const char* name;
    std::vector<int64_t> shape;
    int64_t dim;
    bool success;
};
class SituGluGradInferShapeTest : public testing::TestWithParam<InferCase> {};

TEST_P(SituGluGradInferShapeTest, InfersShapeOrRejectsInvalidInput)
{
    const auto& test = GetParam();
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("SituGluGrad");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::Shape inputShape;
    inputShape.SetDimNum(test.shape.size());
    for (size_t i = 0; i < test.shape.size(); ++i) {
        inputShape.SetDim(i, test.shape[i]);
    }
    gert::Shape gradYShape = {4, 4};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SituGluGrad")
                      .NodeIoNum(2, 1)
                      .InputShapes({&gradYShape, &inputShape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(test.dim)}})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    const auto status = impl->infer_shape(context);
    ASSERT_EQ(status, test.success ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED);
    if (!test.success) {
        return;
    }
    const auto* output = context->GetOutputShape(0);
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->GetDimNum(), test.shape.size());
    for (size_t i = 0; i < test.shape.size(); ++i) {
        int64_t expected = test.shape[i];

        EXPECT_EQ(output->GetDim(i), expected);
    }
}

INSTANTIATE_TEST_SUITE_P(InputValidation, SituGluGradInferShapeTest,
                         testing::Values(InferCase{"ValidNegativeDim", {4, 8}, -1, true},
                                         InferCase{"DynamicSplitDimension", {4, -1}, -1, true},
                                         InferCase{"UnknownRank", {-2}, -1, true}),
                         [](const testing::TestParamInfo<InferCase>& info) { return info.param.name; });
} // namespace
