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
const char* const kOpType = "SituGluGroupGrad";

struct InferCase {
    const char* name;
    std::vector<int64_t> xShape;
    int64_t dim;
};

class SituGluGroupGradInferShapeTest : public testing::TestWithParam<InferCase> {};

// grad_x has the same shape as x; grad_weight (weight present) has the same shape as weight, whose
// element count is the product of the x dims before the dim axis.
TEST_P(SituGluGroupGradInferShapeTest, InfersGradXAndGradWeightShapes)
{
    const auto& test = GetParam();
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);

    const int64_t rank = static_cast<int64_t>(test.xShape.size());
    const int64_t axis = test.dim < 0 ? test.dim + rank : test.dim;

    gert::Shape xShape;
    xShape.SetDimNum(test.xShape.size());
    for (size_t i = 0; i < test.xShape.size(); ++i) {
        xShape.SetDim(i, test.xShape[i]);
    }
    // grad_y equals x with the dim axis halved (only the positive, static case is exercised here).
    gert::Shape gradYShape;
    gradYShape.SetDimNum(test.xShape.size());
    for (size_t i = 0; i < test.xShape.size(); ++i) {
        int64_t d = test.xShape[i];
        if (static_cast<int64_t>(i) == axis && d > 0) {
            d /= 2;
        }
        gradYShape.SetDim(i, d);
    }
    // weight is 1-D with numel = product of the x dims before the dim axis.
    int64_t rows = 1;
    for (int64_t i = 0; i < axis; ++i) {
        if (test.xShape[i] > 0) {
            rows *= test.xShape[i];
        }
    }
    gert::Shape weightShape;
    weightShape.SetDimNum(1);
    weightShape.SetDim(0, rows);

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(kOpType)
                      .NodeIoNum(3, 2)
                      .InputShapes({&gradYShape, &xShape, &weightShape})
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(test.dim)}})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_shape(context), ge::GRAPH_SUCCESS);

    const auto* gradX = context->GetOutputShape(0);
    ASSERT_NE(gradX, nullptr);
    ASSERT_EQ(gradX->GetDimNum(), test.xShape.size());
    for (size_t i = 0; i < test.xShape.size(); ++i) {
        EXPECT_EQ(gradX->GetDim(i), test.xShape[i]);
    }
    const auto* gradWeight = context->GetOutputShape(1);
    ASSERT_NE(gradWeight, nullptr);
    ASSERT_EQ(gradWeight->GetDimNum(), 1);
    EXPECT_EQ(gradWeight->GetDim(0), rows);
}

INSTANTIATE_TEST_SUITE_P(InputValidation, SituGluGroupGradInferShapeTest,
                         testing::Values(InferCase{"ValidNegativeDim", {4, 8}, -1},
                                         InferCase{"ValidNonLastDim", {2, 8, 16}, 1},
                                         InferCase{"DynamicSplitDimension", {4, -1}, -1}, InferCase{"Rank1", {16}, 0}),
                         [](const testing::TestParamInfo<InferCase>& info) { return info.param.name; });
} // namespace
