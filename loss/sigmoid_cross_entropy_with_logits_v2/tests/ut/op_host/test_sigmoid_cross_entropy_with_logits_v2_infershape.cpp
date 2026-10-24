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

#include "infer_shape_context_faker.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"

namespace {
struct InferShapeCase {
    const char* reduction;
    ge::DataType dtype;
    gert::Shape expectedShape;
};

void CheckInferShape(const InferShapeCase& testCase)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SigmoidCrossEntropyWithLogitsV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);

    gert::StorageShape predictShape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape targetShape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape weightShape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape posWeightShape = {{375}, {375}};
    gert::StorageShape outputShape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SigmoidCrossEntropyWithLogitsV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1}, {1})
                      .InputShapes({&predictShape, &targetShape, &weightShape, &posWeightShape})
                      .OutputShapes({&outputShape})
                      .Attr("reduction", ge::AscendString(testCase.reduction))
                      .NodeInputTd(0, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, testCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(opImpl->infer_shape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* actualShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(actualShape, nullptr);
    ASSERT_EQ(actualShape->GetDimNum(), testCase.expectedShape.GetDimNum());
    for (size_t i = 0; i < actualShape->GetDimNum(); ++i) {
        EXPECT_EQ(actualShape->GetDim(i), testCase.expectedShape.GetDim(i));
    }
}

} // namespace

TEST(SigmoidCrossEntropyWithLogitsV2InferShapeTest, reduction_shape_semantics)
{
    const std::vector<InferShapeCase> testCases = {
        {"none", ge::DT_FLOAT16, {16, 8, 375}},
        {"mean", ge::DT_FLOAT, {}},
        {"sum", ge::DT_BF16, {}},
    };
    for (const auto& testCase : testCases) {
        CheckInferShape(testCase);
    }
}

TEST(SigmoidCrossEntropyWithLogitsV2InferShapeTest, reduction_absent_falls_back_to_declared_default_mean)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SigmoidCrossEntropyWithLogitsV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);

    gert::StorageShape predictShape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape targetShape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape weightShape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape posWeightShape = {{375}, {375}};
    gert::StorageShape outputShape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SigmoidCrossEntropyWithLogitsV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1}, {1})
                      .InputShapes({&predictShape, &targetShape, &weightShape, &posWeightShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(opImpl->infer_shape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    const auto* actualShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(actualShape, nullptr);
    EXPECT_EQ(actualShape->GetDimNum(), 0U);
}

TEST(SigmoidCrossEntropyWithLogitsV2InferShapeTest, reduction_invalid_value_failed)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SigmoidCrossEntropyWithLogitsV2");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);

    gert::StorageShape predictShape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape targetShape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape weightShape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape posWeightShape = {{375}, {375}};
    gert::StorageShape outputShape = {{1}, {1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SigmoidCrossEntropyWithLogitsV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1}, {1})
                      .InputShapes({&predictShape, &targetShape, &weightShape, &posWeightShape})
                      .OutputShapes({&outputShape})
                      .Attr("reduction", ge::AscendString("invalid"))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    EXPECT_EQ(opImpl->infer_shape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
