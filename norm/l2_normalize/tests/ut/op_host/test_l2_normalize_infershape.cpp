/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {

constexpr const char* OP_TYPE = "L2Normalize";

ge::graphStatus RunInferShape(gert::Shape& inputShape, gert::Shape& outputShape, ge::Format inputFormat = ge::FORMAT_ND,
                              ge::Format outputFormat = ge::FORMAT_ND)
{
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1}, {1})
                      .NodeInputTd(0, ge::DT_FLOAT, inputFormat, inputFormat)
                      .NodeOutputTd(0, ge::DT_FLOAT, outputFormat, outputFormat)
                      .InputShapes({&inputShape})
                      .OutputShapes({&outputShape})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // The context builder allocates output shapes; OutputShapes() does not initialize them.
    *context->GetOutputShape(0) = outputShape;
    const ge::graphStatus status = opImpl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS && context != nullptr && context->GetOutputShape(0) != nullptr) {
        outputShape = *context->GetOutputShape(0);
    }
    return status;
}

std::vector<int64_t> ShapeToVector(const gert::Shape& shape)
{
    std::vector<int64_t> result;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        result.push_back(shape.GetDim(i));
    }
    return result;
}

} // namespace

TEST(L2NormalizeInferShape, CopiesStaticShape)
{
    gert::Shape inputShape = {2, 3, 4, 5};
    gert::Shape outputShape = {};
    ASSERT_EQ(RunInferShape(inputShape, outputShape), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(outputShape), (std::vector<int64_t>{2, 3, 4, 5}));
}

TEST(L2NormalizeInferShape, CopiesDynamicDimensionAndUnknownRank)
{
    gert::Shape dynamicShape = {-1, 32};
    gert::Shape dynamicOutput = {};
    ASSERT_EQ(RunInferShape(dynamicShape, dynamicOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(dynamicOutput), (std::vector<int64_t>{-1, 32}));

    gert::Shape unknownRankShape = {-2};
    gert::Shape unknownRankOutput = {};
    ASSERT_EQ(RunInferShape(unknownRankShape, unknownRankOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(unknownRankOutput), (std::vector<int64_t>{-2}));
}

TEST(L2NormalizeInferShape, RejectsScalarAndRankAboveEight)
{
    gert::Shape scalarShape = {};
    gert::Shape scalarOutput = {};
    EXPECT_EQ(RunInferShape(scalarShape, scalarOutput), ge::GRAPH_FAILED);

    gert::Shape rankNineShape = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    gert::Shape rankNineOutput = {};
    EXPECT_EQ(RunInferShape(rankNineShape, rankNineOutput), ge::GRAPH_FAILED);
}

TEST(L2NormalizeInferShape, OverwritesReusedOutputShape)
{
    gert::Shape inputShape = {2, 3};
    gert::Shape previousOutput = {2, 4};
    ASSERT_EQ(RunInferShape(inputShape, previousOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(previousOutput), (std::vector<int64_t>{2, 3}));
    gert::Shape newInput = {6};
    ASSERT_EQ(RunInferShape(newInput, previousOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(previousOutput), (std::vector<int64_t>{6}));
}

TEST(L2NormalizeInferShape, OverwritesPartialAndUnknownOutputShapes)
{
    gert::Shape inputShape = {2, 3};
    gert::Shape declaredShape = {2, -1};
    ASSERT_EQ(RunInferShape(inputShape, declaredShape), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(declaredShape), (std::vector<int64_t>{2, 3}));
    gert::Shape unknownOutput = {-2};
    ASSERT_EQ(RunInferShape(inputShape, unknownOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(unknownOutput), (std::vector<int64_t>{2, 3}));
    gert::Shape unknownInput = {-2};
    gert::Shape declaredOutput = {2, 3};
    ASSERT_EQ(RunInferShape(unknownInput, declaredOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(declaredOutput), (std::vector<int64_t>{-2}));
}

TEST(L2NormalizeInferShape, RejectsNonNDInputAndOutputFormats)
{
    gert::Shape inputShape = {2, 3};
    gert::Shape outputShape = {};
    EXPECT_EQ(RunInferShape(inputShape, outputShape, ge::FORMAT_NCHW), ge::GRAPH_FAILED);
    EXPECT_EQ(RunInferShape(inputShape, outputShape, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ), ge::GRAPH_FAILED);
    EXPECT_EQ(RunInferShape(inputShape, outputShape, ge::FORMAT_RESERVED, ge::FORMAT_RESERVED), ge::GRAPH_SUCCESS);
}

// FR-P1-011：零元素张量在 GE 动态图下以未知秩 desc（GE_UNKNOWN_RANK，dims={-2}）到达
// 推导（L0_empty_027/034：data_shape=(0,)/(0,0)、desc/ori_shape=[-2]）；未知秩须无条件
// 透传（不逐维校验），已知秩零元素 shape 同样恒等复制。
TEST(L2NormalizeInferShape, PassesUnknownRankZeroElementTensor)
{
    gert::Shape unknownRankZeroElemShape = {-2};
    gert::Shape unknownRankZeroElemOutput = {};
    ASSERT_EQ(RunInferShape(unknownRankZeroElemShape, unknownRankZeroElemOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(unknownRankZeroElemOutput), (std::vector<int64_t>{-2}));

    gert::Shape zeroElemShape = {0};
    gert::Shape zeroElemOutput = {};
    ASSERT_EQ(RunInferShape(zeroElemShape, zeroElemOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(zeroElemOutput), (std::vector<int64_t>{0}));

    gert::Shape zeroElemRankTwoShape = {0, 0};
    gert::Shape zeroElemRankTwoOutput = {};
    ASSERT_EQ(RunInferShape(zeroElemRankTwoShape, zeroElemRankTwoOutput), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ShapeToVector(zeroElemRankTwoOutput), (std::vector<int64_t>{0, 0}));
}
