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

#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "../../../op_graph/l2_normalize_proto.h"

namespace ops {
ge::graphStatus InferDataTypeForL2Normalize(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
gert::ContextHolder<gert::InferDataTypeContext> BuildContext(ge::DataType inputType)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("L2Normalize").OpName("L2Normalize");
    builder.IONum(1, 1);
    builder.InputTensorDesc(0, inputType, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    return builder.Build();
}

ge::graphStatus RunVerify(const ge::Shape& inputShape, const ge::Shape& outputShape,
                          ge::DataType inputType = ge::DT_FLOAT, ge::DataType outputType = ge::DT_FLOAT,
                          ge::Format inputFormat = ge::FORMAT_ND, ge::Format outputFormat = ge::FORMAT_ND)
{
    ge::op::L2Normalize op("l2_normalize_verify_ut");
    EXPECT_EQ(op.UpdateInputDesc("x", ge::TensorDesc(inputShape, inputFormat, inputType)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateOutputDesc("y", ge::TensorDesc(outputShape, outputFormat, outputType)), ge::GRAPH_SUCCESS);
    return op.VerifyAllAttr(true);
}

} // namespace

TEST(L2NormalizeGraphInfer, CopiesInputDataType)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        auto holder = BuildContext(dtype);
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
        ASSERT_EQ(ops::InferDataTypeForL2Normalize(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), dtype);
        EXPECT_EQ(ops::InferDataTypeForL2Normalize(context), ge::GRAPH_SUCCESS);
    }
}

TEST(L2NormalizeGraphInfer, OverwritesReusedOutputDataType)
{
    auto holder = BuildContext(ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_FLOAT), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeForL2Normalize(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(L2NormalizeGraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeForL2Normalize(nullptr), ge::GRAPH_FAILED);
}

TEST(L2NormalizeGraphVerify, RejectsConflictingDeclaredShape)
{
    EXPECT_EQ(RunVerify(ge::Shape({2, 3}), ge::Shape({2, 4})), ge::GRAPH_FAILED);
    EXPECT_EQ(RunVerify(ge::Shape({2, 3}), ge::Shape({6})), ge::GRAPH_FAILED);
}

TEST(L2NormalizeGraphVerify, RejectsConflictingDeclaredDataType)
{
    EXPECT_EQ(RunVerify(ge::Shape({17}), ge::Shape({17}), ge::DT_FLOAT16, ge::DT_FLOAT), ge::GRAPH_FAILED);
}

TEST(L2NormalizeGraphVerify, RejectsNonNDFormats)
{
    EXPECT_EQ(RunVerify(ge::Shape({2, 3}), ge::Shape({2, 3}), ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_NCHW),
              ge::GRAPH_FAILED);
    EXPECT_EQ(RunVerify(ge::Shape({2, 3}), ge::Shape({2, 3}), ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND,
                        ge::FORMAT_FRACTAL_NZ),
              ge::GRAPH_FAILED);
}

TEST(L2NormalizeGraphVerify, RejectsScalarAndRankAboveEight)
{
    EXPECT_EQ(RunVerify(ge::Shape(std::vector<int64_t>{}), ge::Shape(std::vector<int64_t>{})), ge::GRAPH_FAILED);
    EXPECT_EQ(RunVerify(ge::Shape({1, 1, 1, 1, 1, 1, 1, 1, 1}), ge::Shape({1, 1, 1, 1, 1, 1, 1, 1, 1})),
              ge::GRAPH_FAILED);
}

TEST(L2NormalizeGraphVerify, AcceptsCompatibleAndUndeclaredOutputs)
{
    EXPECT_EQ(RunVerify(ge::Shape({2, 3}), ge::Shape({2, -1})), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunVerify(ge::Shape({17}), ge::Shape(std::vector<int64_t>{}), ge::DT_FLOAT16, ge::DT_UNDEFINED),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunVerify(ge::Shape({17}), ge::Shape({17}), ge::DT_FLOAT16, ge::DT_FLOAT16, ge::FORMAT_RESERVED,
                        ge::FORMAT_RESERVED),
              ge::GRAPH_SUCCESS);
}

TEST(L2NormalizeGraphVerify, AcceptsUnknownRankAndEmptyInputs)
{
    EXPECT_EQ(RunVerify(ge::Shape(ge::UNKNOWN_RANK), ge::Shape({2, 3})), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunVerify(ge::Shape({2, 3}), ge::Shape(ge::UNKNOWN_RANK)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunVerify(ge::Shape({0}), ge::Shape({0})), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunVerify(ge::Shape({0, 0}), ge::Shape({0, 0})), ge::GRAPH_SUCCESS);
}
