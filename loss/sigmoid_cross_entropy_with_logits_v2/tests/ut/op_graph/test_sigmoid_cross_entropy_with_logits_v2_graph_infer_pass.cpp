/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataType4SigmoidCrossEntropyWithLogitsV2(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
void CheckInferDataType(ge::DataType predictDtype, ge::DataType targetDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SigmoidCrossEntropyWithLogitsV2").OpName("SigmoidCrossEntropyWithLogitsV2").IONum(4, 1);
    builder.InputTensorDesc(0, predictDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, targetDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, predictDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, predictDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(ops::InferDataType4SigmoidCrossEntropyWithLogitsV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), predictDtype);
}
} // namespace

TEST(SigmoidCrossEntropyWithLogitsV2GraphInfer, InferDataTypeFollowsPredict)
{
    CheckInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT16);
    CheckInferDataType(ge::DT_FLOAT, ge::DT_FLOAT);
    CheckInferDataType(ge::DT_BF16, ge::DT_BF16);
}

namespace {
void CheckPredeclaredOutput(ge::DataType predictDtype, ge::DataType declaredDtype, ge::DataType expectedDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SigmoidCrossEntropyWithLogitsV2").OpName("SigmoidCrossEntropyWithLogitsV2").IONum(4, 1);
    builder.InputTensorDesc(0, predictDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, predictDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, predictDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, predictDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, declaredDtype), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ops::InferDataType4SigmoidCrossEntropyWithLogitsV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), expectedDtype);
}
} // namespace
TEST(SigmoidCrossEntropyWithLogitsV2GraphInfer, InferDataTypeKeepsPredeclaredLegalOutput)
{
    CheckPredeclaredOutput(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT);
    CheckPredeclaredOutput(ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT);
    CheckPredeclaredOutput(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);
    CheckPredeclaredOutput(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
    CheckPredeclaredOutput(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16);
}
TEST(SigmoidCrossEntropyWithLogitsV2GraphInfer, InferDataTypeNormalizesIllegalPredeclaredOutput)
{
    CheckPredeclaredOutput(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT);
    CheckPredeclaredOutput(ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT);
    CheckPredeclaredOutput(ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16);
    CheckPredeclaredOutput(ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16);
}
