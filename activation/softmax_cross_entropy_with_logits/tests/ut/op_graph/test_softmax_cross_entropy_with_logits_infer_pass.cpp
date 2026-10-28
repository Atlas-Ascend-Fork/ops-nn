/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_softmax_cross_entropy_with_logits_infer_pass.cpp
 * \brief SoftmaxCrossEntropyWithLogits InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeSoftmaxCrossEntropyWithLogits(gert::InferDataTypeContext* context);
}

namespace {
void RunSoftmaxCrossEntropyWithLogitsInferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SoftmaxCrossEntropyWithLogits");
    builder.OpName("SoftmaxCrossEntropyWithLogits");
    builder.IONum(2, 2);
    for (int32_t i = 0; i < 2; ++i) {
        builder.InputTensorDesc(i, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 2; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeSoftmaxCrossEntropyWithLogits(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype);
    EXPECT_EQ(context->GetOutputDataType(1), inDtype);
}
} // namespace

TEST(SoftmaxCrossEntropyWithLogitsGraphInfer, InferDataTypeFloat)
{
    RunSoftmaxCrossEntropyWithLogitsInferDataType(ge::DT_FLOAT);
}

TEST(SoftmaxCrossEntropyWithLogitsGraphInfer, InferDataTypeFloat16)
{
    RunSoftmaxCrossEntropyWithLogitsInferDataType(ge::DT_FLOAT16);
}
