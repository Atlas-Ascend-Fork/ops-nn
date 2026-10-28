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
 * \file test_binary_cross_entropy_grad_infer_pass.cpp
 * \brief BinaryCrossEntropyGrad InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeBinaryCrossEntropyGrad(gert::InferDataTypeContext* context);
}

namespace {
void RunBinaryCrossEntropyGradInferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("BinaryCrossEntropyGrad");
    builder.OpName("BinaryCrossEntropyGrad");
    builder.IONum(4, 1);
    for (int32_t i = 0; i < 4; ++i) {
        builder.InputTensorDesc(i, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 1; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBinaryCrossEntropyGrad(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype);
}
} // namespace

TEST(BinaryCrossEntropyGradGraphInfer, InferDataTypeFloat) { RunBinaryCrossEntropyGradInferDataType(ge::DT_FLOAT); }

TEST(BinaryCrossEntropyGradGraphInfer, InferDataTypeFloat16) { RunBinaryCrossEntropyGradInferDataType(ge::DT_FLOAT16); }
