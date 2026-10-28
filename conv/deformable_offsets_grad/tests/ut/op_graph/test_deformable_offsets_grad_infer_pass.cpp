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
 * \file test_deformable_offsets_grad_infer_pass.cpp
 * \brief DeformableOffsetsGrad InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeDeformableOffsetsGrad(gert::InferDataTypeContext* context);
}

namespace {
void RunDeformableOffsetsGradInferDataType(ge::DataType gradDtype, ge::DataType xDtype, ge::DataType offsetsDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("DeformableOffsetsGrad");
    builder.OpName("DeformableOffsetsGrad");
    builder.IONum(3, 2);
    builder.InputTensorDesc(0, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, offsetsDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    for (int32_t i = 0; i < 2; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeDeformableOffsetsGrad(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), xDtype);
    EXPECT_EQ(context->GetOutputDataType(1), offsetsDtype);
}
} // namespace

TEST(DeformableOffsetsGradGraphInfer, InferDataTypeFloat16)
{
    RunDeformableOffsetsGradInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);
}

TEST(DeformableOffsetsGradGraphInfer, InferDataTypePerInputMapping)
{
    RunDeformableOffsetsGradInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16);
}
