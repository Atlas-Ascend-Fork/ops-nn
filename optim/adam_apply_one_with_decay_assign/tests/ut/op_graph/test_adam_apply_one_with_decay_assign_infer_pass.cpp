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
 * \file test_adam_apply_one_with_decay_assign_infer_pass.cpp
 * \brief AdamApplyOneWithDecayAssign InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeAdamApplyOneWithDecayAssign(gert::InferDataTypeContext* context);
}

namespace {
void RunAdamApplyOneWithDecayAssignInferDataType(ge::DataType input1Dtype, ge::DataType input2Dtype,
                                                 ge::DataType input3Dtype)
{
    const ge::DataType otherDtype = ge::DT_FLOAT16;
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AdamApplyOneWithDecayAssign");
    builder.OpName("AdamApplyOneWithDecayAssign");
    builder.IONum(11, 3);
    builder.InputTensorDesc(0, otherDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, input1Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, input2Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, input3Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    for (int32_t i = 4; i < 11; ++i) {
        builder.InputTensorDesc(i, otherDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 3; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeAdamApplyOneWithDecayAssign(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), input1Dtype);
    EXPECT_EQ(context->GetOutputDataType(1), input2Dtype);
    EXPECT_EQ(context->GetOutputDataType(2), input3Dtype);
}
} // namespace

TEST(AdamApplyOneWithDecayAssignGraphInfer, InferDataTypeAllFloat16)
{
    RunAdamApplyOneWithDecayAssignInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);
}

TEST(AdamApplyOneWithDecayAssignGraphInfer, InferDataTypePerInputMapping)
{
    RunAdamApplyOneWithDecayAssignInferDataType(ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16);
}
