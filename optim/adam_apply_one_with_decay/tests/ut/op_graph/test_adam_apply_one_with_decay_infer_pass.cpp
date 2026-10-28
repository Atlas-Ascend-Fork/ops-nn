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
 * \file test_adam_apply_one_with_decay_infer_pass.cpp
 * \brief AdamApplyOneWithDecay InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeAdamApplyOneWithDecay(gert::InferDataTypeContext* context);
}

namespace {
void RunAdamApplyOneWithDecayInferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AdamApplyOneWithDecay");
    builder.OpName("AdamApplyOneWithDecay");
    builder.IONum(11, 3);
    for (int32_t i = 0; i < 11; ++i) {
        builder.InputTensorDesc(i, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 3; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeAdamApplyOneWithDecay(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype);
    EXPECT_EQ(context->GetOutputDataType(1), inDtype);
    EXPECT_EQ(context->GetOutputDataType(2), inDtype);
}
} // namespace

TEST(AdamApplyOneWithDecayGraphInfer, InferDataTypeFloat) { RunAdamApplyOneWithDecayInferDataType(ge::DT_FLOAT); }

TEST(AdamApplyOneWithDecayGraphInfer, InferDataTypeFloat16) { RunAdamApplyOneWithDecayInferDataType(ge::DT_FLOAT16); }
