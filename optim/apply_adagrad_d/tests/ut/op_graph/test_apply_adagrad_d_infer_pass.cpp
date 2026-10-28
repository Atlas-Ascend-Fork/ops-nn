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
 * \file test_apply_adagrad_d_infer_pass.cpp
 * \brief ApplyAdagradD InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeApplyAdagradD(gert::InferDataTypeContext* context);
}

namespace {
void RunApplyAdagradDInferDataType(ge::DataType inDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ApplyAdagradD");
    builder.OpName("ApplyAdagradD");
    builder.IONum(4, 2);
    for (int32_t i = 0; i < 4; ++i) {
        builder.InputTensorDesc(i, inDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 2; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeApplyAdagradD(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), inDtype);
    EXPECT_EQ(context->GetOutputDataType(1), inDtype);
}
} // namespace

TEST(ApplyAdagradDGraphInfer, InferDataTypeFloat) { RunApplyAdagradDInferDataType(ge::DT_FLOAT); }

TEST(ApplyAdagradDGraphInfer, InferDataTypeFloat16) { RunApplyAdagradDInferDataType(ge::DT_FLOAT16); }
