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
 * \file test_apply_adam_d_infer_pass.cpp
 * \brief ApplyAdamD InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeApplyAdamD(gert::InferDataTypeContext* context);
}

namespace {
void RunApplyAdamDInferDataType(ge::DataType varDtype, ge::DataType mDtype, ge::DataType vDtype)
{
    const ge::DataType otherDtype = ge::DT_FLOAT16;
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("ApplyAdamD");
    builder.OpName("ApplyAdamD");
    builder.IONum(10, 3);
    builder.InputTensorDesc(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, mDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(2, vDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    for (int32_t i = 3; i < 10; ++i) {
        builder.InputTensorDesc(i, otherDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (int32_t i = 0; i < 3; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeApplyAdamD(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), varDtype);
    EXPECT_EQ(context->GetOutputDataType(1), mDtype);
    EXPECT_EQ(context->GetOutputDataType(2), vDtype);
}
} // namespace

TEST(ApplyAdamDGraphInfer, InferDataTypeAllFloat16)
{
    RunApplyAdamDInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16);
}

TEST(ApplyAdamDGraphInfer, InferDataTypePerInputMapping)
{
    RunApplyAdamDInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16);
}
