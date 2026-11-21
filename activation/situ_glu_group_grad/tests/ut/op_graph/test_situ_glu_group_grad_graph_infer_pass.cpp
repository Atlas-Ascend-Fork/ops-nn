/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForSituGluGroupGrad(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
// grad_x follows the x dtype (input index 1); grad_weight is always float32.
void CheckDtype(ge::DataType dtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("SituGluGroupGrad").OpName("situ_glu_group_grad_dtype").IONum(2, 2);
    builder.InputTensorDesc(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(1, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(1, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeForSituGluGroupGrad(holder.GetContext()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), dtype);
    EXPECT_EQ(holder.GetContext()->GetOutputDataType(1), ge::DT_FLOAT);
}
} // namespace

TEST(SituGluGroupGradGraphInfer, PreservesAllSupportedDataTypes)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        CheckDtype(dtype);
    }
}
