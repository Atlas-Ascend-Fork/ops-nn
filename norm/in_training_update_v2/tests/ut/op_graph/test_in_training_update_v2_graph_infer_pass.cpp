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
ge::graphStatus InferDataTypeForINTrainingUpdateV2(gert::InferDataTypeContext* context);
} // namespace ops

TEST(INTrainingUpdateV2InferDataType, YFollowsXAndStatisticsStayFloat)
{
    for (auto xType : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        gert::OpInferDataTypeContextBuilder builder;
        builder.OpType("INTrainingUpdateV2").OpName("in_training_update_v2_dtype").IONum(3, 3);
        builder.InputTensorDesc(0, xType, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
        builder.InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_ND);
        builder.InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_ND);
        builder.OutputTensorDesc(0, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
        builder.OutputTensorDesc(1, ge::FORMAT_NCHW, ge::FORMAT_ND);
        builder.OutputTensorDesc(2, ge::FORMAT_NCHW, ge::FORMAT_ND);
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        for (size_t output = 0; output < 3; ++output) {
            ASSERT_EQ(context->SetOutputDataType(output, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
        }
        ASSERT_EQ(ops::InferDataTypeForINTrainingUpdateV2(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), xType);
        EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
        EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
    }
}
