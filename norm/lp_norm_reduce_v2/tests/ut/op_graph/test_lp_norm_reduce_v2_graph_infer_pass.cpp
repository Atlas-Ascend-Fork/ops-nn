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
 * \file test_lp_norm_reduce_v2_graph_infer_pass.cpp
 * \brief LpNormReduceV2 图模式 InferDataType UT（op_graph 共享 runner，直测注册函数）
 *
 * 覆盖（spec dtype_policy.same_as_first_input / DESIGN §6.6，源工程 inferdtype 用例迁移）：
 *   - fp32 / fp16 / bf16 → y.dtype = x.dtype（GRAPH_SUCCESS）
 *   - 非法 dtype（INT32 / DOUBLE）→ dtype_not_supported（GRAPH_FAILED）
 *   - 空上下文防御分支（nullptr → GRAPH_FAILED）
 */

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForLpNormReduceV2(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
const char* OP_TYPE = "LpNormReduceV2";

void CheckInferDataType(ge::DataType inputDtype, ge::graphStatus expectStatus, ge::DataType expectOutputDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType(OP_TYPE).OpName("lp_norm_reduce_v2_dtype").IONum(1, 1);
    builder.InputTensorDesc(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    ASSERT_NE(holder.GetContext(), nullptr);
    ASSERT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeForLpNormReduceV2(holder.GetContext()), expectStatus);
    if (expectStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(holder.GetContext()->GetOutputDataType(0), expectOutputDtype);
    }
}
} // namespace

TEST(LpNormReduceV2GraphInfer, fp32_passthrough) { CheckInferDataType(ge::DT_FLOAT, ge::GRAPH_SUCCESS, ge::DT_FLOAT); }

TEST(LpNormReduceV2GraphInfer, fp16_passthrough)
{
    CheckInferDataType(ge::DT_FLOAT16, ge::GRAPH_SUCCESS, ge::DT_FLOAT16);
}

TEST(LpNormReduceV2GraphInfer, bf16_passthrough) { CheckInferDataType(ge::DT_BF16, ge::GRAPH_SUCCESS, ge::DT_BF16); }

// 黑盒路由 lpnorm_l2_018: x dtype=int32 拒收（InferDataType dtype_not_supported，spec dtype_set
// = {float16, float32, bfloat16}；real 链路编译期拒收由 verify.cpp/graph_infer 双侧保证）
TEST(LpNormReduceV2GraphInfer, int32_rejected) { CheckInferDataType(ge::DT_INT32, ge::GRAPH_FAILED, ge::DT_UNDEFINED); }

TEST(LpNormReduceV2GraphInfer, double_rejected)
{
    CheckInferDataType(ge::DT_DOUBLE, ge::GRAPH_FAILED, ge::DT_UNDEFINED);
}

TEST(LpNormReduceV2GraphInfer, null_context_rejected)
{
    EXPECT_EQ(ops::InferDataTypeForLpNormReduceV2(nullptr), ge::GRAPH_FAILED);
}
