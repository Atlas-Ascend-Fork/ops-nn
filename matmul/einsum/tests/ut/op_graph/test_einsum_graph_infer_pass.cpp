/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_einsum_graph_infer_pass.cpp
 * \brief Einsum InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeEinsum(gert::InferDataTypeContext* context);
}

namespace {
constexpr size_t kOutputNum = 1;

gert::ContextHolder<gert::InferDataTypeContext> BuildEinsumDtypeContext(const std::vector<ge::DataType>& inputDtypes)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("Einsum").OpName("Einsum");
    builder.IONum(inputDtypes.size(), kOutputNum);
    for (size_t i = 0; i < inputDtypes.size(); i++) {
        builder.InputTensorDesc(i, inputDtypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    // builder不提供输出dtype设置,输出槽残留DT_MAX;归一为DT_UNDEFINED模拟真实GE未声明输出的语义
    EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    return holder;
}
} // namespace

TEST(EinsumGraphInfer, InferDataTypeFp32OutFp32)
{
    auto holder = BuildEinsumDtypeContext({ge::DT_FLOAT});
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeEinsum(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(EinsumGraphInfer, InferDataTypeFp16OutFp16)
{
    auto holder = BuildEinsumDtypeContext({ge::DT_FLOAT16});
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeEinsum(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(EinsumGraphInfer, InferDataTypeInt32OutInt32)
{
    auto holder = BuildEinsumDtypeContext({ge::DT_INT32});
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeEinsum(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(EinsumGraphInfer, InferDataTypeFollowsFirstInput)
{
    // 输入dtype一致时y必须跟随第一个输入
    auto holder = BuildEinsumDtypeContext({ge::DT_FLOAT, ge::DT_FLOAT});
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeEinsum(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(EinsumGraphInfer, InferDataTypeMixedInputsRejected)
{
    // 输入间dtype不一致（fp16+fp32）必须被拒绝：einsum语义要求全部输入同dtype
    auto holder = BuildEinsumDtypeContext({ge::DT_FLOAT16, ge::DT_FLOAT});
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(ops::InferDataTypeEinsum(context), ge::GRAPH_FAILED);
}

TEST(EinsumGraphInfer, InferDataTypeMixedInputsRejectedReverse)
{
    // 反向混合（fp32+fp16）同样拒绝
    auto holder = BuildEinsumDtypeContext({ge::DT_FLOAT, ge::DT_FLOAT16});
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(ops::InferDataTypeEinsum(context), ge::GRAPH_FAILED);
}

TEST(EinsumGraphInfer, InferDataTypeOutputOverriddenToInput)
{
    // 输出dtype被强制覆写为输入dtype（即使builder残留DT_UNDEFINED）
    auto holder = BuildEinsumDtypeContext({ge::DT_FLOAT16});
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeEinsum(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(EinsumGraphInfer, InferDataTypeNullContext) { ASSERT_EQ(ops::InferDataTypeEinsum(nullptr), ge::GRAPH_FAILED); }
