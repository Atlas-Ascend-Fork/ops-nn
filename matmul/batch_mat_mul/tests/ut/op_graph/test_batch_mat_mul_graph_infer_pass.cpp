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
 * \file test_batch_mat_mul_graph_infer_pass.cpp
 * \brief BatchMatMul InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeBatchMatMul(gert::InferDataTypeContext* context);
}

namespace {
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 1;

gert::ContextHolder<gert::InferDataTypeContext> BuildBatchMatMulDtypeContext(ge::DataType x1Dtype, ge::DataType x2Dtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("BatchMatMul").OpName("BatchMatMul");
    builder.IONum(kInputNum, kOutputNum);
    builder.InputTensorDesc(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    return holder;
}
} // namespace

TEST(BatchMatMulGraphInfer, InferDataTypeFp32OutFp32)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(BatchMatMulGraphInfer, InferDataTypeFp16OutFp16)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(BatchMatMulGraphInfer, InferDataTypeBf16OutBf16)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_BF16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST(BatchMatMulGraphInfer, InferDataTypeInt32OutInt32)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_INT32, ge::DT_INT32);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(BatchMatMulGraphInfer, InferDataTypeHiFloat8OutHiFloat8)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_HIFLOAT8, ge::DT_HIFLOAT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_HIFLOAT8);
}

TEST(BatchMatMulGraphInfer, RejectMixedFp16Bf16Input)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(ops::InferDataTypeBatchMatMul(context), ge::GRAPH_FAILED);
}

// int8输入走整数累加,y为int32(对齐950组合表)
TEST(BatchMatMulGraphInfer, InferDataTypeInt8InInt32Out)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_INT8, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

// 对齐canndev RT1行为: 声明的y(fp32)被按x1的推导结果覆盖为fp16
TEST(BatchMatMulGraphInfer, DeclaredFp32OutOverriddenToFp16)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_FLOAT), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// 输出槽未初始化(builder残留DT_MAX)时视同未声明y,y跟随x1,不因残留值推导失败
TEST(BatchMatMulGraphInfer, InferDataTypeUninitializedOutputSlot)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("BatchMatMul").OpName("BatchMatMul");
    builder.IONum(kInputNum, kOutputNum);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetOutputDataType(0), ge::DT_MAX);
    ASSERT_EQ(ops::InferDataTypeBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(BatchMatMulGraphInfer, RejectsNullContext) { EXPECT_EQ(ops::InferDataTypeBatchMatMul(nullptr), ge::GRAPH_FAILED); }
