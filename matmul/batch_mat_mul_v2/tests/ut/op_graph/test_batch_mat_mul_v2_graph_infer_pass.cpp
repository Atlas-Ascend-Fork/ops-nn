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
 * \file test_batch_mat_mul_v2_graph_infer_pass.cpp
 * \brief BatchMatMulV2 InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeBatchMatMulV2(gert::InferDataTypeContext* context);
}

namespace {
constexpr size_t kRequiredInputNum = 2;
constexpr size_t kInputNum = 3; // x1/x2/bias
constexpr size_t kOutputNum = 1;

gert::ContextHolder<gert::InferDataTypeContext> BuildBatchMatMulDtypeContext(ge::DataType x1Dtype, ge::DataType x2Dtype,
                                                                             bool withBias = false,
                                                                             ge::DataType biasDtype = ge::DT_UNDEFINED)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("BatchMatMulV2").OpName("BatchMatMulV2");
    builder.IONum(withBias ? kInputNum : kRequiredInputNum, kOutputNum);
    builder.InputTensorDesc(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    if (withBias) {
        builder.InputTensorDesc(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    return holder;
}
} // namespace

TEST(BatchMatMulV2GraphInfer, InferDataTypeFp32OutFp32)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT, ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeFp16OutFp16)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeBf16OutBf16)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_BF16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeInt32OutInt32)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_INT32, ge::DT_INT32);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeHiFloat8OutHiFloat8)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_HIFLOAT8, ge::DT_HIFLOAT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_HIFLOAT8);
}

TEST(BatchMatMulV2GraphInfer, RejectMixedFp16Bf16Input)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_FAILED);
}

// 对齐canndev RT1行为: 声明的y(fp32)被按x1的推导结果覆盖为fp16
TEST(BatchMatMulV2GraphInfer, DeclaredFp32OutOverriddenToFp16)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_FLOAT), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// 输出槽未初始化(builder残留DT_MAX)时视同未声明y,y跟随x1,不因残留值推导失败
TEST(BatchMatMulV2GraphInfer, InferDataTypeUninitializedOutputSlot)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("BatchMatMulV2").OpName("BatchMatMulV2");
    builder.IONum(kRequiredInputNum, kOutputNum);
    builder.InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = builder.Build();
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetOutputDataType(0), ge::DT_MAX);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// v2支持int8/int4输入,整数累加y为int32
TEST(BatchMatMulV2GraphInfer, InferDataTypeInt8InInt32Out)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_INT8, ge::DT_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeInt4InInt32Out)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_INT4, ge::DT_INT4);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT32);
}

TEST(BatchMatMulV2GraphInfer, InferDataTypeFp16InWithFp16Bias)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(BatchMatMulV2GraphInfer, RejectInt4BiasWithFp16Input)
{
    auto holder = BuildBatchMatMulDtypeContext(ge::DT_FLOAT16, ge::DT_FLOAT16, true, ge::DT_INT4);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(ops::InferDataTypeBatchMatMulV2(context), ge::GRAPH_FAILED);
}

TEST(BatchMatMulV2GraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeBatchMatMulV2(nullptr), ge::GRAPH_FAILED);
}
