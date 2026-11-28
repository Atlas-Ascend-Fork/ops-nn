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
 * \file test_transpose_quant_batch_mat_mul_graph_infer_pass.cpp
 * \brief TransposeQuantBatchMatMul InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForTransposeQuantBatchMatMul(gert::InferDataTypeContext* context);
}

namespace {
constexpr size_t kTqbmmIrInputNum = 5;
constexpr size_t kTqbmmOutputNum = 1;
constexpr size_t kDtypeAttrIdx = 0;

gert::ContextHolder<gert::InferDataTypeContext> BuildTqbmmDtypeContext(ge::DataType dtype, bool withDtypeAttr = true)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("TransposeQuantBatchMatMul").OpName("TransposeQuantBatchMatMul");
    builder.IONum(kTqbmmIrInputNum, kTqbmmOutputNum);
    builder.InputTensorDesc(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    if (withDtypeAttr) {
        builder.AppendAttr(static_cast<int64_t>(dtype));
    }
    auto holder = builder.Build();
    EXPECT_EQ(holder.GetContext()->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    return holder;
}
} // namespace

TEST(TransposeQuantBatchMatMulGraphInfer, InferDataTypeFp16)
{
    auto holder = BuildTqbmmDtypeContext(ge::DT_FLOAT16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeQuantBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST(TransposeQuantBatchMatMulGraphInfer, InferDataTypeBf16)
{
    auto holder = BuildTqbmmDtypeContext(ge::DT_BF16);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeQuantBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}

TEST(TransposeQuantBatchMatMulGraphInfer, InferDataTypeHifloat8)
{
    auto holder = BuildTqbmmDtypeContext(ge::DT_HIFLOAT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeQuantBatchMatMul(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_HIFLOAT8);
}

TEST(TransposeQuantBatchMatMulGraphInfer, InferDataTypeInvalidDtype)
{
    auto holder = BuildTqbmmDtypeContext(ge::DT_FLOAT);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeQuantBatchMatMul(context), ge::GRAPH_FAILED);
}

TEST(TransposeQuantBatchMatMulGraphInfer, InferDataTypeAttrsNull)
{
    auto holder = BuildTqbmmDtypeContext(ge::DT_FLOAT16, false);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(ops::InferDataTypeForTransposeQuantBatchMatMul(context), ge::GRAPH_FAILED);
}

TEST(TransposeQuantBatchMatMulGraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeForTransposeQuantBatchMatMul(nullptr), ge::GRAPH_FAILED);
}
