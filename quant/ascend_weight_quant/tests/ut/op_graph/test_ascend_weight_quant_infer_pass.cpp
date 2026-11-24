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
 * \file test_ascend_weight_quant_infer_pass.cpp
 * \brief AscendWeightQuant InferDataType UT (op_graph)
 */

#include <gtest/gtest.h>
#include "op_infer_datatype_context_builder.h"

namespace ops {
ge::graphStatus InferDataTypeForAscendWeightQuant(gert::InferDataTypeContext* context);
} // namespace ops

namespace {
constexpr int64_t DST_TYPE_INT8 = 2;
constexpr int64_t DST_TYPE_INT4 = 29;
constexpr int64_t DST_TYPE_UNSUPPORTED = 0;

gert::ContextHolder<gert::InferDataTypeContext> BuildDtypeContext(bool withDstType, int64_t dstType)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AscendWeightQuant").OpName("AscendWeightQuant");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_INT8, ge::FORMAT_HWCN, ge::FORMAT_HWCN);
    builder.InputTensorDesc(1, ge::DT_INT8, ge::FORMAT_HWCN, ge::FORMAT_HWCN);
    builder.OutputTensorDesc(0, ge::FORMAT_HWCN, ge::FORMAT_HWCN);
    if (withDstType) {
        builder.AppendAttr(dstType);
    }
    return builder.Build();
}
} // namespace

TEST(AscendWeightQuantGraphInfer, InferDataTypeDefaultInt8)
{
    auto holder = BuildDtypeContext(false, DST_TYPE_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeForAscendWeightQuant(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
}

TEST(AscendWeightQuantGraphInfer, InferDataTypeInt8)
{
    auto holder = BuildDtypeContext(true, DST_TYPE_INT8);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeForAscendWeightQuant(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
}

TEST(AscendWeightQuantGraphInfer, InferDataTypeInt4)
{
    auto holder = BuildDtypeContext(true, DST_TYPE_INT4);
    auto* context = holder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->SetOutputDataType(0, ge::DT_UNDEFINED), ge::GRAPH_SUCCESS);
    ASSERT_EQ(ops::InferDataTypeForAscendWeightQuant(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT4);
}

TEST(AscendWeightQuantGraphInfer, RejectsUnsupportedDstType)
{
    const std::vector<int64_t> unsupported = {DST_TYPE_UNSUPPORTED, 1, 3, 4, 5, 6};
    for (int64_t dstType : unsupported) {
        auto holder = BuildDtypeContext(true, dstType);
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(ops::InferDataTypeForAscendWeightQuant(context), ge::GRAPH_FAILED);
    }
}

TEST(AscendWeightQuantGraphInfer, RejectsNullContext)
{
    EXPECT_EQ(ops::InferDataTypeForAscendWeightQuant(nullptr), ge::GRAPH_FAILED);
}
