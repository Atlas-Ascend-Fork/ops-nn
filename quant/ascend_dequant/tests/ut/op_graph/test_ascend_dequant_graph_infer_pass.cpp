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
 * \file test_ascend_dequant_graph_infer.cpp
 * \brief AscendDequant graph InferDataType UT（op_graph 侧注册）
 */

#include <cstdint>

#include <gtest/gtest.h>
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "graph/types.h"

namespace ops {
ge::graphStatus InferDataTypeForAscendDequant(gert::InferDataTypeContext* context);
} // namespace ops

namespace {

constexpr int64_t kAttrDtypeFloat32 = 0;
constexpr int64_t kAttrDtypeFloat16 = 1;

// 构造 InferDataType 上下文：x=int32、deq_scale=uint64、y 待推导。
// withDtypeAttr=false 时不追加 dtype 属性（对应属性缺失场景）。
ge::graphStatus RunInferDataType(bool withDtypeAttr, int64_t dtypeAttr, ge::DataType xDtype, ge::DataType& outDtype)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpType("AscendDequant").OpName("ascend_dequant_dtype").IONum(2, 1);
    builder.InputTensorDesc(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.AppendAttr(false).AppendAttr(false); // sqrt_mode, relu_flag
    if (withDtypeAttr) {
        builder.AppendAttr(dtypeAttr);
    }
    auto holder = builder.Build();
    auto* ctx = holder.GetContext();
    if (ctx == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::graphStatus ret = ops::InferDataTypeForAscendDequant(ctx);
    if (ret == ge::GRAPH_SUCCESS) {
        outDtype = ctx->GetOutputDataType(0);
    }
    return ret;
}

} // namespace

// dtype=0 -> float32（老版默认）
TEST(AscendDequantGraphInferTest, InferDataTypeDtypeAttrFloat32)
{
    ge::DataType out = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(true, kAttrDtypeFloat32, ge::DT_INT32, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, ge::DT_FLOAT);
}

// dtype=1 -> float16
TEST(AscendDequantGraphInferTest, InferDataTypeDtypeAttrFloat16)
{
    ge::DataType out = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(true, kAttrDtypeFloat16, ge::DT_INT32, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, ge::DT_FLOAT16);
}

// 与老版一致：attr dtype 不做值域校验，按属性值原样 cast
TEST(AscendDequantGraphInferTest, InferDataTypeOutOfRangeAttrIsCastVerbatim)
{
    ge::DataType out = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(true, 2, ge::DT_INT32, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, static_cast<ge::DataType>(2));
}

// 属性缺失时沿用 x 的 dtype
TEST(AscendDequantGraphInferTest, InferDataTypeMissingAttrFallsBackToInputDtype)
{
    ge::DataType out = ge::DT_UNDEFINED;
    ASSERT_EQ(RunInferDataType(false, 0, ge::DT_INT32, out), ge::GRAPH_SUCCESS);
    EXPECT_EQ(out, ge::DT_INT32);
}
