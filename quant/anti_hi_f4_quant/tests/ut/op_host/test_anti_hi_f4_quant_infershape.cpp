/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ut_op_util.h"
#include "infershape_test_util.h"
#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../op_graph/anti_hi_f4_quant_proto.h"
#include "ut_op_common.h"

class AntiHiF4QuantInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AntiHiF4QuantInferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AntiHiF4QuantInferShape TearDown" << std::endl; }
};

TEST_F(AntiHiF4QuantInferShape, AntiHiF4Quant_infershape_2d)
{
    constexpr int32_t DST_TYPE = 1;
    ge::op::AntiHiF4Quant op;
    op.UpdateInputDesc("x", create_desc({64, 128}, ge::DT_HIFLOAT4));
    op.UpdateInputDesc("scale", create_desc({1, 128, 1}, ge::DT_HIFLOAT4_SCALE));
    op.SetAttr("dst_type", DST_TYPE);
    Runtime2TestParam param{{"dst_type"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    std::vector<int64_t> expectedYShape = {64, 128};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(AntiHiF4QuantInferShape, AntiHiF4Quant_infershape_3d)
{
    ge::op::AntiHiF4Quant op;
    op.UpdateInputDesc("x", create_desc({2, 64, 128}, ge::DT_HIFLOAT4));
    op.UpdateInputDesc("scale", create_desc({2, 1, 128, 1}, ge::DT_HIFLOAT4_SCALE));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc("y");
    std::vector<int64_t> expectedYShape = {2, 64, 128};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
}

TEST_F(AntiHiF4QuantInferShape, AntiHiF4Quant_infershape_invalid_rank)
{
    ge::op::AntiHiF4Quant op;
    op.UpdateInputDesc("x", create_desc({128}, ge::DT_HIFLOAT4));
    op.UpdateInputDesc("scale", create_desc({2, 1}, ge::DT_HIFLOAT4_SCALE));
    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

TEST_F(AntiHiF4QuantInferShape, AntiHiF4Quant_InferDtype_bf16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AntiHiF4Quant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AntiHiF4Quant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType inXDtype = ge::DT_HIFLOAT4;
        ge::DataType inScaleDtype = ge::DT_HIFLOAT4_SCALE;
        ge::DataType outDtype = ge::DT_BF16;
        int64_t dstTypeAttr = 27;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 1)
                                  .NodeInputTd(0, inXDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, inScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&inXDtype, &inScaleDtype})
                                  .OutputDataTypes({&outDtype})
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom((int64_t)outDtype)}})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    }
}

TEST_F(AntiHiF4QuantInferShape, AntiHiF4Quant_InferDtype_fp16)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("AntiHiF4Quant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AntiHiF4Quant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType inXDtype = ge::DT_HIFLOAT4;
        ge::DataType inScaleDtype = ge::DT_HIFLOAT4_SCALE;
        ge::DataType outDtype = ge::DT_FLOAT16;
        int64_t dstTypeAttr = 1;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 1)
                                  .NodeInputTd(0, inXDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, inScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&inXDtype, &inScaleDtype})
                                  .OutputDataTypes({&outDtype})
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom((int64_t)outDtype)}})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
    }
}
