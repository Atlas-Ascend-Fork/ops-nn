/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_dynamic_mx_quant_infershape.cpp
 * \brief
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
#include "../../../op_graph/grouped_dynamic_hi_f4_quant_proto.h"
#include "ut_op_common.h"
#include "any_value.h"

class GroupedDynamicHiF4Quant : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupedDynamicHiF4Quant SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupedDynamicHiF4Quant TearDown" << std::endl; }
};

TEST_F(GroupedDynamicHiF4Quant, GroupedDynamicHiF4Quant_infershape_case_0)
{
    ge::op::GroupedDynamicHiF4Quant op;
    op.UpdateInputDesc("x", create_desc({32, 128}, ge::DT_BF16));
    op.UpdateInputDesc("group_index", create_desc({1}, ge::DT_INT32));
    op.SetAttr("scale_alg", 0);
    Runtime2TestParam param{{"scale_alg"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {32, 128};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    std::vector<int64_t> expectedScaleShape = {1, 128, 1};
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(GroupedDynamicHiF4Quant, GroupedDynamicHiF4Quant_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicHiF4Quant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicHiF4Quant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType inDtype = ge::DT_BF16;
        ge::DataType in2Dtype = ge::DT_INT32;
        ge::DataType outDtype = ge::DT_HIFLOAT4;
        ge::DataType out2Dtype = ge::DT_HIFLOAT4_SCALE;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, in2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, out2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                                  .InputDataTypes({&inDtype, &in2Dtype})
                                  .OutputDataTypes({&outDtype, &out2Dtype})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), inDtype);
        EXPECT_EQ(context->GetInputDataType(1), in2Dtype);
        EXPECT_EQ(context->GetOutputDataType(0), outDtype);
        EXPECT_EQ(context->GetOutputDataType(1), out2Dtype);
    }
}
