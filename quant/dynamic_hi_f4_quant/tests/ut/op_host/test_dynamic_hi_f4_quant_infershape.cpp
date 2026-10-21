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
#include "../../../op_graph/dynamic_hi_f4_quant_proto.h"
#include "ut_op_common.h"

using namespace ge;
using namespace std;

class DynamicHiF4Quant : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "DynamicHiF4Quant SetUp" << endl; }
    static void TearDownTestCase() { cout << "DynamicHiF4Quant TearDown" << endl; }
};

TEST_F(DynamicHiF4Quant, infer_shape_axis_neg1)
{
    constexpr int32_t AXIS = -1;
    ge::op::DynamicHiF4Quant op;
    op.UpdateInputDesc("x", create_desc({32, 128}, ge::DT_BF16));
    op.SetAttr("axis", AXIS);
    Runtime2TestParam param{{"axis", "scale_alg"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {32, 128};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    std::vector<int64_t> expectedScaleShape = {32, 2, 1};
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(DynamicHiF4Quant, infer_shape_axis_neg2)
{
    constexpr int32_t AXIS = -2;
    ge::op::DynamicHiF4Quant op;
    op.UpdateInputDesc("x", create_desc({64, 128}, ge::DT_BF16));
    op.SetAttr("axis", AXIS);
    Runtime2TestParam param{{"axis", "scale_alg"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    auto outputY = op.GetOutputDesc(0);
    auto outputScale = op.GetOutputDesc(1);
    std::vector<int64_t> expectedYShape = {64, 128};
    EXPECT_EQ(outputY.GetShape().GetDims(), expectedYShape);
    std::vector<int64_t> expectedScaleShape = {1, 128, 1};
    EXPECT_EQ(outputScale.GetShape().GetDims(), expectedScaleShape);
}

TEST_F(DynamicHiF4Quant, DynamicHiF4Quant_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicHiF4Quant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicHiF4Quant")->infer_datatype;

    int64_t axis = -1;
    if (data_type_func != nullptr) {
        ge::DataType inDtype = ge::DT_BF16;
        ge::DataType outDtype = ge::DT_HIFLOAT4;
        ge::DataType out2Dtype = ge::DT_HIFLOAT4_SCALE;
        int64_t blockSize = 32;
        int64_t scaleAlg = 0;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(1, 2)
                                  .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, out2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom(axis)},
                                              {"scale_alg", Ops::NN::AnyValue::CreateFrom(scaleAlg)}})
                                  .InputDataTypes({&inDtype})
                                  .OutputDataTypes({&outDtype, &out2Dtype})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), inDtype);
        EXPECT_EQ(context->GetOutputDataType(0), outDtype);
        EXPECT_EQ(context->GetOutputDataType(1), out2Dtype);
    }
}
