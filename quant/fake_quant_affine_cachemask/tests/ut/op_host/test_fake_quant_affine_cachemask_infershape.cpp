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
 * \file test_fake_quant_affine_cachemask_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "fake_quant_error_code_ut_util.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "../../../op_graph/fake_quant_affine_cachemask_proto.h"

namespace {
constexpr const char* kOpType = "FakeQuantAffineCachemask";
constexpr const char* kSuiteName = "FakeQuantAffineCachemaskTest";

ge::graphStatus RunInferShapeCase(gert::StorageShape& xShape, gert::StorageShape& scaleShape,
                                  gert::StorageShape& zpShape, int64_t axis, int64_t quantMin, int64_t quantMax)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::StorageShape yShape{};
    gert::StorageShape maskShape{};
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(kOpType)
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &scaleShape, &zpShape})
                      .OutputShapes({&yShape, &maskShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                                  {"quant_min", Ops::NN::AnyValue::CreateFrom<int64_t>(quantMin)},
                                  {"quant_max", Ops::NN::AnyValue::CreateFrom<int64_t>(quantMax)}})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) {
        ADD_FAILURE() << "failed to build infer shape context";
        return ge::GRAPH_FAILED;
    }
    return opImpl->infer_shape(context);
}

ge::graphStatus RunInferDtypeCase(ge::DataType xDtype, ge::DataType scaleDtype, ge::DataType zpDtype)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr || opImpl->infer_datatype == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType yDtype = ge::DT_FLOAT;
    ge::DataType maskDtype = ge::DT_BOOL;
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType(kOpType)
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, zpDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&xDtype, &scaleDtype, &zpDtype})
                      .OutputDataTypes({&yDtype, &maskDtype})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    if (context == nullptr) {
        ADD_FAILURE() << "failed to build infer data type context";
        return ge::GRAPH_FAILED;
    }
    return opImpl->infer_datatype(context);
}

const auto g_infershapeListenerInstalled = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new fake_quant_ut::NoEz9999Listener(kSuiteName));
    return 0;
}();
} // namespace

class FakeQuantAffineCachemaskTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "fake_quant_affine_cachemask test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "fake_quant_affine_cachemask test TearDown" << std::endl; }
};

TEST_F(FakeQuantAffineCachemaskTest, fake_quant_affine_cachemask_test_infer_shape)
{
    ge::op::FakeQuantAffineCachemask op;
    op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));

    Runtime2TestParam param;
    auto ret = InferShapeTest(op, param);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    auto output_y = op.GetOutputDescByName("y");
    auto output_mask = op.GetOutputDescByName("mask");
    std::vector<int64_t> expected_output_shape = {4, 3, 4};
    EXPECT_EQ(output_y.GetShape().GetDims(), expected_output_shape);
    EXPECT_EQ(output_mask.GetShape().GetDims(), expected_output_shape);
}

TEST_F(FakeQuantAffineCachemaskTest, fake_quant_affine_cachemask_test_infer_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantAffineCachemask"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantAffineCachemask")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType zero_point_ref = ge::DT_INT32;
        ge::DataType output_y_ref = ge::DT_FLOAT;
        ge::DataType output_mask_ref = ge::DT_BOOL;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(3, 2)
                                  .IrInstanceNum({1, 1, 1})
                                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &zero_point_ref})
                                  .OutputDataTypes({&output_y_ref, &output_mask_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), zero_point_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_mask_ref);
    }
}

// x rank > 8 → EZ0012（x rank must be in range [0, 8]）
TEST_F(FakeQuantAffineCachemaskTest, infershape_x_rank_too_large_ez0012)
{
    gert::StorageShape xShape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape zpShape = {{1}, {1}};
    EXPECT_EQ(RunInferShapeCase(xShape, scaleShape, zpShape, 0, 0, 255), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0012", "infershape_x_rank_too_large");
}

// scale 维度 > 1 → EZ0012（scale must be a 0-D or 1-D tensor）
TEST_F(FakeQuantAffineCachemaskTest, infershape_scale_rank_invalid_ez0012)
{
    gert::StorageShape xShape = {{4, 3}, {4, 3}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    gert::StorageShape zpShape = {{1}, {1}};
    EXPECT_EQ(RunInferShapeCase(xShape, scaleShape, zpShape, 0, 0, 255), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0012", "infershape_scale_rank_invalid");
}

// scale size 与 axis 维长度不匹配 → EZ0015
TEST_F(FakeQuantAffineCachemaskTest, infershape_scale_size_mismatch_ez0015)
{
    gert::StorageShape xShape = {{4, 3}, {4, 3}};
    gert::StorageShape scaleShape = {{3}, {3}};
    gert::StorageShape zpShape = {{1}, {1}};
    EXPECT_EQ(RunInferShapeCase(xShape, scaleShape, zpShape, 0, 0, 255), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0015", "infershape_scale_size_mismatch");
}

// x dtype 不在支持集合内 → EZ0020（InferDataType）
TEST_F(FakeQuantAffineCachemaskTest, infershape_x_dtype_unsupported_ez0020)
{
    EXPECT_EQ(RunInferDtypeCase(ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_INT32), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0020", "infershape_x_dtype_unsupported");
}

// scale 与 x dtype 不一致 → EZ0021（InferDataType）
TEST_F(FakeQuantAffineCachemaskTest, infershape_scale_dtype_mismatch_ez0021)
{
    EXPECT_EQ(RunInferDtypeCase(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0021", "infershape_scale_dtype_mismatch");
}

// quant_min > quant_max → EZ0027
TEST_F(FakeQuantAffineCachemaskTest, infershape_quant_min_gt_quant_max_ez0027)
{
    gert::StorageShape xShape = {{4, 3}, {4, 3}};
    gert::StorageShape scaleShape = {{4}, {4}};
    gert::StorageShape zpShape = {{1}, {1}};
    EXPECT_EQ(RunInferShapeCase(xShape, scaleShape, zpShape, 0, 10, 0), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0027", "infershape_quant_min_gt_quant_max");
}

// axis 越界 → EZ0026
TEST_F(FakeQuantAffineCachemaskTest, infershape_axis_out_of_range_ez0026)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape scaleShape = {{2}, {2}};
    gert::StorageShape zpShape = {{1}, {1}};
    EXPECT_EQ(RunInferShapeCase(xShape, scaleShape, zpShape, 2, 0, 255), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0026", "infershape_axis_out_of_range");
}
