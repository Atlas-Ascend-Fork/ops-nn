/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "exe_graph/runtime/kernel_context.h"
#include "infer_shape_context_faker.h"
#include "op_impl_registry.h"
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "../../../op_graph/rms_norm_dynamic_mx_quant_proto.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

class RmsNormDynamicMxQuantInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmsNormDynamicMxQuantInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RmsNormDynamicMxQuantInferShapeTest TearDown" << std::endl; }
};

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_case1)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({8, 64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({64}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({64}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {8, 64};
    std::vector<int64_t> expected_mxscale_shape = {8, 1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

static void CheckInferShapeWithRstd(std::initializer_list<int64_t> xShape, std::initializer_list<int64_t> gammaShape,
                                    const std::vector<int64_t>& expectedYShape,
                                    const std::vector<int64_t>& expectedMxScaleShape,
                                    const std::vector<int64_t>& expectedRstdShape)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc(xShape, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc(gammaShape, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc(gammaShape, ge::DT_FLOAT16));

    op.SetAttr("epsilon", static_cast<float>(1e-6));
    op.SetAttr("scale_alg", 0);
    op.SetAttr("round_mode", "rint");
    op.SetAttr("dst_type", 40);
    op.SetAttr("output_rstd", true);
    Runtime2TestParam param{{"epsilon", "scale_alg", "round_mode", "dst_type", "output_rstd"}, {}, {}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    auto output_rstd_desc = op.GetOutputDesc(2);
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expectedYShape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expectedMxScaleShape);
    EXPECT_EQ(output_rstd_desc.GetShape().GetDims(), expectedRstdShape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_case2)
{
    CheckInferShapeWithRstd({8, 64}, {64}, {8, 64}, {8, 1, 2}, {8, 1});
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infer_dtype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType mx_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                                  .NodeAttrs({{"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")}})
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)}})
                                  .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&y_ref, &mx_scale_ref, &rstd_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), mx_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), rstd_ref);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_3d_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({4}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({4}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {4, 3, 4};
    std::vector<int64_t> expected_mxscale_shape = {4, 3, 1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_4d_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({2, 3, 4, 5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({5}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({5}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {2, 3, 4, 5};
    std::vector<int64_t> expected_mxscale_shape = {2, 3, 4, 1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_single_element_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({1}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({1}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {1};
    std::vector<int64_t> expected_mxscale_shape = {1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_aligned_boundary_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({256, 256}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({256}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({256}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {256, 256};
    std::vector<int64_t> expected_mxscale_shape = {256, 4, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_unaligned_fp16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({33, 17}, ge::DT_FLOAT16));
    op.UpdateInputDesc("gamma", create_desc({17}, ge::DT_FLOAT16));
    op.UpdateInputDesc("beta", create_desc({17}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {33, 17};
    std::vector<int64_t> expected_mxscale_shape = {33, 1, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_single_element_bf16)
{
    ge::op::RmsNormDynamicMxQuant op;
    op.UpdateInputDesc("x", create_desc({1024}, ge::DT_BF16));
    op.UpdateInputDesc("gamma", create_desc({1024}, ge::DT_BF16));
    op.UpdateInputDesc("beta", create_desc({1024}, ge::DT_BF16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

    auto output_y_desc = op.GetOutputDesc(0);
    auto output_mxscale_desc = op.GetOutputDesc(1);
    std::vector<int64_t> expected_y_shape = {1024};
    std::vector<int64_t> expected_mxscale_shape = {16, 2};
    EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_mxscale_desc.GetShape().GetDims(), expected_mxscale_shape);
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_m_not0_n_0_true)
{
    CheckInferShapeWithRstd({176, 0}, {0}, {176, 0}, {176, 0, 2}, {176, 1});
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_m_0_n_not0_true)
{
    CheckInferShapeWithRstd({0, 100}, {100}, {0, 100}, {0, 2, 2}, {0, 1});
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_m_0_n_0_true)
{
    CheckInferShapeWithRstd({0, 0}, {0}, {0, 0}, {0, 0, 2}, {0, 1});
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_unknown_rank)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_shape;

    if (infer_shape_func != nullptr) {
        gert::StorageShape x_shape = {{-2}, {-2}};
        gert::StorageShape y_shape = {{-2}, {-2}};
        gert::StorageShape mxscale_shape = {{-2}, {-2}};
        gert::StorageShape rstd_shape = {{-2}, {-2}};

        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &x_shape, &x_shape})
                          .OutputShapes({&y_shape, &mxscale_shape, &rstd_shape})
                          .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                      {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(40)},
                                      {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();

        auto context = holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), -2);
        EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), -2);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_unknown_rank_with_rstd)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_shape;

    if (infer_shape_func != nullptr) {
        gert::StorageShape x_shape = {{-2}, {-2}};
        gert::StorageShape y_shape = {{-2}, {-2}};
        gert::StorageShape mxscale_shape = {{-2}, {-2}};
        gert::StorageShape rstd_shape = {{-2}, {-2}};

        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &x_shape, &x_shape})
                          .OutputShapes({&y_shape, &mxscale_shape, &rstd_shape})
                          .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                      {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(40)},
                                      {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();

        auto context = holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), -2);
        EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), -2);
        EXPECT_EQ(context->GetOutputShape(2)->GetDim(0), -2);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_unknown_dim)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_shape;

    if (infer_shape_func != nullptr) {
        gert::StorageShape x_shape = {{8, -1}, {8, -1}};
        gert::StorageShape gamma_shape = {{-1}, {-1}};
        gert::StorageShape y_shape = {{8, -1}, {8, -1}};
        gert::StorageShape mxscale_shape = {{8, -1, 2}, {8, -1, 2}};
        gert::StorageShape rstd_shape = {{8, 1}, {8, 1}};

        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &gamma_shape})
                          .OutputShapes({&y_shape, &mxscale_shape, &rstd_shape})
                          .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                      {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(40)},
                                      {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();

        auto context = holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 8);
        EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), -1);
        EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), 8);
        EXPECT_EQ(context->GetOutputShape(1)->GetDim(1), -1);
        EXPECT_EQ(context->GetOutputShape(1)->GetDim(2), 2);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_unknown_dim_with_rstd)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_shape;

    if (infer_shape_func != nullptr) {
        gert::StorageShape x_shape = {{8, -1}, {8, -1}};
        gert::StorageShape gamma_shape = {{-1}, {-1}};
        gert::StorageShape y_shape = {{8, -1}, {8, -1}};
        gert::StorageShape mxscale_shape = {{8, -1, 2}, {8, -1, 2}};
        gert::StorageShape rstd_shape = {{8, 1}, {8, 1}};

        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &gamma_shape})
                          .OutputShapes({&y_shape, &mxscale_shape, &rstd_shape})
                          .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                      {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(40)},
                                      {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();

        auto context = holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 8);
        EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), -1);
        EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), 8);
        EXPECT_EQ(context->GetOutputShape(1)->GetDim(1), -1);
        EXPECT_EQ(context->GetOutputShape(1)->GetDim(2), 2);
        EXPECT_EQ(context->GetOutputShape(2)->GetDim(0), 8);
        EXPECT_EQ(context->GetOutputShape(2)->GetDim(1), 1);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infer_dtype_float4_e2m1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType y_ref = ge::DT_FLOAT4_E2M1;
        ge::DataType mx_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                                  .NodeAttrs({{"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")}})
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(40)}})
                                  .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&y_ref, &mx_scale_ref, &rstd_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), mx_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), rstd_ref);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infer_dtype_float4_e1m2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType y_ref = ge::DT_FLOAT4_E1M2;
        ge::DataType mx_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT4_E1M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                                  .NodeAttrs({{"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")}})
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(41)}})
                                  .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&y_ref, &mx_scale_ref, &rstd_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), mx_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), rstd_ref);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infer_dtype_float8_e4m3fn)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType y_ref = ge::DT_FLOAT8_E4M3FN;
        ge::DataType mx_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                                  .NodeAttrs({{"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")}})
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)}})
                                  .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&y_ref, &mx_scale_ref, &rstd_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), mx_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), rstd_ref);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infer_dtype_invalid)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType y_ref = ge::DT_FLOAT16;
        ge::DataType mx_scale_ref = ge::DT_FLOAT8_E8M0;
        ge::DataType rstd_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)}})
                                  .NodeAttrs({{"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                                  .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")}})
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(999)}})
                                  .NodeAttrs({{"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&y_ref, &mx_scale_ref, &rstd_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(RmsNormDynamicMxQuantInfershape, RmsNormDynamicMxQuant_infershape_debug_log)
{
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", true);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant")->infer_shape;

    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape y_shape = {{8, 64}, {8, 64}};
    gert::StorageShape mxscale_shape = {{8, 1, 2}, {8, 1, 2}};
    gert::StorageShape rstd_shape = {{8, 1}, {8, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&y_shape, &mxscale_shape, &rstd_shape})
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6)},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(40)},
                                  {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    auto context = holder.GetContext<gert::InferShapeContext>();
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 8);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 64);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), 8);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(1), 1);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(2), 2);

    setenv("ASCEND_GLOBAL_LOG_LEVEL", "3", true);
}

TEST(RmsNormDynamicMxQuantMetadataError, MissingXMetadataReportsExecutionErrorOnce)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("RmsNormDynamicMxQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::StorageShape x = {{2, 64}, {2, 64}};
    gert::StorageShape gamma = {{64}, {64}};
    auto faker = gert::InferShapeContextFaker();
    faker.SetOpType("RmsNormDynamicMxQuant")
        .NodeIoNum(3, 3)
        .IrInstanceNum({1, 1, 1}, {1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputShapes(std::vector<gert::StorageShape*>{&x, &gamma, &gamma})
        .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6f)},
                    {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                    {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                    {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(40)},
                    {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(false)}});
    auto holder = faker.Build();
    auto* ctx = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(ctx, nullptr);
    ASSERT_NE(ctx->GetInputShape(0), nullptr);
    // Internal-context fault injection, not a claim that a user empty tensor yields nullptr.
    auto* kernel = reinterpret_cast<gert::KernelContext*>(ctx);
    auto* chain = kernel->MutableInput(0);
    ASSERT_NE(chain, nullptr);
    void*& slot = chain->GetValue<void*>();
    void* saved = slot;
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    (void)manager.GetRawErrorMessages();
    slot = nullptr;
    const auto status = impl->infer_shape(ctx);
    slot = saved; // Preserve holder ownership and deleter.
    const auto errors = manager.GetRawErrorMessages();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1014");
    const auto& fields = errors[0].args_map;
    ASSERT_EQ(fields.count("opName"), 1U);
    EXPECT_EQ(fields.at("opName"), ctx->GetNodeName());
    ASSERT_EQ(fields.count("reason"), 1U);
    EXPECT_NE(fields.at("reason").find("input x shape metadata"), std::string::npos);
}
