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
 * \file test_in_training_update_grad_infershape.cpp
 * \brief InferShape/InferDataType UT for INTrainingUpdateGrad。
 *        两个输出的 shape 取自 variance，dtype 恒为 FLOAT。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "../../../op_graph/in_training_update_grad_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class INTrainingUpdateGradTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "INTrainingUpdateGrad Proto Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "INTrainingUpdateGrad Proto Test TearDown" << std::endl; }
};

// 输出 shape 跟随 variance（空间维为 1）。
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_shape_from_variance)
{
    ge::op::INTrainingUpdateGrad op;

    ge::Format format = ge::FORMAT_NDC1HWC0;
    auto full_tensor = create_desc_with_ori({2, 4, 2, 8, 8, 16}, ge::DT_FLOAT, format, {2, 4, 2, 8, 8, 16}, format);
    auto param_tensor = create_desc_with_ori({2, 1, 2, 1, 1, 16}, ge::DT_FLOAT, format, {2, 1, 2, 1, 1, 16}, format);
    op.UpdateInputDesc("dy", full_tensor);
    op.UpdateInputDesc("x", full_tensor);
    op.UpdateInputDesc("variance", param_tensor);
    op.UpdateInputDesc("mean", param_tensor);

    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

    auto res_gamma_desc = op.GetOutputDescByName("res_gamma");
    auto res_beta_desc = op.GetOutputDescByName("res_beta");
    EXPECT_EQ(res_gamma_desc.GetShape().GetDimNum(), 6);
    EXPECT_EQ(res_gamma_desc.GetShape().GetDim(0), 2);
    EXPECT_EQ(res_gamma_desc.GetShape().GetDim(1), 1);
    EXPECT_EQ(res_gamma_desc.GetShape().GetDim(2), 2);
    EXPECT_EQ(res_gamma_desc.GetShape().GetDim(5), 16);
    EXPECT_EQ(res_beta_desc.GetShape().GetDimNum(), 6);
    EXPECT_EQ(res_beta_desc.GetShape().GetDim(5), 16);
}

// dy/x 为 fp16 时，两个输出仍恒为 fp32。
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_data_type_fp16_input)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateGrad")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType dy_dt = ge::DT_FLOAT16;
    ge::DataType param_dt = ge::DT_FLOAT;
    ge::DataType out_dt = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 2)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .InputDataTypes({&dy_dt, &dy_dt, &param_dt, &param_dt})
                              .OutputDataTypes({&out_dt, &out_dt})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
}

// -2 UNKNOWN_RANK：红线 R4 点名的必验项。两个输出都取自 variance 的整体拷贝，
// variance 为 unknown-rank 时应原样透传，不得推成非法形状。
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_shape_unknown_rank)
{
    ge::op::INTrainingUpdateGrad op;
    op.UpdateInputDesc("dy", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("variance", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("mean", create_desc({-2}, ge::DT_FLOAT));
    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    std::vector<int64_t> expected{-2};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected);
}

// 混合未知秩：dy 具体而 x/variance/mean 为 [-2]（GEIR 动态图投影场景，FR-005）——
// 必须透传 unknown，不得因 dy 具体而进入严格校验拒绝
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_shape_mixed_unknown_rank)
{
    ge::op::INTrainingUpdateGrad op;
    ge::Format format = ge::FORMAT_NDC1HWC0;
    auto full_tensor = create_desc_with_ori({2, 4, 2, 8, 8, 16}, ge::DT_FLOAT, format, {2, 4, 2, 8, 8, 16}, format);
    auto unknown = create_desc({-2}, ge::DT_FLOAT);
    op.UpdateInputDesc("dy", full_tensor);
    op.UpdateInputDesc("x", unknown);
    op.UpdateInputDesc("variance", unknown);
    op.UpdateInputDesc("mean", unknown);
    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    std::vector<int64_t> expected{-2};
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), expected);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), expected);
}

// N=0：README 承诺拒绝，Host 门应独立拒绝（FR-006）
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_shape_reject_non_positive_n)
{
    ge::op::INTrainingUpdateGrad op;
    ge::Format format = ge::FORMAT_NDC1HWC0;
    auto full_tensor = create_desc_with_ori({0, 4, 2, 8, 8, 16}, ge::DT_FLOAT, format, {0, 4, 2, 8, 8, 16}, format);
    auto param_tensor = create_desc_with_ori({0, 1, 2, 1, 1, 16}, ge::DT_FLOAT, format, {0, 1, 2, 1, 1, 16}, format);
    op.UpdateInputDesc("dy", full_tensor);
    op.UpdateInputDesc("x", full_tensor);
    op.UpdateInputDesc("variance", param_tensor);
    op.UpdateInputDesc("mean", param_tensor);
    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// C0=65 超 64 lane 上界：Host 门独立拒绝（FR-006）
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_shape_reject_c0_over_64)
{
    ge::op::INTrainingUpdateGrad op;
    ge::Format format = ge::FORMAT_NDC1HWC0;
    auto full_tensor = create_desc_with_ori({2, 4, 2, 8, 8, 65}, ge::DT_FLOAT, format, {2, 4, 2, 8, 8, 65}, format);
    auto param_tensor = create_desc_with_ori({2, 1, 2, 1, 1, 65}, ge::DT_FLOAT, format, {2, 1, 2, 1, 1, 65}, format);
    op.UpdateInputDesc("dy", full_tensor);
    op.UpdateInputDesc("x", full_tensor);
    op.UpdateInputDesc("variance", param_tensor);
    op.UpdateInputDesc("mean", param_tensor);
    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// ---------------- Host 图构建门校验：静态形状/_dtype 违约在此拒绝（原放行至 tiling 门） ----------------

// dy 已知秩但非 6 维：Host 门拒绝
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_shape_reject_rank_not_six)
{
    ge::op::INTrainingUpdateGrad op;
    ge::Format format = ge::FORMAT_NDC1HWC0;
    auto full_tensor = create_desc_with_ori({2, 2, 8, 8, 16}, ge::DT_FLOAT, format, {2, 2, 8, 8, 16}, format);
    auto param_tensor = create_desc_with_ori({2, 1, 2, 1, 1, 16}, ge::DT_FLOAT, format, {2, 1, 2, 1, 1, 16}, format);
    op.UpdateInputDesc("dy", full_tensor);
    op.UpdateInputDesc("x", full_tensor);
    op.UpdateInputDesc("variance", param_tensor);
    op.UpdateInputDesc("mean", param_tensor);
    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// x 与 dy 形状不一致：Host 门拒绝
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_shape_reject_x_mismatch)
{
    ge::op::INTrainingUpdateGrad op;
    ge::Format format = ge::FORMAT_NDC1HWC0;
    auto dy_tensor = create_desc_with_ori({2, 4, 2, 8, 8, 16}, ge::DT_FLOAT, format, {2, 4, 2, 8, 8, 16}, format);
    auto x_tensor = create_desc_with_ori({2, 4, 2, 8, 9, 16}, ge::DT_FLOAT, format, {2, 4, 2, 8, 9, 16}, format);
    auto param_tensor = create_desc_with_ori({2, 1, 2, 1, 1, 16}, ge::DT_FLOAT, format, {2, 1, 2, 1, 1, 16}, format);
    op.UpdateInputDesc("dy", dy_tensor);
    op.UpdateInputDesc("x", x_tensor);
    op.UpdateInputDesc("variance", param_tensor);
    op.UpdateInputDesc("mean", param_tensor);
    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// variance 非期望的 (N,1,C1,1,1,C0)：Host 门拒绝
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_shape_reject_variance_mismatch)
{
    ge::op::INTrainingUpdateGrad op;
    ge::Format format = ge::FORMAT_NDC1HWC0;
    auto full_tensor = create_desc_with_ori({2, 4, 2, 8, 8, 16}, ge::DT_FLOAT, format, {2, 4, 2, 8, 8, 16}, format);
    auto bad_param = create_desc_with_ori({2, 4, 2, 1, 1, 16}, ge::DT_FLOAT, format, {2, 4, 2, 1, 1, 16}, format);
    op.UpdateInputDesc("dy", full_tensor);
    op.UpdateInputDesc("x", full_tensor);
    op.UpdateInputDesc("variance", bad_param);
    op.UpdateInputDesc("mean", bad_param);
    Runtime2TestParam param;
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
}

// dy dtype 非 fp16/fp32：Host 门拒绝（InferDataType）
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_dtype_reject_invalid_dy)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateGrad")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType dy_dt = ge::DT_INT32;
    ge::DataType param_dt = ge::DT_FLOAT;
    ge::DataType out_dt = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 2)
                              .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .InputDataTypes({&dy_dt, &dy_dt, &param_dt, &param_dt})
                              .OutputDataTypes({&out_dt, &out_dt})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
}

// x 与 dy dtype 不一致：Host 门拒绝（InferDataType）
TEST_F(INTrainingUpdateGradTest, in_training_update_grad_infer_dtype_reject_x_mismatch)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateGrad"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("INTrainingUpdateGrad")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);
    ge::DataType dy_dt = ge::DT_FLOAT;
    ge::DataType x_dt = ge::DT_FLOAT16;
    ge::DataType param_dt = ge::DT_FLOAT;
    ge::DataType out_dt = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(4)
                              .NodeIoNum(4, 2)
                              .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0)
                              .InputDataTypes({&dy_dt, &x_dt, &param_dt, &param_dt})
                              .OutputDataTypes({&out_dt, &out_dt})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
}
