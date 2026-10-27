/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/apply_fused_ema_adam_proto.h"

std::vector<int64_t> ToVectorForFused(const gert::Shape& shape)
{
    size_t shape_size = shape.GetDimNum();
    std::vector<int64_t> shape_vec(shape_size, 0);

    for (size_t i = 0; i < shape_size; i++) {
        shape_vec[i] = shape.GetDim(i);
    }
    return shape_vec;
}

class ApplyFusedEmaAdam : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyFusedEmaAdam Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyFusedEmaAdam Proto Test TearDown" << std::endl; }
};

TEST_F(ApplyFusedEmaAdam, apply_fused_ema_dam_infershape_case0)
{
    gert::StorageShape gradShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape varShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape mShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape vShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape sShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape stepShape = {{2, 8}, {2, 8}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ApplyFusedEmaAdam")
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gradShape, &varShape, &mShape, &vShape, &sShape, &stepShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &sShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                                  {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"bias_correction", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFusedEmaAdam")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedOutputShape = {42, 6480};
    auto varOutShape = context->GetOutputShape(0);
    auto mOutShape = context->GetOutputShape(1);
    auto vOutShape = context->GetOutputShape(2);
    auto sOutShape = context->GetOutputShape(3);
    EXPECT_EQ(ToVectorForFused(*varOutShape), expectedOutputShape);
    EXPECT_EQ(ToVectorForFused(*mOutShape), expectedOutputShape);
    EXPECT_EQ(ToVectorForFused(*vOutShape), expectedOutputShape);
    EXPECT_EQ(ToVectorForFused(*sOutShape), expectedOutputShape);
}

TEST_F(ApplyFusedEmaAdam, apply_fused_ema_dam_infershape_bf16_case0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFusedEmaAdam"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFusedEmaAdam")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType input_step_ref = ge::DT_INT64;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .NodeIoNum(6, 4)
                                  .IrInstanceNum({1, 1, 1, 1, 1, 1})
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes(
                                      {&input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_step_ref})
                                  .OutputDataTypes({&output_ref, &output_ref, &output_ref, &output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_ref);
        EXPECT_EQ(context->GetInputDataType(4), input_ref);
        EXPECT_EQ(context->GetInputDataType(5), input_step_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_ref);
        EXPECT_EQ(context->GetOutputDataType(3), output_ref);
    }
}

// 负向场景1：必选输入 step 的实例数置 0（IrInstanceNum 第 6 个输入实例数为 0），
// context 的 input_size（输入实例数之和）变为 5，GetInputShape(INPUT_STEP_INDEX) 按下标越界
// 规则返回空指针，首个触达的整改检查点为 InferShapeForApplyFusedEmaAdam 中 param_name="stepShape"
// （EZ0037 首报），前序 gradShape/varShape/mShape/vShape/sShape 检查点因实例数保留而正常通过
// （见断言），其后序 varRefShape 等输出判空检查点因前置检查已返回而一并覆盖。
// 注意：InferShapeContextFaker 的单参数 IrInstanceNum 是空操作，且 NodeIoNum(IonNum) 与
// IrInstanceNum(IOInstanceNum) 互斥，必须仅用双参数 IrInstanceNum 控制实例数
TEST_F(ApplyFusedEmaAdam, apply_fused_ema_adam_infershape_null_input_shape_001)
{
    gert::StorageShape gradShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape varShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape mShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape vShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape sShape = {{42, 6480}, {42, 6480}};

    // 第 6 个输入 step 实例数置 0（InputShapes 第 6 位 nullptr 占位，faker 不创建该输入链），
    // 保留 4 个输出实例，使输出链存在
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ApplyFusedEmaAdam")
                      .IrInstanceNum({1, 1, 1, 1, 1, 0}, {1, 1, 1, 1})
                      .InputShapes({&gradShape, &varShape, &mShape, &vShape, &sShape, nullptr})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                                  {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"bias_correction", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context->GetInputShape(0), nullptr);
    ASSERT_NE(context->GetInputShape(4), nullptr);
    ASSERT_EQ(context->GetInputShape(5), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFusedEmaAdam")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

// 负向场景2：输出 var_ref 的实例数置 0（IrInstanceNum 输出实例数全 0），
// context 的 output_size 变为 0，GetOutputShape(OUTPUT_VAR_INDEX) 按下标越界规则返回空指针，
// 对应整改检查点 param_name="varRefShape"（InferShapeForApplyFusedEmaAdam 首个输出检查点，
// 其后序 mRefShape/vRefShape/sRefShape 因前置检查已返回而一并覆盖），前序输入 shape 检查点
// 因实例数保留而正常通过（见断言），整改后应上报 EZ0037 首错并返回 ge::GRAPH_FAILED
TEST_F(ApplyFusedEmaAdam, apply_fused_ema_adam_infershape_null_output_shape_002)
{
    gert::StorageShape gradShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape varShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape mShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape vShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape sShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape stepShape = {{2, 8}, {2, 8}};

    // 输出实例数全 0 使 output_size 为 0（不调用 NodeOutputTd，避免为输出创建链），
    // 输入实例数保持 6 个且 shape 全部合法
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ApplyFusedEmaAdam")
                      .IrInstanceNum({1, 1, 1, 1, 1, 1}, {0, 0, 0, 0})
                      .InputShapes({&gradShape, &varShape, &mShape, &vShape, &sShape, &stepShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                                  {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"bias_correction", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context->GetInputShape(5), nullptr);
    ASSERT_EQ(context->GetOutputShape(0), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFusedEmaAdam")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}
