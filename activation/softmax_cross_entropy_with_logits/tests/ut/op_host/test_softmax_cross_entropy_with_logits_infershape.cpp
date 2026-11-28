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
#include <vector>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "infershape_test_util.h"
#include "../../../../../tests/ut/common/any_value.h"

class SoftmaxCrossEntropyWithLogitsTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SoftmaxCrossEntropyWithLogitsTest Proto Test SetUp" << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsTest Proto Test TearDown" << std::endl;
    }
};

TEST_F(SoftmaxCrossEntropyWithLogitsTest, softmax_cross_entropy_with_logits_test_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape features_shape = {{50, 1}, {50, 1}};
    gert::StorageShape labels_shape = {{50, 1}, {50, 1}};

    std::vector<gert::StorageShape> output_shapes(2);
    std::vector<void*> output_shapes_ref(2);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&features_shape, &labels_shape})
                      .OutputShapes(output_shapes_ref)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto loss_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_loss_shape = {50};
    ASSERT_EQ(Ops::Base::ToString(*loss_desc), Ops::Base::ToString(expected_loss_shape));
    auto backprop_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_backprop_shape = {50, 1};
    ASSERT_EQ(Ops::Base::ToString(*backprop_desc), Ops::Base::ToString(expected_backprop_shape));
}

TEST_F(SoftmaxCrossEntropyWithLogitsTest, softmax_cross_entropy_with_logits_dynamic_dim_test)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape features_shape = {{-1, 10}, {-1, 10}};
    gert::StorageShape labels_shape = {{-1, -1}, {-1, -1}};

    std::vector<gert::StorageShape> output_shapes(2);
    std::vector<void*> output_shapes_ref(2);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&features_shape, &labels_shape})
                      .OutputShapes(output_shapes_ref)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto loss_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_loss_shape = {-1};
    ASSERT_EQ(Ops::Base::ToString(*loss_desc), Ops::Base::ToString(expected_loss_shape));
    auto backprop_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_backprop_shape = {-1, 10};
    ASSERT_EQ(Ops::Base::ToString(*backprop_desc), Ops::Base::ToString(expected_backprop_shape));
}

TEST_F(SoftmaxCrossEntropyWithLogitsTest, softmax_cross_entropy_with_logits_dynamic_dim_4d_test)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape features_shape = {{-1, 2, 3, 5}, {-1, 2, 3, 5}};
    gert::StorageShape labels_shape = {{-1, 1, 1, -1}, {-1, 1, 1, -1}};

    std::vector<gert::StorageShape> output_shapes(2);
    std::vector<void*> output_shapes_ref(2);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&features_shape, &labels_shape})
                      .OutputShapes(output_shapes_ref)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto loss_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_loss_shape = {-1, 1, 3, 5};
    ASSERT_EQ(Ops::Base::ToString(*loss_desc), Ops::Base::ToString(expected_loss_shape));
    auto backprop_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_backprop_shape = {-1, 2, 3, 5};
    ASSERT_EQ(Ops::Base::ToString(*backprop_desc), Ops::Base::ToString(expected_backprop_shape));
}

TEST_F(SoftmaxCrossEntropyWithLogitsTest, softmax_cross_entropy_with_logits_unknown_rank_regression_test)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape features_shape = {{-2}, {-2}};
    gert::StorageShape labels_shape = {{4, 1, 1, 2139}, {4, 1, 1, 2139}};

    std::vector<gert::StorageShape> output_shapes(2);
    std::vector<void*> output_shapes_ref(2);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&features_shape, &labels_shape})
                      .OutputShapes(output_shapes_ref)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto loss_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_loss_shape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*loss_desc), Ops::Base::ToString(expected_loss_shape));
    auto backprop_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_backprop_shape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*backprop_desc), Ops::Base::ToString(expected_backprop_shape));
}

TEST_F(SoftmaxCrossEntropyWithLogitsTest, softmax_cross_entropy_with_logits_test_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape features_shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    gert::StorageShape labels_shape = {{2, 3, 4, 5}, {2, 3, 4, 5}};

    std::vector<gert::StorageShape> output_shapes(2);
    std::vector<void*> output_shapes_ref(2);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&features_shape, &labels_shape})
                      .OutputShapes(output_shapes_ref)
                      .Build();

    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto loss_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_loss_shape = {2, 1, 4, 5};
    ASSERT_EQ(Ops::Base::ToString(*loss_desc), Ops::Base::ToString(expected_loss_shape));
    auto backprop_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_backprop_shape = {2, 3, 4, 5};
    ASSERT_EQ(Ops::Base::ToString(*backprop_desc), Ops::Base::ToString(expected_backprop_shape));
}

namespace {
// 覆盖 CannBot #12：未知维(-1)与维度 1 广播时，输出维度必须取"非 1 的那一侧"而不是逐维取较大值。
// 规则与仓内广播语义一致：(-1, 1) 与 (1, -1) -> -1；(-1, d>1) -> d；(0, 1) -> 0。
struct ScewlBroadcastCase {
    const char* name;
    std::vector<int64_t> features;
    std::vector<int64_t> labels;
    ge::graphStatus expected_status;
    std::vector<int64_t> expected_loss;
    std::vector<int64_t> expected_backprop;
};

class SoftmaxCrossEntropyWithLogitsBroadcastTest : public testing::TestWithParam<ScewlBroadcastCase> {};

TEST_P(SoftmaxCrossEntropyWithLogitsBroadcastTest, infer_shape_merges_dims_by_broadcast_rule)
{
    const auto& param = GetParam();
    const auto* op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxCrossEntropyWithLogits");
    ASSERT_NE(op_impl, nullptr);
    auto infer_shape_func = op_impl->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape features_shape = CreateStorageShape(param.features);
    gert::StorageShape labels_shape = CreateStorageShape(param.labels);

    std::vector<gert::StorageShape> output_shapes(2);
    std::vector<void*> output_shapes_ref(2);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&features_shape, &labels_shape})
                      .OutputShapes(output_shapes_ref)
                      .Build();

    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_EQ(infer_shape_func(context), param.expected_status);
    if (param.expected_status != ge::GRAPH_SUCCESS) {
        return;
    }
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(0)), Ops::Base::ToString(CreateShape(param.expected_loss)));
    EXPECT_EQ(Ops::Base::ToString(*context->GetOutputShape(1)),
              Ops::Base::ToString(CreateShape(param.expected_backprop)));
}

const ScewlBroadcastCase BROADCAST_CASES[] = {
    {"UnknownRowVsBroadcastRow", {-1, 10}, {1, 10}, ge::GRAPH_SUCCESS, {-1}, {-1, 10}},
    {"BroadcastRowVsUnknownRow", {1, 10}, {-1, 10}, ge::GRAPH_SUCCESS, {-1}, {-1, 10}},
    {"UnknownColumnVsBroadcastColumn", {3, -1}, {3, 1}, ge::GRAPH_SUCCESS, {3}, {3, -1}},
    {"BroadcastColumnVsUnknownColumn", {3, 1}, {3, -1}, ge::GRAPH_SUCCESS, {3}, {3, -1}},
    // rank 1 的一侧按前导 1 补齐，补出的 1 不能把未知维压成 1。
    {"UnknownRowVsRank1Labels", {-1, 10}, {10}, ge::GRAPH_SUCCESS, {-1}, {-1, 10}},
    {"Rank1FeaturesVsUnknownRow", {10}, {-1, 10}, ge::GRAPH_SUCCESS, {-1}, {-1, 10}},
    {"UnknownRowVsKnownRow", {-1, 10}, {4, 10}, ge::GRAPH_SUCCESS, {4}, {4, 10}},
    {"KnownRowVsUnknownRow", {4, 10}, {-1, 10}, ge::GRAPH_SUCCESS, {4}, {4, 10}},
    // 空 Tensor：0 与 1 广播得 0。
    {"ZeroRowVsBroadcastRow", {0, 10}, {1, 10}, ge::GRAPH_SUCCESS, {0}, {0, 10}},
    // 4D 的空间维同样按广播规则合并。
    {"Unknown4dSpatialVsBroadcastSpatial",
     {2, 3, -1, 5},
     {2, 3, 1, 5},
     ge::GRAPH_SUCCESS,
     {2, 1, -1, 5},
     {2, 3, -1, 5}},
    {"Broadcast4dBatchVsUnknownBatch", {1, 3, 4, 5}, {-1, 3, 4, 5}, ge::GRAPH_SUCCESS, {-1, 1, 4, 5}, {-1, 3, 4, 5}},
    // 两个已知且不等、均不为 1 的具体维仍应报错。
    {"IncompatibleKnownRows", {2, 10}, {3, 10}, ge::GRAPH_FAILED, {}, {}},
};

INSTANTIATE_TEST_SUITE_P(BroadcastScenarios, SoftmaxCrossEntropyWithLogitsBroadcastTest,
                         testing::ValuesIn(BROADCAST_CASES),
                         [](const testing::TestParamInfo<ScewlBroadcastCase>& info) {
                             return std::string(info.param.name);
                         });
} // namespace
