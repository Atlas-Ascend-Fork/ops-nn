/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "op_impl_registry.h"

namespace {
// 根据维度列表 dims 创建逻辑形状。
gert::Shape MakeShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

// 调用形状推导；filter1/2 为两层逻辑权重，format 为权重格式，output 接收结果。
// useDefaults 选择缺省属性，strides/pads 为显式两层几何属性。
ge::graphStatus InferShape(const std::vector<int64_t>& filter1, const std::vector<int64_t>& filter2, ge::Format format,
                           gert::Shape& output, bool useDefaults = false,
                           const std::vector<int64_t>& strides = {1, 1, 2, 2},
                           const std::vector<int64_t>& pads = {0, 0, 1, 1, 1, 1, 0, 0})
{
    auto x = MakeShape({1, 32, 20, 30});
    auto w1 = MakeShape(filter1);
    auto w2 = MakeShape(filter2);
    auto placeholder = MakeShape({1});
    gert::InferShapeContextFaker faker;
    faker.SetOpType("FusedMultiConv2d")
        .NodeIoNum(9, 1)
        .IrInstanceNum({1, 2, 2, 2, 2}, {1})
        .InputShapes({&x, &w1, &w2, &placeholder, &placeholder, &placeholder, &placeholder, &placeholder, &placeholder})
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW);
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"bias_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>({false, false})},
        {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)}};
    if (!useDefaults) {
        attrs.emplace_back("strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides));
        attrs.emplace_back("pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads));
    }
    faker.NodeAttrs(attrs);
    for (int i = 0; i < 9; ++i) {
        const auto origin = i == 0 ? ge::FORMAT_NCHW : (i <= 2 ? format : ge::FORMAT_ND);
        faker.NodeInputTd(i, ge::DT_FLOAT16, origin, i == 1 || i == 2 ? ge::FORMAT_FRACTAL_Z : origin);
    }
    auto holder = faker.Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMultiConv2d");
    EXPECT_NE(impl, nullptr);
    if (impl == nullptr || impl->infer_shape == nullptr || context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto status = impl->infer_shape(context);
    if (status == ge::GRAPH_SUCCESS) {
        output = *context->GetOutputShape(0);
    }
    return status;
}

// 调用类型推导；input 为输入类型，dtype 为指定输出类型，useDefault 选择省略属性，output 接收结果。
ge::graphStatus InferDataType(ge::DataType input, int64_t dtype, bool useDefault, ge::DataType& output)
{
    gert::InferDataTypeContextFaker faker;
    faker.SetOpType("FusedMultiConv2d").NodeIoNum(9, 1).IrInstanceNum({1, 2, 2, 2, 2}, {1});
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"bias_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>({false, false})}};
    if (!useDefault) {
        attrs.emplace_back("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dtype));
    }
    std::vector<ge::DataType> types(9, input);
    std::vector<void*> pointers;
    for (auto& type : types) {
        pointers.push_back(&type);
    }
    faker.NodeAttrs(attrs).InputDataTypes(pointers).OutputDataTypes({&output});
    auto holder = faker.Build();
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMultiConv2d");
    EXPECT_NE(impl, nullptr);
    if (impl == nullptr || impl->infer_datatype == nullptr || context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto status = impl->infer_datatype(context);
    output = context->GetOutputDataType(0);
    return status;
}
} // namespace

// 验证三种逻辑权重格式与两层不同卷积核的形状推导。
TEST(FusedMultiConv2dInferShape, LogicalFilterFormatsAndMixedKernels)
{
    for (auto format : {ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_HWCN}) {
        const auto w1 = format == ge::FORMAT_NCHW ? std::vector<int64_t>{64, 32, 1, 3} :
                        format == ge::FORMAT_NHWC ? std::vector<int64_t>{64, 1, 3, 32} :
                                                    std::vector<int64_t>{1, 3, 32, 64};
        const auto w2 = format == ge::FORMAT_NCHW ? std::vector<int64_t>{2, 64, 3, 1} :
                        format == ge::FORMAT_NHWC ? std::vector<int64_t>{2, 3, 1, 64} :
                                                    std::vector<int64_t>{3, 1, 64, 2};
        gert::Shape output;
        ASSERT_EQ(InferShape(w1, w2, format, output), ge::GRAPH_SUCCESS);
        EXPECT_EQ(output, MakeShape({1, 2, 10, 15}));
    }
}

// 验证省略步长和补边时使用原型默认值。
TEST(FusedMultiConv2dInferShape, DefaultStridesAndPads)
{
    gert::Shape output;
    ASSERT_EQ(InferShape({64, 32, 3, 3}, {96, 64, 3, 3}, ge::FORMAT_NCHW, output, true), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output, MakeShape({1, 96, 16, 26}));
}

// 验证格式、维度和通道关系不合法的权重被拒绝。
TEST(FusedMultiConv2dInferShape, InvalidFilterMetadata)
{
    gert::Shape output;
    EXPECT_EQ(InferShape({64, 32, 1, 3}, {2, 64, 3, 1}, ge::FORMAT_ND, output), ge::GRAPH_FAILED);
    for (const auto& w1 : std::vector<std::vector<int64_t>>{{64, 32, 1}, {64, 32, 0, 3}, {64, 16, 1, 3}}) {
        EXPECT_EQ(InferShape(w1, {2, 64, 3, 1}, ge::FORMAT_NCHW, output), ge::GRAPH_FAILED);
    }
    EXPECT_EQ(InferShape({64, 32, 1, 3}, {2, 32, 3, 1}, ge::FORMAT_NCHW, output), ge::GRAPH_FAILED);
}

// 验证不等步长和非对称补边被拒绝。
TEST(FusedMultiConv2dInferShape, InvalidStridesAndAsymmetricPads)
{
    gert::Shape output;
    for (const auto& strides : std::vector<std::vector<int64_t>>{{1, 1}, {0, 0, 1, 1}, {1, 2, 1, 1}, {1, 1, 1, 2}}) {
        EXPECT_EQ(InferShape({64, 32, 3, 3}, {96, 64, 3, 3}, ge::FORMAT_NCHW, output, false, strides,
                             {0, 0, 0, 0, 0, 0, 0, 0}),
                  ge::GRAPH_FAILED);
    }
    for (size_t i = 0; i < 8; ++i) {
        std::vector<int64_t> pads(8, 1);
        pads[i] = 0;
        EXPECT_EQ(InferShape({64, 32, 3, 3}, {96, 64, 3, 3}, ge::FORMAT_NCHW, output, false, {1, 1, 1, 1}, pads),
                  ge::GRAPH_FAILED);
    }
}

// 验证缺省及显式输出类型，以及非法类型被拒绝。
TEST(FusedMultiConv2dInferDataType, DefaultAndExplicitOutput)
{
    ge::DataType output = ge::DT_UNDEFINED;
    for (auto input : {ge::DT_FLOAT16, ge::DT_INT8}) {
        ASSERT_EQ(InferDataType(input, -1, true, output), ge::GRAPH_SUCCESS);
        EXPECT_EQ(output, input);
        ASSERT_EQ(InferDataType(input, -1, false, output), ge::GRAPH_SUCCESS);
        EXPECT_EQ(output, input);
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_INT8}) {
            ASSERT_EQ(InferDataType(input, dtype, false, output), ge::GRAPH_SUCCESS);
            EXPECT_EQ(output, dtype);
        }
    }
    EXPECT_EQ(InferDataType(ge::DT_INT8, 0, false, output), ge::GRAPH_FAILED);
    EXPECT_EQ(InferDataType(ge::DT_FLOAT, -1, false, output), ge::GRAPH_FAILED);
}
