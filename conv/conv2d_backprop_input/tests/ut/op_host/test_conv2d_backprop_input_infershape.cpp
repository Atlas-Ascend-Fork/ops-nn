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
 * \file test_conv2d_backprop_input_infershape.cpp
 * \brief 覆盖迁移自1.0(nn_calculation_ops.cc Conv2DBackpropInputInfer)的显式校验：
 *        input_size为const且out_backprop为静态shape时，filter的H/W维度不支持-1/-2
 */

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#include <vector>

namespace {
constexpr size_t kInputSizeEleNum = 4;
} // namespace

class Conv2DBackpropInputRuntimeInferShape : public testing::Test {};

static gert::KernelRunContextHolder BuildConv2DBackpropInputContext(
    std::unique_ptr<uint8_t[]>& tensor_holder, bool with_const_input_size, const std::vector<int64_t>& input_size,
    gert::StorageShape& filter_shape, ge::Format filter_format, gert::StorageShape& dedy_shape,
    ge::Format y_format = ge::FORMAT_NCHW)
{
    gert::StorageShape output_shape = {{}, {}};
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, filter_format, filter_format)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeOutputTd(0, ge::DT_FLOAT16, y_format, y_format);
    if (with_const_input_size) {
        size_t total_size = 0;
        tensor_holder = gert::Tensor::CreateFollowing(kInputSizeEleNum, ge::DT_INT64, total_size);
        auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
        tensor->MutableStorageShape().AppendDim(kInputSizeEleNum);
        tensor->MutableOriginShape().AppendDim(kInputSizeEleNum);
        tensor->SetOriginFormat(ge::FORMAT_ND);
        tensor->SetStorageFormat(ge::FORMAT_ND);
        std::vector<int64_t> inputSizeValues(input_size);
        (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), inputSizeValues.data(),
                       inputSizeValues.size() * sizeof(int64_t));
        return faker.InputShapes({tensor, &filter_shape, &dedy_shape}).OutputShapes({&output_shape}).Build();
    }
    gert::StorageShape input_size_shape = {{kInputSizeEleNum}, {kInputSizeEleNum}};
    return faker.InputShapes({&input_size_shape, &filter_shape, &dedy_shape}).OutputShapes({&output_shape}).Build();
}

static ge::graphStatus RunConv2DBackpropInputInferShape(std::unique_ptr<uint8_t[]>& tensor_holder,
                                                        bool with_const_input_size,
                                                        const std::vector<int64_t>& input_size,
                                                        gert::StorageShape& filter_shape, ge::Format filter_format,
                                                        gert::StorageShape& dedy_shape)
{
    auto holder = BuildConv2DBackpropInputContext(tensor_holder, with_const_input_size, input_size, filter_shape,
                                                  filter_format, dedy_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput")->infer_shape;
    return infer_shape_func(holder.GetContext<gert::InferShapeContext>());
}

TEST_F(Conv2DBackpropInputRuntimeInferShape, constInputSizeStaticDedyFilterNegativeHFail)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput"), nullptr);
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, -1, 3}, {32, 64, -1, 3}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape, ge::FORMAT_NCHW,
                                               dedy_shape),
              ge::GRAPH_FAILED);
}

TEST_F(Conv2DBackpropInputRuntimeInferShape, constInputSizeStaticDedyFilterNegativeWFhwcnFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{-2, 3, 64, 32}, {-2, 3, 64, 32}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape, ge::FORMAT_HWCN,
                                               dedy_shape),
              ge::GRAPH_FAILED);
}

TEST_F(Conv2DBackpropInputRuntimeInferShape, constInputSizeStaticDedyFilterUnknownRankFail)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{-2}, {-2}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape, ge::FORMAT_NCHW,
                                               dedy_shape),
              ge::GRAPH_FAILED);
}

TEST_F(Conv2DBackpropInputRuntimeInferShape, constInputSizeStaticDedyFilterPositiveSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, 3, 3}, {32, 64, 3, 3}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    auto holder = BuildConv2DBackpropInputContext(tensor_holder, true, {2, 32, 32, 32}, filter_shape, ge::FORMAT_NCHW,
                                                  dedy_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)), "[2, 32, 32, 32]");
}

TEST_F(Conv2DBackpropInputRuntimeInferShape, constInputSizeDynamicDedyFilterNegativeSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, -1, 3}, {32, 64, -1, 3}};
    gert::StorageShape dedy_shape = {{-1, 64, 16, 16}, {-1, 64, 16, 16}};
    EXPECT_EQ(RunConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape, ge::FORMAT_NCHW,
                                               dedy_shape),
              ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DBackpropInputRuntimeInferShape, constInputSizeUnknownRankDedyFilterNegativeSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, -1, 3}, {32, 64, -1, 3}};
    gert::StorageShape dedy_shape = {{-2}, {-2}};
    EXPECT_EQ(RunConv2DBackpropInputInferShape(tensor_holder, true, {2, 32, 32, 32}, filter_shape, ge::FORMAT_NCHW,
                                               dedy_shape),
              ge::GRAPH_SUCCESS);
}

TEST_F(Conv2DBackpropInputRuntimeInferShape, dynamicInputSizeStaticDedyFilterNegativeSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, -1, 3}, {32, 64, -1, 3}};
    gert::StorageShape dedy_shape = {{2, 64, 16, 16}, {2, 64, 16, 16}};
    EXPECT_EQ(RunConv2DBackpropInputInferShape(tensor_holder, false, {2, 32, 32, 32}, filter_shape, ge::FORMAT_NCHW,
                                               dedy_shape),
              ge::GRAPH_SUCCESS);
}

// input_size const不可见(动态图)时，y的C维=filter的C维，其余维度保持-1交运行时推导
// 现场用例形态：filter=HWCN[3,3,128,256]静态、dedy动态、y=NHWC → 期望y=[-1,-1,-1,128]
TEST_F(Conv2DBackpropInputRuntimeInferShape, dynamicInputSizePartialInferHwcnNhwcSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{3, 3, 128, 256}, {3, 3, 128, 256}};
    gert::StorageShape dedy_shape = {{-1, -1, -1, 256}, {-1, -1, -1, 256}};
    auto holder = BuildConv2DBackpropInputContext(tensor_holder, false, {}, filter_shape, ge::FORMAT_HWCN, dedy_shape,
                                                  ge::FORMAT_NHWC);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[-1, -1, -1, 128]");
}

// NCHW布局同样补推：filter=HWCN语义NCHW[32,64,3,3]、y=NCHW → 期望y=[-1,64,-1,-1]
TEST_F(Conv2DBackpropInputRuntimeInferShape, dynamicInputSizePartialInferNchwSuccess)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, 64, 3, 3}, {32, 64, 3, 3}};
    gert::StorageShape dedy_shape = {{-1, 64, -1, -1}, {-1, 64, -1, -1}};
    auto holder = BuildConv2DBackpropInputContext(tensor_holder, false, {}, filter_shape, ge::FORMAT_NCHW, dedy_shape,
                                                  ge::FORMAT_NCHW);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[-1, 64, -1, -1]");
}

// filter的C维未知时不补推，y保持全-1
TEST_F(Conv2DBackpropInputRuntimeInferShape, dynamicInputSizeFilterCUnknownKeepAllUnknown)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{32, -1, 3, 3}, {32, -1, 3, 3}};
    gert::StorageShape dedy_shape = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
    auto holder = BuildConv2DBackpropInputContext(tensor_holder, false, {}, filter_shape, ge::FORMAT_NCHW, dedy_shape,
                                                  ge::FORMAT_NCHW);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[-1, -1, -1, -1]");
}

// filter为unknown rank([-2])时不补推，y保持全-1
TEST_F(Conv2DBackpropInputRuntimeInferShape, dynamicInputSizeFilterUnknownRankKeepAllUnknown)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape filter_shape = {{-2}, {-2}};
    gert::StorageShape dedy_shape = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
    auto holder = BuildConv2DBackpropInputContext(tensor_holder, false, {}, filter_shape, ge::FORMAT_NCHW, dedy_shape,
                                                  ge::FORMAT_NCHW);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[-1, -1, -1, -1]");
}

// InferShapeRange对齐1.0语义：N[0,-1]、H/W[1,-1]、C=filter的C维收敛、shape静态维保持[v,v]
TEST_F(Conv2DBackpropInputRuntimeInferShape, inferShapeRangeDynamicYSuccess)
{
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape_range, nullptr);

    gert::Shape filterMin({3, 3, 128, 256});
    gert::Shape filterMax({3, 3, 128, 256});
    gert::Range<gert::Shape> filterRange(&filterMin, &filterMax);
    gert::Shape inputSizeMin({4});
    gert::Shape inputSizeMax({4});
    gert::Range<gert::Shape> inputSizeRange(&inputSizeMin, &inputSizeMax);
    gert::Shape dedyMin({-1, -1, -1, 256});
    gert::Shape dedyMax({-1, -1, -1, 256});
    gert::Range<gert::Shape> dedyRange(&dedyMin, &dedyMax);
    gert::Shape yMin({-1, -1, -1, -1});
    gert::Shape yMax({-1, -1, -1, -1});
    gert::Range<gert::Shape> yRange(&yMin, &yMax);
    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_HWCN)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .InputShapeRanges({&inputSizeRange, &filterRange, &dedyRange})
                      .OutputShapeRanges({&yRange})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeRangeContext>();
    EXPECT_EQ(impl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShapeRange(0)->GetMin(), gert::Shape({0, 1, 1, 128}));
    EXPECT_EQ(*context->GetOutputShapeRange(0)->GetMax(), gert::Shape({-1, -1, -1, 128}));
}

// filter的C维未知时C维range保持-1，其余维仍按语义给下界
TEST_F(Conv2DBackpropInputRuntimeInferShape, inferShapeRangeFilterCUnknownKeepSuccess)
{
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv2DBackpropInput");
    ASSERT_NE(impl->infer_shape_range, nullptr);

    gert::Shape filterMin({3, 3, -1, 256});
    gert::Shape filterMax({3, 3, -1, 256});
    gert::Range<gert::Shape> filterRange(&filterMin, &filterMax);
    gert::Shape inputSizeMin({4});
    gert::Shape inputSizeMax({4});
    gert::Range<gert::Shape> inputSizeRange(&inputSizeMin, &inputSizeMax);
    gert::Shape dedyMin({-1, -1, -1, 256});
    gert::Shape dedyMax({-1, -1, -1, 256});
    gert::Range<gert::Shape> dedyRange(&dedyMin, &dedyMax);
    gert::Shape yMin;
    gert::Shape yMax;
    gert::Range<gert::Shape> yRange(&yMin, &yMax);
    auto holder = gert::InferShapeRangeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_HWCN)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .InputShapeRanges({&inputSizeRange, &filterRange, &dedyRange})
                      .OutputShapeRanges({&yRange})
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeRangeContext>();
    EXPECT_EQ(impl->infer_shape_range(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShapeRange(0)->GetMin(), gert::Shape({0, 1, 1, -1}));
    EXPECT_EQ(*context->GetOutputShapeRange(0)->GetMax(), gert::Shape({-1, -1, -1, -1}));
}
