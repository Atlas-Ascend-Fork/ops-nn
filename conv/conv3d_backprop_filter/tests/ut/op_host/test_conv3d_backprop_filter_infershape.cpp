/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv3d_backprop_filter_infershape.cpp
 * \brief 覆盖legacy Conv3DBackpropFilter geir通路的output_padding元素个数拦截：
 *        3D期望5个，少一位/多一位须在infershape阶段拦截
 */

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#include <string>
#include <vector>

namespace {
constexpr size_t kFilterSizeEleNum = 5;
} // namespace

class Conv3DBackpropFilterRuntimeInferShape : public testing::Test {};

static gert::KernelRunContextHolder BuildConv3DBackpropFilterContext(std::unique_ptr<uint8_t[]>& tensor_holder,
                                                                     const std::vector<int64_t>& output_padding,
                                                                     gert::StorageShape& fmap_shape,
                                                                     gert::StorageShape& out_backprop_shape)
{
    gert::StorageShape output_shape = {{}, {}};
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
        .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
        .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                    {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                    {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                    {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                    {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(output_padding)},
                    {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                    {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}});
    size_t total_size = 0;
    tensor_holder = gert::Tensor::CreateFollowing(kFilterSizeEleNum, ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(kFilterSizeEleNum);
    tensor->MutableOriginShape().AppendDim(kFilterSizeEleNum);
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    std::vector<int64_t> filterSizeValues = {256, 128, 3, 3, 3};
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), filterSizeValues.data(),
                   filterSizeValues.size() * sizeof(int64_t));
    return faker.InputShapes({&fmap_shape, tensor, &out_backprop_shape}).OutputShapes({&output_shape}).Build();
}

// geir通路的output_padding元素个数拦截：3D期望5个，合法5个放行
TEST_F(Conv3DBackpropFilterRuntimeInferShape, outputPaddingValidPass)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape fmap_shape = {{2, 128, 8, 8, 8}, {2, 128, 8, 8, 8}};
    gert::StorageShape out_backprop_shape = {{2, 256, 8, 8, 8}, {2, 256, 8, 8, 8}};
    auto holder = BuildConv3DBackpropFilterContext(tensor_holder, {0, 0, 0, 0, 0}, fmap_shape, out_backprop_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              "[256, 128, 3, 3, 3]");
}

// geir通路的output_padding元素个数拦截：3D期望5个，少一位(4个)须拦截
TEST_F(Conv3DBackpropFilterRuntimeInferShape, outputPaddingLessDimInterceptFailed)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape fmap_shape = {{2, 128, 8, 8, 8}, {2, 128, 8, 8, 8}};
    gert::StorageShape out_backprop_shape = {{2, 256, 8, 8, 8}, {2, 256, 8, 8, 8}};
    auto holder = BuildConv3DBackpropFilterContext(tensor_holder, {0, 0, 0, 0}, fmap_shape, out_backprop_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// geir通路的output_padding元素个数拦截：3D期望5个，多一位(6个)须拦截
TEST_F(Conv3DBackpropFilterRuntimeInferShape, outputPaddingMoreDimInterceptFailed)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape fmap_shape = {{2, 128, 8, 8, 8}, {2, 128, 8, 8, 8}};
    gert::StorageShape out_backprop_shape = {{2, 256, 8, 8, 8}, {2, 256, 8, 8, 8}};
    auto holder = BuildConv3DBackpropFilterContext(tensor_holder, {0, 0, 0, 0, 0, 0}, fmap_shape, out_backprop_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// geir布局下output_padding为空列表(0元素)同样属元素个数不符，须拦截
TEST_F(Conv3DBackpropFilterRuntimeInferShape, outputPaddingEmptyListInterceptFailed)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape fmap_shape = {{2, 128, 8, 8, 8}, {2, 128, 8, 8, 8}};
    gert::StorageShape out_backprop_shape = {{2, 256, 8, 8, 8}, {2, 256, 8, 8, 8}};
    auto holder = BuildConv3DBackpropFilterContext(tensor_holder, {}, fmap_shape, out_backprop_shape);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// aclnn L0直调布局兼容：L0的INFER_SHAPE属性链由调用方手工组装且不含output_padding
// (7属性: strides/pads/dilations/groups/data_format/padding/_op_impl_mode_enum，比geir布局8个少一)，
// 校验按属性总数识别布局后跳过，不得误读槽5的非列表内容
TEST_F(Conv3DBackpropFilterRuntimeInferShape, outputPaddingL0LayoutNotIntercepted)
{
    std::unique_ptr<uint8_t[]> tensor_holder;
    gert::StorageShape x_shape = {{2, 128, 8, 8, 8}, {2, 128, 8, 8, 8}};
    gert::StorageShape dedy_shape = {{2, 256, 8, 8, 8}, {2, 256, 8, 8, 8}};
    gert::StorageShape output_shape = {{}, {}};
    gert::InferShapeContextFaker faker;
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
        .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
        // L0调用方属性链：无output_padding，槽5为padding(字符串)
        .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
                    {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
                    {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                    {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                    {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("")},
                    {"_op_impl_mode_enum", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}});
    size_t total_size = 0;
    tensor_holder = gert::Tensor::CreateFollowing(kFilterSizeEleNum, ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(kFilterSizeEleNum);
    tensor->MutableOriginShape().AppendDim(kFilterSizeEleNum);
    tensor->SetOriginFormat(ge::FORMAT_ND);
    tensor->SetStorageFormat(ge::FORMAT_ND);
    std::vector<int64_t> filterSizeValues = {256, 128, 3, 3, 3};
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), filterSizeValues.data(),
                   filterSizeValues.size() * sizeof(int64_t));
    auto holder = faker.InputShapes({&x_shape, tensor, &dedy_shape}).OutputShapes({&output_shape}).Build();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DBackpropFilter")->infer_shape;
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
