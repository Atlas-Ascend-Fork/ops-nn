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
 * \file test_conv3d_transpose_v2_infershape.cpp
 * \brief
 */
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "base/err_mgr.h"

using ge::FORMAT_DHWCN;
using ge::FORMAT_NCDHW;
using ge::FORMAT_ND;
using ge::FORMAT_NDC1HWC0;
using ge::FORMAT_NDHWC;

/*
 * get debug string of vector
 * param[in] v vector
 * return vector's debug string
 */
template <typename T>
std::string DebugString(const std::vector<T>& v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    oss << "]";
    return oss.str();
}

struct Conv3DTransposeProtoTestParam {
    string case_name;
    std::initializer_list<int64_t> input_size_ori_shape;
    std::initializer_list<int64_t> x_ori_shape;
    std::initializer_list<int64_t> filter_ori_shape;
    std::initializer_list<int64_t> y_ori_shape;

    ge::Format input_size_ori_format;
    ge::Format x_ori_format;
    ge::Format filter_ori_format;
    ge::Format y_ori_format;

    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    int64_t groups;
    std::string data_format;
    std::vector<int64_t> output_padding;
    std::string padding;

    bool result;
};

class Conv3DTransposeV2ProtoTest : public testing::TestWithParam<Conv3DTransposeProtoTestParam> {};

TEST_P(Conv3DTransposeV2ProtoTest, general_cases)
{
    Conv3DTransposeProtoTestParam param = GetParam();
    std::cout << "run case " << param.case_name << std::endl;
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTransposeV2")->infer_shape;

    gert::StorageShape input_size_shape = {param.input_size_ori_shape, param.input_size_ori_shape};
    gert::StorageShape x_shape = {param.x_ori_shape, param.x_ori_shape};
    gert::StorageShape filter_shape = {param.filter_ori_shape, param.x_ori_shape};
    gert::StorageShape y_shape = {{}, {}};

    size_t total_size = 0;
    std::vector<int64_t> input_size(param.input_size_ori_shape);
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size.size(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(param.input_size_ori_format);
    tensor->SetStorageFormat(param.input_size_ori_format);

    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, param.input_size_ori_format, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, param.x_ori_format, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, param.filter_ori_format, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, param.y_ori_format, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(param.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(param.data_format)},
                                  {"output_padding",
                                   Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(param.output_padding)},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(param.padding)}})
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&y_shape})
                      .Build();

    if (param.result) {
        ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
        gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Ops::Base::ToString(*output), DebugString(std::vector<int64_t>(param.y_ori_shape)));
    } else {
        ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    }
}

// 负向用例：属性校验失败时断言 InferShape 返回 GRAPH_FAILED，且 ErrorManager 上报预期预置错误码与字段。
static void TestTransposeNegativeCase(const std::string& case_name,
                                      const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs,
                                      const std::string& expect_code, const std::string& expect_field,
                                      ge::Format x_format = FORMAT_NCDHW, ge::Format filter_format = FORMAT_NCDHW)
{
    std::cout << "run negative case " << case_name << std::endl;
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrErrorMessage(); // 读取一次以清空容器，避免其它用例遗留的首错干扰

    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("Conv3DTransposeV2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);

    // input_size 置 0，使输出全 0，从而进入 zero output 分支的属性校验路径
    std::initializer_list<int64_t> zero_input_size = {0, 0, 0, 0, 0};
    gert::StorageShape input_size_shape = {zero_input_size, zero_input_size};
    gert::StorageShape x_shape = {{32, 6, 3, 8, 224}, {32, 6, 3, 8, 224}};
    gert::StorageShape filter_shape = {{5, 7, 1, 24, 224}, {5, 7, 1, 24, 224}};
    gert::StorageShape y_shape = {{}, {}};

    size_t total_size = 0;
    std::vector<int64_t> input_size(zero_input_size);
    auto tensor_holder = gert::Tensor::CreateFollowing(input_size.size(), ge::DT_INT64, total_size);
    auto tensor = reinterpret_cast<gert::Tensor*>(tensor_holder.get());
    tensor->MutableStorageShape().AppendDim(input_size_shape.MutableStorageShape().GetDimNum());
    tensor->MutableOriginShape().AppendDim(input_size_shape.MutableOriginShape().GetDimNum());
    tensor->SetOriginFormat(FORMAT_NCDHW);
    tensor->SetStorageFormat(FORMAT_NCDHW);
    (void)memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), input_size.data(),
                   input_size.size() * sizeof(int64_t));

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, x_format, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT16, filter_format, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT16, FORMAT_NCDHW, ge::Format::FORMAT_RESERVED)
                      .NodeAttrs(attrs)
                      .InputShapes({tensor, &x_shape, &filter_shape})
                      .OutputShapes({&y_shape})
                      .Build();

    ASSERT_EQ(impl->infer_shape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED) << case_name;

    auto errMsg = error_message::GetErrMgrErrorMessage();
    ASSERT_NE(errMsg, nullptr) << case_name;
    std::string msg = errMsg.get();
    std::cout << "[" << case_name << "] error msg: " << msg << std::endl;
    EXPECT_TRUE(msg.find(expect_code) != std::string::npos)
        << "[" << case_name << "] expect code " << expect_code << " in: " << msg;
    EXPECT_TRUE(msg.find(expect_field) != std::string::npos)
        << "[" << case_name << "] expect field " << expect_field << " in: " << msg;
}

TEST_F(Conv3DTransposeV2ProtoTest, negative_attr_error_codes)
{
    std::vector<int64_t> validStrides = {1, 1, 1, 1, 1};
    std::vector<int64_t> validPads = {0, 0, 0};
    std::vector<int64_t> validDilations = {1, 1, 1, 1, 1};
    std::vector<int64_t> validOutputPadding = {0, 0, 0, 0, 0};

    // EZ0025: strides 元素个数不为 5
    TestTransposeNegativeCase(
        "err_EZ0025_strides_listsize",
        {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{1, 1, 1, 1})},
         {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validPads)},
         {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validDilations)},
         {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
         {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
         {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validOutputPadding)}},
        "EZ0025", "strides");

    // EZ0025: dilations 元素个数不为 5
    TestTransposeNegativeCase(
        "err_EZ0025_dilations_listsize",
        {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validStrides)},
         {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validPads)},
         {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{1, 1, 1})},
         {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
         {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
         {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validOutputPadding)}},
        "EZ0025", "dilations");

    // EZ0025: pads 元素个数不在 {1, 3, 6}
    TestTransposeNegativeCase(
        "err_EZ0025_pads_listsize",
        {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validStrides)},
         {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{0, 0})},
         {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validDilations)},
         {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
         {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
         {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validOutputPadding)}},
        "EZ0025", "pads");

    // EZ0025: output_padding 元素个数不为 5
    TestTransposeNegativeCase(
        "err_EZ0025_output_padding_listsize",
        {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validStrides)},
         {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validPads)},
         {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validDilations)},
         {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
         {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
         {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{0, 0, 0, 0})}},
        "EZ0025", "output_padding");

    // EZ0004: 必选属性 groups 未下发（attr 索引越界）
    TestTransposeNegativeCase("err_EZ0004_groups_null",
                              {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validStrides)},
                               {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validPads)},
                               {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validDilations)}},
                              "EZ0004", "groups");

    // EZ0004: 必选属性 output_padding 未下发（attr 索引越界）
    TestTransposeNegativeCase("err_EZ0004_output_padding_null",
                              {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validStrides)},
                               {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validPads)},
                               {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validDilations)},
                               {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}},
                              "EZ0004", "output_padding");

    // EZ0004: 必选属性 dilations 未下发（attr 索引越界）
    TestTransposeNegativeCase("err_EZ0004_dilations_null",
                              {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validStrides)},
                               {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validPads)}},
                              "EZ0004", "dilations");

    // EZ0017: x 的 format 不在支持范围（仅支持 NCDHW / NDHWC）
    TestTransposeNegativeCase(
        "err_EZ0017_x_format",
        {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validStrides)},
         {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validPads)},
         {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validDilations)},
         {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
         {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validOutputPadding)}},
        "EZ0017", "x", ge::FORMAT_ND, FORMAT_NCDHW);

    // EZ0017: filter 的 format 不在支持范围（仅支持 NCDHW / NDHWC / DHWCN）
    TestTransposeNegativeCase(
        "err_EZ0017_filter_format",
        {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validStrides)},
         {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validPads)},
         {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validDilations)},
         {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
         {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validOutputPadding)}},
        "EZ0017", "filter", FORMAT_NCDHW, ge::FORMAT_ND);

    // EZ0026: pads 含负值（非 SAME 场景）
    TestTransposeNegativeCase(
        "err_EZ0026_pads_negative",
        {{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validStrides)},
         {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, 0, 0})},
         {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validDilations)},
         {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
         {"output_padding", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(validOutputPadding)}},
        "EZ0026", "pads");
}

static Conv3DTransposeProtoTestParam general_cases_params[] = {
    {"valid_input_sizes",
     {32, 16, 8, 8, 24},
     {32, 6, 3, 8, 224},
     {5, 7, 1, 24, 224},
     {32, 16, 8, 8, 24},
     FORMAT_NDHWC,
     FORMAT_DHWCN,
     FORMAT_NDHWC,
     FORMAT_NDHWC,
     {1, 3, 3, 1, 1},
     {-1, -1, -1, -1, -1, -1},
     {1, 1, 1, 1, 1},
     1,
     "NDHWC",
     {0, 0, 0, 0, 0},
     "SAME",
     true},

    {"invalid_input_sizes_NCDHW_SAME_4",
     {0, 0, 0, 0, 0},
     {8, 512, 24, 13, 7},
     {512, 28, 3, 5, 2},
     {8, 112, 48, 13, 13},
     FORMAT_NCDHW,
     FORMAT_NCDHW,
     FORMAT_NCDHW,
     FORMAT_NCDHW,
     {1, 1, 2, 1, 2},
     {-1, -1, -1, -1, -1, -1},
     {1, 1, 1, 1, 1},
     4,
     "NCDHW",
     {0, 0, 0, 0, 0},
     "SAME",
     true},

    {"invalid_input_sizes_NDHWC_SAME_1",
     {0, 0, 0, 0, 0},
     {9, 8, 7, 7, 128},
     {5, 1, 1, 22, 128},
     {9, 8, 13, 13, 22},
     FORMAT_NDHWC,
     FORMAT_NDHWC,
     FORMAT_DHWCN,
     FORMAT_NDHWC,
     {1, 1, 2, 2, 1},
     {-1, -1, -1, -1, -1, -1},
     {1, 1, 1, 1, 1},
     1,
     "NDHWC",
     {0, 0, 0, 0, 0},
     "SAME",
     true},

    {"2d_extend_3d_NCDHW_SAME_4_static",
     {9, 13, 13, 22},
     {9, 8, 7, 7, 128},
     {5, 1, 1, 22, 128},
     {9, 13, 1, 13, 22},
     FORMAT_NCDHW,
     FORMAT_NCDHW,
     FORMAT_NCDHW,
     FORMAT_NCDHW,
     {1, 1, 2, 2, 1},
     {-1, -1, -1, -1, -1, -1},
     {1, 1, 1, 1, 1},
     1,
     "NDHWC",
     {0, 0, 0, 0, 0},
     "SAME",
     true},
};

INSTANTIATE_TEST_CASE_P(Conv3DTransposeV2, Conv3DTransposeV2ProtoTest, testing::ValuesIn(general_cases_params));
