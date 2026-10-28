/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "error_util.h"
#include "../../../op_graph/dynamic_quant_v2_proto.h"

class DynamicQuantV2 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DynamicQuantV2 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DynamicQuantV2 TearDown" << std::endl; }
};

TEST_F(DynamicQuantV2, DynamicQuantV2_infershape_case_0)
{
    ge::op::DynamicQuantV2 op;
    op.UpdateInputDesc("x", create_desc({1, 128, 1024}, ge::DT_FLOAT16));
    op.UpdateInputDesc("smooth_scales", create_desc({1024}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    auto output_y = op.GetOutputDesc(0);
    auto output_scale = op.GetOutputDesc(1);
    auto output_offset = op.GetOutputDesc(2);
    std::vector<int64_t> expected_y_shape = {1, 128, 1024};
    std::vector<int64_t> expected_scale_shape = {1, 128};
    std::vector<int64_t> expected_offset_shape = {1, 128};
    EXPECT_EQ(output_y.GetShape().GetDims(), expected_y_shape);
    EXPECT_EQ(output_scale.GetShape().GetDims(), expected_scale_shape);
    EXPECT_EQ(output_offset.GetShape().GetDims(), expected_offset_shape);
}

TEST_F(DynamicQuantV2, DynamicQuantV2_InferDtype_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_smooth_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_INT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        ge::DataType output_offset_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(2, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_x_ref, &input_smooth_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref, &output_offset_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_offset_ref);
    }
}

TEST_F(DynamicQuantV2, DynamicQuantV2_InferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_smooth_ref = ge::DT_FLOAT16;
        ge::DataType output_y_ref = ge::DT_INT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        ge::DataType output_offset_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(1)
                                  .NodeIoNum(2, 3)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                                  .InputDataTypes({&input_x_ref, &input_smooth_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref, &output_offset_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_x_ref);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_offset_ref);
    }
}

TEST_F(DynamicQuantV2, DynamicQuantV2_infershape_case_pertensor)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2")->infer_shape;

    if (infer_shape_func != nullptr) {
        gert::StorageShape x_shape = {{1, 128, 1024}, {1, 128, 1024}};
        gert::StorageShape smooth_shape = {{1024}, {1024}};
        gert::StorageShape y_shape = {{1, 128, 1024}, {1, 128, 1024}};
        gert::StorageShape scale_shape = {{1}, {1}};

        auto context_holder = gert::InferShapeContextFaker()
                                  .NodeIoNum(2, 3)
                                  .IrInstanceNum({1, 1})
                                  .InputShapes({&x_shape, &smooth_shape})
                                  .OutputShapes({&y_shape, &scale_shape, &scale_shape})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                              {"is_symmetrical", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>("pertensor")}})
                                  .Build();

        auto context = context_holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        auto shape1 = context->GetOutputShape(0);
        auto shape2 = context->GetOutputShape(1);
        auto shape3 = context->GetOutputShape(2);
        ASSERT_EQ(Shape2String(*shape1), "[1, 128, 1024]");
        ASSERT_EQ(Shape2String(*shape2), "[1]");
        ASSERT_EQ(Shape2String(*shape3), "[1]");
    }
}

TEST_F(DynamicQuantV2, DynamicQuantV2_infershape_case_perchannel)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2")->infer_shape;
    if (infer_shape_func != nullptr) {
        gert::StorageShape x_shape = {{1, 128, 1024}, {1, 128, 1024}};
        gert::StorageShape smooth_shape = {{128}, {128}};
        gert::StorageShape y_shape = {{1, 128, 1024}, {1, 128, 1024}};
        gert::StorageShape scale_shape = {{1, 1024}, {1, 1024}};

        auto context_holder = gert::InferShapeContextFaker()
                                  .NodeIoNum(2, 3)
                                  .IrInstanceNum({1, 1})
                                  .InputShapes({&x_shape, &smooth_shape})
                                  .OutputShapes({&y_shape, &scale_shape, &scale_shape})
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                              {"is_symmetrical", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                              {"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>("perchannel")}})
                                  .Build();

        auto context = context_holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        auto shape1 = context->GetOutputShape(0);
        auto shape2 = context->GetOutputShape(1);
        auto shape3 = context->GetOutputShape(2);
        ASSERT_EQ(Shape2String(*shape1), "[1, 128, 1024]");
        ASSERT_EQ(Shape2String(*shape2), "[1, 1024]");
        ASSERT_EQ(Shape2String(*shape3), "[1, 1024]");
    }
}
// ============================================================================
// Error code verification UT for E*9*** remediation (2026-10-22)
// Captures ErrorManager output via weak-symbol override; asserts no EZ9999.
// ============================================================================
namespace error_message {
struct CapturedError {
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};
static thread_local CapturedError g_last_error;
static thread_local int g_report_count = 0;
static thread_local std::vector<std::string> g_all_codes; // 记录用例全程上报过的全部错误码
__attribute__((weak)) void ResetCapture()
{
    g_last_error.code.clear();
    g_last_error.keys.clear();
    g_last_error.values.clear();
    g_report_count = 0;
    g_all_codes.clear();
}
__attribute__((weak)) std::string GetLastError() { return g_last_error.code; }
__attribute__((weak)) int GetReportCount() { return g_report_count; }
__attribute__((weak)) bool HasCode(const std::string& c)
{
    for (const auto& code : g_all_codes) {
        if (code == c) {
            return true;
        }
    }
    return false;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code, const std::vector<const char*>& key,
                                                     const std::vector<const char*>& value)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_last_error.keys.clear();
    g_last_error.values.clear();
    for (const auto& k : key)
        g_last_error.keys.push_back(k ? k : "");
    for (const auto& v : value)
        g_last_error.values.push_back(v ? v : "");
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportInnerErrMsg(const char* f, const char* fn, uint32_t l, const char* ec,
                                                const char* fmt, ...)
{
    g_last_error.code = ec ? ec : "";
    if (ec != nullptr) {
        g_all_codes.push_back(ec);
    }
    g_report_count++;
    return 0;
}
} // namespace error_message

// 验证弱符号截获通道在本二进制内生效: 若链接格局变化导致强定义压制弱定义, 该用例显式失败,
// 避免EZ9999守护用例静默失效
TEST_F(DynamicQuantV2, error_capture_channel_self_check)
{
    error_message::ResetCapture();
    const std::vector<const char*> selfKey = {"self_check"};
    const std::vector<const char*> selfValue = {"capture channel liveness"};
    error_message::ReportPredefinedErrMsg("EZ0000", selfKey, selfValue);
    EXPECT_TRUE(error_message::HasCode("EZ0000")) << "capture channel is dead: weak override suppressed";
    EXPECT_EQ(error_message::GetReportCount(), 1);
    error_message::ResetCapture();
}
#define EXPECT_ERROR_CODE(expected)                    \
    EXPECT_EQ(error_message::GetLastError(), expected) \
        << "Expected " << expected << " got [" << error_message::GetLastError() << "]"

// After each test, verify no EZ9999 internal code was captured
class Dqv2NoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        error_message::ResetCapture(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
    }
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        // 只守护本文件的测试套件, 框架层或其他算子用例的EZ9999不跨套件追责
        static const char* const guardedSuites[] = {"DynamicQuantV2"};
        bool guarded = false;
        for (const char* s : guardedSuites) {
            if (std::string(test_info.test_suite_name()) == s) {
                guarded = true;
                break;
            }
        }
        if (!guarded) {
            return;
        }
        if (error_message::HasCode("EZ9999")) {
            ADD_FAILURE() << "EZ9999 captured during test [" << test_info.name() << "] in " << test_info.file();
        }
    }
};
static const auto _inst_dqv2 = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new Dqv2NoEz9999Check());
    return 0;
}();

TEST_F(DynamicQuantV2, err_dst_type_unsupported_ez0026)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicQuantV2")->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_x_ref = ge::DT_FLOAT16;
    ge::DataType input_smooth_ref = ge::DT_FLOAT16;
    ge::DataType output_y_ref = ge::DT_INT8;
    ge::DataType output_scale_ref = ge::DT_FLOAT;
    ge::DataType output_offset_ref = ge::DT_FLOAT;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .IrInputNum(1)
                              .NodeIoNum(2, 3)
                              .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(11)}})
                              .InputDataTypes({&input_x_ref, &input_smooth_ref})
                              .OutputDataTypes({&output_y_ref, &output_scale_ref, &output_offset_ref})
                              .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();

    error_message::ResetCapture();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0026");
}
