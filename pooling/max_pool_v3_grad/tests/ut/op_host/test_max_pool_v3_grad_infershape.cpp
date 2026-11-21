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
 * \file test_max_pool_v3_grad_infershape.cpp
 * \brief InferShape UT for max_pool_v3_grad operator (nn 仓 InferShapeContextFaker 框架, 4D NCHW)
 *
 * 测试场景 (4D NCHW):
 *   - float32 非重叠: orig_input (1,1,4,4) -> out_grad shape = (1,1,4,4)
 *   - float16 重叠: orig_input (1,1,5,5) -> out_grad shape = (1,1,5,5)
 */

#include <iostream>
#include <sstream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace {
template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

class MaxPoolV3GradInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPoolV3GradInfershapeTest SetUp" << std::endl;
        fe::PlatformInfo platformInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
    static void TearDownTestCase() { std::cout << "MaxPoolV3GradInfershapeTest TearDown" << std::endl; }
};

void RunInfershape(const std::initializer_list<int64_t>& origInputShape,
                   const std::initializer_list<int64_t>& origOutputShape,
                   const std::initializer_list<int64_t>& gradShape, ge::DataType dt, const std::vector<int64_t>& ksize,
                   const std::vector<int64_t>& strides, const std::string& paddingMode,
                   const std::vector<int64_t>& pads, const std::string& dataFormat, bool globalPooling, bool ceilMode,
                   ge::graphStatus expectResult, const std::string& expectShapeStr = "")
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3Grad")->infer_shape;
    gert::StorageShape origInputS = {origInputShape, origInputShape};
    gert::StorageShape origOutputS = {origOutputShape, origOutputShape};
    gert::StorageShape gradS = {gradShape, gradShape};
    gert::StorageShape outGradS = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_RESERVED)
                      .NodeInputTd(1, dt, ge::FORMAT_ND, ge::FORMAT_RESERVED)
                      .NodeInputTd(2, dt, ge::FORMAT_ND, ge::FORMAT_RESERVED)
                      .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"padding_mode", Ops::NN::AnyValue::CreateFrom<std::string>(paddingMode)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)},
                                  {"global_pooling", Ops::NN::AnyValue::CreateFrom<bool>(globalPooling)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceilMode)}})
                      .InputShapes({&origInputS, &origOutputS, &gradS})
                      .OutputShapes({&outGradS})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), expectResult);
    if (!expectShapeStr.empty()) {
        gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
        ASSERT_EQ(Shape2String(*output), expectShapeStr);
    }
}
} // namespace

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fp32_non_overlap)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_SUCCESS, "[1, 1, 4, 4]");
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fp16_overlap)
{
    RunInfershape({1, 1, 5, 5}, {1, 1, 5, 5}, {1, 1, 5, 5}, ge::DT_FLOAT16, {1, 1, 3, 3}, {1, 1, 1, 1}, "CALCULATED",
                  {1, 1, 1, 1}, "NCHW", false, false, ge::GRAPH_SUCCESS, "[1, 1, 5, 5]");
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_dtype_mismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3Grad")->infer_shape;
    gert::StorageShape origInputS = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape origOutputS = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape gradS = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape outGradS = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_RESERVED)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_RESERVED)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_RESERVED)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})},
                                  {"padding_mode", Ops::NN::AnyValue::CreateFrom<std::string>("CALCULATED")},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0})},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                  {"global_pooling", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&origInputS, &origOutputS, &gradS})
                      .OutputShapes({&outGradS})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_global_pooling_true)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 1, 1}, {1, 1, 1, 1}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", true, false, ge::GRAPH_SUCCESS, "[1, 1, 4, 4]");
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_non_4d_shape)
{
    RunInfershape({2, 3, 4}, {2, 3, 2}, {2, 3, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED", {0, 0, 0, 0},
                  "NCHW", false, false, ge::GRAPH_FAILED);
}

// ======================== Unknown Rank (-2) ========================
TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_unknown_rank_orig_input)
{
    RunInfershape({-2}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_SUCCESS, "[-2]");
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_unknown_rank_grad)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {-2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_SUCCESS, "[1, 1, 4, 4]");
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_unknown_rank_orig_output)
{
    RunInfershape({1, 1, 4, 4}, {-2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_SUCCESS, "[1, 1, 4, 4]");
}

// ======================== Unknown Dim (-1) ========================
TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_unknown_dim_grad_skip_compare)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, -1, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_SUCCESS, "[1, 1, 4, 4]");
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_unknown_dim_orig_output_skip_compare)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, -1, 2}, {1, 1, -1, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_SUCCESS, "[1, 1, 4, 4]");
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_unknown_dim_nc_skip_compare)
{
    RunInfershape({-1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_SUCCESS, "[-1, 1, 4, 4]");
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_unknown_dim_c_skip_compare)
{
    RunInfershape({1, -1, 4, 4}, {1, 2, 2, 2}, {1, 2, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_SUCCESS, "[1, -1, 4, 4]");
}

namespace {
void RunInferDataType(ge::DataType inputDtype, ge::DataType outputDtype, ge::DataType gradDtype,
                      ge::graphStatus expectResult, ge::DataType expectOutputDtype = ge::DT_UNDEFINED)
{
    ge::DataType resultDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&inputDtype, &outputDtype, &gradDtype})
                      .OutputDataTypes({&resultDtype})
                      .Build();
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPoolV3Grad")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), expectResult);
    if (expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(context->GetOutputDataType(0), expectOutputDtype);
    }
}
} // namespace

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_nhwc_shape)
{
    RunInfershape({1, 4, 4, 2}, {1, 2, 2, 2}, {1, 2, 2, 2}, ge::DT_FLOAT, {1, 2, 2, 1}, {1, 2, 2, 1}, "CALCULATED",
                  {0, 0, 0, 0}, "NHWC", false, false, ge::GRAPH_SUCCESS, "[1, 4, 4, 2]");
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_3d_orig_output)
{
    RunInfershape({1, 1, 4, 4}, {1, 2, 2}, {1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_3d_grad)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_illegal_dim)
{
    RunInfershape({1, 1, -3, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_nc_mismatch)
{
    RunInfershape({2, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_grad_output_mismatch)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 3}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_invalid_data_format)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCDHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_invalid_padding_mode)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "INVALID",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_invalid_stride)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {2, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_invalid_ksize)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {2, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_pads_negative)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {-1, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_pads_ge_ksize)
{
    RunInfershape({1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {2, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_known_rank_with_unknown_rank_dim)
{
    // -2 只能作为 unknown rank 的占位（[-2]），不能出现在已知 4D shape 中。
    RunInfershape({-2, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_unknown_rank_orig_output_grad_nc_mismatch)
{
    // 对齐 canndev VerifyMaxPoolV3GradTensor：orig_output 为 unknown rank 时仍需校验 input/grad 的 N/C。
    RunInfershape({1, 2, 4, 4}, {-2}, {2, 2, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_unknown_output_n_input_grad_mismatch)
{
    // orig_output N 为 -1 时，input 与 grad 已知的 N 仍需一致。
    RunInfershape({1, 2, 4, 4}, {-1, 2, 2, 2}, {2, 2, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV3GradInfershapeTest, max_pool_v3_grad_fail_unknown_output_c_input_grad_mismatch)
{
    // orig_output C 为 -1 时，input 与 grad 已知的 C 仍需一致。
    RunInfershape({1, 2, 4, 4}, {1, -1, 2, 2}, {1, 3, 2, 2}, ge::DT_FLOAT, {1, 1, 2, 2}, {1, 1, 2, 2}, "CALCULATED",
                  {0, 0, 0, 0}, "NCHW", false, false, ge::GRAPH_FAILED);
}
