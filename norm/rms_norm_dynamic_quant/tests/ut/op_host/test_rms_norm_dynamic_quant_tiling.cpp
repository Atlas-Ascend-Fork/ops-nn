/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../op_host/rms_norm_dynamic_quant_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
const std::string OP_TYPE = "RmsNormDynamicQuant";
constexpr int kInputNum = 4;  // x, gamma, smooth_scales, beta
constexpr int kOutputNum = 2; // y, scale

const std::string COMPILE_INFO_910B = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 40
      }
    })";
} // namespace

class RmsNormDynamicQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmsNormDynamicQuantTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RmsNormDynamicQuantTilingTest TearDown" << std::endl; }
};

// 通用 RunTiling
//   irInstanceNum: 4 个输入的实例数，可选输入不存在填 0 (如 {1,1,0,0})
//   inputShapes:   只填存在的输入 (如 {&x, &gamma})
//   NodeInputTd 全量调用 4 次，faker 内部根据 IrInstanceNum 自动处理多余调用
static ge::graphStatus RunTiling(const std::string& compileInfoJson, const vector<uint32_t>& irInstanceNum,
                                 ge::DataType inDtype, const vector<ge::DataType>& outputDtypes,
                                 const vector<gert::StorageShape*>& inputShapes,
                                 const vector<gert::StorageShape*>& outputShapes,
                                 const vector<pair<string, Ops::NN::AnyValue>>& attrs,
                                 uint32_t* outputTilingKey = nullptr)
{
    map<string, string> socInfos, aicoreSpec, intrinsics;
    GetPlatFormInfos(compileInfoJson.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::RmsNormDynamicQuantCompileInfo compileInfo;

    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr)
        return ge::GRAPH_FAILED;
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoJson.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    auto* parseCtx = kernelHolder.GetContext<gert::TilingParseContext>();
    EXPECT_TRUE(parseCtx->GetPlatformInfo()->Init());
    parseCtx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseCtx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    EXPECT_NE(tilingData, nullptr);

    gert::TilingContextFaker faker;
    faker.NodeIoNum(kInputNum, kOutputNum)
        .IrInstanceNum(irInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, outputDtypes[0], ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, outputDtypes[1], ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs(attrs)
        .TilingData(tilingData.get())
        .Workspace(wsSize);

    auto holder = faker.Build();
    auto* tilingCtx = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tilingCtx->GetPlatformInfo(), nullptr);

    ge::graphStatus status = tilingFunc(tilingCtx);
    if (outputTilingKey != nullptr)
        *outputTilingKey = tilingCtx->GetTilingKey();
    return status;
}

// ========================================================================
// {1,1,1,1}: 全部输入存在
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, normal_fp16_all_exist)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape smooth = {{16}, {16}};
    gert::StorageShape beta = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 1}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth, &beta},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u);
}

// ========================================================================
// {1,1,0,0}: smooth=None, beta=None
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, opt_no_smooth_no_beta)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 0, 0}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma}, {&y, &scale},
                        {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u);
}

// ========================================================================
// {1,1,1,0}: smooth=Exist, beta=None
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, opt_has_smooth_no_beta)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape smooth = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 0}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u);
}

// ========================================================================
// {1,1,1,1}: BF16 dtype coverage
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, normal_bf16_all_exist)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape smooth = {{16}, {16}};
    gert::StorageShape beta = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 1}, DT_BF16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth, &beta},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 1u);
}

// ========================================================================
// Slice-D tiling: 大列场景触发 SliceD 路径
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, slice_d_large_col)
{
    gert::StorageShape x = {{1, 1, 30000}, {1, 1, 30000}};
    gert::StorageShape gamma = {{30000}, {30000}};
    gert::StorageShape smooth = {{30000}, {30000}};
    gert::StorageShape beta = {{30000}, {30000}};
    gert::StorageShape y = {{1, 1, 30000}, {1, 1, 30000}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    uint32_t key = 0;
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 1}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth, &beta},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}, &key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 3u);
}

// ========================================================================
// Error case: gamma 最后一维与 x 不匹配
// ========================================================================
TEST_F(RmsNormDynamicQuantTilingTest, error_gamma_lastdim_mismatch)
{
    gert::StorageShape x = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma = {{2000}, {2000}};
    gert::StorageShape smooth = {{2000}, {2000}};
    gert::StorageShape beta = {{2000}, {2000}};
    gert::StorageShape y = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape scale = {{24, 1}, {24, 1}};
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 1}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth, &beta},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}),
              ge::GRAPH_FAILED);
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
TEST_F(RmsNormDynamicQuantTilingTest, error_capture_channel_self_check)
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
class RmsNormDynamicQuantNoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        error_message::ResetCapture(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
    }
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        // 只守护本文件的测试套件, 框架层或其他算子用例的EZ9999不跨套件追责
        static const char* const guardedSuites[] = {"RmsNormDynamicQuantTilingTest"};
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
static const auto _inst_rms_norm_dynamic_quant = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new RmsNormDynamicQuantNoEz9999Check());
    return 0;
}();

namespace {
// UB_SIZE=0 平台信息, 用于触发 ValidateBaseParameters 中 EZ1008 (ubSize<=0) 分支
const std::string COMPILE_INFO_UB_ZERO = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 0, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 40
      }
    })";
} // namespace

TEST_F(RmsNormDynamicQuantTilingTest, err_dst_type_invalid_ez0024)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    error_message::ResetCapture();
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 0, 0}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma}, {&y, &scale},
                        {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                         {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(3)}}),
              ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0024");
}

TEST_F(RmsNormDynamicQuantTilingTest, err_epsilon_invalid_ez0024)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    error_message::ResetCapture();
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 0, 0}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma}, {&y, &scale},
                        {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(-1.0)},
                         {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}}),
              ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0024");
}

TEST_F(RmsNormDynamicQuantTilingTest, err_x_dtype_unsupported_ez0019)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    error_message::ResetCapture();
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 0, 0}, DT_FLOAT, {DT_INT8, DT_FLOAT}, {&x, &gamma}, {&y, &scale},
                        {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}),
              ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0019");
}

TEST_F(RmsNormDynamicQuantTilingTest, err_ub_size_zero_ez1008)
{
    gert::StorageShape x = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma = {{16}, {16}};
    gert::StorageShape y = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape scale = {{1, 1}, {1, 1}};
    error_message::ResetCapture();
    EXPECT_EQ(RunTiling(COMPILE_INFO_UB_ZERO, {1, 1, 0, 0}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma}, {&y, &scale},
                        {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}),
              ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ1008");
}

TEST_F(RmsNormDynamicQuantTilingTest, err_gamma_x_mismatch_ez0010)
{
    gert::StorageShape x = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma = {{2000}, {2000}};
    gert::StorageShape smooth = {{2000}, {2000}};
    gert::StorageShape beta = {{2000}, {2000}};
    gert::StorageShape y = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape scale = {{24, 1}, {24, 1}};
    error_message::ResetCapture();
    EXPECT_EQ(RunTiling(COMPILE_INFO_910B, {1, 1, 1, 1}, DT_FLOAT16, {DT_INT8, DT_FLOAT}, {&x, &gamma, &smooth, &beta},
                        {&y, &scale}, {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}}),
              ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0010");
}
