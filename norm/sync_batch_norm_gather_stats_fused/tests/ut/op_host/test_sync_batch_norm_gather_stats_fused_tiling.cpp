/**
 * Copyright (256) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "any_value.h"

using namespace std;
using namespace Ops::NN;

struct SyncBatchNormGatherStatsFusedCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t blockSize = 0;
    int64_t vlFp32 = 0;
    bool isRegBase = false;
};

class SyncBatchNormGatherStatsFusedTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SyncBatchNormGatherStatsFusedTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SyncBatchNormGatherStatsFusedTiling TearDown" << std::endl; }
};

TEST_F(SyncBatchNormGatherStatsFusedTiling, test_case_101_fp32)
{
    // 输入shape定义
    gert::StorageShape mean_shape = {{4, 256}, {4, 256}};   // mean: [4, 256]
    gert::StorageShape invstd_shape = {{4, 256}, {4, 256}}; // invstd: [4, 256]
    gert::StorageShape counts_shape = {{4}, {4}};           // counts: [4]
    gert::StorageShape runningMean_shape = {{256}, {256}};
    gert::StorageShape runningVar_shape = {{256}, {256}}; // runningVar: [256]

    // 输出shape定义
    gert::StorageShape meanAllOut_shape = {{256}, {256}};    // meanAllOut: [256]
    gert::StorageShape invstdAllOut_shape = {{256}, {256}};  // invstdAllOut: [256]
    gert::StorageShape runningVarOut_shape = {{256}, {256}}; // runningVarOut: [256]
    gert::StorageShape runningMeanOut_shape = {{256}, {256}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48}
                         })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    SyncBatchNormGatherStatsFusedCompileInfo compile_info;

    std::string op_type("SyncBatchNormGatherStatsFused");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("SyncBatchNormGatherStatsFused")
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&mean_shape, &invstd_shape, &counts_shape, &runningMean_shape, &runningVar_shape})
                      .OutputShapes(
                          {&meanAllOut_shape, &invstdAllOut_shape, &runningMeanOut_shape, &runningVarOut_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)  // mean
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)  // invstd
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)  // counts (always FLOAT)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)  // runningMean
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)  // runningVar
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // meanAllOut
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // invstdAllOut
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // runningMeanOut
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // runningVarOut
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(0.1f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-5f)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 101);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
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
TEST_F(SyncBatchNormGatherStatsFusedTiling, error_capture_channel_self_check)
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
class SbngsNoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        error_message::ResetCapture(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
    }
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        // 只守护本文件的测试套件, 框架层或其他算子用例的EZ9999不跨套件追责
        static const char* const guardedSuites[] = {"SyncBatchNormGatherStatsFusedTiling"};
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
static const auto _inst_sbngs = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new SbngsNoEz9999Check());
    return 0;
}();

// 错误码用例公共入口: 复用原用例的 faker 搭建, 允许覆盖 mean 形状/数据类型、attr 值与 coreNum
static ge::graphStatus RunSbngsTilingCase(gert::StorageShape& meanShape, ge::DataType meanDtype, float momentum,
                                          float eps, uint64_t coreNum)
{
    gert::StorageShape invstd_shape = {{4, 256}, {4, 256}};
    gert::StorageShape counts_shape = {{4}, {4}};
    gert::StorageShape runningMean_shape = {{256}, {256}};
    gert::StorageShape runningVar_shape = {{256}, {256}};
    gert::StorageShape meanAllOut_shape = {{256}, {256}};
    gert::StorageShape invstdAllOut_shape = {{256}, {256}};
    gert::StorageShape runningMeanOut_shape = {{256}, {256}};
    gert::StorageShape runningVarOut_shape = {{256}, {256}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48}
                         })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    SyncBatchNormGatherStatsFusedCompileInfo compile_info;

    std::string op_type("SyncBatchNormGatherStatsFused");
    auto* op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    EXPECT_NE(op_impl, nullptr);
    if (op_impl == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    auto* parse_ctx = kernel_holder.GetContext<gert::TilingParseContext>();
    EXPECT_TRUE(parse_ctx->GetPlatformInfo()->Init());
    parse_ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(op_impl->tiling_parse(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    compile_info.coreNum = coreNum; // 覆盖 parse 结果以构造平台异常场景

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType("SyncBatchNormGatherStatsFused")
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&meanShape, &invstd_shape, &counts_shape, &runningMean_shape, &runningVar_shape})
                      .OutputShapes(
                          {&meanAllOut_shape, &invstdAllOut_shape, &runningMeanOut_shape, &runningVarOut_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, meanDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, meanDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, meanDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, meanDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, meanDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, meanDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, meanDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, meanDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(momentum)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    return op_impl->tiling(tiling_context);
}

TEST_F(SyncBatchNormGatherStatsFusedTiling, err_mean_dim_not_2_ez0013)
{
    gert::StorageShape mean = {{4, 256, 3}, {4, 256, 3}};
    error_message::ResetCapture();
    EXPECT_EQ(RunSbngsTilingCase(mean, ge::DT_FLOAT, 0.1f, 1e-5f, 48), ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0013");
}

TEST_F(SyncBatchNormGatherStatsFusedTiling, err_mean_dtype_unsupported_ez0019)
{
    gert::StorageShape mean = {{4, 256}, {4, 256}};
    error_message::ResetCapture();
    EXPECT_EQ(RunSbngsTilingCase(mean, ge::DT_DOUBLE, 0.1f, 1e-5f, 48), ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0019");
}

TEST_F(SyncBatchNormGatherStatsFusedTiling, err_momentum_out_of_range_ez0026)
{
    gert::StorageShape mean = {{4, 256}, {4, 256}};
    error_message::ResetCapture();
    EXPECT_EQ(RunSbngsTilingCase(mean, ge::DT_FLOAT, 2.0f, 1e-5f, 48), ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0026");
}

TEST_F(SyncBatchNormGatherStatsFusedTiling, err_eps_not_positive_ez0026)
{
    gert::StorageShape mean = {{4, 256}, {4, 256}};
    error_message::ResetCapture();
    EXPECT_EQ(RunSbngsTilingCase(mean, ge::DT_FLOAT, 0.1f, -1.0f, 48), ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0026");
}

TEST_F(SyncBatchNormGatherStatsFusedTiling, err_core_num_zero_ez1008)
{
    gert::StorageShape mean = {{4, 256}, {4, 256}};
    error_message::ResetCapture();
    EXPECT_EQ(RunSbngsTilingCase(mean, ge::DT_FLOAT, 0.1f, 1e-5f, 0), ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ1008");
}
