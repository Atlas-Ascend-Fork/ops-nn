/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

struct GroupQuantCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

class GroupQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupQuantTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupQuantTiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tilingData)
{
    auto data = tilingData->GetData();
    string result;
    for (size_t i = 0; i < tilingData->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tilingData->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void InitPlatForm(fe::PlatFormInfos& platformInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    platformInfo.Init();
}

void GroupQuantTilingSimpleTest(const int64_t dimS, const int64_t dimE, const int64_t dimH, const ge::DataType xDtype,
                                const ge::DataType scaleDtype, const ge::DataType groupIndexDtype,
                                const ge::DataType offsetDtype, const ge::DataType yDtype, const int64_t attrDstType,
                                const bool hasOffset, const ge::graphStatus expectStatus,
                                const uint64_t expectTilingKey, const int64_t expectBlockDim,
                                const string expectTilingData)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

    std::initializer_list<int64_t> xShape = {dimS, dimH};
    std::initializer_list<int64_t> scaleShape = {dimE, dimH};
    std::initializer_list<int64_t> groupIndexShape = {
        dimE,
    };
    std::initializer_list<int64_t> offsetShape = {
        1,
    };
    std::initializer_list<int64_t> yShape = {dimS, dimH};
    gert::StorageShape xStorageShape = {xShape, xShape};
    gert::StorageShape scaleStorageShape = {scaleShape, scaleShape};
    gert::StorageShape groupIndexStorageShape = {groupIndexShape, groupIndexShape};
    gert::StorageShape offsetStorageShape = {offsetShape, offsetShape};
    gert::StorageShape outStorageShape = {yShape, yShape};

    GroupQuantCompileInfo compileInfo;
    std::string opType("GroupQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);

    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 1024 * 1024);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&xStorageShape, &scaleStorageShape, &groupIndexStorageShape, &offsetStorageShape})
                      .OutputShapes({&outStorageShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(attrDstType)},
                      })
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tilingFunc(tilingContext), expectStatus);

    auto blockDim = tilingContext->GetBlockDim();
    ASSERT_EQ(blockDim, expectBlockDim);

    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
    ASSERT_EQ(tilingDataResult, expectTilingData);
}

TEST_F(GroupQuantTiling, tiling_demo_case)
{
    int64_t dimS = 6;
    int64_t dimE = 4;
    int64_t dimH = 4;
    int64_t attrDstType = 2;
    bool hasOffset = true;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    uint64_t expectTilingKey = 0;
    int64_t expectBlockDim = 6;
    string expectTilingData = "6 4 4 1 6 6 1 1 ";
    GroupQuantTilingSimpleTest(dimS, dimE, dimH, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT8,
                               attrDstType, hasOffset, expectStatus, expectTilingKey, expectBlockDim, expectTilingData);
}

TEST_F(GroupQuantTiling, tiling_net_case_full_s5120_e32_h1024)
{
    int64_t dimS = 5120;
    int64_t dimE = 32;
    int64_t dimH = 1024;
    int64_t attrDstType = 2;
    bool hasOffset = true;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    uint64_t expectTilingKey = 0;
    int64_t expectBlockDim = 48;
    string expectTilingData = "5120 32 1024 1 48 32 107 106 ";
    GroupQuantTilingSimpleTest(dimS, dimE, dimH, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT8,
                               attrDstType, hasOffset, expectStatus, expectTilingKey, expectBlockDim, expectTilingData);
}

TEST_F(GroupQuantTiling, tiling_net_case_incremental_s80_e32_h1024)
{
    int64_t dimS = 80;
    int64_t dimE = 32;
    int64_t dimH = 1024;
    int64_t attrDstType = 2;
    bool hasOffset = true;
    const ge::graphStatus expectStatus = ge::GRAPH_SUCCESS;
    uint64_t expectTilingKey = 0;
    int64_t expectBlockDim = 48;
    string expectTilingData = "80 32 1024 1 48 32 2 1 ";
    GroupQuantTilingSimpleTest(dimS, dimE, dimH, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT8,
                               attrDstType, hasOffset, expectStatus, expectTilingKey, expectBlockDim, expectTilingData);
}

// ============================================================================
// Error code verification UT for E*9*** remediation (2026-10-21)
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
TEST_F(GroupQuantTiling, error_capture_channel_self_check)
{
    error_message::ResetCapture();
    const std::vector<const char*> selfKey = {"self_check"};
    const std::vector<const char*> selfValue = {"capture channel liveness"};
    error_message::ReportPredefinedErrMsg("EZ0000", selfKey, selfValue);
    EXPECT_TRUE(error_message::HasCode("EZ0000")) << "capture channel is dead: weak override suppressed";
    EXPECT_EQ(error_message::GetReportCount(), 1);
    error_message::ResetCapture();
}
// Global listener: after each test, verify no EZ9999 was captured
class GROUP_QUANTNoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        error_message::ResetCapture(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
    }
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        // 只守护本文件的测试套件, 框架层或其他算子用例的EZ9999不跨套件追责
        static const char* const guardedSuites[] = {"GroupQuantTiling"};
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
static const auto _inst_GROUP_QUANT = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new GROUP_QUANTNoEz9999Check());
    return 0;
}();
