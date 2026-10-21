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
 * \file test_apply_ftrl_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "../../../../op_host/apply_ftrl_tiling.h"
#include "ut_op_util.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class TestApplyFtrlTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TestApplyFtrlTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TestApplyFtrlTiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                         map<string, string>& socVersion)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false,
                           "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true,
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    // platform info
    platFormInfo.Init();
}

struct ApplyFtrlUtCompileInfo {};

static void DoApplyFtrlTilingCase(std::initializer_list<int64_t>& inputShape, ge::DataType inputDtype,
                                  ge::Format inputFormat, std::string& expectStr)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("ApplyFtrl");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape tensorShape = {inputShape, inputShape};
    gert::StorageShape oneShape = {{1}, {1}};

    ApplyFtrlUtCompileInfo compileInfo;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(8, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&tensorShape, &tensorShape, &tensorShape, &tensorShape, &oneShape, &oneShape,
                                    &oneShape, &oneShape})
                      .OutputShapes({&tensorShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(1, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(2, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(3, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(4, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(5, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(6, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(7, inputDtype, inputFormat, inputFormat)
                      .NodeOutputTd(0, inputDtype, inputFormat, inputFormat)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    EXPECT_EQ(tiling_data_result, expectStr);
}

TEST_F(TestApplyFtrlTiling, apply_ftrl_testcase_float32)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape = {16, 26, 16, 19};
    auto inputDtype = ge::DT_FLOAT;
    auto inputFormat = ge::FORMAT_ND;
    std::string expectStr = "126464 12094627905567 4096 31 2 2 1280 768 2816 1 ";
    DoApplyFtrlTilingCase(inputShape, inputDtype, inputFormat, expectStr);
}

TEST_F(TestApplyFtrlTiling, apply_ftrl_testcase_float16)
{
    // FLOAT16
    std::initializer_list<int64_t> inputShape = {3761, 4, 44, 4};
    auto inputDtype = ge::DT_FLOAT16;
    auto inputFormat = ge::FORMAT_ND;
    std::string expectStr = "2647744 10995116277824 41472 64 17 14 512 1728 2560 1 ";
    DoApplyFtrlTilingCase(inputShape, inputDtype, inputFormat, expectStr);
}

TEST_F(TestApplyFtrlTiling, apply_ftrl_testcase_bfloat16)
{
    // BFLOAT16
    std::initializer_list<int64_t> inputShape = {7, 2, 7, 8, 10};
    auto inputDtype = ge::DT_BF16;
    auto inputFormat = ge::FORMAT_ND;
    std::string expectStr = "7840 10995116277762 4096 2 2 2 1536 1184 2560 1 ";
    DoApplyFtrlTilingCase(inputShape, inputDtype, inputFormat, expectStr);
}
class ApplyFtrlTilingAccessor : public optiling::ApplyFtrlRegbaseTiling {
public:
    using optiling::ApplyFtrlRegbaseTiling::ApplyFtrlRegbaseTiling;
    using optiling::ApplyFtrlRegbaseTiling::DoElewiseTiling;
    using optiling::ApplyFtrlRegbaseTiling::SetTilingData;
};

// Exercise the registered tiling entry so outer wrappers cannot hide duplicate reports.
static void CheckApplyFtrlError(int scenario, const std::string& code, const std::string& parameter,
                                const std::string& actual, const std::string& reason)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    gert::StorageShape tensorShape = {{2, 3}, {2, 3}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape invalidScalar = {{2}, {2}};
    gert::StorageShape differentShape = {{6}, {6}};
    auto data = gert::TilingData::CreateCap(scenario == 6 ? 1 : 4096);
    auto workspace = gert::ContinuousVector::Create<size_t>(scenario == 7 ? 0 : 1);
    auto dtype = scenario == 3 ? ge::DT_INT32 : ge::DT_FLOAT;
    ApplyFtrlUtCompileInfo compileInfo;
    fe::PlatFormInfos platform;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platform, socInfos, aicoreSpec, intrinsics, socVersion);
    platform.SetPlatformRes("SoCInfo", socInfos);
    platform.SetPlatformRes("AICoreSpec", aicoreSpec);
    platform.SetCoreNumByCoreType("AICore");
    platform.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platform.SetPlatformRes("version", socVersion);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ApplyFtrl")
                      .NodeIoNum(8, 1)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platform))
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&tensorShape, scenario == 1 ? &differentShape : &tensorShape, &tensorShape,
                                    &tensorShape,
                                    scenario == 4 ? nullptr : (scenario == 0 ? &invalidScalar : &scalarShape),
                                    &scalarShape, &scalarShape, &scalarShape})
                      .OutputShapes({&tensorShape})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, scenario == 2 ? ge::DT_FLOAT16 : dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(data.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                      .Build();
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    auto context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    if (scenario == 4) {
        auto kernelContext = holder.GetContext<gert::KernelContext>();
        ASSERT_NE(kernelContext->MutableInput(4), nullptr);
        kernelContext->MutableInput(4)->Set(nullptr, nullptr);
        ASSERT_EQ(context->GetInputShape(4), nullptr);
    }
    if (scenario == 5) {
        *holder.GetContext<gert::KernelContext>()->GetOutputPointer<gert::TilingData*>(
            gert::TilingContext::kOutputTilingData) = nullptr;
        ASSERT_EQ(context->GetRawTilingData(), nullptr);
    }
    if (scenario == 9) {
        // Compare the dependency's own diagnostics with the registered wrapper's diagnostics.
        auto nodeInfo = context->GetComputeNodeInfo();
        auto kernelContext = holder.GetContext<gert::KernelContext>();
        const size_t platformIndex = nodeInfo->GetInputsNum() + nodeInfo->GetOutputsNum() + 1;
        ASSERT_NE(kernelContext->MutableInput(platformIndex), nullptr);
        kernelContext->MutableInput(platformIndex)->Set(nullptr, nullptr);
        ASSERT_EQ(context->GetPlatformInfo(), nullptr);
        ApplyFtrlTilingAccessor tiling(context);
        tiling.tiling_ = context->GetTilingData<ApplyFtrlOpTiling::ApplyFtrlRegbaseTilingData>();
        ASSERT_NE(tiling.tiling_, nullptr);
        auto lowerStatus = tiling.DoElewiseTiling();
        auto lowerErrors = manager.GetRawErrorMessages();
        ASSERT_NE(lowerStatus, ge::GRAPH_SUCCESS);
        ASSERT_FALSE(lowerErrors.empty());
        EXPECT_EQ(impl->tiling(context), lowerStatus);
        auto wrapperErrors = manager.GetRawErrorMessages();
        ASSERT_EQ(wrapperErrors.size(), lowerErrors.size());
        for (size_t i = 0; i < lowerErrors.size(); ++i) {
            EXPECT_EQ(wrapperErrors[i].error_id, lowerErrors[i].error_id);
            EXPECT_EQ(wrapperErrors[i].error_message, lowerErrors[i].error_message);
        }
        return;
    }
    if (scenario == 8) {
        ApplyFtrlTilingAccessor tiling(context);
        ASSERT_EQ(tiling.RunTiling(), ge::GRAPH_SUCCESS);
        ASSERT_TRUE(manager.GetRawErrorMessages().empty());
        ASSERT_GT(tiling.tiling_->elewiseTiling.blockNum, 0);
        // Corrupt only the successful result, keeping legal inputs, resources and storage.
        tiling.tiling_->elewiseTiling.blockNum = 0;
        EXPECT_EQ(tiling.SetTilingData(), ge::GRAPH_FAILED);
    } else {
        EXPECT_EQ(impl->tiling(context), ge::GRAPH_FAILED);
    }
    auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, code);
    EXPECT_NE(errors[0].error_message.find("ApplyFtrl"), std::string::npos);
    EXPECT_NE(errors[0].error_message.find(parameter), std::string::npos);
    EXPECT_NE(errors[0].error_message.find(actual), std::string::npos);
    EXPECT_NE(errors[0].error_message.find(reason), std::string::npos);
    if (scenario == 5 || scenario == 6) {
        EXPECT_NE(errors[0].error_message.find("requiredBytes=" +
                                               std::to_string(sizeof(ApplyFtrlOpTiling::ApplyFtrlRegbaseTilingData))),
                  std::string::npos);
    }
    if (scenario == 8) {
        for (const auto* field :
             {"GRAPH_SUCCESS", "expected blockNum>0", "dim0=6", "coreNum=", "blockFormer=", "tilingKey=", "shape="}) {
            EXPECT_NE(errors[0].error_message.find(field), std::string::npos);
        }
    }
}

TEST_F(TestApplyFtrlTiling, invalid_scalar_reports_shape_once)
{
    CheckApplyFtrlError(0, "EZ0009", "lr", "2", "scalar(0D) or have shape size 1");
}

TEST_F(TestApplyFtrlTiling, mismatched_shape_reports_once)
{
    CheckApplyFtrlError(1, "EZ0010", "var and accum", "6", "same shape as var");
}

TEST_F(TestApplyFtrlTiling, mismatched_dtype_reports_once)
{
    CheckApplyFtrlError(2, "EZ0021", "var and accum", "DT_FLOAT16", "same dtype as var");
}

TEST_F(TestApplyFtrlTiling, unsupported_dtype_reports_once)
{
    CheckApplyFtrlError(3, "EZ0019", "var", "DT_INT32", "fp32, fp16 or bf16");
}

TEST_F(TestApplyFtrlTiling, missing_scalar_reports_input_once)
{
    CheckApplyFtrlError(4, "EZ0004", "lr", "", "required");
}

TEST_F(TestApplyFtrlTiling, null_context_reports_tiling_failure_once)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrl");
    ASSERT_NE(impl, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->tiling(nullptr), ge::GRAPH_FAILED);
    auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_NE(errors[0].error_message.find("ApplyFtrl"), std::string::npos);
    EXPECT_NE(errors[0].error_message.find("TilingContext is nullptr"), std::string::npos);
}

TEST_F(TestApplyFtrlTiling, missing_tiling_buffer_reports_stage_failure_once)
{
    CheckApplyFtrlError(5, "EZ1008", "raw tiling buffer",
                        "requiredBytes=", "Check the GE TilingContext output buffer allocation");
}

TEST_F(TestApplyFtrlTiling, insufficient_tiling_buffer_reports_capacity_once)
{
    CheckApplyFtrlError(6, "EZ1008", "tiling buffer capacity", "actualBytes=1", "requiredBytes=");
}

TEST_F(TestApplyFtrlTiling, insufficient_workspace_slots_reports_stage_failure_once)
{
    CheckApplyFtrlError(7, "EZ1008", "GetWorkspaceSizes(1)", "required output-slot count=1", "workspace container");
}

TEST_F(TestApplyFtrlTiling, corrupted_successful_block_count_reports_tiling_failure_once)
{
    CheckApplyFtrlError(8, "EZ1008", "blockNum=0", "dtype=DT_FLOAT", "opbase implementation owner");
}

TEST_F(TestApplyFtrlTiling, lower_tiling_failure_preserves_reports_without_adding_another)
{
    CheckApplyFtrlError(9, "", "", "", "");
}
