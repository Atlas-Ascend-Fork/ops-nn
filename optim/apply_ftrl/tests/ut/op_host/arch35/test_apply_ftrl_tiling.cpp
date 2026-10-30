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

static void CheckApplyFtrlCoverage(gert::TilingContext* context, const std::initializer_list<int64_t>& shape)
{
    int64_t elements = 1;
    for (auto dimension : shape) {
        elements *= dimension;
    }
    auto raw = context->GetRawTilingData();
    ASSERT_NE(raw, nullptr);
    ASSERT_EQ(raw->GetDataSize(), sizeof(ApplyFtrlOpTiling::ApplyFtrlRegbaseTilingData));
    const auto& tiling = static_cast<const ApplyFtrlOpTiling::ApplyFtrlRegbaseTilingData*>(raw->GetData())
                             ->elewiseTiling;
    EXPECT_EQ(tiling.dim0, elements);
    EXPECT_EQ(context->GetBlockDim(), tiling.blockNum);
    ASSERT_GT(tiling.blockNum, 0);
    EXPECT_LE(tiling.blockNum, 64); // The fixture platform declares 64 cores.
    if (elements == 0) {
        EXPECT_EQ(tiling.blockNum, 1);
        return;
    }
    EXPECT_LE(tiling.blockNum, elements);
    ASSERT_GT(tiling.ubFormer, 0);
    ASSERT_GT(tiling.ubLoopOfFormerBlock, 0);
    ASSERT_GT(tiling.ubLoopOfTailBlock, 0);
    ASSERT_GT(tiling.ubTailOfFormerBlock, 0);
    ASSERT_GT(tiling.ubTailOfTailBlock, 0);
    EXPECT_LE(tiling.ubTailOfFormerBlock, tiling.ubFormer);
    EXPECT_LE(tiling.ubTailOfTailBlock, tiling.ubFormer);
    const auto former = (tiling.ubLoopOfFormerBlock - 1) * tiling.ubFormer + tiling.ubTailOfFormerBlock;
    const auto tail = (tiling.ubLoopOfTailBlock - 1) * tiling.ubFormer + tiling.ubTailOfTailBlock;
    EXPECT_EQ(former, tiling.blockFormer);
    EXPECT_LE(tail, former);
    EXPECT_EQ((tiling.blockNum - 1) * former + tail, elements);
}

static void DoApplyFtrlTilingCase(std::initializer_list<int64_t>& inputShape, ge::DataType inputDtype,
                                  ge::Format inputFormat, std::string& expectStr, uint32_t expectBlockDim = 0)
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
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    CheckApplyFtrlCoverage(tiling_context, inputShape);

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    EXPECT_EQ(tiling_data_result, expectStr);
    if (expectBlockDim != 0) {
        EXPECT_EQ(tiling_context->GetBlockDim(), expectBlockDim);
    }
}

TEST_F(TestApplyFtrlTiling, apply_ftrl_empty_states_have_launchable_noop_tiling)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}) {
        for (auto shape : {std::initializer_list<int64_t>{0}, std::initializer_list<int64_t>{0, 1, 1, 1, 1, 1, 1, 1},
                           std::initializer_list<int64_t>{2, 0, 3}}) {
            std::string expected = "0 0 0 1 0 0 0 0 0 0 ";
            DoApplyFtrlTilingCase(shape, dtype, ge::FORMAT_ND, expected, 1);
        }
    }
}

TEST_F(TestApplyFtrlTiling, apply_ftrl_testcase_float32)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape = {16, 26, 16, 19};
    auto inputDtype = ge::DT_FLOAT;
    auto inputFormat = ge::FORMAT_ND;
    // The promoted tensor DAG changes UB planning: ubFormer=2560, while the
    // 31-block partition and total coverage remain unchanged (checked above).
    std::string expectStr = "126464 10995116277791 4096 31 2 2 1536 1024 2560 1 ";
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
// Exercise validation before platform-dependent tiling, including malformed contexts.
class ApplyFtrlValidation : public optiling::ApplyFtrlRegbaseTiling {
public:
    using ApplyFtrlRegbaseTiling::ApplyFtrlRegbaseTiling;
    using ApplyFtrlRegbaseTiling::CheckShapeAndType;
};

static ge::graphStatus ValidateApplyFtrl(int changedInput = -1, gert::StorageShape changedShape = {{2, 3}, {2, 3}},
                                         ge::DataType changedDtype = ge::DT_FLOAT, bool missingShape = false,
                                         bool useLocking = false, bool changeOutput = false)
{
    // The native builder requires real context output storage even when this
    // fixture only calls validation before platform-dependent tiling.
    fe::PlatFormInfos platform;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platform, socInfos, aicoreSpec, intrinsics, socVersion);
    ApplyFtrlUtCompileInfo compileInfo;
    auto parameter = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    if (parameter == nullptr || workspaceHolder == nullptr) {
        ADD_FAILURE() << "Cannot allocate native validation fixture storage";
        return ge::GRAPH_FAILED;
    }
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    gert::StorageShape tensor = {{2, 3}, {2, 3}};
    gert::StorageShape scalar = {{}, {}};
    std::vector<gert::StorageShape*> shapes = {&tensor, &tensor, &tensor, &tensor, &scalar, &scalar, &scalar, &scalar};
    std::vector<ge::DataType> dtypes(8, ge::DT_FLOAT);
    if (changedInput >= 0) {
        // Build a valid complete context first. A nullptr InputShapes entry
        // packs optional inputs instead of creating a missing required pointer.
        if (!missingShape) {
            shapes[changedInput] = &changedShape;
        }
        dtypes[changedInput] = changedDtype;
    }
    auto outputShape = changeOutput ? &changedShape : &tensor;
    auto outputDtype = changeOutput ? changedDtype : ge::DT_FLOAT;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ApplyFtrl")
                      .NodeIoNum(8, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(shapes)
                      .OutputShapes({outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platform))
                      .TilingData(parameter.get())
                      .Workspace(workspace)
                      .NodeInputTd(0, dtypes[0], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtypes[1], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtypes[2], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dtypes[3], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dtypes[4], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, dtypes[5], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, dtypes[6], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, dtypes[7], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(useLocking)}})
                      .Build();
    auto context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr || context->GetComputeNodeInfo() == nullptr) {
        ADD_FAILURE() << "Native validation fixture did not build a valid context";
        return ge::GRAPH_FAILED; // ADD_FAILURE also fails negative tests.
    }
    for (size_t index = 0; index < 8; ++index) {
        if (context->GetInputShape(index) == nullptr || context->GetInputDesc(index) == nullptr) {
            ADD_FAILURE() << "Native fixture input missing before fault injection: " << index;
            return ge::GRAPH_FAILED;
        }
    }
    if (missingShape) {
        // Use the holder's public KernelContext view: ExtendedKernelContext
        // inherits KernelContext with protected access in SDK 9.2.
        auto kernelContext = holder.GetContext<gert::KernelContext>();
        if (kernelContext == nullptr) {
            ADD_FAILURE() << "Native fixture has no KernelContext view";
            return ge::GRAPH_FAILED;
        }
        auto slot = kernelContext->MutableInput(static_cast<size_t>(changedInput));
        if (slot == nullptr) {
            ADD_FAILURE() << "Native fixture has no input slot for fault injection";
            return ge::GRAPH_FAILED;
        }
        slot->Set(nullptr, nullptr);
        if (context->GetInputShape(static_cast<size_t>(changedInput)) != nullptr) {
            ADD_FAILURE() << "Native input pointer fault injection did not take effect";
            return ge::GRAPH_FAILED;
        }
    }
    ApplyFtrlValidation tiling(context);
    return tiling.CheckShapeAndType();
}

TEST_F(TestApplyFtrlTiling, validation_accepts_scalar_and_singleton_shapes)
{
    EXPECT_EQ(ValidateApplyFtrl(), ge::GRAPH_SUCCESS);
    for (int i = 4; i < 8; ++i) {
        EXPECT_EQ(ValidateApplyFtrl(i, {{1}, {1}}), ge::GRAPH_SUCCESS);
        EXPECT_EQ(ValidateApplyFtrl(i, {{1, 1}, {1, 1}}), ge::GRAPH_SUCCESS);
    }
}

TEST_F(TestApplyFtrlTiling, validation_rejects_missing_input_shapes)
{
    for (int i = 0; i < 8; ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(ValidateApplyFtrl(i, {}, ge::DT_FLOAT, true), ge::GRAPH_FAILED);
    }
}

TEST_F(TestApplyFtrlTiling, validation_rejects_dtype_mismatch)
{
    for (int i = 0; i < 8; ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(
            ValidateApplyFtrl(i, i < 4 ? gert::StorageShape{{2, 3}, {2, 3}} : gert::StorageShape{}, ge::DT_FLOAT16),
            ge::GRAPH_FAILED);
    }
    EXPECT_EQ(ValidateApplyFtrl(0, {{2, 3}, {2, 3}}, ge::DT_INT32), ge::GRAPH_FAILED);
}

TEST_F(TestApplyFtrlTiling, validation_rejects_tensor_shape_mismatch)
{
    for (int i = 1; i < 4; ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(ValidateApplyFtrl(i, {{3, 2}, {3, 2}}), ge::GRAPH_FAILED);
        EXPECT_EQ(ValidateApplyFtrl(i, {{6}, {6}}), ge::GRAPH_FAILED);
    }
}

TEST_F(TestApplyFtrlTiling, validation_rejects_invalid_scalar_shapes)
{
    for (int i = 4; i < 8; ++i) {
        SCOPED_TRACE(i);
        for (const auto& dims : {std::initializer_list<int64_t>{0}, {2}, {-1}, {-1, -1}}) {
            EXPECT_EQ(ValidateApplyFtrl(i, {dims, dims}), ge::GRAPH_FAILED);
        }
    }
}

TEST_F(TestApplyFtrlTiling, validation_rejects_invalid_output_and_locking)
{
    EXPECT_EQ(ValidateApplyFtrl(-1, {{3, 2}, {3, 2}}, ge::DT_FLOAT, false, false, true), ge::GRAPH_FAILED);
    EXPECT_EQ(ValidateApplyFtrl(-1, {{2, 3}, {2, 3}}, ge::DT_FLOAT16, false, false, true), ge::GRAPH_FAILED);
    EXPECT_EQ(ValidateApplyFtrl(-1, {}, ge::DT_FLOAT, false, true), ge::GRAPH_FAILED);
}

TEST_F(TestApplyFtrlTiling, validation_rejects_unknown_and_overflowing_runtime_shape)
{
    EXPECT_EQ(ValidateApplyFtrl(0, {{-1, 3}, {-1, 3}}), ge::GRAPH_FAILED);
    EXPECT_EQ(ValidateApplyFtrl(0, {{INT64_MAX, 2}, {INT64_MAX, 2}}), ge::GRAPH_FAILED);
    optiling::ApplyFtrlRegbaseTiling tiling(nullptr);
    EXPECT_EQ(tiling.RunTiling(), ge::GRAPH_FAILED);
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

TEST_F(TestApplyFtrlTiling, RemainingValidationErrorsReportSpecificCodeOnce)
{
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    const char* codes[] = {"EZ0008", "EZ0019", "EZ0024", "EZ0009"};
    for (int scenario = 0; scenario < 4; ++scenario) {
        SCOPED_TRACE(scenario);
        manager.GetRawErrorMessages();
        ge::graphStatus status;
        if (scenario == 0) {
            status = ValidateApplyFtrl(-1, {{3, 2}, {3, 2}}, ge::DT_FLOAT, false, false, true);
        } else if (scenario == 1) {
            status = ValidateApplyFtrl(-1, {{2, 3}, {2, 3}}, ge::DT_FLOAT16, false, false, true);
        } else if (scenario == 2) {
            status = ValidateApplyFtrl(-1, {}, ge::DT_FLOAT, false, true);
        } else {
            status = ValidateApplyFtrl(0, {{INT64_MAX, 2}, {INT64_MAX, 2}});
        }
        EXPECT_EQ(status, ge::GRAPH_FAILED);
        const auto errors = manager.GetRawErrorMessages();
        ASSERT_EQ(errors.size(), 1U);
        EXPECT_EQ(errors[0].error_id, codes[scenario]);
    }
}
