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
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/foreach_cosh_tiling_data.h"
#include "../../../../op_kernel/arch35/foreach_cosh_tiling_key.h"
#include <map>
#include <string>
#include "base/err_mgr.h"
#include "graph/utils/type_utils.h"

using namespace ut_util;

class ForeachCoshTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachCoshTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachCoshTilingTest TearDown" << std::endl; }
};

static void DoCase(std::initializer_list<int64_t> xShape, ge::DataType dt, uint64_t expectKey)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc, ai, intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachCoshCompileInfo {
    } ci;
    auto opType = std::string("ForeachCosh");
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFn, nullptr);
    ASSERT_NE(tilingParseFn, nullptr);

    std::string ciStr = R"({"device_id":null})";
    auto kh = gert::KernelRunContextFaker()
                  .KernelIONum(2, 1)
                  .Inputs({const_cast<char*>(ciStr.c_str()), reinterpret_cast<void*>(&pf)})
                  .Outputs({&ci})
                  .Build();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    ASSERT_EQ(tilingParseFn(kh.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xS = {xShape, xShape};
    gert::StorageShape yS = {xShape, xShape};
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .NodeIoNum(1, 1)
                  .IrInstanceNum({1})
                  .InputShapes({&xS})
                  .OutputShapes({&yS})
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    EXPECT_EQ(tilingFn(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetTilingKey(), expectKey);
}

TEST_F(ForeachCoshTilingTest, foreach_cosh_float32)
{
    DoCase({4, 4}, ge::DT_FLOAT, GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_FLOAT));
}

TEST_F(ForeachCoshTilingTest, foreach_cosh_float16)
{
    DoCase({4, 4}, ge::DT_FLOAT16, GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_FLOAT16));
}

TEST_F(ForeachCoshTilingTest, foreach_cosh_bfloat16)
{
    DoCase({4, 4}, ge::DT_BF16, GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_BF16));
}

namespace {
constexpr int64_t MOCK_CORE_NUM = 64;
constexpr int64_t ALIGN_SIZE_UT = 32;

int64_t CeilDivExpect(int64_t a, int64_t b) { return (a + b - 1) / b; }

int64_t CeilAlignExpect(int64_t a, int64_t b) { return CeilDivExpect(a, b) * b; }
} // namespace

// list-shaped tiling: per-tensor shapes are 1-D of the given numel, mirroring the
// scalar-list / multi-tensor workloads of issue #65; expected values are derived from
// MOCK_CORE_NUM so they stay coupled to the mocked platform instead of hard-coded
static void DoListCase(const std::vector<int64_t>& tensorNumels, ge::DataType dt, uint64_t expectKey)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc, ai, intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachCoshCompileInfo {
    } ci;
    auto opType = std::string("ForeachCosh");
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFn, nullptr);
    ASSERT_NE(tilingParseFn, nullptr);

    std::string ciStr = R"({"device_id":null})";
    auto kh = gert::KernelRunContextFaker()
                  .KernelIONum(2, 1)
                  .Inputs({const_cast<char*>(ciStr.c_str()), reinterpret_cast<void*>(&pf)})
                  .Outputs({&ci})
                  .Build();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    ASSERT_EQ(tilingParseFn(kh.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    const uint32_t tensorNum = static_cast<uint32_t>(tensorNumels.size());
    std::vector<gert::StorageShape> inputShapes(tensorNum);
    std::vector<gert::StorageShape> outputShapes(tensorNum);
    int64_t totalElements = 0;
    for (uint32_t i = 0; i < tensorNum; i++) {
        inputShapes[i] = {{tensorNumels[i]}, {tensorNumels[i]}};
        outputShapes[i] = {{tensorNumels[i]}, {tensorNumels[i]}};
        totalElements += tensorNumels[i];
    }
    std::vector<gert::StorageShape*> inputShapeRefs;
    std::vector<gert::StorageShape*> outputShapeRefs;
    for (auto& shape : inputShapes) {
        inputShapeRefs.push_back(&shape);
    }
    for (auto& shape : outputShapes) {
        outputShapeRefs.push_back(&shape);
    }

    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .NodeIoNum(1, 1)
                  .IrInstanceNum({tensorNum}, {tensorNum})
                  .InputShapes(inputShapeRefs)
                  .OutputShapes(outputShapeRefs)
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);

    EXPECT_EQ(tilingFn(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetTilingKey(), expectKey);

    auto rawTilingData = ctx->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    ASSERT_GE(rawTilingData->GetDataSize(), sizeof(ForeachCoshTilingData));
    const auto* tilingData = reinterpret_cast<const ForeachCoshTilingData*>(rawTilingData->GetData());
    EXPECT_EQ(tilingData->tensorCount, static_cast<int32_t>(tensorNum));
    EXPECT_EQ(tilingData->totalElements, totalElements);

    int64_t expectPerCore = CeilDivExpect(totalElements, MOCK_CORE_NUM);
    if (expectPerCore >= ALIGN_SIZE_UT) {
        expectPerCore = CeilAlignExpect(expectPerCore, ALIGN_SIZE_UT);
    }
    const int64_t expectNeedCore = CeilDivExpect(totalElements, expectPerCore);
    const int64_t expectLastCore = totalElements - expectPerCore * (expectNeedCore - 1);
    EXPECT_EQ(tilingData->perCoreElements, expectPerCore);
    EXPECT_EQ(tilingData->needCoreNum, static_cast<int32_t>(expectNeedCore));
    EXPECT_EQ(tilingData->lastCoreElements, expectLastCore);

    // prefix starts: tensorStarts[i] is the flat start of tensor i, tensorStarts[N] the total
    int64_t prefix = 0;
    for (uint32_t i = 0; i < tensorNum; i++) {
        EXPECT_EQ(tilingData->tensorStarts[i], prefix);
        prefix += tensorNumels[i];
    }
    EXPECT_EQ(tilingData->tensorStarts[tensorNum], totalElements);
}

// 128 0-dim-scalar-like tensors: the small-list regime of issue #65 must keep
// needCoreNum close to min(coreNum, totalElements) instead of being squashed to 1
TEST_F(ForeachCoshTilingTest, foreach_cosh_scalar_list_multicore)
{
    std::vector<int64_t> numels(128, 1);
    DoListCase(numels, ge::DT_FLOAT16, GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_FLOAT16));
}

// mid-size list: perCoreElements crosses the 32-alignment threshold
TEST_F(ForeachCoshTilingTest, foreach_cosh_mid_list_aligned_chunk)
{
    DoListCase({1000, 1000, 1000}, ge::DT_FLOAT, GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_FLOAT));
}

// single large tensor: aligned full-core split with a shorter last core
TEST_F(ForeachCoshTilingTest, foreach_cosh_large_tensor_full_cores)
{
    DoListCase({125898626}, ge::DT_BF16, GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_BF16));
}

// empty tensor inside the list holds equal prefix starts and must not break the table
TEST_F(ForeachCoshTilingTest, foreach_cosh_list_with_empty_tensor)
{
    DoListCase({4, 0, 8}, ge::DT_FLOAT, GET_TPL_TILING_KEY(FOREACH_COSH_TPL_SCH_MODE_FLOAT));
}

// ---------------------------------------------------------------------------
// Error-code remediation negative tests: formal first-error reporting for the
// user-parameter root causes in ForeachCoshTilingFunc.
// ---------------------------------------------------------------------------
#include <map>
#include <string>
#include "base/err_mgr.h"
#include "graph/utils/type_utils.h"

namespace {

ge::graphStatus RunCoshTiling(const std::vector<gert::StorageShape>& xShapes, const std::vector<ge::DataType>& xDtypes,
                              ge::DataType yDtype)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc;
    std::map<std::string, std::string> ai;
    std::map<std::string, std::string> intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    if (!pf.Init()) {
        return ge::GRAPH_FAILED;
    }
    struct ForeachCoshCompileInfo {
    } ci;
    auto opType = std::string("ForeachCosh");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    std::vector<gert::StorageShape> yShapes(xShapes.size());
    std::vector<gert::StorageShape*> outPtr;
    for (size_t i = 0; i < xShapes.size(); i++) {
        yShapes[i] = xShapes[i];
        outPtr.push_back(&yShapes[i]);
    }
    // per-instance dtype inside one dynamic input requires TensorV2-level inputs
    gert::StorageFormat storageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType());
    std::vector<gert::TensorV2> inputTensorsStorage;
    for (size_t i = 0; i < xShapes.size(); i++) {
        inputTensorsStorage.emplace_back(xShapes[i], storageFormat, gert::TensorPlacement::kOnHost, xDtypes[i],
                                         nullptr);
    }
    std::vector<gert::TensorV2*> inputTensors;
    for (auto& t : inputTensorsStorage) {
        inputTensors.push_back(&t);
    }
    gert::TilingContextFaker builder;
    builder.SetOpType("ForeachCosh")
        .NodeIoNum(1, static_cast<uint32_t>(xShapes.size()))
        .IrInstanceNum({static_cast<uint32_t>(xShapes.size())}, {static_cast<uint32_t>(xShapes.size())})
        .InputTensors(inputTensors)
        .OutputShapes(outPtr)
        .CompileInfo(&ci)
        .PlatformInfo(reinterpret_cast<char*>(&pf));
    for (size_t i = 0; i < xShapes.size(); i++) {
        builder.NodeOutputTd(static_cast<int32_t>(i), yDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto th = std::move(builder)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    return opImpl->tiling(ctx);
}

void ExpectSingleRawError(const char* expectedCode, const std::map<std::string, std::string>& expectedMap)
{
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(std::string(errors.front().error_id.get()), expectedCode);
    std::map<std::string, std::string> actual;
    for (size_t j = 0; j < errors.front().args_key.size(); ++j) {
        EXPECT_TRUE(actual.emplace(errors.front().args_key[j].get(), errors.front().args_value[j].get()).second);
    }
    EXPECT_EQ(actual, expectedMap);
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

} // namespace

TEST_F(ForeachCoshTilingTest, mixed_dtype_list_reports_single_formal_error)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    gert::StorageShape x0 = {{4, 4}, {4, 4}};
    gert::StorageShape x1 = {{2, 2}, {2, 2}};
    EXPECT_EQ(RunCoshTiling({x0, x1}, {ge::DT_FLOAT, ge::DT_FLOAT16}, ge::DT_FLOAT), ge::GRAPH_FAILED);
    ExpectSingleRawError("EZ0020",
                         {{"param_name", "x"},
                          {"op_name", "ForeachCosh"},
                          {"incorrect_dtype", ge::TypeUtils::DataTypeToSerialString(ge::DT_FLOAT16)},
                          {"reason", "The dtypes of all tensors in the tensor list must be the same. Currently, the "
                                     "dtype of the 1th tensor is inconsistent with that (" +
                                         ge::TypeUtils::DataTypeToSerialString(ge::DT_FLOAT) + ") of other tensors"}});
}

TEST_F(ForeachCoshTilingTest, rank_over_limit_reports_single_formal_error)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    gert::StorageShape x = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    EXPECT_EQ(RunCoshTiling({x}, {ge::DT_FLOAT}, ge::DT_FLOAT), ge::GRAPH_FAILED);
    ExpectSingleRawError("EZ0012",
                         {{"param_name", "x"},
                          {"op_name", "ForeachCosh"},
                          {"incorrect_dim", "9"},
                          {"reason", "The shape dim of the 0th tensor in the tensor list should be less than or "
                                     "equal to 8"}});
}

TEST_F(ForeachCoshTilingTest, unsupported_dtype_reports_single_formal_error)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    gert::StorageShape x = {{4, 4}, {4, 4}};
    EXPECT_EQ(RunCoshTiling({x}, {ge::DT_INT32}, ge::DT_INT32), ge::GRAPH_FAILED);
    ExpectSingleRawError("EZ0019", {{"param_name", "x"},
                                    {"op_name", "ForeachCosh"},
                                    {"incorrect_dtype", ge::TypeUtils::DataTypeToSerialString(ge::DT_INT32)},
                                    {"correct_dtype", "float, float16 or bfloat16"}});
}
