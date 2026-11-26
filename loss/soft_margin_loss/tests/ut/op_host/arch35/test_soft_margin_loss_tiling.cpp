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
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base/err_mgr.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;

namespace {
// 注册入口 TilingForSoftMarginLoss 仅要求 GetCompileInfo() 非空（负向路径不读取其字段），
// 此处以本地桩结构替代 op_host 头文件（其头文件链依赖 op_kernel 路径，UT 目标不可见）。
struct SoftMarginLossUtCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};
constexpr int64_t DEFAULT_CORE_NUM = 64;

gert::StorageShape MakeStorageShapeFromVector(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t d : dims) {
        shape.MutableStorageShape().AppendDim(d);
        shape.MutableOriginShape().AppendDim(d);
    }
    return shape;
}
constexpr uint64_t DEFAULT_UB_SIZE = 253952;
constexpr uint64_t DEFAULT_TILING_DATA_SIZE = 4096;
const char* DEFAULT_COMPILE_INFO = R"({
    "hardware_info": {"UB_SIZE": 253952, "CORE_NUM": 64}
})";

struct SoftMarginCaseResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = std::numeric_limits<uint64_t>::max();
};

SoftMarginCaseResult RunSoftMarginLossTiling(const vector<int64_t>& selfShape, const vector<int64_t>& targetShape,
                                             const vector<int64_t>& outputShape, ge::DataType inputDtype,
                                             ge::DataType outputDtype, const string& reduction)
{
    SoftMarginCaseResult result;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(DEFAULT_COMPILE_INFO, socInfos, aicoreSpec, intrinsics);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    SoftMarginLossUtCompileInfo compileInfo{};
    compileInfo.coreNum = static_cast<uint64_t>(DEFAULT_CORE_NUM);
    compileInfo.ubSize = DEFAULT_UB_SIZE;
    const string opType("SoftMarginLoss");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return result;
    }

    gert::StorageShape selfStorage = MakeStorageShapeFromVector(selfShape);
    gert::StorageShape targetStorage = MakeStorageShapeFromVector(targetShape);
    gert::StorageShape outputStorage = MakeStorageShapeFromVector(outputShape);

    auto tilingData = gert::TilingData::CreateCap(DEFAULT_TILING_DATA_SIZE);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(DEFAULT_TILING_DATA_SIZE);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (tilingData == nullptr || workspace == nullptr) {
        return result;
    }

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&selfStorage, &targetStorage})
                      .OutputShapes({&outputStorage})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return result;
    }
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    result.status = opImpl->tiling(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.tilingKey = context->GetTilingKey();
    }
    return result;
}

struct RawErrorEntry {
    std::string errorId;
    std::map<std::string, std::string> fields;
};

std::vector<RawErrorEntry> DrainAllErrorFields()
{
    std::vector<RawErrorEntry> entries;
    auto errors = error_message::GetErrMgrRawErrorMessages();
    for (size_t e = 0; e < errors.size(); ++e) {
        RawErrorEntry entry;
        if (errors[e].error_id.get() != nullptr) {
            entry.errorId = std::string(errors[e].error_id.get());
        }
        for (size_t i = 0; i < errors[e].args_key.size(); ++i) {
            entry.fields.emplace(std::string(errors[e].args_key[i].get()), std::string(errors[e].args_value[i].get()));
        }
        entries.push_back(entry);
    }
    return entries;
}
} // namespace

class SoftMarginLossTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SoftMarginLossTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SoftMarginLossTilingTest TearDown" << std::endl; }
};

TEST_F(SoftMarginLossTilingTest, tiling_ele_fp32_success)
{
    auto result = RunSoftMarginLossTiling({4, 8}, {4, 8}, {4, 8}, ge::DT_FLOAT, ge::DT_FLOAT, "none");
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
}

TEST_F(SoftMarginLossTilingTest, tiling_reduce_fp32_success)
{
    auto result = RunSoftMarginLossTiling({4, 8}, {4, 8}, {}, ge::DT_FLOAT, ge::DT_FLOAT, "sum");
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
}

// ---------------------------------------------------------------------------
// 错误码整改负向用例：RunTiling/CheckShape/TilingEle/TilingReduce 的 WITHOUT_REPORT
// 失败分支与下层正式宏首报的联合覆盖。
// ---------------------------------------------------------------------------

// CheckShape 失败 -> OP_LOGE_WITHOUT_REPORT("check shape failed")，下层首报 EZ0010
TEST_F(SoftMarginLossTilingTest, check_shape_failed_reports_ez0010)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    auto result = RunSoftMarginLossTiling({2, 3}, {2, 4}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, "none");
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
    const auto errors = DrainAllErrorFields();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].errorId, "EZ0010");
    EXPECT_EQ(errors[0].fields.at("param_names"), "input_x and input_y");
    EXPECT_EQ(errors[0].fields.at("reason"), "The known shapes of input_x and input_y must be the same");
    EXPECT_FALSE(errors[0].fields.at("op_name").empty());
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

// reduction=none + fp16 输出 + 0 维 shape -> ElewiseTiling 失败，
// 覆盖 TilingEle fp16 分支的 WITHOUT_REPORT 与 RunTiling elewise 包装
TEST_F(SoftMarginLossTilingTest, elewise_zero_dim_failed_fp16)
{
    auto result = RunSoftMarginLossTiling({0, 10}, {0, 10}, {0, 10}, ge::DT_FLOAT16, ge::DT_FLOAT16, "none");
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// reduction=none + fp32 输出 + 0 维 shape -> 覆盖 TilingEle fp32 分支的 WITHOUT_REPORT
TEST_F(SoftMarginLossTilingTest, elewise_zero_dim_failed_fp32)
{
    auto result = RunSoftMarginLossTiling({0, 10}, {0, 10}, {0, 10}, ge::DT_FLOAT, ge::DT_FLOAT, "none");
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// reduction=sum + 动态维(-1) -> ReduceOpTmpl::GetInputParam 失败（EZ0012 首报），
// 覆盖 TilingReduce get self input 分支与 RunTiling reduce 包装
TEST_F(SoftMarginLossTilingTest, reduce_get_input_param_failed_reports_ez0012)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    auto result = RunSoftMarginLossTiling({2, -1}, {2, -1}, {}, ge::DT_FLOAT, ge::DT_FLOAT, "sum");
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
    const auto errors = DrainAllErrorFields();
    // 首错为正式宏 EZ0012；ReduceOpTmpl 库内层另有普通 OP_LOGE 包装产生的 EZ9999，
    // 本算子的 WITHOUT_REPORT 出口不再追加第三条
    ASSERT_EQ(errors.size(), 2U);
    EXPECT_EQ(errors[0].errorId, "EZ0012");
    EXPECT_EQ(errors[0].fields.at("param_name"), "x");
    EXPECT_EQ(errors[0].fields.at("incorrect_dim"), "-1");
    EXPECT_FALSE(errors[0].fields.at("op_name").empty());
    EXPECT_EQ(errors[1].errorId, "EZ9999");
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

// 输入 dtype 未定义 -> reduce ParamCheck 失败（EZ0020 首报），覆盖
// half/sum、half/mean、float/sum、float/mean 四个 Tiling4ReduceOp WITHOUT_REPORT 分支
TEST_F(SoftMarginLossTilingTest, reduce_half_sum_dtype_illegal_reports_ez0020)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    auto result = RunSoftMarginLossTiling({4, 8}, {4, 8}, {}, ge::DT_UNDEFINED, ge::DT_FLOAT16, "sum");
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
    const auto errors = DrainAllErrorFields();
    // 首错为正式宏 EZ0020；ReduceOpTiling::DoTiling 的普通 OP_LOGE 包装产生 EZ9999，
    // 本算子的 WITHOUT_REPORT 出口不再追加第三条
    ASSERT_EQ(errors.size(), 2U);
    EXPECT_EQ(errors[0].errorId, "EZ0020");
    EXPECT_EQ(errors[0].fields.at("param_name"), "x");
    EXPECT_EQ(errors[0].fields.at("reason"), "dtype of input x is illegal");
    EXPECT_FALSE(errors[0].fields.at("op_name").empty());
    EXPECT_EQ(errors[1].errorId, "EZ9999");
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST_F(SoftMarginLossTilingTest, reduce_half_mean_dtype_illegal_failed)
{
    auto result = RunSoftMarginLossTiling({4, 8}, {4, 8}, {}, ge::DT_UNDEFINED, ge::DT_BF16, "mean");
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(SoftMarginLossTilingTest, reduce_float_sum_dtype_illegal_failed)
{
    auto result = RunSoftMarginLossTiling({4, 8}, {4, 8}, {}, ge::DT_UNDEFINED, ge::DT_FLOAT, "sum");
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(SoftMarginLossTilingTest, reduce_float_mean_dtype_illegal_failed)
{
    auto result = RunSoftMarginLossTiling({4, 8}, {4, 8}, {}, ge::DT_UNDEFINED, ge::DT_FLOAT, "mean");
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}
