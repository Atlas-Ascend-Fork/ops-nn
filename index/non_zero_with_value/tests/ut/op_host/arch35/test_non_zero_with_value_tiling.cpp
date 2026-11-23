/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "ut_op_util.h"
#include "index/non_zero_with_value/op_host/arch35/non_zero_with_value_tiling_arch35.h"
#include "test_cube_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "base/err_mgr.h"

using namespace ge;
using namespace ut_util;

namespace nzv_ut {
// 每个用例独占一个 ErrorManager 上下文（work_stream_id 取本对象地址），生命周期内记录只落在该
// 上下文中；析构时先 drain 掉本次记录，再恢复进入前的上下文，避免同一 Host UT 进程内跨用例串扰。
// 对齐 rnn/gru_grad/tests/ut/op_host/test_gru_grad_tiling.cpp 的同名实现。
class ScopedErrorContext {
public:
    ScopedErrorContext() : savedContext_(error_message::GetErrMgrContext())
    {
        auto context = savedContext_;
        context.work_stream_id = reinterpret_cast<std::uintptr_t>(this);
        error_message::SetErrMgrContext(context);
        (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
        (void)error_message::GetErrMgrRawErrorMessages();
    }

    ~ScopedErrorContext()
    {
        (void)error_message::GetErrMgrRawErrorMessages();
        error_message::SetErrMgrContext(savedContext_);
    }

    ScopedErrorContext(const ScopedErrorContext&) = delete;
    ScopedErrorContext& operator=(const ScopedErrorContext&) = delete;

private:
    error_message::ErrorManagerContext savedContext_;
};

struct CapturedError {
    size_t count = 0;
    std::string code;
    std::vector<std::string> ids;
};

inline CapturedError CaptureError()
{
    CapturedError captured;
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    captured.count = errors.size();
    for (const auto& error : errors) {
        captured.ids.emplace_back(error.error_id == nullptr ? std::string() : std::string(error.error_id.get()));
    }
    if (!captured.ids.empty()) {
        captured.code = captured.ids.back();
    }
    return captured;
}

inline int CountErrorId(const CapturedError& captured, const std::string& errorId)
{
    int count = 0;
    for (const auto& id : captured.ids) {
        if (id == errorId) {
            ++count;
        }
    }
    return count;
}
} // namespace nzv_ut

class NonZeroWithValueTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NonZeroWithValueTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NonZeroWithValueTilingTest TearDown" << std::endl; }
};

static void InitPlatForm(fe::PlatFormInfos& platformInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

static ge::graphStatus RunTiling(gert::StorageShape& xShape, bool transpose, int64_t& tilingKeyOut)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    int64_t numel = 1;
    for (size_t i = 0; i < xShape.GetStorageShape().GetDimNum(); i++) {
        numel *= xShape.GetStorageShape().GetDim(i);
    }
    gert::StorageShape valueShape = {{numel}, {numel}};
    gert::StorageShape indexShape = {{2 * numel}, {2 * numel}};
    gert::StorageShape countShape = {{1}, {1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> keysToValue;
    keysToValue.push_back(std::make_pair("transpose", Ops::NN::AnyValue::CreateFrom<bool>(transpose)));
    keysToValue.push_back(std::make_pair("dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)));

    optiling::NonZeroWithValueCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 253952;
    compileInfo.vRegSize = 256;

    std::string opType("NonZeroWithValue");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tilingFunc = opImpl->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(64 * 8);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&valueShape, &indexShape, &countShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(keysToValue)
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = tilingFunc(tilingContext);
    tilingKeyOut = static_cast<int64_t>(tilingContext->GetTilingKey());
    return ret;
}

// 非空 2D → general 路径,tilingKey = 1020
TEST_F(NonZeroWithValueTilingTest, general_small_2d)
{
    gert::StorageShape xShape = {{4, 8}, {4, 8}};
    int64_t tilingKey = -1;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 1020);
}

// 大形状多核 → general 路径,tilingKey = 1020
TEST_F(NonZeroWithValueTilingTest, general_large_2d)
{
    gert::StorageShape xShape = {{1024, 1024}, {1024, 1024}};
    int64_t tilingKey = -1;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 1020);
}

// 空张量 numel==0 → null 路径,tilingKey = 1000
TEST_F(NonZeroWithValueTilingTest, null_empty_2d)
{
    gert::StorageShape xShape = {{0, 8}, {0, 8}};
    int64_t tilingKey = -1;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, 1000);
}

// transpose=false 不支持 → 失败；仅上报 1 条 EZ0026（L74 首报 + L264 WITHOUT_REPORT 去重，不得出现 EZ9999）
TEST_F(NonZeroWithValueTilingTest, transpose_false_fail)
{
    gert::StorageShape xShape = {{4, 8}, {4, 8}};
    int64_t tilingKey = -1;
    nzv_ut::ScopedErrorContext errorContext;
    EXPECT_EQ(RunTiling(xShape, false, tilingKey), ge::GRAPH_FAILED);
    const auto error = nzv_ut::CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0026");
    EXPECT_EQ(nzv_ut::CountErrorId(error, std::string("EZ") + "9999"), 0);
}

// 3D 输入 → 失败；仅上报 1 条 EZ0012（L94 首报 + L266 WITHOUT_REPORT 去重）
TEST_F(NonZeroWithValueTilingTest, input_3d_fail_ez0012)
{
    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    int64_t tilingKey = -1;
    nzv_ut::ScopedErrorContext errorContext;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_FAILED);
    const auto error = nzv_ut::CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0012");
}

// numel 超过 INT32_MAX（65536*65536=2^32）→ 失败；上报 EZ0015（L109）
TEST_F(NonZeroWithValueTilingTest, numel_exceeds_int32_fail_ez0015)
{
    gert::StorageShape xShape = {{65536, 65536}, {65536, 65536}};
    int64_t tilingKey = -1;
    nzv_ut::ScopedErrorContext errorContext;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_FAILED);
    const auto error = nzv_ut::CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0015");
}

// col <= 0 → 失败；上报 EZ0009（L111）
TEST_F(NonZeroWithValueTilingTest, col_non_positive_fail_ez0009)
{
    gert::StorageShape xShape = {{2, -1}, {2, -1}};
    int64_t tilingKey = -1;
    nzv_ut::ScopedErrorContext errorContext;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_FAILED);
    const auto error = nzv_ut::CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
}

// 正例：合法 2D 输入 tiling 成功且不产生任何错误记录（无回归）
TEST_F(NonZeroWithValueTilingTest, valid_2d_success_no_error)
{
    gert::StorageShape xShape = {{4, 8}, {4, 8}};
    int64_t tilingKey = -1;
    nzv_ut::ScopedErrorContext errorContext;
    EXPECT_EQ(RunTiling(xShape, true, tilingKey), ge::GRAPH_SUCCESS);
    const auto error = nzv_ut::CaptureError();
    EXPECT_EQ(error.count, 0U);
    EXPECT_EQ(tilingKey, 1020);
}
