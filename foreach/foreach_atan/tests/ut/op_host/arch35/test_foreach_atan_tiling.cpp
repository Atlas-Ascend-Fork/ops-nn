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
#include "../../../../op_kernel/arch35/foreach_atan_tiling_data.h"
#include "../../../../op_kernel/arch35/foreach_atan_tiling_key.h"

using namespace ut_util;

class ForeachAtanTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachAtanTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachAtanTilingTest TearDown" << std::endl; }
};

static void DoCase(std::initializer_list<int64_t> xShape, ge::DataType dt, uint64_t expectKey)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc, ai, intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachAtanCompileInfo {
    } ci;
    auto opType = std::string("ForeachAtan");
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

TEST_F(ForeachAtanTilingTest, foreach_atan_float32)
{
    DoCase({4, 4}, ge::DT_FLOAT, GET_TPL_TILING_KEY(FOREACH_ATAN_TPL_SCH_MODE_FLOAT));
}

TEST_F(ForeachAtanTilingTest, foreach_atan_float16)
{
    DoCase({4, 4}, ge::DT_FLOAT16, GET_TPL_TILING_KEY(FOREACH_ATAN_TPL_SCH_MODE_FLOAT16));
}

TEST_F(ForeachAtanTilingTest, foreach_atan_bfloat16)
{
    DoCase({4, 4}, ge::DT_BF16, GET_TPL_TILING_KEY(FOREACH_ATAN_TPL_SCH_MODE_BF16));
}

// ---------------------------------------------------------------------------
// Error-code remediation negative tests: formal first-error reporting for the
// user-parameter dtype root cause in ForeachAtanTilingFunc.
// ---------------------------------------------------------------------------
#include <map>
#include <string>
#include "base/err_mgr.h"
#include "graph/utils/type_utils.h"

namespace {

ge::graphStatus RunAtanTiling(const gert::StorageShape& xShape, ge::DataType dt)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc;
    std::map<std::string, std::string> ai;
    std::map<std::string, std::string> intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    if (!pf.Init()) {
        return ge::GRAPH_FAILED;
    }
    struct ForeachAtanCompileInfo {
    } ci;
    auto opType = std::string("ForeachAtan");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::StorageShape yS = xShape;
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .SetOpType("ForeachAtan")
                  .NodeIoNum(1, 1)
                  .IrInstanceNum({1})
                  .InputShapes({const_cast<gert::StorageShape*>(&xShape)})
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
    return opImpl->tiling(ctx);
}

} // namespace

TEST_F(ForeachAtanTilingTest, unsupported_dtype_reports_single_formal_error)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    gert::StorageShape x = {{4, 4}, {4, 4}};
    EXPECT_EQ(RunAtanTiling(x, ge::DT_INT32), ge::GRAPH_FAILED);
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(std::string(errors.front().error_id.get()), "EZ0019");
    std::map<std::string, std::string> actual;
    for (size_t j = 0; j < errors.front().args_key.size(); ++j) {
        EXPECT_TRUE(actual.emplace(errors.front().args_key[j].get(), errors.front().args_value[j].get()).second);
    }
    const std::map<std::string, std::string> expected = {
        {"param_name", "x"},
        {"op_name", "ForeachAtan"},
        {"incorrect_dtype", ge::TypeUtils::DataTypeToSerialString(ge::DT_INT32)},
        {"correct_dtype", "FP32, FP16 or BF16"}};
    EXPECT_EQ(actual, expected);
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}
