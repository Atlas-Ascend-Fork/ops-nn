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

using namespace ut_util;

class ForeachAddListTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachAddListTilingArch35 SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachAddListTilingArch35 TearDown" << std::endl; }
};

static void DoCase(std::initializer_list<int64_t> xShape, ge::DataType dt, ge::DataType alphaDt, uint64_t expectKey)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc, ai, intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachAddListCompileInfo {
    } ci;
    auto opType = std::string("ForeachAddList");
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
    gert::StorageShape scalarS = {{1}, {1}};
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .NodeIoNum(3, 1)
                  .IrInstanceNum({1, 1, 1})
                  .InputShapes({&xS, &xS, &scalarS})
                  .OutputShapes({&yS})
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(1, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(2, alphaDt, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(ForeachAddListTilingArch35, test_tiling_float16_arch35) { DoCase({32, 4}, ge::DT_FLOAT16, ge::DT_FLOAT16, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_float32_arch35) { DoCase({32, 4}, ge::DT_FLOAT, ge::DT_FLOAT, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_float16_float_alpha_arch35)
{
    DoCase({32, 4}, ge::DT_FLOAT16, ge::DT_FLOAT, 0);
}

TEST_F(ForeachAddListTilingArch35, test_tiling_int32_arch35) { DoCase({32, 4}, ge::DT_INT32, ge::DT_INT32, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_bfloat16_arch35) { DoCase({32, 4}, ge::DT_BF16, ge::DT_FLOAT, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_int16_arch35) { DoCase({32, 4}, ge::DT_INT16, ge::DT_INT32, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_int8_arch35) { DoCase({32, 4}, ge::DT_INT8, ge::DT_INT32, 0); }

TEST_F(ForeachAddListTilingArch35, test_tiling_uint8_arch35) { DoCase({32, 4}, ge::DT_UINT8, ge::DT_INT32, 0); }

// ---------------------------------------------------------------------------
// Error-code remediation negative tests: formal first-error reporting for the
// alpha dtype mapping and one-element root causes in ValidateForeachAddList.
// ---------------------------------------------------------------------------
#include <map>
#include <string>
#include "base/err_mgr.h"
#include "graph/utils/type_utils.h"

namespace {

ge::graphStatus RunAddListAlphaCase(ge::DataType xDtype, ge::DataType alphaDtype, const gert::StorageShape& alphaShape)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc;
    std::map<std::string, std::string> ai;
    std::map<std::string, std::string> intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    if (!pf.Init()) {
        return ge::GRAPH_FAILED;
    }
    struct ForeachAddListCompileInfo {
    } ci;
    auto opType = std::string("ForeachAddList");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::StorageShape xS = {{4, 4}, {4, 4}};
    gert::StorageShape yS = {{4, 4}, {4, 4}};
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .SetOpType("ForeachAddList")
                  .NodeIoNum(3, 1)
                  .IrInstanceNum({1, 1, 1})
                  .InputShapes({&xS, &xS, const_cast<gert::StorageShape*>(&alphaShape)})
                  .OutputShapes({&yS})
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(2, alphaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(ForeachAddListTilingArch35, alpha_dtype_mismatch_reports_single_formal_error)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    gert::StorageShape alphaShape = {{1}, {1}};
    // x1 is FLOAT (expected alpha FLOAT) but alpha is INT32
    EXPECT_EQ(RunAddListAlphaCase(ge::DT_FLOAT, ge::DT_INT32, alphaShape), ge::GRAPH_FAILED);
    ExpectSingleRawError("EZ0020",
                         {{"param_name", "alpha"},
                          {"op_name", "ForeachAddList"},
                          {"incorrect_dtype", ge::TypeUtils::DataTypeToSerialString(ge::DT_INT32)},
                          {"reason", "the dtype of alpha must follow the scalar mapping of x1 dtype: the same as x1 "
                                     "for FLOAT/FLOAT16/INT32, FLOAT when x1 is BF16, and INT32 when x1 is "
                                     "INT16/INT8/UINT8"}});
}

TEST_F(ForeachAddListTilingArch35, alpha_not_one_element_reports_single_formal_error)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    gert::StorageShape alphaShape = {{2}, {2}};
    EXPECT_EQ(RunAddListAlphaCase(ge::DT_FLOAT, ge::DT_FLOAT, alphaShape), ge::GRAPH_FAILED);
    ExpectSingleRawError("EZ0015", {{"param_name", "alpha"},
                                    {"op_name", "ForeachAddList"},
                                    {"incorrect_size", "2"},
                                    {"reason", "alpha must contain exactly one element"}});
}

// x1 and x2 carry different tensor-list sizes: the shared flat-regbase validator
// reports a single formal EZ0037 first error before any per-tensor validation.
TEST_F(ForeachAddListTilingArch35, flow_list_size_mismatch_reports_single_formal_error)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc;
    std::map<std::string, std::string> ai;
    std::map<std::string, std::string> intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    ASSERT_TRUE(pf.Init());
    struct ForeachAddListCompileInfo {
    } ci;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachAddList");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);

    // x1 holds 2 tensors while x2 holds 1; the list-size equality is checked
    // before any per-tensor validation in ValidateHomogeneous<2>.
    gert::StorageShape x1Tensor0 = {{4, 4}, {4, 4}};
    gert::StorageShape x1Tensor1 = {{2, 2}, {2, 2}};
    gert::StorageShape x2Tensor0 = {{4, 4}, {4, 4}};
    gert::StorageShape scalarS = {{1}, {1}};
    gert::StorageShape y0 = {{4, 4}, {4, 4}};
    gert::StorageShape y1 = {{2, 2}, {2, 2}};
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .SetOpType("ForeachAddList")
                  .NodeIoNum(3, 2)
                  .IrInstanceNum({2, 1, 1}, {2})
                  .InputShapes({&x1Tensor0, &x1Tensor1, &x2Tensor0, &scalarS})
                  .OutputShapes({&y0, &y1})
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    EXPECT_EQ(opImpl->tiling(ctx), ge::GRAPH_FAILED);
    ExpectSingleRawError("EZ0037", {{"param_name", "x"},
                                    {"op_name", "ForeachAddList"},
                                    {"reason", "all tensor-list inputs must have the same list size"}});
}
