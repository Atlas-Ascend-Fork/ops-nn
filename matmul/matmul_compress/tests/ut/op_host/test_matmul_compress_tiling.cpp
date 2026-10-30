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
#include <string>

#include "base/err_mgr.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "../../../op_host/op_tiling/matmul_compress_tiling_base.h"

namespace {
class MatmulCompressErrorReportTest : public testing::Test {
protected:
    void SetUp() override
    {
        savedContext_ = error_message::GetErrMgrContext();
        auto context = savedContext_;
        context.work_stream_id = reinterpret_cast<std::uintptr_t>(this);
        error_message::SetErrMgrContext(context);
        (void)error_message::GetErrMgrRawErrorMessages();
    }

    void TearDown() override
    {
        (void)error_message::GetErrMgrRawErrorMessages();
        error_message::SetErrMgrContext(savedContext_);
    }

private:
    error_message::ErrorManagerContext savedContext_;
};

TEST_F(MatmulCompressErrorReportTest, ZeroShapeDimensionReportsOneParameterError)
{
    gert::StorageShape x1 = {{0, 32}, {0, 32}};
    gert::StorageShape x2 = {{32, 16}, {32, 16}};
    gert::StorageShape output = {{0, 16}, {0, 16}};
    fe::PlatFormInfos platformInfo;
    ASSERT_TRUE(platformInfo.Init());
    std::map<std::string, std::string> version = {{"SoC_version", "Ascend310P"}, {"Short_SoC_version", "Ascend310P"}};
    std::map<std::string, std::string> socInfo = {{"ai_core_cnt", "8"}, {"l2_size", "33554432"}};
    std::map<std::string, std::string> coreSpec = {
        {"l0_a_size", "65536"}, {"l0_b_size", "65536"}, {"l0_c_size", "262144"}, {"l1_size", "1048576"}};
    platformInfo.SetPlatformRes("version", version);
    platformInfo.SetPlatformRes("SoCInfo", socInfo);
    platformInfo.SetPlatformRes("AICoreSpec", coreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    optiling::pp_matmul::HardwareInfo compileInfo;
    auto tilingData = gert::TilingData::CreateCap(1);
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MatmulCompress")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x1, &x2})
                      .OutputShapes({&output})
                      .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"tilingK", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"tilingN", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(&platformInfo)
                      .TilingData(tilingData.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetRawTilingData(), reinterpret_cast<gert::TilingData*>(tilingData.get()));
    auto tiling = gert::OpImplRegistry::GetInstance().GetOpImpl("MatmulCompress")->tiling;
    ASSERT_NE(tiling, nullptr);
    // Only observe the real tiling call, not diagnostics from context construction.
    (void)error_message::GetErrMgrRawErrorMessages();
    EXPECT_EQ(tiling(context), ge::GRAPH_FAILED);
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_NE(errors[0].error_id, nullptr);
    EXPECT_STREQ(errors[0].error_id.get(), "EZ1001");
    ASSERT_EQ(errors[0].args_key.size(), 1U);
    ASSERT_EQ(errors[0].args_value.size(), 1U);
    EXPECT_STREQ(errors[0].args_key[0].get(), "message");
    ASSERT_NE(errors[0].args_value[0], nullptr);
    EXPECT_NE(std::string(errors[0].args_value[0].get()).find("should not be zero"), std::string::npos);
}

TEST_F(MatmulCompressErrorReportTest, InsufficientTilingCapacityReportsOneExecutionError)
{
    gert::StorageShape x1 = {{16, 32}, {16, 32}};
    gert::StorageShape x2 = {{32, 16}, {32, 16}};
    gert::StorageShape output = {{16, 16}, {16, 16}};
    fe::PlatFormInfos platformInfo;
    optiling::pp_matmul::HardwareInfo compileInfo;
    auto tilingData = gert::TilingData::CreateCap(1);
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MatmulCompress")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x1, &x2})
                      .OutputShapes({&output})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(&platformInfo)
                      .TilingData(tilingData.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetRawTilingData(), reinterpret_cast<gert::TilingData*>(tilingData.get()));
    ASSERT_EQ(context->GetRawTilingData()->GetCapacity(), 1U);
    optiling::matmul_compress::MatmulCompressTilingBase tiling(context);
    (void)error_message::GetErrMgrRawErrorMessages();
    EXPECT_EQ(tiling.PostTiling(), ge::GRAPH_FAILED);
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_NE(errors[0].error_id, nullptr);
    EXPECT_STREQ(errors[0].error_id.get(), "EZ1008");
    ASSERT_EQ(errors[0].args_key.size(), 2U);
    ASSERT_EQ(errors[0].args_value.size(), 2U);
    EXPECT_STREQ(errors[0].args_key[0].get(), "opName");
    EXPECT_STREQ(errors[0].args_key[1].get(), "reason");
    ASSERT_NE(errors[0].args_value[1], nullptr);
    const std::string reason = errors[0].args_value[1].get();
    EXPECT_NE(reason.find("memcpy_s failed"), std::string::npos);
    EXPECT_NE(reason.find("required bytes="), std::string::npos);
    EXPECT_NE(reason.find("capacity bytes=1"), std::string::npos);
}
} // namespace
