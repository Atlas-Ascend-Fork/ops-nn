/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "../../../op_host/add_rms_norm_cast_tiling.h"

TEST(AddRmsNormCastRegbaseTilingErrorReporting, UnsupportedDtypeReportsSupportedTypesOnce)
{
    gert::StorageShape x({8, 64}, {8, 64});
    gert::StorageShape weight({64}, {64});
    optiling::AddRmsNormCastCompileInfo compileInfo{};
    fe::PlatFormInfos platform;
    ASSERT_TRUE(platform.Init());
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(workspaceHolder, nullptr);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AddRmsNormCast")
                      .NodeIoNum(4, 5)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x, &x, &weight, &weight})
                      .OutputShapes({&x, &x, &x, &x, &x})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(&platform)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    optiling::AddRmsNormCastRegbaseTiling tiling(context);
    ge::DataType dtype = ge::DT_INT32;
    std::vector<ge::DataType> supported{ge::DT_FLOAT16, ge::DT_BF16};
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(tiling.CheckDtypeVaild(dtype, supported, "x1"), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0020");
    EXPECT_EQ(errors[0].args_map.at("param_name"), "x1");
    EXPECT_EQ(errors[0].args_map.at("incorrect_dtype"), Ops::Base::ToString(dtype));
    EXPECT_NE(errors[0].args_map.at("reason").find("supported dtype list"), std::string::npos);
}

TEST(AddRmsNormCastRegbaseTilingErrorReporting, UnsupportedDtypeRejectedThroughTilingPath)
{
    // 走 GetShapeAttrsInfo 的真实路径：x1/x2/gamma 一致但均为白名单外 dtype（INT32），
    // CheckInputDtype 的 CheckDtypeVaild 应拒绝并上报一次 EZ0020，tiling 整体返回失败。
    gert::StorageShape x({8, 64}, {8, 64});
    gert::StorageShape weight({64}, {64});
    gert::StorageShape rstd({8, 1}, {8, 1});
    optiling::AddRmsNormCastCompileInfo compileInfo{64, 245760, platform_ascendc::SocVersion::ASCEND950};
    fe::PlatFormInfos platform;
    ASSERT_TRUE(platform.Init());
    std::map<std::string, std::string> versionInfo = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    platform.SetPlatformRes("version", versionInfo);
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(workspaceHolder, nullptr);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AddRmsNormCast")
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &x, &weight})
                      .OutputShapes({&x, &x, &rstd, &x})
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platform))
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddRmsNormCast");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);
    EXPECT_EQ(opImpl->tiling(context), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0020");
    EXPECT_EQ(errors[0].args_map.at("param_name"), "x1");
    EXPECT_EQ(errors[0].args_map.at("incorrect_dtype"), Ops::Base::ToString(ge::DT_INT32));
    EXPECT_NE(errors[0].args_map.at("reason").find("supported dtype list"), std::string::npos);
}
