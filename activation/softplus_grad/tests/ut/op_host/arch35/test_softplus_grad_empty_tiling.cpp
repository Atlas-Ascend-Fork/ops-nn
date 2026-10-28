/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/softplus_grad_tiling.h"
#include "../../../../op_kernel/arch35/softplus_grad_dag.h"

TEST(SoftplusGradEmptyTilingTest, DispatchesZeroWorkForBroadcastableEmptyOutput)
{
    gert::StorageShape emptyShape = {{0, 3}, {0, 3}};
    gert::StorageShape broadcastShape = {{1, 3}, {1, 3}};
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "UB_SIZE": 245760, "CORE_NUM": 64}})";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    Ops::Base::BroadcastCompileInfo compileInfo;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftplusGrad");
    ASSERT_NE(opImpl, nullptr);

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    auto parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    ASSERT_TRUE(parseContext->GetPlatformInfo()->Init());
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parseContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    ASSERT_EQ(opImpl->tiling_parse(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("SoftplusGrad")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1}, {1})
                      .InputShapes({&emptyShape, &broadcastShape})
                      .OutputShapes({&emptyShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(workspace)
                      .Build();
    auto tilingContext = holder.GetContext<gert::TilingContext>();
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(opImpl->tiling(tilingContext), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingContext->GetBlockDim(), 1U);
    ASSERT_NE(tilingContext->GetWorkspaceSizes(1), nullptr);
    EXPECT_EQ(tilingContext->GetWorkspaceSizes(1)[0], 0U);
    auto tilingData = tilingContext->GetTilingData<
        Ops::Base::BroadcastBaseTilingData<typename SoftplusGradOp::SoftplusGradDag<float>::OpDag>>();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->blockNum, 1U);
    EXPECT_EQ(tilingData->blockFormer, 0U);
    EXPECT_EQ(tilingData->blockTail, 0U);
    EXPECT_EQ(tilingData->elemNum, 1U);
}
