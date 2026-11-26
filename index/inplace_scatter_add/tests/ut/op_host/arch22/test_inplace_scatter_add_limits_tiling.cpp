/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include <gtest/gtest.h>
#include "../../../../op_host/arch22/inplace_scatter_add_tiling.h"
#include "test_cube_util.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"

namespace {
TEST(InplaceScatterAddArch22Tiling, preserves_large_tails)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceScatterAdd");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);

    const char* platformJson = R"({"hardware_info": {"UB_SIZE": 196608, "CORE_NUM": 48}})";
    std::map<std::string, std::string> socInfos, aicoreSpec, intrinsics;
    GetPlatFormInfos(platformJson, socInfos, aicoreSpec, intrinsics);
    for (const int64_t tail : {INT64_C(4294967296), INT64_C(4294967299)}) {
        SCOPED_TRACE(tail);
        gert::StorageShape var = {{1, tail}, {1, tail}};
        gert::StorageShape indices = {{1}, {1}};
        gert::StorageShape updates = {{1, tail}, {1, tail}};
        auto tilingData = gert::TilingData::CreateCap(4096);
        ASSERT_NE(tilingData, nullptr);
        auto workspace = gert::ContinuousVector::Create<size_t>(1);
        ASSERT_NE(workspace, nullptr);
        fe::PlatFormInfos platform;
        platform.Init();
        platform.SetPlatformRes("SoCInfo", socInfos);
        platform.SetPlatformRes("AICoreSpec", aicoreSpec);
        platform.SetCoreNumByCoreType("AICore");
        optiling::InplaceScatterAddCompileInfo compileInfo;
        auto holder = gert::TilingContextFaker()
                          .SetOpType("InplaceScatterAdd")
                          .NodeIoNum(3, 1)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&var, &indices, &updates})
                          .OutputShapes({&var})
                          .CompileInfo(&compileInfo)
                          .PlatformInfo(&platform)
                          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .TilingData(tilingData.get())
                          .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                          .Build();
        auto* context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(impl->tiling(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetTilingKey(), 1U);
        optiling::InplaceScatterAddTilingData expected;
        expected.set_M(1);
        expected.set_N(static_cast<uint64_t>(tail));
        expected.set_K(1);
        expected.set_NAligned((static_cast<uint64_t>(tail) + 7) / 8 * 8);
        // Compare only the serialized shape fields, before the core split fields.
        uint8_t expectedBytes[128] = {};
        expected.SaveToBuffer(expectedBytes, sizeof(expectedBytes));
        constexpr size_t shapeFieldsBytes = 2 * sizeof(uint32_t) + 2 * sizeof(uint64_t);
        ASSERT_GE(context->GetRawTilingData()->GetDataSize(), shapeFieldsBytes);
        EXPECT_EQ(std::memcmp(context->GetRawTilingData()->GetData(), expectedBytes, shapeFieldsBytes), 0);
    }
}
} // namespace
