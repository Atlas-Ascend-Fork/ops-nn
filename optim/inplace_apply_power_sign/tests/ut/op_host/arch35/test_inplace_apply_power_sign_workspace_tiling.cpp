/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "base/context_builder/op_tiling_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/inplace_apply_power_sign_tiling.h"
#include "../../../../op_kernel/arch35/inplace_apply_power_sign_tiling_data.h"

namespace {

class InplaceApplyPowerSignWorkspaceTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        auto& registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance();
        if (!registry.GetSpaceRegistry()) {
            ASSERT_EQ(registry.SetSpaceRegistry(std::make_shared<gert::OpImplSpaceRegistryV2>()), ge::GRAPH_SUCCESS);
        }
    }

    void CheckWorkspace(int64_t elements, size_t workspaceCapacity, ge::graphStatus expectedStatus, int scenario = 0,
                        const char* expectedCode = "EZ1008")
    {
        const auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
        ASSERT_NE(registry, nullptr);
        const auto* implementation = registry->GetOpImpl("InplaceApplyPowerSign");
        ASSERT_NE(implementation, nullptr);
        ASSERT_NE(implementation->tiling, nullptr);

        // Synthetic CPU platform metadata, matching the existing Host fixture.
        fe::PlatFormInfos platform;
        ASSERT_TRUE(platform.Init());
        std::map<std::string, std::string> soc{{"ai_core_cnt", "64"},
                                               {"vector_core_cnt", "64"},
                                               {"cube_core_cnt", "64"},
                                               {"core_type_list", "AICore"},
                                               {"l2_size", "33554432"}};
        std::map<std::string, std::string> spec{
            {"ub_size", "245760"},   {"l1_size", "524288"}, {"l0_a_size", "65536"},     {"l0_b_size", "65536"},
            {"l0_c_size", "131072"}, {"bt_size", "0"},      {"load3d_constraints", "1"}};
        platform.SetPlatformRes("SoCInfo", soc);
        platform.SetPlatformRes("AICoreSpec", spec);
        platform.SetCoreNumByCoreType("AICore");
        optiling::InplaceApplyPowerSignCompileInfo compileInfo{64, 245760};

        std::array<gert::Tensor, 7> inputs;
        std::vector<gert::Tensor*> inputPointers;
        for (size_t index = 0; index < inputs.size(); ++index) {
            auto& input = inputs[index];
            const auto shape = gert::Shape{index == 0 || index == 1 || index == 6 ? elements : 1};
            input.MutableOriginShape() = shape;
            input.MutableStorageShape() = shape;
            input.SetOriginFormat(ge::FORMAT_ND);
            input.SetStorageFormat(ge::FORMAT_ND);
            input.SetDataType(ge::DT_FLOAT);
            inputPointers.push_back(&input);
        }
        std::array<gert::Tensor, 2> outputs;
        for (auto& output : outputs) {
            output.MutableOriginShape() = gert::Shape{elements};
            output.MutableStorageShape() = gert::Shape{elements};
            output.SetOriginFormat(ge::FORMAT_ND);
            output.SetStorageFormat(ge::FORMAT_ND);
            output.SetDataType(ge::DT_FLOAT);
        }

        if (scenario == 1) {
            inputs[1].SetDataType(ge::DT_FLOAT16);
        }
        if (scenario == 2) {
            inputs[0].SetDataType(ge::DT_INT32);
        }
        if (scenario == 3) {
            inputs[2].MutableStorageShape() = gert::Shape{2};
        }
        if (scenario == 4) {
            inputs[1].MutableStorageShape() = gert::Shape{elements + 1};
        }
        if (scenario == 5) {
            inputs[2].MutableStorageShape() = gert::Shape{1, 1, 1, 1, 1, 1, 1, 1, 1};
        }
        if (scenario == 6) {
            inputs[2].MutableStorageShape() = gert::Shape{-1};
        }
        if (scenario == 7) {
            for (size_t i : {0, 1, 6}) {
                inputs[i].MutableStorageShape() = gert::Shape{INT64_MAX};
            }
        }
        if (scenario == 10) {
            spec["ub_size"] = "32";
            platform.SetPlatformRes("AICoreSpec", spec);
        }
        auto rawStorage = gert::TilingData::CreateCap(scenario == 8 ? 1 : sizeof(InplaceApplyPowerSignTilingData));
        auto workspaceStorage = gert::ContinuousVector::Create<size_t>(workspaceCapacity);
        ASSERT_NE(rawStorage, nullptr);
        ASSERT_NE(workspaceStorage, nullptr);
        auto* raw = reinterpret_cast<gert::TilingData*>(rawStorage.get());
        auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceStorage.get());
        constexpr size_t sentinel = 0x5a5a;
        if (workspaceCapacity > 0) {
            // A nonzero sentinel catches a callback that merely leaves initial storage untouched.
            ASSERT_EQ(workspace->SetSize(1), ge::GRAPH_SUCCESS);
            static_cast<size_t*>(workspace->MutableData())[0] = sentinel;
        }

        gert::OpTilingContextBuilder builder;
        builder.OpName("powersign_workspace_cpu")
            .OpType("InplaceApplyPowerSign")
            .IOInstanceNum({1, 1, 1, 1, 1, 1, 1}, {1, 1})
            .InputTensors(inputPointers)
            .OutputTensors({&outputs[0], &outputs[1]})
            .AppendAttr(false)
            .CompileInfo(&compileInfo)
            .PlatformInfo(&platform)
            .TilingData(raw)
            .Workspace(workspace);
        auto holder = builder.Build();
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->GetPlatformInfo(), &platform);
        if (scenario == 9) {
            auto* kernelContext = reinterpret_cast<gert::KernelContext*>(context);
            kernelContext->MutableInput(inputs.size() + outputs.size() + 1)->Set(nullptr, nullptr);
            ASSERT_EQ(context->GetPlatformInfo(), nullptr);
        }
        ASSERT_EQ(context->GetRawTilingData(), raw);

        auto& manager = ErrorManager::GetInstance();
        ASSERT_EQ(manager.Init(), 0);
        manager.GetRawErrorMessages();
        ASSERT_EQ(implementation->tiling(context), expectedStatus);
        const auto errors = manager.GetRawErrorMessages();
        if (expectedStatus != ge::GRAPH_SUCCESS) {
            ASSERT_EQ(errors.size(), 1U);
            EXPECT_EQ(errors[0].error_id, expectedCode);
            return;
        }
        EXPECT_TRUE(errors.empty());
        if (expectedStatus == ge::GRAPH_SUCCESS) {
            ASSERT_EQ(workspace->GetSize(), 1U);
            EXPECT_EQ(static_cast<const size_t*>(workspace->GetData())[0], 0U);
            ASSERT_EQ(raw->GetDataSize(), sizeof(InplaceApplyPowerSignTilingData));
            InplaceApplyPowerSignTilingData data{};
            std::memcpy(&data, raw->GetData(), sizeof(data));
            EXPECT_EQ(data.elemNum, elements);
            EXPECT_EQ(data.dim0, elements);
            EXPECT_EQ(data.blockNum, elements == 0 ? 0 : 1);
            EXPECT_EQ(context->GetBlockDim(), 1U);
        } else {
            EXPECT_EQ(workspace->GetSize(), 0U);
        }
    }
};

TEST_F(InplaceApplyPowerSignWorkspaceTest, NonemptyTilingClearsWorkspaceSentinel)
{
    CheckWorkspace(16, 1, ge::GRAPH_SUCCESS);
}

TEST_F(InplaceApplyPowerSignWorkspaceTest, EmptyTilingClearsWorkspaceSentinel)
{
    CheckWorkspace(0, 1, ge::GRAPH_SUCCESS);
}

TEST_F(InplaceApplyPowerSignWorkspaceTest, RejectsInsufficientWorkspaceCapacity)
{
    CheckWorkspace(16, 0, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyPowerSignWorkspaceTest, ValidationReportsSpecificErrorOnce)
{
    const char* codes[] = {"EZ0019", "EZ0019", "EZ0009", "EZ0010", "EZ0012",
                           "EZ0009", "EZ0009", "EZ1008", "EZ1008", "EZ1008"};
    for (int scenario = 1; scenario <= 10; ++scenario) {
        SCOPED_TRACE(scenario);
        CheckWorkspace(16, 1, ge::GRAPH_FAILED, scenario, codes[scenario - 1]);
    }
}

} // namespace
