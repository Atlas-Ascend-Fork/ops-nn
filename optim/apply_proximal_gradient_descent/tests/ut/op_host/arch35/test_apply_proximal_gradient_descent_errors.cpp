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
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"

TEST(ApplyProximalGradientDescentErrorCodes, TilingReportsSpecificErrorOnce)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyProximalGradientDescent");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    const char* codes[] = {"",       "EZ0019", "EZ0019", "EZ0018", "EZ0010", "EZ0010", "EZ0009", "EZ1008",
                           "EZ1008", "EZ0019", "EZ0018", "EZ1008", "EZ0004", "EZ0009", "EZ1008"};
    for (int scenario = 0; scenario < 15; ++scenario) {
        SCOPED_TRACE(scenario);
        gert::StorageShape tensor = {{16}, {16}};
        gert::StorageShape scalar = {{1}, {1}};
        gert::StorageShape different = {{8}, {8}};
        gert::StorageShape invalidScalar = {{2}, {2}};
        gert::StorageShape overflowing = {{INT64_MAX}, {INT64_MAX}};
        auto* state = scenario == 13 ? &overflowing : &tensor;
        auto* delta = scenario == 4 ? &different : state;
        auto* output = scenario == 5 ? &different : state;
        const auto dtype = scenario == 1 ? ge::DT_INT32 : ge::DT_FLOAT;
        gert::TilingContextFaker builder;
        builder.SetOpType("ApplyProximalGradientDescent")
            .NodeIoNum(5, 1)
            .IrInstanceNum({1, 1, 1, 1, 1})
            .InputShapes({state, scenario == 6 ? &invalidScalar : &scalar, &scalar, &scalar, delta})
            .OutputShapes({output});
        for (size_t i = 0; i < 5; ++i) {
            builder.NodeInputTd(i, scenario == 2 && i == 1 ? ge::DT_FLOAT16 : dtype,
                                scenario == 3 && i == 1 ? ge::FORMAT_NCHW : ge::FORMAT_ND, ge::FORMAT_ND);
        }
        builder.NodeOutputTd(0, scenario == 9 ? ge::DT_FLOAT16 : dtype,
                             scenario == 10 ? ge::FORMAT_NCHW : ge::FORMAT_ND, ge::FORMAT_ND);
        fe::PlatFormInfos platform;
        ASSERT_TRUE(platform.Init());
        std::map<std::string, std::string> soc{
            {"ai_core_cnt", "64"}, {"vector_core_cnt", "64"}, {"cube_core_cnt", "64"}, {"core_type_list", "AICore"}};
        std::map<std::string, std::string> spec{{"ub_size", scenario == 14 ? "32" : "245760"}};
        platform.SetPlatformRes("SoCInfo", soc);
        platform.SetPlatformRes("AICoreSpec", spec);
        platform.SetCoreNumByCoreType("AICore");
        auto data = gert::TilingData::CreateCap(scenario == 8 ? 1 : 4096);
        auto workspace = gert::ContinuousVector::Create<size_t>(scenario == 7 ? 0 : 1);
        uint64_t compileInfo = 0;
        auto holder = builder.CompileInfo(&compileInfo)
                          .PlatformInfo(scenario == 11 ? nullptr : reinterpret_cast<char*>(&platform))
                          .TilingData(data.get())
                          .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                          .Build();
        auto* context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(context, nullptr);
        if (scenario == 12) {
            holder.GetContext<gert::KernelContext>()->MutableInput(1)->Set(nullptr, nullptr);
            ASSERT_EQ(context->GetInputShape(1), nullptr);
        }
        manager.GetRawErrorMessages();
        const auto status = impl->tiling(context);
        const auto errors = manager.GetRawErrorMessages();
        if (scenario == 0) {
            EXPECT_EQ(status, ge::GRAPH_SUCCESS);
            EXPECT_TRUE(errors.empty());
        } else {
            EXPECT_EQ(status, ge::GRAPH_FAILED);
            ASSERT_EQ(errors.size(), 1U);
            EXPECT_EQ(errors[0].error_id, codes[scenario]);
        }
    }
    manager.GetRawErrorMessages();
    EXPECT_EQ(impl->tiling(nullptr), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
}
