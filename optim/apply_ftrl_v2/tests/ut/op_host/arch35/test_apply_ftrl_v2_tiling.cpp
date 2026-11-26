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
#include <map>
#include <vector>
#include "base/err_mgr.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "../../../../op_kernel/arch35/apply_ftrl_v2_tiling_data.h"

namespace {
void RunCase(const std::vector<int64_t>& varDims, const std::vector<int64_t>& accumDims,
             const std::vector<int64_t>& lrDims, ge::graphStatus expected, const std::string& code,
             const std::map<std::string, std::string>& expectedArgs)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    ASSERT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
    const char* platformJson = R"({"hardware_info":{"BT_SIZE":0,"load3d_constraints":"1",
        "Intrinsic_fix_pipe_l0c2out":false,"Intrinsic_data_move_l12ub":true,
        "Intrinsic_data_move_l0c2ub":true,"Intrinsic_data_move_out2l1_nd2nz":false,
        "UB_SIZE":245760,"L2_SIZE":33554432,"L1_SIZE":524288,"L0A_SIZE":65536,
        "L0B_SIZE":65536,"L0C_SIZE":131072,"CORE_NUM":64}})";
    std::map<std::string, std::string> soc, spec, intrinsics;
    GetPlatFormInfos(platformJson, soc, spec, intrinsics);
    fe::PlatFormInfos platform;
    platform.Init();
    platform.SetPlatformRes("SoCInfo", soc);
    platform.SetPlatformRes("AICoreSpec", spec);
    platform.SetCoreNumByCoreType("AICore");
    platform.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    std::map<std::string, std::string> version = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    platform.SetPlatformRes("version", version);
    auto makeShape = [](const std::vector<int64_t>& dims) {
        gert::StorageShape shape;
        for (auto dim : dims) {
            shape.MutableOriginShape().AppendDim(dim);
            shape.MutableStorageShape().AppendDim(dim);
        }
        return shape;
    };
    auto var = makeShape(varDims);
    auto accum = makeShape(accumDims);
    auto lr = makeShape(lrDims);
    auto scalar = makeShape({});
    auto output = makeShape(varDims);
    auto data = gert::TilingData::CreateCap(4096);
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    ASSERT_NE(data, nullptr);
    ASSERT_NE(workspace, nullptr);
    struct CompileInfo {
    } compileInfo;
    auto faker = gert::TilingContextFaker();
    faker.SetOpType("ApplyFtrlV2")
        .NodeIoNum(9, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
        .InputShapes({&var, &accum, &var, &var, &lr, &scalar, &scalar, &scalar, &scalar})
        .OutputShapes({&output})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .NodeAttrs({{"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .TilingData(data.get())
        .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
    for (int i = 0; i < 9; ++i) {
        faker.NodeInputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    faker.NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    auto holder = faker.Build();
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFtrlV2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(impl->tiling(context), expected);
    auto messages = error_message::GetErrMgrRawErrorMessages();
    if (expected == ge::GRAPH_SUCCESS) {
        EXPECT_TRUE(messages.empty());
        EXPECT_EQ(context->GetTilingData<ApplyFtrlV2TilingData>()->totalElements, 8U);
        EXPECT_GT(context->GetBlockDim(), 0U);
        return;
    }
    ASSERT_EQ(messages.size(), 1U);
    ASSERT_NE(messages[0].error_id, nullptr);
    EXPECT_EQ(std::string(messages[0].error_id.get()), code);
    ASSERT_EQ(messages[0].args_key.size(), expectedArgs.size() + 1U);
    ASSERT_EQ(messages[0].args_key.size(), messages[0].args_value.size());
    std::map<std::string, std::string> args;
    for (size_t i = 0; i < messages[0].args_key.size(); ++i) {
        ASSERT_NE(messages[0].args_key[i], nullptr);
        ASSERT_NE(messages[0].args_value[i], nullptr);
        EXPECT_TRUE(args.emplace(messages[0].args_key[i].get(), messages[0].args_value[i].get()).second);
    }
    EXPECT_FALSE(args.at("op_name").empty());
    for (const auto& entry : expectedArgs) {
        EXPECT_EQ(args.at(entry.first), entry.second);
    }
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}
} // namespace

TEST(ApplyFtrlV2TilingReporting, normal_preserves_execution) { RunCase({2, 4}, {2, 4}, {}, ge::GRAPH_SUCCESS, "", {}); }
TEST(ApplyFtrlV2TilingReporting, rank_limit_preserves_single_root)
{
    const std::vector<int64_t> dims(9, 1);
    RunCase(dims, dims, {}, ge::GRAPH_FAILED, "EZ0012",
            {{"param_name", "var"}, {"incorrect_dim", "9"}, {"reason", "rank exceeds the max supported rank 8"}});
}
TEST(ApplyFtrlV2TilingReporting, shape_relation_preserves_single_root)
{
    RunCase({2, 4}, {4, 2}, {}, ge::GRAPH_FAILED, "EZ0010",
            {{"param_names", "accum and var"},
             {"incorrect_shapes", "(4,2) and (2,4)"},
             {"reason", "shapes of var/accum/linear/grad must be identical; accum shape (4,2) vs var shape (2,4), "
                        "first mismatched dim[0]: var=2, accum=4"}});
}
TEST(ApplyFtrlV2TilingReporting, scalar_contract_preserves_single_root)
{
    RunCase(
        {2, 4}, {2, 4}, {2}, ge::GRAPH_FAILED, "EZ0009",
        {{"param_name", "lr"}, {"incorrect_shape", "(2)"}, {"reason", "must be a scalar (0-d or exactly 1 element)"}});
}
