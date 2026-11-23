/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sync_batch_norm_backward_elemt_tiling.cpp
 * \brief
 */

#include "../../../../op_host/arch35/sync_batch_norm_gather_stats_tiling_arch35.h"
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <string>
#include "log/log.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

using namespace ge;
using namespace ut_util;
using namespace std;

class SyncBatchNormGatherStatsTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SyncBatchNormGatherStatsTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SyncBatchNormGatherStatsTiling TearDown" << std::endl; }
};

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling5)
{
    gert::StorageShape total_sum_shape = {{16, 100}, {16, 100}};
    gert::StorageShape total_square_sum_shape = {{16, 100}, {16, 100}};
    gert::StorageShape sample_count_shape = {{16}, {16}};
    gert::StorageShape mean_shape = {{100}, {100}};
    gert::StorageShape var_shape = {{100}, {100}};
    gert::StorageShape batch_mean_shape = {{100}, {100}};
    gert::StorageShape batch_invstd_shape = {{100}, {100}};
    gert::StorageShape running_mean_update = {{100}, {100}};
    gert::StorageShape running_var_update = {{100}, {100}};
    ge::DataType dataType = ge::DT_FLOAT;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;

    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes(
                          {&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling6)
{
    gert::StorageShape total_sum_shape = {{16, 10000}, {16, 10000}};
    gert::StorageShape total_square_sum_shape = {{16, 10000}, {16, 10000}};
    gert::StorageShape sample_count_shape = {{16}, {16}};
    gert::StorageShape mean_shape = {{10000}, {10000}};
    gert::StorageShape var_shape = {{10000}, {10000}};
    gert::StorageShape batch_mean_shape = {{10000}, {10000}};
    gert::StorageShape batch_invstd_shape = {{10000}, {10000}};
    gert::StorageShape running_mean_update = {{10000}, {10000}};
    gert::StorageShape running_var_update = {{10000}, {10000}};
    ge::DataType dataType = ge::DT_FLOAT;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;

    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes(
                          {&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling7)
{
    gert::StorageShape total_sum_shape = {{16, 100}, {16, 100}};
    gert::StorageShape total_square_sum_shape = {{16, 100}, {16, 100}};
    gert::StorageShape sample_count_shape = {{16}, {16}};
    gert::StorageShape mean_shape = {{100}, {100}};
    gert::StorageShape var_shape = {{100}, {100}};
    gert::StorageShape batch_mean_shape = {{100}, {100}};
    gert::StorageShape batch_invstd_shape = {{100}, {100}};
    gert::StorageShape running_mean_update = {{100}, {100}};
    gert::StorageShape running_var_update = {{100}, {100}};
    ge::DataType dataType = ge::DT_FLOAT16;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;

    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes(
                          {&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling8)
{
    gert::StorageShape total_sum_shape = {{16, 10000}, {16, 10000}};
    gert::StorageShape total_square_sum_shape = {{16, 10000}, {16, 10000}};
    gert::StorageShape sample_count_shape = {{16}, {16}};
    gert::StorageShape mean_shape = {{10000}, {10000}};
    gert::StorageShape var_shape = {{10000}, {10000}};
    gert::StorageShape batch_mean_shape = {{10000}, {10000}};
    gert::StorageShape batch_invstd_shape = {{10000}, {10000}};
    gert::StorageShape running_mean_update = {{10000}, {10000}};
    gert::StorageShape running_var_update = {{10000}, {10000}};
    ge::DataType dataType = ge::DT_FLOAT16;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;

    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes(
                          {&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling9)
{
    gert::StorageShape total_sum_shape = {{5089, 1678}, {5089, 1678}};
    gert::StorageShape total_square_sum_shape = {{5089, 1678}, {5089, 1678}};
    gert::StorageShape sample_count_shape = {{5089}, {5089}};
    gert::StorageShape mean_shape = {{1678}, {1678}};
    gert::StorageShape var_shape = {{1678}, {1678}};
    gert::StorageShape batch_mean_shape = {{1678}, {1678}};
    gert::StorageShape batch_invstd_shape = {{1678}, {1678}};
    gert::StorageShape running_mean_update = {{1678}, {1678}};
    gert::StorageShape running_var_update = {{1678}, {1678}};
    ge::DataType dataType = ge::DT_FLOAT16;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;

    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes(
                          {&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling10)
{
    gert::StorageShape total_sum_shape = {{5089, 1678}, {5089, 1678}};
    gert::StorageShape total_square_sum_shape = {{5089, 1678}, {5089, 1678}};
    gert::StorageShape sample_count_shape = {{5089}, {5089}};
    gert::StorageShape mean_shape = {{1678}, {1678}};
    gert::StorageShape var_shape = {{1678}, {1678}};
    gert::StorageShape batch_mean_shape = {{1678}, {1678}};
    gert::StorageShape batch_invstd_shape = {{1678}, {1678}};
    gert::StorageShape running_mean_update = {{1678}, {1678}};
    gert::StorageShape running_var_update = {{1678}, {1678}};
    ge::DataType dataType = ge::DT_FLOAT;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;

    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes(
                          {&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

namespace {
class TestableSyncBatchNormGatherStatsTiling : public optiling::SyncBatchNormGatherStatsTiling {
public:
    using SyncBatchNormGatherStatsTiling::BatchInvstdCheck;
    using SyncBatchNormGatherStatsTiling::BatchMeanCheck;
    using SyncBatchNormGatherStatsTiling::MeanShapeCheck;
    using SyncBatchNormGatherStatsTiling::RunningMeanCheck;
    using SyncBatchNormGatherStatsTiling::RunningVarCheck;
    using SyncBatchNormGatherStatsTiling::SampleCountShapeCheck;
    using SyncBatchNormGatherStatsTiling::SyncBatchNormGatherStatsTiling;
    using SyncBatchNormGatherStatsTiling::TotalSquareSumShapeCheck;
    using SyncBatchNormGatherStatsTiling::TotalSumShapeCheck;
    using SyncBatchNormGatherStatsTiling::ValidateInputShapes;
    using SyncBatchNormGatherStatsTiling::ValidateOutputShapes;
    using SyncBatchNormGatherStatsTiling::VarShapeCheck;
};

constexpr int32_t INPUT_NUM = 5;
constexpr int32_t OUTPUT_NUM = 4;

gert::TilingContextFaker BuildTilingContext(const std::vector<gert::StorageShape*>& inputs,
                                            const std::vector<gert::StorageShape*>& outputs,
                                            const std::vector<ge::DataType>& input_dtypes,
                                            optiling::SyncBatchNormGatherStatsCompileInfo& compile_info,
                                            fe::PlatFormInfos& platform_info, const void* tiling_data,
                                            const gert::ContinuousVector* workspace)
{
    gert::TilingContextFaker faker;
    faker.SetOpType("SyncBatchNormGatherStats").NodeIoNum(inputs.size(), outputs.size());
    if (!inputs.empty()) {
        faker.IrInstanceNum(std::vector<uint32_t>(inputs.size(), 1U));
    }
    faker.InputShapes(inputs).OutputShapes(outputs);
    faker.CompileInfo(&compile_info).PlatformInfo(reinterpret_cast<char*>(&platform_info));
    for (size_t i = 0; i < inputs.size(); ++i) {
        faker.NodeInputTd(static_cast<int32_t>(i), input_dtypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        faker.NodeOutputTd(static_cast<int32_t>(i), ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    faker.TilingData(tiling_data).Workspace(workspace);
    return faker;
}

// 逐个张量直接驱动对应的校验函数，单点回归（某一处判空被改坏）也能被用例捕获
bool RunInputCheck(TestableSyncBatchNormGatherStatsTiling& tiling, int32_t index)
{
    switch (index) {
        case 0:
            return tiling.TotalSumShapeCheck();
        case 1:
            return tiling.TotalSquareSumShapeCheck();
        case 2:
            return tiling.SampleCountShapeCheck();
        case 3:
            return tiling.MeanShapeCheck();
        case 4:
            return tiling.VarShapeCheck();
        default:
            return false;
    }
}

bool RunOutputCheck(TestableSyncBatchNormGatherStatsTiling& tiling, int32_t index)
{
    switch (index) {
        case 0:
            return tiling.BatchMeanCheck();
        case 1:
            return tiling.BatchInvstdCheck();
        case 2:
            return tiling.RunningMeanCheck();
        case 3:
            return tiling.RunningVarCheck();
        default:
            return false;
    }
}
} // namespace

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTilingNullDescRejected)
{
    std::vector<gert::StorageShape> input_shapes = {
        {{16, 100}, {16, 100}}, {{16, 100}, {16, 100}}, {{16}, {16}}, {{100}, {100}}, {{100}, {100}}};
    std::vector<ge::DataType> input_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<gert::StorageShape> output_shapes = {{{100}, {100}}, {{100}, {100}}, {{100}, {100}}, {{100}, {100}}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    std::vector<gert::StorageShape*> all_inputs;
    std::vector<gert::StorageShape*> all_outputs;
    for (auto& shape : input_shapes) {
        all_inputs.push_back(&shape);
    }
    for (auto& shape : output_shapes) {
        all_outputs.push_back(&shape);
    }

    // 完整描述符/形状：9 个校验函数与两级聚合校验全部通过
    {
        auto faker = BuildTilingContext(all_inputs, all_outputs, input_dtypes, compile_info, platform_info, param.get(),
                                        ws_size);
        auto holder = faker.Build();
        gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(context, nullptr);
        TestableSyncBatchNormGatherStatsTiling tiling(context);
        for (int32_t i = 0; i < INPUT_NUM; ++i) {
            EXPECT_TRUE(RunInputCheck(tiling, i)) << "input index " << i;
        }
        for (int32_t j = 0; j < OUTPUT_NUM; ++j) {
            EXPECT_TRUE(RunOutputCheck(tiling, j)) << "output index " << j;
        }
        EXPECT_TRUE(tiling.ValidateInputShapes());
        EXPECT_TRUE(tiling.ValidateOutputShapes());
    }

    // 逐个输入截断：第 n 个输入的 Desc/Shape 同时为空，Desc 判空分支必须拦截
    // （TilingContextFaker 无法构造 0 输入上下文，故从 n=1 起；第 0 个输入的 Shape 判空见 NullShapeRejected 用例）
    for (int32_t n = 1; n < INPUT_NUM; ++n) {
        std::vector<gert::StorageShape*> inputs(all_inputs.begin(), all_inputs.begin() + n);
        auto faker = BuildTilingContext(inputs, all_outputs, input_dtypes, compile_info, platform_info, param.get(),
                                        ws_size);
        auto holder = faker.Build();
        gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->GetInputDesc(static_cast<size_t>(n)), nullptr);
        ASSERT_EQ(context->GetInputShape(static_cast<size_t>(n)), nullptr);
        TestableSyncBatchNormGatherStatsTiling tiling(context);
        EXPECT_FALSE(RunInputCheck(tiling, n)) << "input index " << n;
        EXPECT_FALSE(tiling.ValidateInputShapes());
    }

    // 逐个输出截断：第 m 个输出的 Desc/Shape 同时为空，Desc 判空分支必须拦截
    for (int32_t m = 1; m < OUTPUT_NUM; ++m) {
        std::vector<gert::StorageShape*> outputs(all_outputs.begin(), all_outputs.begin() + m);
        auto faker = BuildTilingContext(all_inputs, outputs, input_dtypes, compile_info, platform_info, param.get(),
                                        ws_size);
        auto holder = faker.Build();
        gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(context->GetOutputDesc(static_cast<size_t>(m)), nullptr);
        ASSERT_EQ(context->GetOutputShape(static_cast<size_t>(m)), nullptr);
        TestableSyncBatchNormGatherStatsTiling tiling(context);
        EXPECT_FALSE(RunOutputCheck(tiling, m)) << "output index " << m;
        EXPECT_FALSE(tiling.ValidateOutputShapes());
    }
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTilingNullShapeRejected)
{
    std::vector<gert::StorageShape> input_shapes = {
        {{16, 100}, {16, 100}}, {{16, 100}, {16, 100}}, {{16}, {16}}, {{100}, {100}}, {{100}, {100}}};
    std::vector<ge::DataType> input_dtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<gert::StorageShape> output_shapes = {{{100}, {100}}, {{100}, {100}}, {{100}, {100}}, {{100}, {100}}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    std::vector<gert::StorageShape*> all_inputs;
    std::vector<gert::StorageShape*> all_outputs;
    for (auto& shape : input_shapes) {
        all_inputs.push_back(&shape);
    }
    for (auto& shape : output_shapes) {
        all_outputs.push_back(&shape);
    }

    // Desc 非空、Shape 为空：逐个覆盖 9 处 shape 判空的失败路径。
    // TilingContextFaker 无法单独构造空 Shape（Desc/Shape 取自同一组 IO 槽位），
    // 故在完整上下文上把目标槽位的 Shape 指针置空（Desc 来自 ComputeNodeInfo，保持非空）。
    for (int32_t i = 0; i < INPUT_NUM; ++i) {
        auto faker = BuildTilingContext(all_inputs, all_outputs, input_dtypes, compile_info, platform_info, param.get(),
                                        ws_size);
        auto holder = faker.Build();
        gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
        gert::KernelContext* kernel_context = holder.GetContext<gert::KernelContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_NE(kernel_context, nullptr);
        gert::Chain* chain = kernel_context->MutableInput(static_cast<size_t>(i));
        ASSERT_NE(chain, nullptr);
        ASSERT_FALSE(chain->HasDeleter());
        chain->Set(nullptr, nullptr);
        ASSERT_NE(context->GetInputDesc(static_cast<size_t>(i)), nullptr);
        ASSERT_EQ(context->GetInputShape(static_cast<size_t>(i)), nullptr);
        TestableSyncBatchNormGatherStatsTiling tiling(context);
        EXPECT_FALSE(RunInputCheck(tiling, i)) << "input index " << i;
        EXPECT_FALSE(tiling.ValidateInputShapes());
    }

    // 输出 Shape 为空（输出槽位位于输入之后：INPUT_NUM + j）
    for (int32_t j = 0; j < OUTPUT_NUM; ++j) {
        auto faker = BuildTilingContext(all_inputs, all_outputs, input_dtypes, compile_info, platform_info, param.get(),
                                        ws_size);
        auto holder = faker.Build();
        gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
        gert::KernelContext* kernel_context = holder.GetContext<gert::KernelContext>();
        ASSERT_NE(context, nullptr);
        ASSERT_NE(kernel_context, nullptr);
        gert::Chain* chain = kernel_context->MutableInput(static_cast<size_t>(INPUT_NUM + j));
        ASSERT_NE(chain, nullptr);
        ASSERT_FALSE(chain->HasDeleter());
        chain->Set(nullptr, nullptr);
        ASSERT_NE(context->GetOutputDesc(static_cast<size_t>(j)), nullptr);
        ASSERT_EQ(context->GetOutputShape(static_cast<size_t>(j)), nullptr);
        TestableSyncBatchNormGatherStatsTiling tiling(context);
        EXPECT_FALSE(RunOutputCheck(tiling, j)) << "output index " << j;
        EXPECT_FALSE(tiling.ValidateOutputShapes());
    }
}
