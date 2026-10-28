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
 * \file test_batch_norm_grad_v3_tiling.cpp
 * \brief
 */

#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <tuple>
#include <utility>
#include <gtest/gtest.h>
#include "securec.h"
// #include "test_common.h"
// #include "pad_ops.h"
#include "array_ops.h"
#include "log/log.h"
#include "tests/ut/common/ut_op_util.h"
#include "platform/platform_infos_def.h"
// #include "op_tiling/op_tiling_util.h"
// #include "common_unittest.h"
#include "norm/batch_norm_grad_v3/op_host/batch_norm_grad_v3_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
// #include "experiment_ops.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class BatchNormGradV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchNormGradV3Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchNormGradV3Tiling TearDown" << std::endl; }
};

template <typename T>
static string to_string(void* buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_full_load_fp32)
{
    gert::StorageShape dy_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape x_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape weight_shape = {{2048}, {2048}};
    gert::StorageShape running_mean_shape = {{2048}, {2048}};
    gert::StorageShape running_var_shape = {{2048}, {2048}};
    gert::StorageShape save_mean_shape = {{2048}, {2048}};
    gert::StorageShape save_rstd_shape = {{2048}, {2048}};
    gert::StorageShape dx_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape dweight_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
              "4 0 2048 0 225 0 904 0 64 0 0 0 32 0 33 0 512 0 0 0 8 0 16 0 2 0 16 0 3 0 1 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_full_load_fp16)
{
    gert::StorageShape dy_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape x_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape weight_shape = {{2048}, {2048}};
    gert::StorageShape running_mean_shape = {{2048}, {2048}};
    gert::StorageShape running_var_shape = {{2048}, {2048}};
    gert::StorageShape save_mean_shape = {{2048}, {2048}};
    gert::StorageShape save_rstd_shape = {{2048}, {2048}};
    gert::StorageShape dx_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape dweight_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
              "4 0 2048 0 225 0 912 0 64 0 0 0 32 0 33 0 512 0 0 0 8 0 22 0 2 0 10 0 2 0 11 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_full_load_bf16)
{
    gert::StorageShape dy_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape x_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape weight_shape = {{2048}, {2048}};
    gert::StorageShape running_mean_shape = {{2048}, {2048}};
    gert::StorageShape running_var_shape = {{2048}, {2048}};
    gert::StorageShape save_mean_shape = {{2048}, {2048}};
    gert::StorageShape save_rstd_shape = {{2048}, {2048}};
    gert::StorageShape dx_shape = {{4, 2048, 15, 15}, {4, 2048, 15, 15}};
    gert::StorageShape dweight_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
              "4 0 2048 0 225 0 912 0 64 0 0 0 32 0 33 0 512 0 0 0 8 0 22 0 2 0 10 0 2 0 11 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_recompute_split_r1_fp32)
{
    gert::StorageShape dy_shape = {{4, 2048, 1, 5000}, {4, 2048, 1, 5000}};
    gert::StorageShape x_shape = {{4, 2048, 1, 5000}, {4, 2048, 1, 5000}};
    gert::StorageShape weight_shape = {{2048}, {2048}};
    gert::StorageShape running_mean_shape = {{2048}, {2048}};
    gert::StorageShape running_var_shape = {{2048}, {2048}};
    gert::StorageShape save_mean_shape = {{2048}, {2048}};
    gert::StorageShape save_rstd_shape = {{2048}, {2048}};
    gert::StorageShape dx_shape = {{4, 2048, 1, 5000}, {4, 2048, 1, 5000}};
    gert::StorageShape dweight_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 31000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
        "4 0 2048 0 5000 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 10000 0 10000 0 2 0 0 0 5000 0 5000 "
        "0 0 0 0 0 5000 0 5000 0 2 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_recompute_split_r1_fp16)
{
    gert::StorageShape dy_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape x_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape weight_shape = {{2048}, {2048}};
    gert::StorageShape running_mean_shape = {{2048}, {2048}};
    gert::StorageShape running_var_shape = {{2048}, {2048}};
    gert::StorageShape save_mean_shape = {{2048}, {2048}};
    gert::StorageShape save_rstd_shape = {{2048}, {2048}};
    gert::StorageShape dx_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape dweight_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 31000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
        "8 0 2048 0 5000 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 10000 0 10000 0 4 0 0 0 5000 0 5000 "
        "0 0 0 0 0 5000 0 5000 0 2 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_recompute_split_r1_bf16)
{
    gert::StorageShape dy_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape x_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape weight_shape = {{2048}, {2048}};
    gert::StorageShape running_mean_shape = {{2048}, {2048}};
    gert::StorageShape running_var_shape = {{2048}, {2048}};
    gert::StorageShape save_mean_shape = {{2048}, {2048}};
    gert::StorageShape save_rstd_shape = {{2048}, {2048}};
    gert::StorageShape dx_shape = {{8, 2048, 1, 5000}, {8, 2048, 1, 5000}};
    gert::StorageShape dweight_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 31000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
        "8 0 2048 0 5000 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 10000 0 10000 0 4 0 0 0 5000 0 5000 "
        "0 0 0 0 0 5000 0 5000 0 2 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_recompute_split_r0_fp32)
{
    gert::StorageShape dy_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape x_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape weight_shape = {{2048}, {2048}};
    gert::StorageShape running_mean_shape = {{2048}, {2048}};
    gert::StorageShape running_var_shape = {{2048}, {2048}};
    gert::StorageShape save_mean_shape = {{2048}, {2048}};
    gert::StorageShape save_rstd_shape = {{2048}, {2048}};
    gert::StorageShape dx_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape dweight_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 32000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
        "4 0 2048 0 89280 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 14848 0 14848 0 4 0 29888 0 7424 0 "
        "7424 0 4 0 192 0 7424 0 7424 0 2 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_recompute_split_r0_fp16)
{
    gert::StorageShape dy_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape x_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape weight_shape = {{2048}, {2048}};
    gert::StorageShape running_mean_shape = {{2048}, {2048}};
    gert::StorageShape running_var_shape = {{2048}, {2048}};
    gert::StorageShape save_mean_shape = {{2048}, {2048}};
    gert::StorageShape save_rstd_shape = {{2048}, {2048}};
    gert::StorageShape dx_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape dweight_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 32000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
        "4 0 2048 0 89280 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 14848 0 14848 0 4 0 29888 0 7424 0 "
        "7424 0 4 0 192 0 7424 0 7424 0 2 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_recompute_split_r0_bf16)
{
    gert::StorageShape dy_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape x_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape weight_shape = {{2048}, {2048}};
    gert::StorageShape running_mean_shape = {{2048}, {2048}};
    gert::StorageShape running_var_shape = {{2048}, {2048}};
    gert::StorageShape save_mean_shape = {{2048}, {2048}};
    gert::StorageShape save_rstd_shape = {{2048}, {2048}};
    gert::StorageShape dx_shape = {{4, 2048, 6, 14880}, {4, 2048, 6, 14880}};
    gert::StorageShape dweight_shape = {{2048}, {2048}};
    gert::StorageShape dbias_shape = {{2048}, {2048}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 32000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
        "4 0 2048 0 89280 0 0 0 64 0 0 0 32 0 33 0 8192 0 1 0 64 0 4096 0 0 0 64 0 14848 0 14848 0 4 0 29888 0 7424 0 "
        "7424 0 4 0 192 0 7424 0 7424 0 2 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_split_core_r1_fp32)
{
    gert::StorageShape dy_shape = {{400, 4, 15, 15000}, {400, 4, 15, 15000}};
    gert::StorageShape x_shape = {{400, 4, 15, 15000}, {400, 4, 15, 15000}};
    gert::StorageShape weight_shape = {{4}, {4}};
    gert::StorageShape running_mean_shape = {{4}, {4}};
    gert::StorageShape running_var_shape = {{4}, {4}};
    gert::StorageShape save_mean_shape = {{4}, {4}};
    gert::StorageShape save_rstd_shape = {{4}, {4}};
    gert::StorageShape dx_shape = {{400, 4, 15, 15000}, {400, 4, 15, 15000}};
    gert::StorageShape dweight_shape = {{4}, {4}};
    gert::StorageShape dbias_shape = {{4}, {4}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 400 0 4 0 8 0 225000 0 58 0 7 0 1 0 12168 0 19 0 5976 0 5976 0 1 0 7 0 1 0 1 "
        "0 1 0 1 0 8 0 4 0 1 0 8 0 1 0 4 0 10224 0 23 0 72 0 72 0 1 0 7 0 1 0 1 0 1 0 1 0 8 0 4 0 1 0 128 0 5 0 8 0 7 "
        "0 16 0 3 0 4 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_split_core_r1_fp16)
{
    gert::StorageShape dy_shape = {{86000, 2, 15, 15}, {86000, 2, 15, 15}};
    gert::StorageShape x_shape = {{86000, 2, 15, 15}, {86000, 2, 15, 15}};
    gert::StorageShape weight_shape = {{2}, {2}};
    gert::StorageShape running_mean_shape = {{2}, {2}};
    gert::StorageShape running_var_shape = {{2}, {2}};
    gert::StorageShape save_mean_shape = {{2}, {2}};
    gert::StorageShape save_rstd_shape = {{2}, {2}};
    gert::StorageShape dx_shape = {{86000, 2, 15, 15}, {86000, 2, 15, 15}};
    gert::StorageShape dweight_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 86000 0 2 0 8 0 225 0 64 0 1344 0 1328 0 240 0 1 0 225 0 240 0 63 0 22 0 21 0 "
        "22 0 5 0 1 0 8 0 2 0 1 0 16 0 1 0 2 0 240 0 1 0 225 0 240 0 85 0 16 0 69 0 16 0 53 0 1 0 8 0 2 0 1 0 16 0 6 0 "
        "5 0 4 0 16 0 6 0 4 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_ra_split_r1_bf16)
{
    gert::StorageShape dy_shape = {{120, 20, 60, 68}, {120, 20, 60, 68}};
    gert::StorageShape x_shape = {{120, 20, 60, 68}, {120, 20, 60, 68}};
    gert::StorageShape weight_shape = {{68}, {68}};
    gert::StorageShape running_mean_shape = {{68}, {68}};
    gert::StorageShape running_var_shape = {{68}, {68}};
    gert::StorageShape save_mean_shape = {{68}, {68}};
    gert::StorageShape save_rstd_shape = {{68}, {68}};
    gert::StorageShape dx_shape = {{120, 20, 60, 68}, {120, 20, 60, 68}};
    gert::StorageShape dweight_shape = {{68}, {68}};
    gert::StorageShape dbias_shape = {{68}, {68}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 50000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
        "144000 0 68 0 64 0 64 0 2250 0 2250 0 36 0 32 0 10 0 36 0 32 0 10 0 68 0 80 0 68 0 1 0 252 0 234 0 9 0 252 0 "
        "234 0 9 0 6 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_infer_ba)
{
    gert::StorageShape dy_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape weight_shape = {{4}, {4}};
    gert::StorageShape running_mean_shape = {{4}, {4}};
    gert::StorageShape running_var_shape = {{4}, {4}};
    gert::StorageShape save_mean_shape = {{4}, {4}};
    gert::StorageShape save_rstd_shape = {{4}, {4}};
    gert::StorageShape dx_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape dweight_shape = {{4}, {4}};
    gert::StorageShape dbias_shape = {{4}, {4}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 900000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
              "1 0 1 0 1 0 4 0 1 0 1 0 128 0 4 0 124 0 30 0 30 0 925353388 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_infer_bab)
{
    gert::StorageShape dy_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape weight_shape = {{2}, {2}};
    gert::StorageShape running_mean_shape = {{2}, {2}};
    gert::StorageShape running_var_shape = {{2}, {2}};
    gert::StorageShape save_mean_shape = {{2}, {2}};
    gert::StorageShape save_rstd_shape = {{2}, {2}};
    gert::StorageShape dx_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape dweight_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 910000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
              "1 0 1 0 1 0 5 0 2 0 12 0 1 0 1 0 1 0 5 0 5 0 2 0 2 0 128 0 12 0 0 0 925353388 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_ra_full_load)
{
    gert::StorageShape dy_shape = {{4, 1637, 1, 1}, {4, 1637, 1, 1}};
    gert::StorageShape x_shape = {{4, 1637, 1, 1}, {4, 1637, 1, 1}};
    gert::StorageShape weight_shape = {{1637}, {1637}};
    gert::StorageShape running_mean_shape = {{1637}, {1637}};
    gert::StorageShape running_var_shape = {{1637}, {1637}};
    gert::StorageShape save_mean_shape = {{1637}, {1637}};
    gert::StorageShape save_rstd_shape = {{1637}, {1637}};
    gert::StorageShape dx_shape = {{4, 1637, 1, 1}, {4, 1637, 1, 1}};
    gert::StorageShape dweight_shape = {{1637}, {1637}};
    gert::StorageShape dbias_shape = {{1637}, {1637}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 20000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
              "4 0 1637 0 1 0 0 0 64 0 37 0 25 0 26 0 0 0 0 0 0 0 26 0 64 0 37 0 25 0 1 0 64 0 64 0 37 0 64 0 1 0 2 0 "
              "0 0 2 0 0 0 2 0 0 0 1 0 ");
}

TEST_F(BatchNormGradV3Tiling, batch_norm_grad_v3_ra_recompute)
{
    gert::StorageShape dy_shape = {{8, 1, 136, 2880}, {8, 1, 136, 2880}};
    gert::StorageShape x_shape = {{8, 1, 136, 2880}, {8, 1, 136, 2880}};
    gert::StorageShape weight_shape = {{2880}, {2880}};
    gert::StorageShape running_mean_shape = {{2880}, {2880}};
    gert::StorageShape running_var_shape = {{2880}, {2880}};
    gert::StorageShape save_mean_shape = {{2880}, {2880}};
    gert::StorageShape save_rstd_shape = {{2880}, {2880}};
    gert::StorageShape dx_shape = {{8, 1, 136, 2880}, {8, 1, 136, 2880}};
    gert::StorageShape dweight_shape = {{2880}, {2880}};
    gert::StorageShape dbias_shape = {{2880}, {2880}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64, "socVersion": "Ascend950"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NHWC, ge::FORMAT_NHWC)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 40000000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_GE(tilingData->GetDataSize(), sizeof(int64_t));
    int64_t outputMask = 0;
    std::memcpy(&outputMask,
                static_cast<const uint8_t*>(tilingData->GetData()) + tilingData->GetDataSize() - sizeof(outputMask),
                sizeof(outputMask));
    EXPECT_EQ(outputMask, 7);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize() - sizeof(int64_t)),
              "1088 0 2880 0 1 0 0 0 64 0 0 0 45 0 46 0 0 0 0 0 0 0 64 0 45 0 0 0 64 0 0 0 64 0 64 0 64 0 17 0 0 0 16 "
              "0 17 0 0 0 5 0 980480241 0 ");
}

// A2/A3 must keep their original tiling keys and complete serialized layouts,
// both without output_mask (legacy L0) and with each shared-OpDef mask value.
// Explicit 910B/910_93 platform information makes CANN reserve 256 bytes of UB;
// serialized expectations below use 196352 available bytes from 196608 physical bytes.
using BatchNormGradV3LegacyParam = std::tuple<std::string, int64_t>;

class BatchNormGradV3LegacyTiling : public testing::TestWithParam<BatchNormGradV3LegacyParam> {
protected:
    std::map<std::string, std::string>& LegacyPlatform()
    {
        legacyPlatform_ = {{"Short_SoC_version", std::get<0>(GetParam())}, {"NpuArch", "2201"}};
        return legacyPlatform_;
    }

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> LegacyAttrs(bool training) const
    {
        std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
            {"is_training", Ops::NN::AnyValue::CreateFrom<bool>(training)},
            {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)}};
        auto bits = std::get<1>(GetParam());
        if (bits >= 0) {
            std::vector<bool> mask(optiling::BNG_V3_OUTPUT_MASK_NUM);
            for (size_t i = 0; i < mask.size(); ++i) {
                mask[i] = (bits & (int64_t{1} << i)) != 0;
            }
            attrs.emplace_back("output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(mask));
        }
        return attrs;
    }

private:
    std::map<std::string, std::string> legacyPlatform_;
};

INSTANTIATE_TEST_SUITE_P(LegacyPlatformsAndMasks, BatchNormGradV3LegacyTiling,
                         testing::Combine(testing::Values(std::string("Ascend910B"), std::string("Ascend910_93")),
                                          testing::Range(int64_t{-1}, optiling::BNG_V3_OUTPUT_MASK_ALL + 1)),
                         [](const testing::TestParamInfo<BatchNormGradV3LegacyParam>& info) {
                             auto soc = std::get<0>(info.param) == "Ascend910B" ? "A2_" : "A3_";
                             auto bits = std::get<1>(info.param);
                             return std::string(soc) + (bits < 0 ? "Default" : "Mask" + std::to_string(bits));
                         });

TEST_P(BatchNormGradV3LegacyTiling, batch_norm_grad_v3_legacy_infer_full_load)
{
    gert::StorageShape dy_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape weight_shape = {{2}, {2}};
    gert::StorageShape running_mean_shape = {{2}, {2}};
    gert::StorageShape running_var_shape = {{2}, {2}};
    gert::StorageShape save_mean_shape = {{2}, {2}};
    gert::StorageShape save_rstd_shape = {{2}, {2}};
    gert::StorageShape dx_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape dweight_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            LegacyPlatform());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(LegacyAttrs(is_training))
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", LegacyPlatform());
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 10001);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
              "12 0 2 0 5 0 80 0 1 0 1 0 182 0 2 0 925353388 0 ");
}

TEST_P(BatchNormGradV3LegacyTiling, batch_norm_grad_v3_legacy_infer_split_load)
{
    gert::StorageShape dy_shape = {{50, 40, 30, 40}, {50, 40, 30, 40}};
    gert::StorageShape x_shape = {{50, 40, 30, 40}, {50, 40, 30, 40}};
    gert::StorageShape weight_shape = {{40}, {40}};
    gert::StorageShape running_mean_shape = {{40}, {40}};
    gert::StorageShape running_var_shape = {{40}, {40}};
    gert::StorageShape save_mean_shape = {{40}, {40}};
    gert::StorageShape save_rstd_shape = {{40}, {40}};
    gert::StorageShape dx_shape = {{50, 40, 30, 40}, {50, 40, 30, 40}};
    gert::StorageShape dweight_shape = {{40}, {40}};
    gert::StorageShape dbias_shape = {{40}, {40}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            LegacyPlatform());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(LegacyAttrs(is_training))
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", LegacyPlatform());
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 20002);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
              "1200 0 40 0 50 0 60000 0 1 0 1 0 12264 0 10 0 5 0 0 0 10 0 40 0 925353388 0 ");
}

TEST_P(BatchNormGradV3LegacyTiling, batch_norm_grad_v3_legacy_infer_split_load_cross_core)
{
    gert::StorageShape dy_shape = {{50, 2, 30, 40}, {50, 2, 30, 40}};
    gert::StorageShape x_shape = {{50, 2, 30, 40}, {50, 2, 30, 40}};
    gert::StorageShape weight_shape = {{2}, {2}};
    gert::StorageShape running_mean_shape = {{2}, {2}};
    gert::StorageShape running_var_shape = {{2}, {2}};
    gert::StorageShape save_mean_shape = {{2}, {2}};
    gert::StorageShape save_rstd_shape = {{2}, {2}};
    gert::StorageShape dx_shape = {{50, 2, 30, 40}, {50, 2, 30, 40}};
    gert::StorageShape dweight_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = false;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            LegacyPlatform());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(LegacyAttrs(is_training))
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", LegacyPlatform());
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 20012);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "1200 0 2 0 50 0 60000 0 0 0 0 0 12272 0 10 0 1 0 1 0 0 0 3 0 2 0 34 0 17 0 3 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 5 0 10 0 40 0 925353388 0 ");
}

TEST_P(BatchNormGradV3LegacyTiling, batch_norm_grad_v3_legacy_train_full_load)
{
    gert::StorageShape dy_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape x_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape weight_shape = {{2}, {2}};
    gert::StorageShape running_mean_shape = {{2}, {2}};
    gert::StorageShape running_var_shape = {{2}, {2}};
    gert::StorageShape save_mean_shape = {{2}, {2}};
    gert::StorageShape save_rstd_shape = {{2}, {2}};
    gert::StorageShape dx_shape = {{5, 2, 3, 4}, {5, 2, 3, 4}};
    gert::StorageShape dweight_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            LegacyPlatform());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(LegacyAttrs(is_training))
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", LegacyPlatform());
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 30001);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
              "12 0 2 0 5 0 80 0 1 0 1 0 136 0 2 0 925353388 0 ");
}

TEST_P(BatchNormGradV3LegacyTiling, batch_norm_grad_v3_legacy_train_split_load)
{
    gert::StorageShape dy_shape = {{50, 40, 30, 40}, {50, 40, 30, 40}};
    gert::StorageShape x_shape = {{50, 40, 30, 40}, {50, 40, 30, 40}};
    gert::StorageShape weight_shape = {{40}, {40}};
    gert::StorageShape running_mean_shape = {{40}, {40}};
    gert::StorageShape running_var_shape = {{40}, {40}};
    gert::StorageShape save_mean_shape = {{40}, {40}};
    gert::StorageShape save_rstd_shape = {{40}, {40}};
    gert::StorageShape dx_shape = {{50, 40, 30, 40}, {50, 40, 30, 40}};
    gert::StorageShape dweight_shape = {{40}, {40}};
    gert::StorageShape dbias_shape = {{40}, {40}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_NE(tiling_func, nullptr);
    ASSERT_NE(tiling_parse_func, nullptr);

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            LegacyPlatform());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(LegacyAttrs(is_training))
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", LegacyPlatform());
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 40002);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
              "1200 0 40 0 50 0 60000 0 1 0 1 0 12264 0 10 0 5 0 0 0 10 0 40 0 925353388 0 ");
}

TEST_P(BatchNormGradV3LegacyTiling, batch_norm_grad_v3_legacy_train_split_load_corss_core)
{
    gert::StorageShape dy_shape = {{50, 2, 30, 40}, {50, 2, 30, 40}};
    gert::StorageShape x_shape = {{50, 2, 30, 40}, {50, 2, 30, 40}};
    gert::StorageShape weight_shape = {{2}, {2}};
    gert::StorageShape running_mean_shape = {{2}, {2}};
    gert::StorageShape running_var_shape = {{2}, {2}};
    gert::StorageShape save_mean_shape = {{2}, {2}};
    gert::StorageShape save_rstd_shape = {{2}, {2}};
    gert::StorageShape dx_shape = {{50, 2, 30, 40}, {50, 2, 30, 40}};
    gert::StorageShape dweight_shape = {{2}, {2}};
    gert::StorageShape dbias_shape = {{2}, {2}};
    float eps = 1e-5;
    bool is_training = true;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_NE(tiling_func, nullptr);
    ASSERT_NE(tiling_parse_func, nullptr);

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            LegacyPlatform());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape,
                                    &save_mean_shape, &save_rstd_shape})
                      .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(LegacyAttrs(is_training))
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", LegacyPlatform());
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 40012);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(
        to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
        "1200 0 2 0 50 0 60000 0 0 0 0 0 12272 0 10 0 1 0 1 0 0 0 3 0 2 0 34 0 17 0 3 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 5 0 10 0 40 0 925353388 0 ");
}

// ra_split_r fused 路径 tilingKey 派发与合计 UB 兜底用例(NCHW (N,C,1,1) HW=1×1)
// 50000000 generic / 51000000 fused-single / 52000000 fused-pair
namespace {
constexpr uint32_t TEST_UB_BYTES_248_KIB = 253952;
constexpr uint32_t TEST_UB_BYTES_240_KIB = 245760;
constexpr uint32_t TEST_AIV_CORE_NUM_56 = 56;
constexpr uint32_t TEST_AIV_CORE_NUM_64 = 64;
constexpr uint64_t RA_FULL_LOAD_KEY = 20000000;
constexpr uint64_t RA_PACKED_FULL_LOAD_KEY = 21000000;
constexpr uint64_t RA_RECOMPUTE_KEY = 40000000;
constexpr uint64_t RA_SPLIT_R_GENERIC_KEY = 50000000;
constexpr uint64_t RA_SPLIT_R_FUSED_SINGLE_KEY = 51000000;
constexpr uint64_t RA_SPLIT_R_FUSED_PAIR_KEY = 52000000;
constexpr uint64_t RA_SPLIT_R_PACKED_KEY = 53000000;
} // namespace

static void RunSplitRTilingKeyCase(int64_t n, int64_t c, ge::DataType dyDtype, uint32_t ubSize, uint32_t coreNum,
                                   uint64_t expectedKey, bool expectNotSplitR = false,
                                   const std::vector<bool>* outputMask = nullptr,
                                   const std::vector<int64_t>* expectedFullLoadSplit = nullptr,
                                   ge::DataType weightDtype = ge::DT_FLOAT,
                                   const std::vector<int64_t>* expectedFullLoadReduction = nullptr)
{
    const std::string trace = "R=" + std::to_string(n) + ", C=" + std::to_string(c) +
                              ", dtype=" + std::to_string(static_cast<int>(dyDtype)) +
                              ", weight=" + std::to_string(static_cast<int>(weightDtype)) +
                              ", UB=" + std::to_string(ubSize) + ", cores=" + std::to_string(coreNum);
    SCOPED_TRACE(trace);
    gert::StorageShape dy_shape = {{n, c, 1, 1}, {n, c, 1, 1}};
    gert::StorageShape x_shape = {{n, c, 1, 1}, {n, c, 1, 1}};
    gert::StorageShape weight_shape = {{c}, {c}};
    gert::StorageShape running_mean_shape = {{c}, {c}};
    gert::StorageShape running_var_shape = {{c}, {c}};
    gert::StorageShape save_mean_shape = {{c}, {c}};
    gert::StorageShape save_rstd_shape = {{c}, {c}};
    gert::StorageShape dx_shape = {{n, c, 1, 1}, {n, c, 1, 1}};
    gert::StorageShape dweight_shape = {{c}, {c}};
    gert::StorageShape dbias_shape = {{c}, {c}};

    std::string compile_info_string =
        std::string(
            R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
                "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": )") +
        std::to_string(ubSize) +
        R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                "CORE_NUM": )" +
        std::to_string(coreNum) + R"(, "socVersion": "Ascend950"}})";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::BatchNormGradV3CompileInfo compile_info;

    std::string op_type("BatchNormGradV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    gert::TilingContextFaker faker;
    faker.SetOpType(op_type)
        .NodeIoNum(7, 3)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
        .InputShapes({&dy_shape, &x_shape, &weight_shape, &running_mean_shape, &running_var_shape, &save_mean_shape,
                      &save_rstd_shape})
        .OutputShapes({&dx_shape, &dweight_shape, &dbias_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, dyDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(1, dyDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(2, weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, dyDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeOutputTd(1, weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                    {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
        .TilingData(param.get())
        .Workspace(ws_size);
    if (outputMask != nullptr) {
        faker.NodeAttrs({{"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(*outputMask)}});
    }
    auto holder = faker.Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    if (outputMask != nullptr && outputMask->size() != 3) {
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
        return;
    }
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto raw = tiling_context->GetRawTilingData();
    int64_t actualMask = 0;
    ASSERT_GE(raw->GetDataSize(), sizeof(actualMask));
    ASSERT_EQ(memcpy_s(&actualMask, sizeof(actualMask),
                       static_cast<const uint8_t*>(raw->GetData()) + raw->GetDataSize() - sizeof(actualMask),
                       sizeof(actualMask)),
              EOK);
    int64_t expectedMask = 7;
    if (outputMask != nullptr) {
        expectedMask = 0;
        for (size_t i = 0; i < outputMask->size(); ++i) {
            expectedMask |= int64_t((*outputMask)[i]) << i;
        }
    }
    EXPECT_EQ(actualMask, expectedMask);
    if (expectNotSplitR) {
        // 只断言"没被切R模板抢走",不绑定具体落到哪个对手模板,避免对手改 key 时误伤
        uint64_t actualKey = tiling_context->GetTilingKey();
        EXPECT_NE(actualKey, static_cast<uint64_t>(50000000));
        EXPECT_NE(actualKey, static_cast<uint64_t>(51000000));
        EXPECT_NE(actualKey, static_cast<uint64_t>(52000000));
    } else {
        EXPECT_EQ(tiling_context->GetTilingKey(), expectedKey);
    }
    if (expectedFullLoadSplit != nullptr) {
        constexpr size_t FULL_LOAD_SPLIT_FIELD_COUNT = 5;
        constexpr size_t FULL_LOAD_SPLIT_FIELD_OFFSET = 11;
        ASSERT_EQ(tiling_context->GetTilingKey(), static_cast<uint64_t>(20000000));
        ASSERT_EQ(expectedFullLoadSplit->size(), FULL_LOAD_SPLIT_FIELD_COUNT);
        ASSERT_GE(raw->GetDataSize(), (FULL_LOAD_SPLIT_FIELD_OFFSET + FULL_LOAD_SPLIT_FIELD_COUNT) * sizeof(int64_t));
        int64_t split[FULL_LOAD_SPLIT_FIELD_COUNT] = {};
        ASSERT_EQ(memcpy_s(split, sizeof(split),
                           static_cast<const uint8_t*>(raw->GetData()) + FULL_LOAD_SPLIT_FIELD_OFFSET * sizeof(int64_t),
                           sizeof(split)),
                  EOK);
        for (size_t i = 0; i < FULL_LOAD_SPLIT_FIELD_COUNT; ++i) {
            EXPECT_EQ(split[i], (*expectedFullLoadSplit)[i]);
        }
        EXPECT_EQ(split[1] * split[3] + split[2] * split[4], c);
        EXPECT_EQ(split[0], split[3] + split[4]);
        EXPECT_LE(split[0], coreNum);
    }
    if (expectedFullLoadReduction != nullptr) {
        constexpr size_t FULL_LOAD_REDUCTION_FIELD_COUNT = 8;
        constexpr size_t FULL_LOAD_REDUCTION_FIELD_OFFSET = 20;
        ASSERT_TRUE(tiling_context->GetTilingKey() == 20000000 || tiling_context->GetTilingKey() == 21000000);
        ASSERT_EQ(expectedFullLoadReduction->size(), FULL_LOAD_REDUCTION_FIELD_COUNT);
        ASSERT_GE(raw->GetDataSize(),
                  (FULL_LOAD_REDUCTION_FIELD_OFFSET + FULL_LOAD_REDUCTION_FIELD_COUNT) * sizeof(int64_t));
        int64_t reduction[FULL_LOAD_REDUCTION_FIELD_COUNT] = {};
        ASSERT_EQ(
            memcpy_s(reduction, sizeof(reduction),
                     static_cast<const uint8_t*>(raw->GetData()) + FULL_LOAD_REDUCTION_FIELD_OFFSET * sizeof(int64_t),
                     sizeof(reduction)),
            EOK);
        for (size_t i = 0; i < FULL_LOAD_REDUCTION_FIELD_COUNT; ++i) {
            EXPECT_EQ(reduction[i], (*expectedFullLoadReduction)[i]);
        }
    }
}

TEST_F(BatchNormGradV3Tiling, ra_full_load_vector_core_partition)
{
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        // numBlocks, mainBlockFactor, tailBlockFactor, mainBlockCount, tailBlockCount.
        for (int64_t channels : {64, 96, 128, 160, 176, 256, 512, 3504, 3584}) {
            int64_t cores = (channels + 63) / 64;
            int64_t mainFactor = (channels + cores - 1) / cores;
            int64_t blockChannels = dtype == ge::DT_FLOAT ? 8 : 16;
            mainFactor = std::min(channels, (mainFactor + blockChannels - 1) / blockChannels * blockChannels);
            int64_t tail = channels % mainFactor;
            std::vector<int64_t> split{(channels + mainFactor - 1) / mainFactor, mainFactor, tail,
                                       channels / mainFactor, tail > 0 ? 1 : 0};
            RunSplitRTilingKeyCase(128, channels, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_FULL_LOAD_KEY,
                                   false, nullptr, &split);
        }
        // Existing wider per-core blocks keep the original partition.
        std::vector<int64_t> wideSplit{56, 128, 0, 56, 0};
        RunSplitRTilingKeyCase(128, 7168, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_FULL_LOAD_KEY, false,
                               nullptr, &wideSplit);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_full_load_unaligned_rows_keep_core_partition)
{
    std::vector<int64_t> fp32Split{17, 8, 1, 16, 1};
    RunSplitRTilingKeyCase(128, 129, ge::DT_FLOAT, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_FULL_LOAD_KEY, false,
                           nullptr, &fp32Split);
    std::vector<int64_t> fp16Split{9, 15, 9, 8, 1};
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        RunSplitRTilingKeyCase(128, 129, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_FULL_LOAD_KEY, false,
                               nullptr, &fp16Split);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_full_load_vector_core_partition_ub_fallback)
{
    // 64 FP32 channels at R=128 need 215552B; retain the original 8-core split below that budget.
    std::vector<int64_t> originalSplit{8, 8, 0, 8, 0};
    RunSplitRTilingKeyCase(128, 64, ge::DT_FLOAT, 215551, TEST_AIV_CORE_NUM_56, RA_FULL_LOAD_KEY, false, nullptr,
                           &originalSplit);
    std::vector<int64_t> vectorSplit{1, 64, 0, 1, 0};
    RunSplitRTilingKeyCase(128, 64, ge::DT_FLOAT, 215552, TEST_AIV_CORE_NUM_56, RA_FULL_LOAD_KEY, false, nullptr,
                           &vectorSplit);
}

TEST_F(BatchNormGradV3Tiling, ra_full_load_small_r_partition_and_reduction)
{
    const std::vector<std::pair<ge::DataType, ge::DataType>> dtypePairs{{ge::DT_FLOAT, ge::DT_FLOAT},
                                                                        {ge::DT_FLOAT16, ge::DT_FLOAT16},
                                                                        {ge::DT_BF16, ge::DT_BF16},
                                                                        {ge::DT_FLOAT16, ge::DT_FLOAT},
                                                                        {ge::DT_BF16, ge::DT_FLOAT}};
    // foldStep1, foldOffset1, foldStep2, foldOffset2, foldStep3, foldOffset3, recursion, reduceRows.
    const std::vector<std::pair<int64_t, std::vector<int64_t>>> reductionCases{
        {1, {0, 1, 0, 0, 0, 0, 0, 0}}, {2, {0, 1, 0, 0, 0, 0, 0, 0}}, {3, {0, 2, 1, 0, 0, 2, 0, 1}},
        {4, {1, 2, 0, 2, 0, 2, 0, 1}}, {5, {0, 4, 1, 0, 1, 2, 0, 2}}, {9, {0, 8, 1, 0, 3, 2, 1, 4}}};
    const std::vector<int64_t> smallSplit{3, 48, 33, 2, 1};
    for (const auto& dtype : dtypePairs) {
        for (const auto& testCase : reductionCases) {
            const auto* expectedSplit = testCase.first <= 4 ? &smallSplit : nullptr;
            RunSplitRTilingKeyCase(testCase.first, 129, dtype.first, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56,
                                   RA_FULL_LOAD_KEY, false, nullptr, expectedSplit, dtype.second, &testCase.second);
        }
    }
}

TEST_F(BatchNormGradV3Tiling, ra_packed_full_load_channel_boundary)
{
    RunSplitRTilingKeyCase(256, 31, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
    RunSplitRTilingKeyCase(256, 32, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
    RunSplitRTilingKeyCase(256, 33, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_FULL_LOAD_KEY);
}

TEST_F(BatchNormGradV3Tiling, ra_packed_full_load_weight_types_and_tail_groups)
{
    const std::vector<std::pair<ge::DataType, ge::DataType>> dtypePairs{{ge::DT_FLOAT, ge::DT_FLOAT},
                                                                        {ge::DT_FLOAT16, ge::DT_FLOAT16},
                                                                        {ge::DT_BF16, ge::DT_BF16},
                                                                        {ge::DT_FLOAT16, ge::DT_FLOAT},
                                                                        {ge::DT_BF16, ge::DT_FLOAT}};
    const std::vector<int64_t> fiveGroups{0, 4, 1, 0, 1, 2, 0, 2};
    const std::vector<int64_t> sixGroups{1, 4, 0, 2, 1, 2, 0, 2};
    const std::vector<int64_t> seventeenGroups{0, 16, 1, 0, 7, 2, 2, 8};
    const std::vector<int64_t> eighteenGroups{1, 16, 0, 2, 7, 2, 2, 8};
    for (const auto& dtype : dtypePairs) {
        RunSplitRTilingKeyCase(257, 1, dtype.first, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                               RA_PACKED_FULL_LOAD_KEY, false, nullptr, nullptr, dtype.second, &fiveGroups);
        RunSplitRTilingKeyCase(321, 1, dtype.first, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                               RA_PACKED_FULL_LOAD_KEY, false, nullptr, nullptr, dtype.second, &sixGroups);
        RunSplitRTilingKeyCase(257, 3, dtype.first, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                               RA_PACKED_FULL_LOAD_KEY, false, nullptr, nullptr, dtype.second, &seventeenGroups);
        RunSplitRTilingKeyCase(273, 3, dtype.first, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                               RA_PACKED_FULL_LOAD_KEY, false, nullptr, nullptr, dtype.second, &eighteenGroups);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_packed_full_load_exact_ub_boundary_by_weight_type)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        RunSplitRTilingKeyCase(256, 16, dtype, 59136, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY, false, nullptr,
                               nullptr, dtype);
        RunSplitRTilingKeyCase(256, 16, dtype, 59135, TEST_AIV_CORE_NUM_64, RA_FULL_LOAD_KEY, false, nullptr, nullptr,
                               dtype);
        RunSplitRTilingKeyCase(256, 16, dtype, 59904, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY, false, nullptr,
                               nullptr, ge::DT_FLOAT);
        RunSplitRTilingKeyCase(256, 16, dtype, 59903, TEST_AIV_CORE_NUM_64, RA_FULL_LOAD_KEY, false, nullptr, nullptr,
                               ge::DT_FLOAT);
    }
}

// fp32: C≤64 → fused-single(51); C=128/96(部分尾块) → fused-pair(52); C≥192 → generic(50)
TEST_F(BatchNormGradV3Tiling, ra_split_r_fused_single_fp32_c64)
{
    RunSplitRTilingKeyCase(8192, 64, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                           RA_SPLIT_R_FUSED_SINGLE_KEY);
}
TEST_F(BatchNormGradV3Tiling, ra_split_r_fused_pair_fp32_c128)
{
    RunSplitRTilingKeyCase(8192, 128, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                           RA_SPLIT_R_FUSED_PAIR_KEY);
}
TEST_F(BatchNormGradV3Tiling, ra_split_r_fused_pair_fp32_c96_partial_tail)
{
    RunSplitRTilingKeyCase(8192, 96, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                           RA_SPLIT_R_FUSED_PAIR_KEY);
}
TEST_F(BatchNormGradV3Tiling, ra_split_r_generic_fp32_c256)
{
    RunSplitRTilingKeyCase(8192, 256, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                           RA_SPLIT_R_GENERIC_KEY);
}
// fp16/bf16: single 扩到 C≤128(51);C>128 因 pair 仅 fp32 退 generic(50)
TEST_F(BatchNormGradV3Tiling, ra_split_r_fused_single_fp16_c64)
{
    RunSplitRTilingKeyCase(8192, 64, ge::DT_FLOAT16, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                           RA_SPLIT_R_FUSED_SINGLE_KEY);
}
TEST_F(BatchNormGradV3Tiling, ra_split_r_fused_single_bf16_c128)
{
    RunSplitRTilingKeyCase(8192, 128, ge::DT_BF16, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                           RA_SPLIT_R_FUSED_SINGLE_KEY);
}
TEST_F(BatchNormGradV3Tiling, ra_split_r_generic_fp16_c256)
{
    RunSplitRTilingKeyCase(8192, 256, ge::DT_FLOAT16, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                           RA_SPLIT_R_GENERIC_KEY);
}
// IsCapable 核利用率判据边界:只有"自己能用的核比对手多"才该抢，否则核数不升反降
// (对手 RA_ALL_LOAD/RA_RECOMPUTE 只切A;切R 的 blockFactor 有 R_LOOP_FACTOR=64 下限,
//  r1Dim < 64*coreNum 时切R 自己也填不满核)
TEST_F(BatchNormGradV3Tiling, ra_split_r_not_capable_own_cores_less_than_rival)
{
    // (256,256,1,1) fp32: 对手可用 32 核，切R 仅 4 核 → 不得调度到切R
    RunSplitRTilingKeyCase(256, 256, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, 0, true);
}
TEST_F(BatchNormGradV3Tiling, ra_split_r_not_capable_single_core)
{
    // (64,64,1,1) fp32: 对手可用 8 核，切R 仅 1 核 → 不得调度到切R
    RunSplitRTilingKeyCase(64, 64, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, 0, true);
}
TEST_F(BatchNormGradV3Tiling, ra_split_r_capable_own_cores_more_than_rival)
{
    // (320,32,1,1) fp32: 对手可用 4 核，切R 可用 5 核，且对手>=2核 → 应调度到切R(generic 50)
    RunSplitRTilingKeyCase(320, 32, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_SPLIT_R_GENERIC_KEY);
}
TEST_F(BatchNormGradV3Tiling, ra_split_r_not_capable_single_rival_core)
{
    // (320,1,1,1) fp32: 对手仅 1 核(窄通道单核全载),中 R 下切R 反而更慢 → 交还全载,不调度到切R。
    // rivalCores==1 时第一分支被 rivalCores>=2 门槛挡下,r1=320 又不够第二分支(>16*coreNum) → 不进切R。
    RunSplitRTilingKeyCase(320, 1, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, 0, true);
}
// 合计 UB 兜底:40 核/196608B 下 fp32 C128 的 fused-pair 占用超 UB，应退回 generic(50)
TEST_F(BatchNormGradV3Tiling, ra_split_r_fused_pair_ub_fallback_generic)
{
    RunSplitRTilingKeyCase(5120, 128, ge::DT_FLOAT, 196608, 40, RA_SPLIT_R_GENERIC_KEY);
}

TEST_F(BatchNormGradV3Tiling, ra_packed_full_load_dispatch)
{
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t channels : {1, 3, 8}) {
            RunSplitRTilingKeyCase(257, channels, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                                   RA_PACKED_FULL_LOAD_KEY);
        }
        // Split-R yields when a packed FullLoad tile fits, regardless of the old R threshold.
        RunSplitRTilingKeyCase(8192, 1, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
        const uint64_t boundaryKey = dtype == ge::DT_FLOAT ? RA_SPLIT_R_GENERIC_KEY : RA_PACKED_FULL_LOAD_KEY;
        RunSplitRTilingKeyCase(320, 32, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, boundaryKey);
    }
    RunSplitRTilingKeyCase(256, 32, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
    RunSplitRTilingKeyCase(257, 16, ge::DT_FLOAT16, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                           RA_PACKED_FULL_LOAD_KEY);
    RunSplitRTilingKeyCase(257, 16, ge::DT_BF16, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
}

TEST_F(BatchNormGradV3Tiling, ra_packed_full_load_admission_boundaries)
{
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        RunSplitRTilingKeyCase(127, 1, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(128, 1, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(255, 1, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(256, 1, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
    }
    // Packed capacity replaces the original ordinary FullLoad R thresholds.
    RunSplitRTilingKeyCase(957, 1, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
    RunSplitRTilingKeyCase(958, 1, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        RunSplitRTilingKeyCase(765, 1, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(766, 1, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_narrow_rows_amortize_copy_width)
{
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        // Use the same coarse six-core threshold for all input dtypes.
        RunSplitRTilingKeyCase(256, 256, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(320, 256, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(384, 256, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_SPLIT_R_GENERIC_KEY);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_rejects_narrow_full_load_tiles)
{
    // A minimal aligned tile fits, but it is too narrow to prefer FullLoad.
    RunSplitRTilingKeyCase(900, 7168, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_RECOMPUTE_KEY);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        RunSplitRTilingKeyCase(765, 7168, dtype, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64, RA_RECOMPUTE_KEY);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_keeps_short_reductions_on_full_load)
{
    // A single R64 recompute tile has no reduction-cache entry. Keep these
    // reductions on FullLoad across both supported platform core counts.
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t cores : {TEST_AIV_CORE_NUM_56, TEST_AIV_CORE_NUM_64}) {
            for (int64_t rows : {63, 64, 65}) {
                for (int64_t channels : {2048, 4096, 11515}) {
                    RunSplitRTilingKeyCase(rows, channels, dtype, TEST_UB_BYTES_240_KIB, cores, RA_FULL_LOAD_KEY);
                }
            }
        }
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_resident_vector_and_packed_layouts)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (int64_t rows : {128, 192, 255, 256}) {
            RunSplitRTilingKeyCase(rows, 32, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_FULL_LOAD_KEY);
        }
        RunSplitRTilingKeyCase(257, 32, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(12352, 1, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(16384, 1, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(16448, 1, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_SPLIT_R_PACKED_KEY);
        RunSplitRTilingKeyCase(772, 16, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(896, 16, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(1024, 16, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(1025, 16, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_SPLIT_R_GENERIC_KEY);
        RunSplitRTilingKeyCase(512, 32, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(513, 32, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_SPLIT_R_GENERIC_KEY);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_full_load_exact_ub_boundary)
{
    // R552/C16 packs into 138 groups and requires exactly 247296 bytes.
    RunSplitRTilingKeyCase(552, 16, ge::DT_FLOAT, 247295, TEST_AIV_CORE_NUM_56, RA_SPLIT_R_GENERIC_KEY);
    RunSplitRTilingKeyCase(552, 16, ge::DT_FLOAT, 247296, TEST_AIV_CORE_NUM_56, RA_PACKED_FULL_LOAD_KEY);
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_r_less_than_c)
{
    RunSplitRTilingKeyCase(1280, 2048, ge::DT_FLOAT, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56,
                           RA_SPLIT_R_GENERIC_KEY);
    RunSplitRTilingKeyCase(1024, 2048, ge::DT_FLOAT, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56,
                           RA_SPLIT_R_GENERIC_KEY);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        RunSplitRTilingKeyCase(979, 1024, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_SPLIT_R_GENERIC_KEY);
        RunSplitRTilingKeyCase(896, 2048, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_SPLIT_R_GENERIC_KEY);
        RunSplitRTilingKeyCase(768, 2048, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_RECOMPUTE_KEY);
        RunSplitRTilingKeyCase(1280, 4096, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_SPLIT_R_GENERIC_KEY);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_r_less_than_c_work_boundaries)
{
    for (uint32_t cores : {TEST_AIV_CORE_NUM_56, TEST_AIV_CORE_NUM_64}) {
        // Both sides have enough R cores; only the R/C work threshold changes.
        RunSplitRTilingKeyCase(1023, 2048, ge::DT_FLOAT, TEST_UB_BYTES_248_KIB, cores, RA_RECOMPUTE_KEY);
        RunSplitRTilingKeyCase(1024, 2048, ge::DT_FLOAT, TEST_UB_BYTES_248_KIB, cores, RA_SPLIT_R_GENERIC_KEY);
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            // Aligned half rows require R>=C/4; unaligned rows require R>=ceil(C/2).
            RunSplitRTilingKeyCase(1023, 4096, dtype, TEST_UB_BYTES_248_KIB, cores, RA_RECOMPUTE_KEY);
            RunSplitRTilingKeyCase(1024, 4096, dtype, TEST_UB_BYTES_248_KIB, cores, RA_SPLIT_R_GENERIC_KEY);
            RunSplitRTilingKeyCase(2048, 4097, dtype, TEST_UB_BYTES_248_KIB, cores, RA_RECOMPUTE_KEY);
            RunSplitRTilingKeyCase(2049, 4097, dtype, TEST_UB_BYTES_248_KIB, cores, RA_SPLIT_R_GENERIC_KEY);
        }
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_r_less_than_c_core_boundaries)
{
    const std::vector<std::tuple<uint32_t, int64_t, int64_t>> cases{{TEST_AIV_CORE_NUM_56, 832, 833},
                                                                    {TEST_AIV_CORE_NUM_64, 960, 961}};
    for (const auto testCase : cases) {
        // R/C work and nonresident FullLoad already qualify. One additional R
        // row raises occupancy from 13 to 14 cores, or from 15 to 16 cores.
        for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
            RunSplitRTilingKeyCase(std::get<1>(testCase), 1536, dtype, TEST_UB_BYTES_248_KIB, std::get<0>(testCase),
                                   RA_RECOMPUTE_KEY);
            RunSplitRTilingKeyCase(std::get<2>(testCase), 1536, dtype, TEST_UB_BYTES_248_KIB, std::get<0>(testCase),
                                   RA_SPLIT_R_GENERIC_KEY);
        }
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_r_less_than_c_resident_blocks)
{
    const std::vector<std::tuple<uint32_t, int64_t, int64_t>> cases{{TEST_AIV_CORE_NUM_56, 833, 896},
                                                                    {TEST_AIV_CORE_NUM_64, 961, 1024}};
    for (const auto testCase : cases) {
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            // Both points pass the work and occupancy gates. A per-core C16
            // block is resident; the neighboring C17 block needs a C loop.
            RunSplitRTilingKeyCase(std::get<1>(testCase), std::get<2>(testCase), dtype, TEST_UB_BYTES_248_KIB,
                                   std::get<0>(testCase), RA_FULL_LOAD_KEY);
            RunSplitRTilingKeyCase(std::get<1>(testCase), std::get<2>(testCase) + 1, dtype, TEST_UB_BYTES_248_KIB,
                                   std::get<0>(testCase), RA_SPLIT_R_GENERIC_KEY);
        }
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_r_less_than_c_channel_vector_boundaries)
{
    const std::vector<std::tuple<uint32_t, ge::DataType, int64_t, int64_t>> cases{
        {TEST_AIV_CORE_NUM_56, ge::DT_FLOAT, 2048, 3528},   {TEST_AIV_CORE_NUM_64, ge::DT_FLOAT, 2048, 4032},
        {TEST_AIV_CORE_NUM_56, ge::DT_FLOAT16, 4096, 7112}, {TEST_AIV_CORE_NUM_64, ge::DT_FLOAT16, 4096, 8128},
        {TEST_AIV_CORE_NUM_56, ge::DT_BF16, 4096, 7112},    {TEST_AIV_CORE_NUM_64, ge::DT_BF16, 4096, 8128}};
    for (const auto testCase : cases) {
        // The initial C block changes from vector-width-minus-one to exactly
        // one input vector. R/C work, occupancy, and FullLoad capacity qualify.
        RunSplitRTilingKeyCase(std::get<2>(testCase), std::get<3>(testCase), std::get<1>(testCase),
                               TEST_UB_BYTES_248_KIB, std::get<0>(testCase), RA_SPLIT_R_GENERIC_KEY);
        RunSplitRTilingKeyCase(std::get<2>(testCase), std::get<3>(testCase) + 1, std::get<1>(testCase),
                               TEST_UB_BYTES_248_KIB, std::get<0>(testCase), RA_RECOMPUTE_KEY);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_full_load_vector_capacity_boundary)
{
    const std::vector<std::tuple<ge::DataType, ge::DataType, int64_t, uint32_t>> cases{
        {ge::DT_FLOAT, ge::DT_FLOAT, 128, 215552},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, 256, 231168},
        {ge::DT_FLOAT16, ge::DT_FLOAT, 256, 231936},
        {ge::DT_BF16, ge::DT_BF16, 256, 231168},
        {ge::DT_BF16, ge::DT_FLOAT, 256, 231936}};
    for (uint32_t cores : {TEST_AIV_CORE_NUM_56, TEST_AIV_CORE_NUM_64}) {
        for (const auto testCase : cases) {
            // Each core owns C128, so no resident-block shortcut applies.
            // Exactly one vector fits at this UB budget; one byte less cannot.
            RunSplitRTilingKeyCase(std::get<2>(testCase), cores * 128, std::get<0>(testCase), std::get<3>(testCase) - 1,
                                   cores, RA_RECOMPUTE_KEY, false, nullptr, nullptr, std::get<1>(testCase));
            RunSplitRTilingKeyCase(std::get<2>(testCase), cores * 128, std::get<0>(testCase), std::get<3>(testCase),
                                   cores, RA_FULL_LOAD_KEY, false, nullptr, nullptr, std::get<1>(testCase));
        }
    }
}

class RARecomputeCapabilityProbe : public optiling::BatchNormGradV3RARecomputeTilingBase {
public:
    RARecomputeCapabilityProbe() : optiling::BatchNormGradV3RARecomputeTilingBase(nullptr) {}

    bool SupportsReduction(int64_t outerReduction, int64_t innerReduction)
    {
        r0Dim = innerReduction;
        r1Dim = outerReduction;
        return IsCapable();
    }
};

TEST_F(BatchNormGradV3Tiling, ra_recompute_rejects_single_r_tile)
{
    // Test capability directly: registry tests would select FullLoad first
    // and could pass even if Recompute incorrectly admitted a single R tile.
    RARecomputeCapabilityProbe probe;
    EXPECT_FALSE(probe.SupportsReduction(63, 1));
    EXPECT_FALSE(probe.SupportsReduction(64, 1));
    EXPECT_TRUE(probe.SupportsReduction(65, 1));
    EXPECT_FALSE(probe.SupportsReduction(65, 2));
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_coarse_regions)
{
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        // Short R stays resident, including unaligned rows.
        RunSplitRTilingKeyCase(64, 31, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(128, 17, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_PACKED_FULL_LOAD_KEY);
        // Wide per-core C is left to Recompute when FullLoad cannot fill a vector.
        RunSplitRTilingKeyCase(2048, 16384, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_RECOMPUTE_KEY);
        // R<C with resident C blocks stays on FullLoad.
        RunSplitRTilingKeyCase(128, 2048, dtype, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56, RA_FULL_LOAD_KEY);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_dispatch_packed_split_r_preserves_fused_boundary)
{
    for (int64_t cores : {TEST_AIV_CORE_NUM_56, TEST_AIV_CORE_NUM_64}) {
        for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
            RunSplitRTilingKeyCase(cores * 128 - 1, 32, dtype, TEST_UB_BYTES_248_KIB, cores, RA_SPLIT_R_GENERIC_KEY);
            RunSplitRTilingKeyCase(cores * 128, 32, dtype, TEST_UB_BYTES_248_KIB, cores, RA_SPLIT_R_FUSED_SINGLE_KEY);
            RunSplitRTilingKeyCase(cores * 128 + 1, 32, dtype, TEST_UB_BYTES_248_KIB, cores, RA_SPLIT_R_PACKED_KEY);
        }
        RunSplitRTilingKeyCase(cores * 256, 32, ge::DT_FLOAT, TEST_UB_BYTES_248_KIB, cores, RA_SPLIT_R_GENERIC_KEY);
        RunSplitRTilingKeyCase(cores * 256 + 1, 32, ge::DT_FLOAT, TEST_UB_BYTES_248_KIB, cores, RA_SPLIT_R_PACKED_KEY);
    }
    RunSplitRTilingKeyCase(int64_t{1} << 25, 32, ge::DT_FLOAT16, TEST_UB_BYTES_248_KIB, TEST_AIV_CORE_NUM_56,
                           RA_SPLIT_R_PACKED_KEY);
}

TEST_F(BatchNormGradV3Tiling, ra_packed_full_load_ub_fallback)
{
    // Packed fits only the larger budget. The small R partition does not
    // reach six cores, so ordinary FullLoad handles the smaller budget.
    RunSplitRTilingKeyCase(256, 32, ge::DT_FLOAT, 220000, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
    RunSplitRTilingKeyCase(256, 32, ge::DT_FLOAT, 200000, TEST_AIV_CORE_NUM_64, RA_FULL_LOAD_KEY);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        RunSplitRTilingKeyCase(256, 16, dtype, 62000, TEST_AIV_CORE_NUM_64, RA_PACKED_FULL_LOAD_KEY);
        RunSplitRTilingKeyCase(256, 16, dtype, 59000, TEST_AIV_CORE_NUM_64, RA_FULL_LOAD_KEY);
    }
}

TEST_F(BatchNormGradV3Tiling, ra_packed_full_load_output_masks)
{
    for (uint64_t bits = 0; bits < 8; ++bits) {
        std::vector<bool> mask{bool(bits & 1), bool(bits & 2), bool(bits & 4)};
        RunSplitRTilingKeyCase(321, 1, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                               RA_PACKED_FULL_LOAD_KEY, false, &mask);
    }
}

TEST_F(BatchNormGradV3Tiling, output_mask_all_combinations)
{
    for (uint64_t bits = 0; bits < 8; ++bits) {
        std::vector<bool> mask{bool(bits & 1), bool(bits & 2), bool(bits & 4)};
        RunSplitRTilingKeyCase(8192, 64, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                               RA_SPLIT_R_FUSED_SINGLE_KEY, false, &mask);
        RunSplitRTilingKeyCase(8192, 128, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                               RA_SPLIT_R_FUSED_PAIR_KEY, false, &mask);
        RunSplitRTilingKeyCase(8192, 256, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                               RA_SPLIT_R_GENERIC_KEY, false, &mask);
    }
}

TEST_F(BatchNormGradV3Tiling, output_mask_invalid_length)
{
    for (size_t length : {0, 1, 2, 4}) {
        std::vector<bool> mask(length, true);
        RunSplitRTilingKeyCase(8192, 64, ge::DT_FLOAT, TEST_UB_BYTES_240_KIB, TEST_AIV_CORE_NUM_64,
                               RA_SPLIT_R_FUSED_SINGLE_KEY, false, &mask);
    }
}

// ============================================================================
// Error code verification UT for E*9*** remediation (2026-10-21)
// Captures ErrorManager output via weak-symbol override; asserts no EZ9999.
// ============================================================================
namespace error_message {
struct CapturedError {
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};
static thread_local CapturedError g_last_error;
static thread_local int g_report_count = 0;
static thread_local std::vector<std::string> g_all_codes; // 记录用例全程上报过的全部错误码
__attribute__((weak)) void ResetCapture()
{
    g_last_error.code.clear();
    g_last_error.keys.clear();
    g_last_error.values.clear();
    g_report_count = 0;
    g_all_codes.clear();
}
__attribute__((weak)) std::string GetLastError() { return g_last_error.code; }
__attribute__((weak)) int GetReportCount() { return g_report_count; }
__attribute__((weak)) bool HasCode(const std::string& c)
{
    for (const auto& code : g_all_codes) {
        if (code == c) {
            return true;
        }
    }
    return false;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code, const std::vector<const char*>& key,
                                                     const std::vector<const char*>& value)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_last_error.keys.clear();
    g_last_error.values.clear();
    for (const auto& k : key)
        g_last_error.keys.push_back(k ? k : "");
    for (const auto& v : value)
        g_last_error.values.push_back(v ? v : "");
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportInnerErrMsg(const char* f, const char* fn, uint32_t l, const char* ec,
                                                const char* fmt, ...)
{
    g_last_error.code = ec ? ec : "";
    if (ec != nullptr) {
        g_all_codes.push_back(ec);
    }
    g_report_count++;
    return 0;
}
} // namespace error_message

// 验证弱符号截获通道在本二进制内生效: 若链接格局变化导致强定义压制弱定义, 该用例显式失败,
// 避免EZ9999守护用例静默失效
TEST_F(BatchNormGradV3Tiling, error_capture_channel_self_check)
{
    error_message::ResetCapture();
    const std::vector<const char*> selfKey = {"self_check"};
    const std::vector<const char*> selfValue = {"capture channel liveness"};
    error_message::ReportPredefinedErrMsg("EZ0000", selfKey, selfValue);
    EXPECT_TRUE(error_message::HasCode("EZ0000")) << "capture channel is dead: weak override suppressed";
    EXPECT_EQ(error_message::GetReportCount(), 1);
    error_message::ResetCapture();
}
// Global listener: after each test, verify no EZ9999 was captured
class BATCH_NORM_GRAD_V3NoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        error_message::ResetCapture(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
    }
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        // 只守护本文件的测试套件, 框架层或其他算子用例的EZ9999不跨套件追责
        static const char* const guardedSuites[] = {"BatchNormGradV3Tiling"};
        bool guarded = false;
        for (const char* s : guardedSuites) {
            if (std::string(test_info.test_suite_name()) == s) {
                guarded = true;
                break;
            }
        }
        if (!guarded) {
            return;
        }
        if (error_message::HasCode("EZ9999")) {
            ADD_FAILURE() << "EZ9999 captured during test [" << test_info.name() << "] in " << test_info.file();
        }
    }
};
static const auto _inst_BATCH_NORM_GRAD_V3 = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new BATCH_NORM_GRAD_V3NoEz9999Check());
    return 0;
}();
