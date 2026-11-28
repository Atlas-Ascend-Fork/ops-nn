/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/apply_adam_w_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;
using namespace ut_util;

class ApplyAdamWTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyAdamWTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyAdamWTilingTest TearDown" << std::endl; }
};

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1000)
{
    std::string op_type("ApplyAdamW");

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = false;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                                    &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs({{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1001)
{
    std::string op_type("ApplyAdamW");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_FLOAT16;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = false;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                                    &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs({{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1002)
{
    std::string op_type("ApplyAdamW");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_BF16;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = false;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                                    &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs({{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1003)
{
    std::string op_type("ApplyAdamW");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_FLOAT16;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = true;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                                    &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs({{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1004)
{
    std::string op_type("ApplyAdamW");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_BF16;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = true;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                                    &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs({{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

namespace {
// tiling 通用骨架：var/m/v/grad 为 (64,10,10,32)、标量超参 (1,)、可选 mgn 可在场或缺失（v2 实测写法：
// IrInstanceNum 保持全 1、InputShapes 对应位传 nullptr 即空可选输入槽，NodeInputTd 照常设置）；
// dtypeChangeIdx 注入单输入 dtype 偏差以驱动 SCALAR/TENSOR 两循环的 CheckSameDtype 分支。
ge::graphStatus RunApplyAdamWTilingCase(bool amsgrad, bool maximize, bool mgnPresent, int32_t dtypeChangeIdx = -1,
                                        ge::DataType changedDtype = ge::DT_FLOAT, int32_t nonScalarIdx = -1,
                                        int32_t shapeMismatchIdx = -1, bool withTilingData = true,
                                        int32_t missingShapeIdx = -1)
{
    std::string op_type("ApplyAdamW");
    if (gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()) == nullptr) {
        return ge::GRAPH_FAILED + 100;
    }
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape tensorShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;

    gert::StorageShape mismatchShape = {{64, 10, 10, 16}, {64, 10, 10, 16}};
    std::vector<gert::StorageShape*> inputShapes = {&tensorShape, &tensorShape, &tensorShape, &scalarShape,
                                                    &scalarShape, &scalarShape, &scalarShape, &scalarShape,
                                                    &scalarShape, &scalarShape, &tensorShape, nullptr};
    if (mgnPresent) {
        inputShapes[11] = &tensorShape;
    }
    if (nonScalarIdx >= 3 && nonScalarIdx <= 9) {
        inputShapes[nonScalarIdx] = &tensorShape;
    }
    if (shapeMismatchIdx >= 0 && shapeMismatchIdx < 11) {
        inputShapes[shapeMismatchIdx] = &mismatchShape;
    }
    if (missingShapeIdx >= 0 && missingShapeIdx < 11) {
        inputShapes[missingShapeIdx] = nullptr;
    }
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(inputShapes)
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtypeChangeIdx == 0 ? changedDtype : varDtype, format, format)
                      .NodeInputTd(1, dtypeChangeIdx == 1 ? changedDtype : varDtype, format, format)
                      .NodeInputTd(2, dtypeChangeIdx == 2 ? changedDtype : varDtype, format, format)
                      .NodeInputTd(3, dtypeChangeIdx == 3 ? changedDtype : varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dtypeChangeIdx == 4 ? changedDtype : varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, dtypeChangeIdx == 5 ? changedDtype : varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, dtypeChangeIdx == 6 ? changedDtype : varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, dtypeChangeIdx == 7 ? changedDtype : varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, dtypeChangeIdx == 8 ? changedDtype : varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, dtypeChangeIdx == 9 ? changedDtype : varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, dtypeChangeIdx == 10 ? changedDtype : varDtype, format, format)
                      .NodeInputTd(11, dtypeChangeIdx == 11 ? changedDtype : varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs({{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
                      .TilingData(withTilingData ? param.get() : nullptr)
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED + 100;
    }
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    return tiling_func(tiling_context);
}
} // namespace

// #17：amsgrad=true 且 max_grad_norm 缺失时 tiling 必须拦截（红转绿：修复前假绿 SUCCESS）
TEST_F(ApplyAdamWTilingTest, neg_amsgrad_without_mgn)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(true, false, false), ge::GRAPH_FAILED);
}

// 防过度拦截：amsgrad=false 且 mgn 缺失为 proto 合法组合，不得被新校验误伤
TEST_F(ApplyAdamWTilingTest, noamsgrad_without_mgn_pass)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(false, false, false), ge::GRAPH_SUCCESS);
}

// 正常路径回归：amsgrad=true 且 mgn 在场且 dtype/shape 合法
TEST_F(ApplyAdamWTilingTest, amsgrad_with_mgn_pass)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(true, false, true), ge::GRAPH_SUCCESS);
}

// #19：SCALAR_INDEX_LIST 的 CheckSameDtype 失败分支可达（lr dtype != var）
TEST_F(ApplyAdamWTilingTest, scalar_dtype_mismatch_lr)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(false, false, true, 5, ge::DT_FLOAT16), ge::GRAPH_FAILED);
}

// #19：TENSOR_INDEX_LIST 的 CheckSameDtype 失败分支可达（grad dtype != var）
TEST_F(ApplyAdamWTilingTest, tensor_dtype_mismatch_grad)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(false, false, true, 10, ge::DT_FLOAT16), ge::GRAPH_FAILED);
}

// 日志整改：SCALAR_INDEX_LIST 的 CheckIsScalar 失败分支可达（lr 非标量）
TEST_F(ApplyAdamWTilingTest, scalar_not_scalar_shape_lr)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(false, false, true, -1, ge::DT_FLOAT, 5), ge::GRAPH_FAILED);
}

// 日志整改：TENSOR_INDEX_LIST 的 CheckSameShape 失败分支可达（m shape != var）
TEST_F(ApplyAdamWTilingTest, tensor_shape_mismatch_m)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(false, false, true, -1, ge::DT_FLOAT, -1, 1), ge::GRAPH_FAILED);
}

// 判空防御分支直测：TilingForApplyAdamW 入口 context 为空（EZ1008）
TEST_F(ApplyAdamWTilingTest, tiling_func_null_context)
{
    std::string op_type("ApplyAdamW");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    EXPECT_EQ(tiling_func(nullptr), ge::GRAPH_FAILED);
}

// 判空防御分支直测：RunTiling 入口 tilingContext_ 为空（EZ1008）
TEST_F(ApplyAdamWTilingTest, run_tiling_null_context)
{
    optiling::ApplyAdamWTiling tiling(nullptr);
    EXPECT_EQ(tiling.RunTiling(), ge::GRAPH_FAILED);
}

// 判空防御分支直测：GetTilingData 返回空（EZ1008）
TEST_F(ApplyAdamWTilingTest, get_tiling_data_null)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(false, false, true, -1, ge::DT_FLOAT, -1, -1, false), ge::GRAPH_FAILED);
}

// 检视修复：SCALAR_INDEX_LIST 输入 shape 缺失时日志不得解引用空指针，返回 GRAPH_FAILED 而非崩溃
TEST_F(ApplyAdamWTilingTest, scalar_shape_missing_lr)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(false, false, true, -1, ge::DT_FLOAT, -1, -1, true, 5), ge::GRAPH_FAILED);
}

// 检视修复：TENSOR_INDEX_LIST 输入 shape 缺失时日志不得解引用空指针，返回 GRAPH_FAILED 而非崩溃
TEST_F(ApplyAdamWTilingTest, tensor_shape_missing_m)
{
    EXPECT_EQ(RunApplyAdamWTilingCase(false, false, true, -1, ge::DT_FLOAT, -1, -1, true, 1), ge::GRAPH_FAILED);
}
