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
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "base/err_mgr.h"

using namespace ut_util;
using namespace std;
using namespace ge;

// must be layout-compatible with optiling::QuantizeAddLayerNormCompileInfo in tiling.h
struct QuantizeAddLayerNormCompileInfo {
    uint32_t aivCoreNum_ = 0;
    uint64_t sysWorkspaceSize_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t vecRegSize_ = 0;
    uint32_t blockSize_ = 0;
    bool isRegbase = false;
};

class QuantizeAddLayerNormTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QuantizeAddLayerNormTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantizeAddLayerNormTiling TearDown" << std::endl; }
};

// ErrorManager raw message helpers (replaces weak-symbol capture)
static std::vector<std::string> g_ezCodes;
static bool g_ezCodesLoaded = false;

static void InitErrMgr()
{
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrErrorMessage();
    g_ezCodes.clear();
    g_ezCodesLoaded = false;
}

static void LoadEZCodes()
{
    if (g_ezCodesLoaded) {
        return;
    }
    g_ezCodes.clear();
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    for (const auto& e : errors) {
        g_ezCodes.push_back(e.error_id == nullptr ? std::string() : std::string(e.error_id.get()));
    }
    g_ezCodesLoaded = true;
}

static bool HasEZCode(const std::string& code)
{
    LoadEZCodes();
    for (const auto& id : g_ezCodes) {
        if (id == code) {
            return true;
        }
    }
    return false;
}

static int GetEZCodeCount()
{
    LoadEZCodes();
    return static_cast<int>(g_ezCodes.size());
}

TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{8192, 5120}, {8192, 5120}};
    gert::StorageShape gamma_shape = {{
                                          5120,
                                      },
                                      {
                                          5120,
                                      }};
    gert::StorageShape out_shape = {{8192, 5120}, {8192, 5120}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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
    QuantizeAddLayerNormCompileInfo compile_info;

    std::string op_type("QuantizeAddLayerNorm");
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
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape,
                                    &gamma_shape})
                      .OutputShapes({&out_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(65535)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)},
                                  {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 1101);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{40, 11264}, {40, 11264}};
    gert::StorageShape gamma_shape = {{
                                          11264,
                                      },
                                      {
                                          11264,
                                      }};
    gert::StorageShape out_shape = {{40, 11264}, {40, 11264}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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
    QuantizeAddLayerNormCompileInfo compile_info;

    std::string op_type("QuantizeAddLayerNorm");
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
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape,
                                    &gamma_shape})
                      .OutputShapes({&out_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-65535)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)},
                                  {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 1102);
    // dlog_setlevel(0, 3, 0);
}

// ---------- ascend950 (regbase) tiling cases ----------
// faked Ascend950 platform; quant_shape is the shape used for both scales(idx5) and zero_points(idx6)
// (per-channel -> gamma_shape; per_tensor -> scalar [1]). bias(idx4) defaults to gamma_shape (broadcast);
// pass bias_shape = &input_shape for the elewise-bias keys.
static void RunRegbaseTilingCase(gert::StorageShape& input_shape, gert::StorageShape& gamma_shape,
                                 gert::StorageShape& out_shape, gert::StorageShape& quant_shape, int64_t axis,
                                 ge::DataType xDtype, ge::DataType scaleDtype, uint32_t expectedKey,
                                 uint32_t expectedBlockDim, gert::StorageShape* bias_shape = nullptr)
{
    gert::StorageShape& bias_ref = (bias_shape != nullptr) ? *bias_shape : gamma_shape;
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::map<std::string, std::string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    QuantizeAddLayerNormCompileInfo compile_info;

    std::string op_type("QuantizeAddLayerNorm");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &bias_ref, &quant_shape,
                                    &quant_shape})
                      .OutputShapes(std::vector<gert::StorageShape*>{&out_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)},
                                  {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_context->GetTilingKey(), expectedKey);
    ASSERT_EQ(tiling_context->GetBlockDim(), expectedBlockDim);
}

// full_load + per_channel(div, axis=-1) + brc bias, fp16 -> key 8012
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_perchannel_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT16, ge::DT_FLOAT, 8012, 64);
}

// full_load + mul_mode(axis=-65535) + brc bias, fp16 -> key 8002
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_mulmode_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -65535, ge::DT_FLOAT16, ge::DT_FLOAT, 8002,
                         64);
}

// full_load + per_tensor(scalar scale, axis=65535) + brc bias, fp16 -> key 8022
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_pertensor_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape scalar_shape = {{1}, {1}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, scalar_shape, 65535, ge::DT_FLOAT16, ge::DT_FLOAT, 8022,
                         64);
}

// welford + per_channel(div, axis=-1) + brc bias, fp16, large cols -> key 8112
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_welford_perchannel_001)
{
    gert::StorageShape input_shape = {{1, 1024, 32768}, {1, 1024, 32768}};
    gert::StorageShape gamma_shape = {{32768}, {32768}};
    gert::StorageShape out_shape = {{1, 1024, 32768}, {1, 1024, 32768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT16, ge::DT_FLOAT, 8112, 64);
}

// full_load + per_channel(div) + elewise bias, fp16 -> key 8011
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_perchannel_elewise_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT16, ge::DT_FLOAT, 8011, 64,
                         &input_shape);
}

// full_load + mul_mode(axis=-65535) + elewise bias, fp16 -> key 8001
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_mulmode_elewise_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -65535, ge::DT_FLOAT16, ge::DT_FLOAT, 8001,
                         64, &input_shape);
}

// full_load + per_tensor(axis=65535, scalar scale) + elewise bias, fp16 -> key 8021
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_pertensor_elewise_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape scalar_shape = {{1}, {1}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, scalar_shape, 65535, ge::DT_FLOAT16, ge::DT_FLOAT, 8021,
                         64, &input_shape);
}

// welford + per_channel(div) + elewise bias, fp16, large cols -> key 8111
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_welford_perchannel_elewise_001)
{
    gert::StorageShape input_shape = {{1, 1024, 32768}, {1, 1024, 32768}};
    gert::StorageShape gamma_shape = {{32768}, {32768}};
    gert::StorageShape out_shape = {{1, 1024, 32768}, {1, 1024, 32768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT16, ge::DT_FLOAT, 8111, 64,
                         &input_shape);
}

// full_load + per_channel(div) + brc bias, bf16 x with bf16 scales -> key 8012
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_perchannel_bf16_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_BF16, ge::DT_BF16, 8012, 64);
}

// full_load + per_channel(div) + brc bias, fp32 x with fp32 scales -> key 8012
TEST_F(QuantizeAddLayerNormTiling, quantize_add_layer_norm_tiling_regbase_fullload_perchannel_fp32_001)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    RunRegbaseTilingCase(input_shape, gamma_shape, out_shape, gamma_shape, -1, ge::DT_FLOAT, ge::DT_FLOAT, 8012, 64);
}

// namespace error_message

#define EXPECT_ERROR_CODE(expected) EXPECT_TRUE(HasEZCode(expected)) << "Expected error code " << expected

// After each test, verify no EZ9999 internal code was captured
class QalnNoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        InitErrMgr();
        (void)error_message::GetErrMgrErrorMessage(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
    }
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        // 只守护本文件的测试套件, 框架层或其他算子用例的EZ9999不跨套件追责
        static const char* const guardedSuites[] = {"QuantizeAddLayerNormTiling"};
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
        if (HasEZCode("EZ9999")) {
            ADD_FAILURE() << "EZ9999 captured during test [" << test_info.name() << "] in " << test_info.file();
        }
    }
};
static const auto _inst_qaln = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new QalnNoEz9999Check());
    return 0;
}();

// 错误码用例公共入口: 沿用 RunRegbaseTilingCase 的搭建, 支持覆盖 x1 dtype、epsilon、UB_SIZE
static ge::graphStatus RunQalnTilingErrCase(ge::DataType xDtype, float eps, uint64_t ubSize)
{
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::string ub_json = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": )" + std::to_string(ubSize) +
                          R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    GetPlatFormInfos(ub_json.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    QuantizeAddLayerNormCompileInfo compile_info;

    std::string op_type("QuantizeAddLayerNorm");
    auto* op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    EXPECT_NE(op_impl, nullptr);
    if (op_impl == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(ub_json.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    if (!kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init()) {
        return ge::GRAPH_FAILED;
    }
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    EXPECT_EQ(op_impl->tiling_parse(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(7, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape,
                                    &gamma_shape})
                      .OutputShapes(std::vector<gert::StorageShape*>{&out_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                                  {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    return op_impl->tiling(tiling_context);
}

TEST_F(QuantizeAddLayerNormTiling, err_x1_dtype_unsupported_ez0019)
{
    InitErrMgr();
    EXPECT_EQ(RunQalnTilingErrCase(ge::DT_DOUBLE, 1e-5F, 245760), ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0019");
}

TEST_F(QuantizeAddLayerNormTiling, err_epsilon_not_positive_ez0024)
{
    InitErrMgr();
    EXPECT_EQ(RunQalnTilingErrCase(ge::DT_FLOAT16, -1.0F, 245760), ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ0024");
}

TEST_F(QuantizeAddLayerNormTiling, err_ub_size_zero_ez1008)
{
    InitErrMgr();
    EXPECT_EQ(RunQalnTilingErrCase(ge::DT_FLOAT16, 1e-5F, 0), ge::GRAPH_FAILED);
    EXPECT_ERROR_CODE("EZ1008");
}
