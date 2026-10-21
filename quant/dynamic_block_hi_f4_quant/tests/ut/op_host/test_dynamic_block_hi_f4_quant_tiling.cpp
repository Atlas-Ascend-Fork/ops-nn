/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_block_hi_f4_quant_tiling.cpp
 * \brief Tiling unit tests for DynamicBlockHiF4Quant
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "quant/dynamic_block_hi_f4_quant/op_host/arch35/dynamic_block_hi_f4_quant_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "any_value.h"

using namespace std;
using namespace ge;

class DynamicBlockHiF4QuantTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DynamicBlockHiF4QuantTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DynamicBlockHiF4QuantTiling TearDown" << std::endl; }
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

static void ExecuteTestCase(ge::DataType inDtype, ge::DataType yDtype, ge::DataType scaleDtype,
                            gert::StorageShape shape, gert::StorageShape scale1Shape, gert::StorageShape scale2Shape,
                            string roundMode, string expectTilingData, int64_t scaleAlg = 0,
                            ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_versions = {{"Short_SoC_version", "Ascend960DT"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::DynamicBlockHiF4QuantCompileInfo compile_info;

    std::string op_type("DynamicBlockHiF4Quant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_versions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&shape, &scale1Shape, &scale2Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(scaleAlg)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_versions);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

void ExecuteTestCaseFailed(ge::DataType inDtype, ge::DataType yDtype, ge::DataType scaleDtype, gert::StorageShape shape,
                           gert::StorageShape scale1Shape, gert::StorageShape scale2Shape, string roundMode)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_versions = {{"Short_SoC_version", "Ascend960DT"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::DynamicBlockHiF4QuantCompileInfo compile_info;

    std::string op_type("DynamicBlockHiF4Quant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_versions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&shape, &scale1Shape, &scale2Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 正常用例：tilingKey 的预期值待算子完整实现后跑通回填
TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_bf16_rint_2d)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    string roundMode = "rint";
    string expectTilingData = "0 64 2 253952 4 0 64 512 128 128 2 1 128 0 1 1 ";
    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape, roundMode,
                    expectTilingData);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_bf16_rint_scaleAlg1_2d)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    string roundMode = "rint";
    string expectTilingData = "0 64 2 253952 4 1 64 512 128 128 2 1 128 0 1 1 ";
    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape, roundMode,
                    expectTilingData, 1);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_fp16_rint_2d)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    string roundMode = "rint";
    string expectTilingData = "0 64 2 253952 4 0 64 512 128 128 2 1 128 0 1 1 ";
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape, roundMode,
                    expectTilingData);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_bf16_trunc_2d)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    string roundMode = "trunc";
    string expectTilingData = "1 64 2 253952 3 0 64 512 128 128 2 1 128 0 1 1 ";
    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape, roundMode,
                    expectTilingData);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_bf16_3d)
{
    // shape=[2,128,128]：合轴后 rowNum=2*128=256, colNum=128
    gert::StorageShape shape = {{2, 128, 128}, {2, 128, 128}};
    gert::StorageShape scale1Shape = {{2, 128, 2, 1}, {2, 128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 2, 128, 1}, {2, 2, 128, 1}};
    string roundMode = "rint";
    string expectTilingData = "0 64 4 253952 4 0 64 512 256 128 4 1 128 0 1 1 ";
    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape, roundMode,
                    expectTilingData);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_xDtype_failed)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    string roundMode = "rint";
    ExecuteTestCaseFailed(ge::DT_FLOAT, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape,
                          roundMode);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_yDtype_failed)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    string roundMode = "rint";
    ExecuteTestCaseFailed(ge::DT_BF16, ge::DT_FLOAT16, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape,
                          roundMode);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_scaleDtype_failed)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    string roundMode = "rint";
    ExecuteTestCaseFailed(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_FLOAT16, shape, scale1Shape, scale2Shape, roundMode);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_roundMode_failed)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    // round_mode 仅支持 rint 与 trunc，传入 "floor" 应当校验失败
    string roundMode = "floor";
    ExecuteTestCaseFailed(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape,
                          roundMode);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_scaleAlg_failed)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    // scale_alg 仅支持 0 与 1，传入 2 应当校验失败
    string roundMode = "rint";
    string expectTilingData = "";
    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape, roundMode,
                    expectTilingData, 2, ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_shape_dim_not_64_failed)
{
    gert::StorageShape shape = {{128, 32}, {128, 32}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 32, 1}, {2, 32, 1}};
    string roundMode = "rint";
    ExecuteTestCaseFailed(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape,
                          roundMode);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_shape_rank_failed)
{
    gert::StorageShape shape = {{128}, {128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    string roundMode = "rint";
    ExecuteTestCaseFailed(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape,
                          roundMode);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_scale1_shape_failed)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 4, 1}, {128, 4, 1}};
    gert::StorageShape scale2Shape = {{2, 128, 1}, {2, 128, 1}};
    string roundMode = "rint";
    ExecuteTestCaseFailed(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape,
                          roundMode);
}

TEST_F(DynamicBlockHiF4QuantTiling, DynamicBlockHiF4Quant_scale2_shape_failed)
{
    gert::StorageShape shape = {{128, 128}, {128, 128}};
    gert::StorageShape scale1Shape = {{128, 2, 1}, {128, 2, 1}};
    gert::StorageShape scale2Shape = {{4, 128, 1}, {4, 128, 1}};
    string roundMode = "rint";
    ExecuteTestCaseFailed(ge::DT_BF16, ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, shape, scale1Shape, scale2Shape,
                          roundMode);
}
