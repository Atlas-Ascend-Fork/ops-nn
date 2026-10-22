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
 * \file test_softmax_grad_ext_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <map>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "activation/softmax_grad_ext/op_host/arch35/softmax_grad_ext_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SoftmaxGradExtTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
        setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", true);
    }

    static void TearDownTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "0", true);
        setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", true);
    }
};

TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_float32_ar_base)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{2, 2, 20}, {2, 2, 20}};
    gert::StorageShape y_shape = {{2, 2, 20}, {2, 2, 20}};

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
    optiling::SoftmaxGradExtCompileInfo compile_info;
    std::string op_type("SoftmaxGradExt");
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

    int64_t axes = -1;
    bool keep_dims = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &x_shape, &x_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axes", Ops::NN::AnyValue::CreateFrom<int64_t>(axes)},
                                  {"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keep_dims)}})
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
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_bfloat16_ar_base)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{9, 20}, {9, 20}};
    gert::StorageShape y_shape = {{9, 20}, {9, 20}};

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
    optiling::SoftmaxGradExtCompileInfo compile_info;
    std::string op_type("SoftmaxGradExt");
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

    int64_t axes = -1;
    bool keep_dims = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &x_shape, &x_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axes", Ops::NN::AnyValue::CreateFrom<int64_t>(axes)},
                                  {"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keep_dims)}})
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
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ar_recompute)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{16, 20000}, {16, 20000}};
    gert::StorageShape y_shape = {{16, 20000}, {16, 20000}};

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
    optiling::SoftmaxGradExtCompileInfo compile_info;
    std::string op_type("SoftmaxGradExt");
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

    int64_t axes = -1;
    bool keep_dims = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &x_shape, &x_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axes", Ops::NN::AnyValue::CreateFrom<int64_t>(axes)},
                                  {"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keep_dims)}})
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
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ar_small_r)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{20000, 4}, {20000, 4}};
    gert::StorageShape y_shape = {{20000, 4}, {20000, 4}};

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
    optiling::SoftmaxGradExtCompileInfo compile_info;
    std::string op_type("SoftmaxGradExt");
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

    int64_t axes = -1;
    bool keep_dims = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &x_shape, &x_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axes", Ops::NN::AnyValue::CreateFrom<int64_t>(axes)},
                                  {"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keep_dims)}})
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
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // dlog_setlevel(0, 3, 0);
}

// ============================================================================
// ARA 模板用例（U1-U12），依据《softmax_grad_ext 增加 ARA 模板 开发方案设计》§7 落地。
// 用例 shape 记法 (a1, r, a0)：三维输入，axes=1 对中间 R 轴 reduce，reduce 轴之后保留
// A0 尾轴（a0 > 1 时 AR 族 IsCapable 不通过，由 ARA 模板承接，AR 优先 ARA 兜底）。
// 预期 tilingKey：ara_full_load = TILINGKEY_ARA(10000)、ara_recompute = 20000。
// 字段断言值按 tiling 实现公式核算（softmax_grad_ext_tiling_ara_*_arch35.cpp）：
// Ascend950 faker 环境 UB_SIZE=245760、CORE_NUM=64、blockSize=32、vlFp32=64、vlFp16=128。
// ============================================================================

namespace {
constexpr uint64_t ARA_UT_TKEY_AR_FULL_LOAD = 1000U;
constexpr uint64_t ARA_UT_TKEY_ARA_FULL_LOAD = 10000U;
constexpr uint64_t ARA_UT_TKEY_ARA_RECOMPUTE = 20000U;

// ARA full load TilingData raw 布局：int64*9 + uint16*6 + uint32*11（偏移均天然满足对齐）
constexpr size_t ARA_UT_U16_BASE = 9U * 8U;           // 72
constexpr size_t ARA_UT_U32_BASE = 9U * 8U + 6U * 2U; // 84

struct SoftmaxGradExtTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0U;
    uint32_t blockDim = 0U;
    size_t workspaceSize = 0U;
    std::vector<uint8_t> tilingData;
};

gert::StorageShape MakeAraStorageShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0U; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    gert::StorageShape storageShape;
    storageShape.MutableOriginShape() = shape;
    storageShape.MutableStorageShape() = shape;
    return storageShape;
}

// x2ShapeIn 为空表示 x2 与 grad/x1 同 shape；否则按传入 shape（如 (1,) 标量分支）
SoftmaxGradExtTilingResult RunSoftmaxGradExtAraTiling(const std::vector<int64_t>& shape,
                                                      const std::vector<int64_t>& x2ShapeIn, const ge::DataType dtype,
                                                      const int64_t axes)
{
    SoftmaxGradExtTilingResult result;
    gert::StorageShape xShape = MakeAraStorageShape(shape);
    gert::StorageShape yShape = MakeAraStorageShape(shape);
    gert::StorageShape x2Shape = x2ShapeIn.empty() ? MakeAraStorageShape(shape) : MakeAraStorageShape(x2ShapeIn);

    const std::string compileInfoString = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}
                            })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersion;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::SoftmaxGradExtCompileInfo compileInfo;
    const std::string opType("SoftmaxGradExt");
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr) {
        return result;
    }
    const auto tilingFunc = opImpl->tiling;
    const auto tilingParseFunc = opImpl->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    if (!kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init()) {
        return result;
    }
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    if (tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return result;
    }

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    if (param == nullptr) {
        return result;
    }
    const bool keepDims = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &xShape, &x2Shape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"axes", Ops::NN::AnyValue::CreateFrom<int64_t>(axes)},
                                  {"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext->GetPlatformInfo() == nullptr) {
        return result;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    result.status = tilingFunc(tilingContext);
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }
    result.tilingKey = tilingContext->GetTilingKey();
    result.blockDim = tilingContext->GetBlockDim();
    // 不走 GetWorkspaceNum() 守卫：gert 版本间 GetWorkspaceSizes(count) 的 SetSize(count) 回写行为
    // 不一致（CI 环境 size 恒 0，守卫跳过导致误捕获 0）。与算子 PostTiling 一致直接按 1 槽读取，
    // 返回空时兜底直读 faker 注入的 workspace 向量首槽
    const auto wsSizes = tilingContext->GetWorkspaceSizes(1U);
    if (wsSizes != nullptr) {
        result.workspaceSize = wsSizes[0];
    } else {
        result.workspaceSize = *(reinterpret_cast<const size_t*>(wsSize->GetData()));
    }
    const auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData != nullptr && rawTilingData->GetData() != nullptr && rawTilingData->GetDataSize() > 0U) {
        result.tilingData.resize(rawTilingData->GetDataSize());
        (void)std::memcpy(result.tilingData.data(), rawTilingData->GetData(), rawTilingData->GetDataSize());
    }
    return result;
}

// ---- raw tiling data 反解（字段顺序见 softmax_grad_ext_tiling.h 的 TILING_DATA_FIELD_DEF）----
int64_t TdI64(const std::vector<uint8_t>& data, const size_t idx)
{
    return reinterpret_cast<const int64_t*>(data.data())[idx];
}
uint16_t TdU16(const std::vector<uint8_t>& data, const size_t idx)
{
    return reinterpret_cast<const uint16_t*>(data.data() + ARA_UT_U16_BASE)[idx];
}
uint32_t TdU32(const std::vector<uint8_t>& data, const size_t idx)
{
    return reinterpret_cast<const uint32_t*>(data.data() + ARA_UT_U32_BASE)[idx];
}
} // namespace

// U1: ARA full load fp32 基础场景。
// 合轴 (a1=3, r=64, a0=100)：factorMax = 237568/2322/64 = 1；a0Inner=1、tileA0Len=64、a0Outer=2。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_full_load_fp32)
{
    const auto res = RunSoftmaxGradExtAraTiling({3, 64, 100}, {}, ge::DT_FLOAT, 1);
    // ARA 模板依赖 axes=1（中间轴 reduce，尾轴 a0=100>1）；若 tiling base 拒绝中间轴 reduce，
    // 则 ARA 模板（TKEY 10000/20000）不可达，本断言失败即暴露该问题
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_FULL_LOAD);
    // 切分字段：int64 段 idx 0~8
    EXPECT_EQ(TdI64(res.tilingData, 0), 64);  // totalRLen
    EXPECT_EQ(TdI64(res.tilingData, 1), 100); // totalA0Len
    EXPECT_EQ(TdI64(res.tilingData, 2), 6);   // totalTiles = 3 * CeilDiv(100, 64)
    EXPECT_EQ(TdI64(res.tilingData, 3), 1);   // tilesPerCore = CeilDiv(6, 64)
    EXPECT_EQ(TdI64(res.tilingData, 4), 6);   // usedCoreNums = CeilDiv(6, 1)
    EXPECT_EQ(TdI64(res.tilingData, 5), 2);   // a0Outer = CeilDiv(100, 64)
    EXPECT_EQ(TdI64(res.tilingData, 6), 64);  // tileA0Len = 1 * 64（a0TileBase=64 的整数倍）
    EXPECT_EQ(TdI64(res.tilingData, 7), 36);  // tileA0Tail = 100 - 64 * (2 - 1)
    EXPECT_EQ(TdI64(res.tilingData, 8), 0);   // x2IsScalar = 0（x2 同 shape）
    EXPECT_EQ(res.blockDim, 6U);              // SetBlockDim(usedCoreNums)
    // r=64 > 8 触发二分累加：quotient=32、K=1、last=0（tileA0Len=64）
    EXPECT_EQ(TdU16(res.tilingData, 0), 1);      // binaryAddK
    EXPECT_EQ(TdU16(res.tilingData, 1), 0);      // binaryAddLast
    EXPECT_EQ(TdU16(res.tilingData, 2), 4);      // binaryAddInnerLoop = 32 / 8
    EXPECT_EQ(TdU16(res.tilingData, 3), 4);      // remainderLoopCount = CeilDiv(64-32, 8)
    EXPECT_EQ(TdU16(res.tilingData, 4), 0);      // quotientLoopCount = 32/8 - 4
    EXPECT_EQ(TdU16(res.tilingData, 5), 8);      // remainderTailCount = 32 - (4-1)*8
    EXPECT_EQ(TdU32(res.tilingData, 0), 2048U);  // remainderOffset = 32 * 64
    EXPECT_EQ(TdU32(res.tilingData, 1), 512U);   // baseLineOffset = 8 * 64
    EXPECT_EQ(TdU32(res.tilingData, 2), 4096U);  // validNumInXUb = 64 * 64
    EXPECT_EQ(TdU32(res.tilingData, 3), 1536U);  // quotientTailOffset = (4-1) * 512
    EXPECT_EQ(TdU32(res.tilingData, 4), 3584U);  // remainderTailOffset = 1536 + 2048
    EXPECT_EQ(TdU32(res.tilingData, 5), 3584U);  // remainderTailOffset0
    EXPECT_EQ(TdU32(res.tilingData, 6), 3648U);  // remainderTailOffset1
    EXPECT_EQ(TdU32(res.tilingData, 12), 4032U); // remainderTailOffset7
}

// U2: ARA full load fp16。合轴 (7, 80, 64)：factorMax = 237568/1608/128 = 1。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_full_load_fp16)
{
    const auto res = RunSoftmaxGradExtAraTiling({7, 80, 64}, {}, ge::DT_FLOAT16, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_FULL_LOAD);
    EXPECT_EQ(TdI64(res.tilingData, 2), 7);   // totalTiles = 7 * CeilDiv(64, 128)
    EXPECT_EQ(TdI64(res.tilingData, 3), 1);   // tilesPerCore
    EXPECT_EQ(TdI64(res.tilingData, 4), 7);   // usedCoreNums
    EXPECT_EQ(TdI64(res.tilingData, 5), 1);   // a0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 128); // tileA0Len（a0TileBase=128 的整数倍）
    EXPECT_EQ(TdI64(res.tilingData, 7), 64);  // tileA0Tail = 64
    EXPECT_EQ(TdI64(res.tilingData, 8), 0);   // x2IsScalar
    EXPECT_EQ(res.blockDim, 7U);
    // r=80：quotient=64、binaryAddNum=8、cur 折叠至 16=2*8 → K=1、last=1（tileA0Len=128）
    EXPECT_EQ(TdU16(res.tilingData, 0), 1);       // binaryAddK
    EXPECT_EQ(TdU16(res.tilingData, 1), 1);       // binaryAddLast
    EXPECT_EQ(TdU16(res.tilingData, 2), 8);       // binaryAddInnerLoop = 64 / 8
    EXPECT_EQ(TdU16(res.tilingData, 3), 2);       // remainderLoopCount = CeilDiv(80-64, 8)
    EXPECT_EQ(TdU16(res.tilingData, 4), 6);       // quotientLoopCount = 64/8 - 2
    EXPECT_EQ(TdU16(res.tilingData, 5), 8);       // remainderTailCount = 16 - 8
    EXPECT_EQ(TdU32(res.tilingData, 0), 8192U);   // remainderOffset = 64 * 128
    EXPECT_EQ(TdU32(res.tilingData, 1), 1024U);   // baseLineOffset = 8 * 128
    EXPECT_EQ(TdU32(res.tilingData, 2), 10240U);  // validNumInXUb = 80 * 128
    EXPECT_EQ(TdU32(res.tilingData, 3), 1024U);   // quotientTailOffset = (2-1) * 1024
    EXPECT_EQ(TdU32(res.tilingData, 4), 9216U);   // remainderTailOffset = 1024 + 8192
    EXPECT_EQ(TdU32(res.tilingData, 5), 9216U);   // remainderTailOffset0
    EXPECT_EQ(TdU32(res.tilingData, 12), 10112U); // remainderTailOffset7
}

// U3: ARA full load bf16。合轴 (2, 64, 200)：factorMax = 237568/1288/128 = 1；tileA0Tail=72 非对齐尾块。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_full_load_bf16)
{
    const auto res = RunSoftmaxGradExtAraTiling({2, 64, 200}, {}, ge::DT_BF16, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_FULL_LOAD);
    EXPECT_EQ(TdI64(res.tilingData, 2), 4);   // totalTiles = 2 * CeilDiv(200, 128)
    EXPECT_EQ(TdI64(res.tilingData, 3), 1);   // tilesPerCore
    EXPECT_EQ(TdI64(res.tilingData, 4), 4);   // usedCoreNums
    EXPECT_EQ(TdI64(res.tilingData, 5), 2);   // a0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 128); // tileA0Len
    EXPECT_EQ(TdI64(res.tilingData, 7), 72);  // tileA0Tail = 200 - 128
    EXPECT_EQ(TdI64(res.tilingData, 8), 0);   // x2IsScalar
    EXPECT_EQ(res.blockDim, 4U);
    // r=64（同 U1 结构，tileA0Len=128）：quotient=32、K=1、last=0
    EXPECT_EQ(TdU16(res.tilingData, 0), 1);
    EXPECT_EQ(TdU16(res.tilingData, 1), 0);
    EXPECT_EQ(TdU16(res.tilingData, 3), 4);     // remainderLoopCount
    EXPECT_EQ(TdU16(res.tilingData, 4), 0);     // quotientLoopCount
    EXPECT_EQ(TdU32(res.tilingData, 0), 4096U); // remainderOffset = 32 * 128
    EXPECT_EQ(TdU32(res.tilingData, 1), 1024U); // baseLineOffset = 8 * 128
    EXPECT_EQ(TdU32(res.tilingData, 2), 8192U); // validNumInXUb = 64 * 128
    EXPECT_EQ(TdU32(res.tilingData, 4), 7168U); // remainderTailOffset = 3072 + 4096
}

// U4: ARA full load fp16 x2 标量分支。合轴 (5, 32, 320)：factorMax = 237568/648/128 = 2。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_full_load_x2_scalar)
{
    const auto res = RunSoftmaxGradExtAraTiling({5, 32, 320}, {1}, ge::DT_FLOAT16, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_FULL_LOAD);
    EXPECT_EQ(TdI64(res.tilingData, 8), 1);   // x2IsScalar = 1（x2 shape 为 (1,)）
    EXPECT_EQ(TdI64(res.tilingData, 2), 15);  // totalTiles = 5 * CeilDiv(320, 128)
    EXPECT_EQ(TdI64(res.tilingData, 3), 1);   // tilesPerCore
    EXPECT_EQ(TdI64(res.tilingData, 4), 15);  // usedCoreNums
    EXPECT_EQ(TdI64(res.tilingData, 5), 3);   // a0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 128); // tileA0Len
    EXPECT_EQ(TdI64(res.tilingData, 7), 64);  // tileA0Tail = 320 - 256
    EXPECT_EQ(res.blockDim, 15U);
    // r=32：quotient=16、binaryAddNum=2、cur 折叠至 4=2*2 → K=0、last=1
    EXPECT_EQ(TdU16(res.tilingData, 0), 0);     // binaryAddK
    EXPECT_EQ(TdU16(res.tilingData, 1), 1);     // binaryAddLast
    EXPECT_EQ(TdU16(res.tilingData, 3), 2);     // remainderLoopCount = CeilDiv(32-16, 8)
    EXPECT_EQ(TdU16(res.tilingData, 4), 0);     // quotientLoopCount = 16/8 - 2
    EXPECT_EQ(TdU32(res.tilingData, 0), 2048U); // remainderOffset = 16 * 128
    EXPECT_EQ(TdU32(res.tilingData, 2), 4096U); // validNumInXUb = 32 * 128
    EXPECT_EQ(TdU32(res.tilingData, 4), 3072U); // remainderTailOffset = 1024 + 2048
}

// U5: ARA full load r==8 边界（跳过二分累加分支）。合轴 (4, 8, 256)：factorMax = 237568/296/64 = 12。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_full_load_small_r)
{
    const auto res = RunSoftmaxGradExtAraTiling({4, 8, 256}, {}, ge::DT_FLOAT, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_FULL_LOAD);
    EXPECT_EQ(TdI64(res.tilingData, 0), 8);  // totalRLen（r==8 边界）
    EXPECT_EQ(TdI64(res.tilingData, 2), 16); // totalTiles = 4 * CeilDiv(256, 64)
    EXPECT_EQ(TdI64(res.tilingData, 3), 1);  // tilesPerCore
    EXPECT_EQ(TdI64(res.tilingData, 4), 16); // usedCoreNums
    EXPECT_EQ(TdI64(res.tilingData, 5), 4);  // a0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 64); // tileA0Len
    EXPECT_EQ(TdI64(res.tilingData, 7), 64); // tileA0Tail（整除时等于 tileA0Len）
    EXPECT_EQ(res.blockDim, 16U);
    // r<=8 走 SumRLessThan8 短路径，二分累加字段不写入（保持 0）
    EXPECT_EQ(TdU16(res.tilingData, 0), 0);  // binaryAddK
    EXPECT_EQ(TdU16(res.tilingData, 1), 0);  // binaryAddLast
    EXPECT_EQ(TdU32(res.tilingData, 2), 0U); // validNumInXUb
}

// U6: ARA full load r==9（首次触发二分累加分支）。合轴 (4, 9, 128)：factorMax = 237568/188/128 = 9。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_full_load_r9)
{
    const auto res = RunSoftmaxGradExtAraTiling({4, 9, 128}, {}, ge::DT_FLOAT16, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_FULL_LOAD);
    EXPECT_EQ(TdI64(res.tilingData, 2), 4);   // totalTiles = 4 * CeilDiv(128, 128)
    EXPECT_EQ(TdI64(res.tilingData, 4), 4);   // usedCoreNums
    EXPECT_EQ(TdI64(res.tilingData, 5), 1);   // a0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 128); // tileA0Len
    EXPECT_EQ(TdI64(res.tilingData, 7), 128); // tileA0Tail
    // r=9：quotient=8、binaryAddNum=1、cur=1==binaryAddNum → K=0、last=0
    EXPECT_EQ(TdU16(res.tilingData, 0), 0);      // binaryAddK
    EXPECT_EQ(TdU16(res.tilingData, 1), 0);      // binaryAddLast
    EXPECT_EQ(TdU16(res.tilingData, 2), 1);      // binaryAddInnerLoop = 8 / 8
    EXPECT_EQ(TdU16(res.tilingData, 3), 1);      // remainderLoopCount = CeilDiv(9-8, 8)
    EXPECT_EQ(TdU16(res.tilingData, 4), 0);      // quotientLoopCount = 8/8 - 1
    EXPECT_EQ(TdU16(res.tilingData, 5), 1);      // remainderTailCount = 1
    EXPECT_EQ(TdU32(res.tilingData, 0), 1024U);  // remainderOffset = 8 * 128
    EXPECT_EQ(TdU32(res.tilingData, 1), 1024U);  // baseLineOffset = 8 * 128
    EXPECT_EQ(TdU32(res.tilingData, 2), 1152U);  // validNumInXUb = 9 * 128
    EXPECT_EQ(TdU32(res.tilingData, 3), 0U);     // quotientTailOffset = (1-1) * 1024
    EXPECT_EQ(TdU32(res.tilingData, 4), 1024U);  // remainderTailOffset
    EXPECT_EQ(TdU32(res.tilingData, 5), 1024U);  // remainderTailOffset0（0 > 1 不成立）
    EXPECT_EQ(TdU32(res.tilingData, 6), 1152U);  // remainderTailOffset1（1 > 1 不成立）
    EXPECT_EQ(TdU32(res.tilingData, 7), 1152U);  // remainderTailOffset2 起钳位到 validNumInXUb
    EXPECT_EQ(TdU32(res.tilingData, 12), 1152U); // remainderTailOffset7
}

// U7: ARA recompute fp16。合轴 (2, 65536, 64)：full load factorMax = 237568/1310728/128 = 0 → fallback。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_recompute_fp16)
{
    const auto res = RunSoftmaxGradExtAraTiling({2, 65536, 64}, {}, ge::DT_FLOAT16, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_RECOMPUTE);
    // ARARecomputeTilingData：int64*17，idx 0~16（mc 后追加 idx 17~19）
    EXPECT_EQ(TdI64(res.tilingData, 0), 65536); // totalRLen
    EXPECT_EQ(TdI64(res.tilingData, 1), 64);    // totalA0Len
    EXPECT_EQ(TdI64(res.tilingData, 2), 2);     // totalTiles = a1（mc 满核 R 切分）
    EXPECT_EQ(TdI64(res.tilingData, 3), 1);     // tilesPerCore
    EXPECT_EQ(TdI64(res.tilingData, 4), 64);    // usedCoreNums = a1 * (64 / a1)
    EXPECT_EQ(TdI64(res.tilingData, 5), 1);     // tileA0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 64);    // tileA0Len = a0 全宽（mc）
    EXPECT_EQ(TdI64(res.tilingData, 7), 64);    // tileA0Tail
    EXPECT_EQ(TdI64(res.tilingData, 8), 0);     // x2IsScalar
    EXPECT_EQ(TdI64(res.tilingData, 9), 128);   // binAddRFactor = 128
    EXPECT_EQ(TdI64(res.tilingData, 10), 512);  // binAddRLoop = 65536 / 128
    EXPECT_EQ(TdI64(res.tilingData, 11), 512);  // binAddRTotalLoop = CeilDiv(65536, 128)
    EXPECT_EQ(TdI64(res.tilingData, 12), 128);  // binAddRTail
    EXPECT_EQ(TdI64(res.tilingData, 13), 256);  // binAddBasicBlockLoop = 2^floor(log2(511))
    EXPECT_EQ(TdI64(res.tilingData, 14), 256);  // binAddMainFoldCount = 512 - 256
    EXPECT_EQ(TdI64(res.tilingData, 15), 9);    // binAddCacheBufferCount = 64 - clz(256)
    EXPECT_EQ(TdI64(res.tilingData, 16), 8);    // binAddResultCacheID = GetCacheID(255)
    EXPECT_EQ(TdI64(res.tilingData, 17), 1);    // mcEnabled：a0==64 且 totalTilesMax(8) < 核数(64)
    EXPECT_EQ(TdI64(res.tilingData, 18), 2);    // mcA1Len
    EXPECT_EQ(TdI64(res.tilingData, 19), 32);   // mcChunksPerA1 = 64 / 2
    EXPECT_EQ(res.blockDim, 64U);
}

// U8: ARA recompute fp32。合轴 (2, 32768, 32)：full load factorMax = 237568/1179656/64 = 0 → fallback。
// a0=32 整除 64 lane 且非 mc 占不满核 → mc 启用（64 核，a0<64 掩码段路径）。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_recompute_fp32)
{
    const auto res = RunSoftmaxGradExtAraTiling({2, 32768, 32}, {}, ge::DT_FLOAT, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_RECOMPUTE);
    EXPECT_EQ(TdI64(res.tilingData, 0), 32768); // totalRLen
    EXPECT_EQ(TdI64(res.tilingData, 1), 32);    // totalA0Len
    EXPECT_EQ(TdI64(res.tilingData, 2), 2);     // totalTiles = a1（mc 满核 R 切分）
    EXPECT_EQ(TdI64(res.tilingData, 3), 1);     // tilesPerCore
    EXPECT_EQ(TdI64(res.tilingData, 4), 64);    // usedCoreNums = a1 * (64 / a1)
    EXPECT_EQ(TdI64(res.tilingData, 5), 1);     // tileA0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 32);    // tileA0Len = a0 全宽（mc）
    EXPECT_EQ(TdI64(res.tilingData, 7), 32);    // tileA0Tail
    EXPECT_EQ(TdI64(res.tilingData, 8), 0);     // x2IsScalar
    EXPECT_EQ(TdI64(res.tilingData, 9), 128);   // binAddRFactor（mc 段因子 = min(128, segMax≈218)）
    EXPECT_EQ(TdI64(res.tilingData, 10), 256);  // binAddRLoop
    EXPECT_EQ(TdI64(res.tilingData, 11), 256);  // binAddRTotalLoop
    EXPECT_EQ(TdI64(res.tilingData, 12), 128);  // binAddRTail
    EXPECT_EQ(TdI64(res.tilingData, 13), 128);  // binAddBasicBlockLoop
    EXPECT_EQ(TdI64(res.tilingData, 14), 128);  // binAddMainFoldCount
    EXPECT_EQ(TdI64(res.tilingData, 15), 8);    // binAddCacheBufferCount
    EXPECT_EQ(TdI64(res.tilingData, 16), 7);    // binAddResultCacheID
    EXPECT_EQ(TdI64(res.tilingData, 17), 1);    // mcEnabled：a0=32 整除 64 且 totalTilesMax(8) < 核数(64)
    EXPECT_EQ(TdI64(res.tilingData, 18), 2);    // mcA1Len
    EXPECT_EQ(TdI64(res.tilingData, 19), 32);   // mcChunksPerA1 = 64 / 2
    EXPECT_EQ(res.blockDim, 64U);
}

// U9: ARA recompute bf16 x2 标量分支。合轴 (1, 49152, 128)：full load factorMax = 237568/983048/128 = 0。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_recompute_x2_scalar)
{
    const auto res = RunSoftmaxGradExtAraTiling({1, 49152, 128}, {1}, ge::DT_BF16, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_RECOMPUTE);
    EXPECT_EQ(TdI64(res.tilingData, 8), 1);     // x2IsScalar = 1
    EXPECT_EQ(TdI64(res.tilingData, 0), 49152); // totalRLen
    EXPECT_EQ(TdI64(res.tilingData, 1), 128);   // totalA0Len
    EXPECT_EQ(TdI64(res.tilingData, 2), 8);     // totalTiles = 1 * CeilDiv(128, 16)
    EXPECT_EQ(TdI64(res.tilingData, 4), 8);     // usedCoreNums
    EXPECT_EQ(TdI64(res.tilingData, 5), 8);     // tileA0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 16);    // tileA0Len（bf16 a0TileBase=16）
    EXPECT_EQ(TdI64(res.tilingData, 7), 16);    // tileA0Tail
    EXPECT_EQ(TdI64(res.tilingData, 9), 128);   // binAddRFactor
    EXPECT_EQ(TdI64(res.tilingData, 10), 384);  // binAddRLoop = 49152 / 128
    EXPECT_EQ(TdI64(res.tilingData, 11), 384);  // binAddRTotalLoop
    EXPECT_EQ(TdI64(res.tilingData, 12), 128);  // binAddRTail
    EXPECT_EQ(TdI64(res.tilingData, 13), 256);  // binAddBasicBlockLoop = 2^floor(log2(383))
    EXPECT_EQ(TdI64(res.tilingData, 14), 128);  // binAddMainFoldCount = 384 - 256
    EXPECT_EQ(TdI64(res.tilingData, 15), 9);    // binAddCacheBufferCount
    EXPECT_EQ(TdI64(res.tilingData, 16), 8);    // binAddResultCacheID = GetCacheID(255)
    EXPECT_EQ(res.blockDim, 8U);
}

// U10: 回归——a0==1（二维输入，axes=1 即最后一维）仍选 AR full load，不被 ARA 抢占。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_regression_ar_priority)
{
    const auto res = RunSoftmaxGradExtAraTiling({704, 385}, {}, ge::DT_FLOAT, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_AR_FULL_LOAD);
    // ARTilingData：int64*7，idx 0~6
    EXPECT_EQ(TdI64(res.tilingData, 0), 704); // a = a1
    EXPECT_EQ(TdI64(res.tilingData, 1), 385); // r
    EXPECT_EQ(TdI64(res.tilingData, 2), 392); // rAligned = CeilAlign(385, 8)
    EXPECT_EQ(TdI64(res.tilingData, 3), 11);  // ubFactor = min(18, aBlockFactor)
    EXPECT_EQ(TdI64(res.tilingData, 4), 11);  // aBlockFactor = CeilDiv(704, 64)
    EXPECT_EQ(TdI64(res.tilingData, 5), 7);   // rLoopCount = CeilDiv(392, 64)
    EXPECT_EQ(TdI64(res.tilingData, 6), 0);   // x2IsScalar
    EXPECT_EQ(res.blockDim, 64U);             // usedCoreNums = CeilDiv(704, 11)
}

// U11: ARA recompute r 非 128 整数倍（binAddRTail 尾段路径）。合轴 (3, 4225, 96)：
// full load factorMax = 237568/84508/128 = 0 → fallback。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_recompute_r_tail)
{
    const auto res = RunSoftmaxGradExtAraTiling({3, 4225, 96}, {}, ge::DT_FLOAT16, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_RECOMPUTE);
    EXPECT_EQ(TdI64(res.tilingData, 0), 4225); // totalRLen
    EXPECT_EQ(TdI64(res.tilingData, 1), 96);   // totalA0Len
    EXPECT_EQ(TdI64(res.tilingData, 2), 18);   // totalTiles = 3 * CeilDiv(96, 16)
    EXPECT_EQ(TdI64(res.tilingData, 3), 1);    // tilesPerCore
    EXPECT_EQ(TdI64(res.tilingData, 4), 18);   // usedCoreNums
    EXPECT_EQ(TdI64(res.tilingData, 5), 6);    // tileA0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 16);   // tileA0Len
    EXPECT_EQ(TdI64(res.tilingData, 7), 16);   // tileA0Tail
    EXPECT_EQ(TdI64(res.tilingData, 8), 0);    // x2IsScalar
    EXPECT_EQ(TdI64(res.tilingData, 9), 128);  // binAddRFactor
    EXPECT_EQ(TdI64(res.tilingData, 10), 33);  // binAddRLoop = 4225 / 128
    EXPECT_EQ(TdI64(res.tilingData, 11), 34);  // binAddRTotalLoop = CeilDiv(4225, 128)
    EXPECT_EQ(TdI64(res.tilingData, 12), 1);   // binAddRTail = 4225 - 33*128
    EXPECT_EQ(TdI64(res.tilingData, 13), 32);  // binAddBasicBlockLoop = 2^floor(log2(33))
    EXPECT_EQ(TdI64(res.tilingData, 14), 1);   // binAddMainFoldCount = 33 - 32
    EXPECT_EQ(TdI64(res.tilingData, 15), 6);   // binAddCacheBufferCount = 64 - clz(32)
    EXPECT_EQ(TdI64(res.tilingData, 16), 5);   // binAddResultCacheID = GetCacheID(31)
    EXPECT_EQ(res.blockDim, 18U);
}

// U12: ARA full load 容量上界邻域（fp16 R 上界约 92）。合轴 (2, 92, 128)：
// factorMax = 237568/1848/128 = 1，恰好可装下单 tile。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_full_load_capacity_boundary)
{
    const auto res = RunSoftmaxGradExtAraTiling({2, 92, 128}, {}, ge::DT_FLOAT16, 1);
    ASSERT_EQ(res.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    EXPECT_EQ(res.tilingKey, ARA_UT_TKEY_ARA_FULL_LOAD);
    EXPECT_EQ(TdI64(res.tilingData, 0), 92);  // totalRLen（R 容量边界邻域）
    EXPECT_EQ(TdI64(res.tilingData, 2), 2);   // totalTiles = 2 * CeilDiv(128, 128)
    EXPECT_EQ(TdI64(res.tilingData, 3), 1);   // tilesPerCore
    EXPECT_EQ(TdI64(res.tilingData, 4), 2);   // usedCoreNums
    EXPECT_EQ(TdI64(res.tilingData, 5), 1);   // a0Outer
    EXPECT_EQ(TdI64(res.tilingData, 6), 128); // tileA0Len
    EXPECT_EQ(TdI64(res.tilingData, 7), 128); // tileA0Tail
    EXPECT_EQ(res.blockDim, 2U);
    // r=92：quotient=64、K=1、last=1；remainderTailCount=4 → offset4 之后钳位
    EXPECT_EQ(TdU16(res.tilingData, 0), 1);       // binaryAddK
    EXPECT_EQ(TdU16(res.tilingData, 1), 1);       // binaryAddLast
    EXPECT_EQ(TdU16(res.tilingData, 3), 4);       // remainderLoopCount = CeilDiv(92-64, 8)
    EXPECT_EQ(TdU16(res.tilingData, 4), 4);       // quotientLoopCount = 8 - 4
    EXPECT_EQ(TdU16(res.tilingData, 5), 4);       // remainderTailCount = 28 - 24
    EXPECT_EQ(TdU32(res.tilingData, 0), 8192U);   // remainderOffset = 64 * 128
    EXPECT_EQ(TdU32(res.tilingData, 2), 11776U);  // validNumInXUb = 92 * 128
    EXPECT_EQ(TdU32(res.tilingData, 3), 3072U);   // quotientTailOffset = 3 * 1024
    EXPECT_EQ(TdU32(res.tilingData, 4), 11264U);  // remainderTailOffset = 3072 + 8192
    EXPECT_EQ(TdU32(res.tilingData, 5), 11264U);  // remainderTailOffset0
    EXPECT_EQ(TdU32(res.tilingData, 8), 11648U);  // remainderTailOffset3
    EXPECT_EQ(TdU32(res.tilingData, 9), 11776U);  // remainderTailOffset4（4 > 4 不成立）
    EXPECT_EQ(TdU32(res.tilingData, 10), 11776U); // remainderTailOffset5 钳位到 validNumInXUb
    EXPECT_EQ(TdU32(res.tilingData, 12), 11776U); // remainderTailOffset7
}

// U13: 负轴等价性——axes=1 与 axes=-2 归一化到同一 reduce 轴，tiling 结果应完全一致。
// 取 U7 的 ARA recompute shape (2, 65536, 64) fp16。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_recompute_fp16_neg_axes)
{
    const auto resPos = RunSoftmaxGradExtAraTiling({2, 65536, 64}, {}, ge::DT_FLOAT16, 1);
    const auto resNeg = RunSoftmaxGradExtAraTiling({2, 65536, 64}, {}, ge::DT_FLOAT16, -2);
    ASSERT_EQ(resPos.status, ge::GRAPH_SUCCESS) << "ARA requires middle-axis reduce: axes=1 on 3-dim (a1, r, a0) input";
    ASSERT_EQ(resNeg.status, ge::GRAPH_SUCCESS) << "axes=-2 should be normalized to the same reduce axis as axes=1";
    EXPECT_EQ(resNeg.tilingKey, resPos.tilingKey);
    EXPECT_EQ(resNeg.blockDim, resPos.blockDim);
    ASSERT_EQ(resNeg.tilingData.size(), resPos.tilingData.size());
    EXPECT_EQ(resNeg.tilingData, resPos.tilingData);
}

// U14: axes 越界拒绝——3 维输入下 axes=3（== dim）与 axes=-4（< -dim）均应 tiling 失败。
TEST_F(SoftmaxGradExtTiling, softmax_grad_ext_ara_axes_out_of_range)
{
    const auto resUpper = RunSoftmaxGradExtAraTiling({2, 65536, 64}, {}, ge::DT_FLOAT16, 3);
    EXPECT_EQ(resUpper.status, ge::GRAPH_FAILED) << "axes == dim is out of range and must be rejected";
    const auto resLower = RunSoftmaxGradExtAraTiling({2, 65536, 64}, {}, ge::DT_FLOAT16, -4);
    EXPECT_EQ(resLower.status, ge::GRAPH_FAILED) << "axes < -dim is out of range and must be rejected";
}
