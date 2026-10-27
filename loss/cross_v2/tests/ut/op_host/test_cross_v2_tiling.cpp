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
 * \file test_cross_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/cross_v2_tiling.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "array_ops.h"

using namespace std;
using namespace ge;

class CrossV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CrossV2Tiling Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "CrossV2Tiling TearDown" << std::endl; }
};

TEST_F(CrossV2Tiling, cross_v2_fp32_001)
{
    gert::StorageShape x_shape = {{4096, 3}, {4096, 3}};
    int32_t dim = -4;

    std::vector<gert::StorageShape> output_shapes(1, {{4096, 3}, {4096, 3}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                         "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct CrossV2CompileInfo {};
    CrossV2CompileInfo compile_info;

    std::string op_type("CrossV2");
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
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &x_shape})
                      .OutputShapes({&x_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

namespace {
// 错误码整改负向测试公共构造：与正向用例相同的平台信息/编译信息准备流程
struct CrossV2TilingNegativeEnv {
    fe::PlatFormInfos platform_info;
    struct CrossV2CompileInfo {};
    CrossV2CompileInfo compile_info;
    std::string op_type;

    CrossV2TilingNegativeEnv()
    {
        string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                             "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                             "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                             "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                             "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                             "CORE_NUM": 40}
                              })";
        map<string, string> soc_infos;
        map<string, string> aicore_spec;
        map<string, string> intrinsics;
        GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
        platform_info.Init();
        platform_info.SetPlatformRes("SoCInfo", soc_infos);
        platform_info.SetPlatformRes("AICoreSpec", aicore_spec);
        platform_info.SetCoreNumByCoreType("AICore");
        platform_info.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
        op_type = "CrossV2";
    }
};
} // namespace

// 负向场景1：必选输入 x1 的输入实例缺失（IrInstanceNum 全 0，inputs_num_ 为 0），
// GetInputShape(0) 按下标越界规则返回 nullptr，
// 整改后应通过 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON 上报 EZ0037 首错并返回失败，不产生内部码首报
TEST_F(CrossV2Tiling, cross_v2_tiling_null_input_shape_001)
{
    CrossV2TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape x_shape = {{4096, 3}, {4096, 3}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    // 输入实例数全 0 使 ComputeNodeInfo 的 inputs_num_ 为 0，GetInputShape(0)/GetInputDesc(0) 返回空指针；
    // 仍传入 1 个输出 shape 保持输出链存在，使 compile/platform 等扩展链的槽位按声明序对齐
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({0, 0}, {1})
                      .OutputShapes({&x_shape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(1))}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_context->GetInputShape(0), nullptr);
    ASSERT_EQ(tiling_context->GetInputDesc(0), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景覆盖说明（TilingContextFaker 能力限制，未单独构造用例）：
// 1. attrs 判空检查点（cross_v2_tiling.cpp 中 GetAttrs()）：TilingContext::GetAttrs() 仅在
//    ComputeNodeInfo 缺失时返回空，而 TilingContextFaker::Build 必然构造 ComputeNodeInfo，
//    且 GetAttrs 为固定偏移计算永不为空，该检查点无法通过 faker 构造（与场景1同属 EZ0037 上报路由）；
// 2. x1Desc 判空检查点：GetInputDesc(0) 与 GetInputShape(0) 同受 inputs_num_ 越界约束，
//    desc 为空时 shape 必然先为空并在首个检查点提前返回，故由场景1一并覆盖（见上方对 GetInputDesc 的断言）。

TEST_F(CrossV2Tiling, cross_v2_fp32_002)
{
    gert::StorageShape x_shape = {{4096, 3, 1000}, {4096, 3, 1000}};
    int32_t dim = 1;

    std::vector<gert::StorageShape> output_shapes(1, {{4096, 3, 1000}, {4096, 3, 1000}});
    std::vector<void*> output_shapes_ref(1);
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                         "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct CrossV2CompileInfo {};
    CrossV2CompileInfo compile_info;

    std::string op_type("CrossV2");
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
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x_shape, &x_shape})
                      .OutputShapes({&x_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

// 负向场景2：TilingParse 阶段 context 为空（tilingParseFunc 直接收 nullptr），
// 对应整改检查点 ReportTilingFailure("TilingPrepareForCrossV2", "Tiling context is nullptr")（EZ1008 首报）
TEST_F(CrossV2Tiling, cross_v2_tiling_parse_null_context_002)
{
    CrossV2TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling_parse;

    EXPECT_EQ(tiling_parse_func(nullptr), ge::GRAPH_FAILED);
}

// 负向场景3：平台 SoCInfo 中移除 ai_core_cnt 使 GetCoreNumAiv 返回 0，
// 对应整改检查点 ReportTilingFailure(context->GetNodeName(), "coreNum should not be 0!")（EZ1008 首报）
TEST_F(CrossV2Tiling, cross_v2_tiling_core_num_zero_003)
{
    CrossV2TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape x_shape = {{4096, 3}, {4096, 3}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1}, {1})
                      .InputShapes({&x_shape, &x_shape})
                      .OutputShapes({&x_shape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(1))}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 与正向用例一致先补齐平台资源，再用移除 ai_core_cnt 的 SoCInfo 覆盖，使 GetCoreNumAiv 返回 0
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                         "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                          })";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    soc_infos.erase("ai_core_cnt");
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景4：dim 属性为 5 越界（x1 为 2 维），DimProcess 返回 5 >= dimNum，
// 对应整改检查点 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON（dim 取值非法分支）
TEST_F(CrossV2Tiling, cross_v2_tiling_invalid_dim_004)
{
    CrossV2TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape x_shape = {{4096, 3}, {4096, 3}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1}, {1})
                      .InputShapes({&x_shape, &x_shape})
                      .OutputShapes({&x_shape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(5))}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
