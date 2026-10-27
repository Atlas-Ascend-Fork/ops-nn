/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_host/fused_adamw_tiling.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class FusedAdamWTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedAdamWTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FusedAdamWTiling TearDown" << std::endl; }
};

TEST_F(FusedAdamWTiling, test_tiling_float32)
{
    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape maxExpAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stateStepsShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                        "Intrinsic_fix_pipe_l0c2out": false,
                                                        "Intrinsic_data_move_l12ub": true,
                                                        "Intrinsic_data_move_l0c2ub": true,
                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                        "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct FusedAdamwCompileInfo {};
    FusedAdamwCompileInfo compile_info;

    std::string op_type("FusedAdamw");
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stateStepsShape, &gradScaleShape, &foundInfShape})
                      .OutputShapes({&paramsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.99f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

TEST_F(FusedAdamWTiling, test_tiling_float16)
{
    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape maxExpAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stateStepsShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                        "Intrinsic_fix_pipe_l0c2out": false,
                                                        "Intrinsic_data_move_l12ub": true,
                                                        "Intrinsic_data_move_l0c2ub": true,
                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                        "CORE_NUM": 48}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct FusedAdamwCompileInfo {};
    FusedAdamwCompileInfo compile_info;

    std::string op_type("FusedAdamw");
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stateStepsShape, &gradScaleShape, &foundInfShape})
                      .OutputShapes({&paramsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.99f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

namespace {
// 错误码整改负向测试公共构造：与正向用例相同的平台信息/编译信息准备流程
struct FusedAdamwTilingNegativeEnv {
    fe::PlatFormInfos platform_info;
    struct FusedAdamwCompileInfo {};
    FusedAdamwCompileInfo compile_info;
    std::string op_type;

    FusedAdamwTilingNegativeEnv()
    {
        string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                        "Intrinsic_fix_pipe_l0c2out": false,
                                                        "Intrinsic_data_move_l12ub": true,
                                                        "Intrinsic_data_move_l0c2ub": true,
                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                        "CORE_NUM": 48}
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
        op_type = "FusedAdamw";
    }
};
} // namespace

// 负向场景1：tiling 上下文未设置平台信息，GetPlatformInfo 返回空指针，
// 整改后应通过 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON 上报 EZ0037（param_name=platformInfo）首错并返回失败
TEST_F(FusedAdamWTiling, fused_adamw_tiling_null_platform_info_001)
{
    FusedAdamwTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape maxExpAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stateStepsShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};

    // 注入法构造：正常构建（含 PlatformInfo）后向 platform 槽位（inputs_num_+outputs_num_+1 = 13）
    // 注入空值使 GetPlatformInfo 返回空指针；直接省略 PlatformInfo 会使
    // OpTilingContextBuilder::Build 返回空 holder（GetContext 为空，用例自身段错误），不可行
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 4)
                      .InputShapes({&paramsShape, &gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stateStepsShape, &gradScaleShape, &foundInfShape})
                      .OutputShapes({&paramsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.99f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    // 向 platform 槽位注入空值（先例：test_ctc_loss_v3_tiling.cpp parse_null_platform_info_005）；
    // MutableInput 为 KernelContext 公有接口，经 holder 取 KernelContext 视图访问
    auto* kernelContext = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(kernelContext, nullptr);
    auto* platformSlot = kernelContext->MutableInput(13);
    ASSERT_NE(platformSlot, nullptr);
    platformSlot->Set(nullptr, nullptr);
    ASSERT_EQ(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景覆盖说明：attrs 判空检查点（fused_adamw_tiling.cpp GetAttrInfo 中 GetAttrs() 判空）
// 不可构造：TilingContextFaker::Build 必然构造 ComputeNodeInfo，GetAttrs 为固定偏移计算永不为空；
// attrs 内各 attr 指针判空由场景3（缺 maximize 属性）代表覆盖

// 负向场景3：属性列表缺少 maximize（第 7 个属性），
// GetAttrPointer<bool>(ATTR_MAXIMIZE_IDX) 返回空指针，
// 整改后应上报 EZ0037（param_name=attrMaximize）首错并返回失败
TEST_F(FusedAdamWTiling, fused_adamw_tiling_null_attr_maximize_003)
{
    FusedAdamwTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape maxExpAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stateStepsShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // 只设置前 6 个属性，缺少第 7 个属性 maximize
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 4)
                      .InputShapes({&paramsShape, &gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stateStepsShape, &gradScaleShape, &foundInfShape})
                      .OutputShapes({&paramsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.99f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetAttrs(), nullptr);
    ASSERT_EQ(tiling_context->GetAttrs()->GetAttrPointer<bool>(6), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景4：grads 输入的 StorageShape 为空（GetDynamicInputShape 返回空指针），
// GetInputTensorInfo 的 checkShape 中 shapePtr 检查触发，
// 整改后应上报 EZ0037（param_name=shapePtr）首错并返回失败
TEST_F(FusedAdamWTiling, fused_adamw_tiling_null_input_shape_004)
{
    FusedAdamwTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape maxExpAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stateStepsShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // grads 位置不放 shape，使 GetDynamicInputShape(INPUT_GRADS_IDX, 0) 返回空指针
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 4)
                      .InputShapes({&paramsShape, nullptr, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stateStepsShape, &gradScaleShape, &foundInfShape})
                      .OutputShapes({&paramsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.99f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_context->GetDynamicInputShape(1, 0), nullptr);
    // desc 与 shape 同受输入实例数约束（实例数为 0 时 desc/shape 同空），
    // CheckInputDtype 中 dtypeInput/dtypeInputInner 判空检查点由本场景一并覆盖
    ASSERT_EQ(tiling_context->GetDynamicInputDesc(1, 0), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景覆盖说明：GetDynamicInputDesc 判空检查点（CheckInputDtype 中 dtypeInput/dtypeInputInner）
// 不可单独构造：desc 与 shape 同受输入实例数约束（实例数为 0 时 desc/shape 同空，
// 且源码 shape 检查先于 desc 触发），由场景4 的断言一并覆盖

// 负向场景覆盖说明：computeNodeInfo 判空（GetInputTensorInfo 中 GetComputeNodeInfo）与
// anchorInstanceInfo 判空（GetInputInstanceInfo(INPUT_PARAMS_IDX=0)）检查点不可构造：
// TilingContextFaker::Build 必然构造 ComputeNodeInfo；使 GetInputInstanceInfo(0) 为空需要
// IR 输入原型数为 0，此时 faker 构建空 holder，用例自身段错误；
// 两检查点属 EZ0037 同上报路由，由场景4 的 shape 判空用例代表覆盖

// 负向场景5：tiling data 容量（1 字节）小于 sizeof(FusedAdamWTilingData)，
// GetTilingData<FusedAdamWTilingData>() 因容量不足返回空指针，
// 整改后应上报 EZ0037（param_name=tilingData）首错并返回失败
TEST_F(FusedAdamWTiling, fused_adamw_tiling_null_tiling_data_005)
{
    FusedAdamwTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape maxExpAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stateStepsShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};

    // 容量 1 字节小于 sizeof(FusedAdamWTilingData)，GetTilingData<T> 内
    // GetCapacity() < sizeof(T) 判定成立返回空指针（见 tiling_context.h GetTilingData）
    auto param = gert::TilingData::CreateCap(1);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 4)
                      .InputShapes({&paramsShape, &gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stateStepsShape, &gradScaleShape, &foundInfShape})
                      .OutputShapes({&paramsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.99f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_context->GetTilingData<FusedAdamWTilingData>(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景7：平台 SoCInfo 中移除 ai_core_cnt 使 GetCoreNumAiv 返回 0，
// 对应整改检查点 ReportTilingFailure(context_->GetNodeName(), "coreNum is 0")（EZ1008 首报）
TEST_F(FusedAdamWTiling, fused_adamw_tiling_core_num_zero_007)
{
    FusedAdamwTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape maxExpAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stateStepsShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stateStepsShape, &gradScaleShape, &foundInfShape})
                      .OutputShapes({&paramsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.99f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                        "Intrinsic_fix_pipe_l0c2out": false,
                                                        "Intrinsic_data_move_l12ub": true,
                                                        "Intrinsic_data_move_l0c2ub": true,
                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                        "CORE_NUM": 48}
                                    })";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    soc_infos.erase("ai_core_cnt");
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景8：params 输入实例个数为 0（IrInstanceNum 首项为 0），
// 对应整改检查点 OP_LOGE_FOR_INVALID_TENSORNUM（tensorNum_ == 0 分支，EZ0022 首报）
TEST_F(FusedAdamWTiling, fused_adamw_tiling_params_instance_num_zero_008)
{
    FusedAdamwTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape expAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape maxExpAvgSqsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stateStepsShape = {{1}, {1}};
    gert::StorageShape gradScaleShape = {{1}, {1}};
    gert::StorageShape foundInfShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // params 实例个数置 0（形状槽位传 nullptr 占位），其余输入保持 1 个实例
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(8, 4)
                      .IrInstanceNum({0, 1, 1, 1, 1, 1})
                      .InputShapes({nullptr, &gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape,
                                    &stateStepsShape, &gradScaleShape, &foundInfShape})
                      .OutputShapes({&gradsShape, &expAvgsShape, &expAvgSqsShape, &maxExpAvgSqsShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.99f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_EQ(tiling_context->GetDynamicInputShape(0, 0), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
