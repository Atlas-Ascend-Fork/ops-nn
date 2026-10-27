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
#include "../../../op_host/fused_sgd_tiling.h"
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

class FusedSgdTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedSgdTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FusedSgdTiling TearDown" << std::endl; }
};

TEST_F(FusedSgdTiling, test_tiling_float32)
{
    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape momentumShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

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

    struct FusedSgdCompileInfo {};
    FusedSgdCompileInfo compile_info;

    std::string op_type("FusedSgd");
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
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape, &gradScaleShape})
                      .OutputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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

TEST_F(FusedSgdTiling, test_tiling_float16)
{
    gert::StorageShape paramsShape = {{6400, 42}, {6400, 42}};
    gert::StorageShape gradsShape = {{6400, 42}, {6400, 42}};
    gert::StorageShape momentumShape = {{6400, 42}, {6400, 42}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

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

    struct FusedSgdCompileInfo {};
    FusedSgdCompileInfo compile_info;

    std::string op_type("FusedSgd");
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
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape, &gradScaleShape})
                      .OutputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
struct FusedSgdTilingNegativeEnv {
    fe::PlatFormInfos platform_info;
    struct FusedSgdCompileInfo {};
    FusedSgdCompileInfo compile_info;
    std::string op_type;

    FusedSgdTilingNegativeEnv()
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
        op_type = "FusedSgd";
    }
};
} // namespace

// 负向场景1：tiling 上下文未设置平台信息，GetPlatformInfo 返回空指针，
// 整改后应通过 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON 上报 EZ0037（param_name=platformInfo）首错并返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_null_platform_info_001)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape momentumShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

    // 注入法构造：正常构建（含 PlatformInfo）后向 platform 槽位（inputs_num_+outputs_num_+1 = 9）
    // 注入空值使 GetPlatformInfo 返回空指针；直接省略 PlatformInfo 会使
    // OpTilingContextBuilder::Build 返回空 holder（GetContext 为空，用例自身段错误），不可行
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape, &gradScaleShape})
                      .OutputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    // 向 platform 槽位注入空值（先例：test_ctc_loss_v3_tiling.cpp parse_null_platform_info_005）；
    // MutableInput 为 KernelContext 公有接口，经 holder 取 KernelContext 视图访问
    auto* kernelContext = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(kernelContext, nullptr);
    auto* platformSlot = kernelContext->MutableInput(9);
    ASSERT_NE(platformSlot, nullptr);
    platformSlot->Set(nullptr, nullptr);
    ASSERT_EQ(tiling_context->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景覆盖说明：attrs 判空检查点（fused_sgd_tiling.cpp GetAttrInfo 中 GetAttrs() 判空）
// 不可构造：TilingContextFaker::Build 必然构造 ComputeNodeInfo，GetAttrs 为固定偏移计算永不为空；
// attrs 内各 attr 指针判空由场景3（缺 is_first_step 属性）代表覆盖

// 负向场景3：属性列表缺少 is_first_step（第 7 个属性），
// GetAttrPointer<bool>(ATTR_IS_FIRST_STEP_IDX) 返回空指针，
// 整改后应上报 EZ0037（param_name=attrIsFirstStep）首错并返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_null_attr_is_first_step_003)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape momentumShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // 只设置前 6 个属性，缺少第 7 个属性 is_first_step
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape, &gradScaleShape})
                      .OutputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetAttrs(), nullptr);
    ASSERT_EQ(tiling_context->GetAttrs()->GetAttrPointer<bool>(6), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景4：grads 输入的 StorageShape 为空（GetDynamicInputShape 返回空指针），
// GetInputTensorInfo 中 gradsShapePtr 检查触发，
// 整改后应上报 EZ0037（param_name=gradsShapePtr）首错并返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_null_input_shape_004)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape momentumShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // grads 位置不放 shape，使 GetDynamicInputShape(INPUT_GRADS_IDX, 0) 返回空指针
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&paramsShape, nullptr, &momentumShape, &gradScaleShape})
                      .OutputShapes({&paramsShape, &momentumShape, &momentumShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_context->GetDynamicInputShape(1, 0), nullptr);
    // desc 与 shape 同受输入实例数约束（实例数为 0 时 desc/shape 同空），
    // CheckInputDtype 中 dtypeInput 判空检查点由本场景一并覆盖
    ASSERT_EQ(tiling_context->GetDynamicInputDesc(1, 0), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景5：params/grads 各 2 个实例而 momentum_buffer_list 仅 1 个实例，i=1 时
// GetDynamicInputShape(INPUT_MOMENTUM_BUFFER_IDX, 1) 因实例数越界返回空指针，
// GetInputTensorInfo 中 momentumShapePtr 检查触发，
// 整改后应上报 EZ0037（param_name=momentumShapePtr）首错并返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_null_momentum_shape_005)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape tensorShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // params/grads 各 2 个实例、momentum_buffer_list 1 个实例：CheckOptionalInputs 中
    // GetDynamicInputShape(INPUT_MOMENTUM_BUFFER_IDX, 0) 非空使 useMomentum_=1，
    // 循环 i=1 时 momentum 第 2 个实例越界返回空指针
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({2, 2, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({&tensorShape, &tensorShape, &tensorShape, &tensorShape, &tensorShape,
                                    &gradScaleShape, &gradScaleShape})
                      .OutputShapes({&tensorShape, &tensorShape, &tensorShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetDynamicInputShape(2, 0), nullptr);
    ASSERT_EQ(tiling_context->GetDynamicInputShape(2, 1), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景覆盖说明：GetDynamicInputDesc 判空检查点（CheckInputDtype 中 params/grads/momentum 的
// dtypeInput）不可单独构造：desc 与 shape 同受输入实例数约束（实例数为 0 时 desc/shape 同空，
// 且源码 shape 检查先于 desc 触发），由场景4 的断言一并覆盖；
// momentum 的 desc 判空（useMomentum!=0 时 GetDynamicInputDesc(INPUT_MOMENTUM_BUFFER_IDX, 0)）
// 由场景5 的实例数约束代表覆盖（momentum 实例数减为 0 时 CheckOptionalInputs 判定 useMomentum_=0，
// 该 desc 检查不可达，只能经场景5 的越界路径触达 momentumShapePtr 检查）

// 负向场景覆盖说明：computeNodeInfo 判空（GetInputTensorInfo 中 GetComputeNodeInfo）与
// anchorInstanceInfo 判空（GetInputInstanceInfo(INPUT_PARAMS_IDX=0)）检查点不可构造：
// TilingContextFaker::Build 必然构造 ComputeNodeInfo；使 GetInputInstanceInfo(0) 为空需要
// IR 输入原型数为 0，此时 faker 构建空 holder，用例自身段错误；
// 两检查点属 EZ0037 同上报路由，由场景4/5 的 shape 判空用例代表覆盖

// 负向场景6：tiling data 容量（1 字节）小于 sizeof(FusedSgdTilingData)，
// GetTilingData<FusedSgdTilingData>() 因容量不足返回空指针，
// 整改后应上报 EZ0037（param_name=tilingData）首错并返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_null_tiling_data_006)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape momentumShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

    // 容量 1 字节小于 sizeof(FusedSgdTilingData)，GetTilingData<T> 内
    // GetCapacity() < sizeof(T) 判定成立返回空指针（见 tiling_context.h GetTilingData）
    auto param = gert::TilingData::CreateCap(1);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape, &gradScaleShape})
                      .OutputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(tiling_context->GetTilingData<FusedSgdTilingData>(), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景7：tiling 上下文平台信息无核数（SoCInfo 擦除 ai_core_cnt，GetCoreNumAiv() 返回 0），
// GetPlatformInfo 中 coreNum_==0 触发 ReportTilingFailure 上报 EZ1008 首报并返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_core_num_zero_007)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape momentumShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape, &gradScaleShape})
                      .OutputShapes({&paramsShape, &gradsShape, &momentumShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    // GetCoreNumAiv 读取 SoCInfo.ai_core_cnt，擦除后返回 0 -> coreNum_==0 分支
    // （参照 max_pool3d_grad 用例先例：擦除后跳过 SetCoreNumByCoreType）
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    string compile_info_string = R"({"hardware_info": {"CORE_NUM": 48, "UB_SIZE": 196608}})";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    soc_infos.erase("ai_core_cnt");
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景8：grads 与 params dtype 不一致（params FLOAT / grads FLOAT16），
// momentum_buffer_list 与 grad_scale 均不传（useMomentum_=0 / useGradScale_=0 空指针分支），
// CheckInputDtype 中 isDiffDtype 触发（useMomentum_==0 分支的 EZ 上报），返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_dtype_mismatch_grads_008)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // momentum(位置2)/grad_scale(位置3) 不传 shape：CheckOptionalInputs 中
    // GetDynamicInputShape(2,0) 因实例数为 0 返回空指针 -> useMomentum_=0（空指针分支），
    // GetOptionalInputTensor(3) 返回空指针 -> useGradScale_=0（空指针分支）
    // 注意：需补齐 NodeInputTd(4) 使实际传入 tensor 数与 IOInstanceNum 声明的实例总数
    // （1+1+0+0+1=3）一致，否则 platform 槽位索引错位导致 GetPlatformInfo 空指针
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&paramsShape, &gradsShape, nullptr, nullptr})
                      .OutputShapes({&paramsShape, &gradsShape, &gradsShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetDynamicInputShape(1, 0), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景9：momentum_buffer_list 的 dtype 与 params 不一致（params/grads FLOAT，
// momentum FLOAT16），grad_scale 传 0 维 shape（覆盖 useGradScale_=0 的维度含 0 分支），
// CheckInputDtype 中 isDiffDtype 触发（useMomentum_!=0 分支的 EZ 上报），返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_dtype_mismatch_momentum_009)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape tensorShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{0}, {0}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // grad_scale shape 为 {0}：CheckOptionalInputs 中 flag=false -> useGradScale_=0（含 0 维分支）
    // 注意：需补齐 NodeInputTd(4) 使传入 tensor 数（5）与声明的实例总数（5）一致
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&tensorShape, &tensorShape, &tensorShape, &gradScaleShape})
                      .OutputShapes({&tensorShape, &tensorShape, &tensorShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetDynamicInputShape(2, 0), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景10：grads 与 params shape 不一致，momentum 含 0 维（useMomentum_=0 的 0 维分支）、
// grad_scale 不传（useGradScale_=0），GetInputTensorInfo 中 isDiffSize 触发
// （useMomentum_==0 分支的 EZ 上报），返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_shape_mismatch_grads_010)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradsShape = {{300, 4, 3}, {300, 4, 3}};
    gert::StorageShape momentumShape = {{0, 4, 2}, {0, 4, 2}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // momentum 首维为 0：CheckOptionalInputs 中 flag=false -> useMomentum_=0（0 维分支），
    // grad_scale 不传 -> useGradScale_=0；需补齐 NodeInputTd(4) 使传入 tensor 数（4）
    // 与声明的实例总数（1+1+1+0+1=4）一致
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&paramsShape, &gradsShape, &momentumShape, nullptr})
                      .OutputShapes({&paramsShape, &gradsShape, &gradsShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景11：momentum_buffer_list 与 params shape 不一致（useMomentum_!=0），
// GetInputTensorInfo 中 isDiffSize 触发（useMomentum_!=0 分支的 EZ 上报），返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_shape_mismatch_momentum_011)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape paramsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape momentumShape = {{300, 4, 3}, {300, 4, 3}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&paramsShape, &paramsShape, &momentumShape, &gradScaleShape})
                      .OutputShapes({&paramsShape, &paramsShape, &momentumShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetDynamicInputShape(2, 0), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景12：params 输入实例数为 0（IrInstanceNum 第 1 位为 0），
// GetInputInstanceInfo(INPUT_PARAMS_IDX) 的实例数为 0 -> tensorNum_==0，
// 触发 OP_LOGE_FOR_INVALID_TENSORNUM（EZ0022）上报并返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_tensor_num_zero_012)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape gradsShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape momentumShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    // params 实例数置 0：GetInputInstanceInfo(0) 的 GetInstanceNum() 为 0 -> tensorNum_==0 分支
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({0, 1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({nullptr, &gradsShape, &momentumShape, &gradScaleShape})
                      .OutputShapes({&gradsShape, &gradsShape, &momentumShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    auto* anchorInstanceInfo = tiling_context->GetComputeNodeInfo()->GetInputInstanceInfo(0);
    ASSERT_NE(anchorInstanceInfo, nullptr);
    ASSERT_EQ(anchorInstanceInfo->GetInstanceNum(), 0u);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景13：params/grads/momentum 均为 DT_INT32（dtype 一致但不在支持列表
// float16/bfloat16/float 中），CheckInputDtype 中 isInvalidType 触发（EZ 上报），返回失败
TEST_F(FusedSgdTiling, fused_sgd_tiling_invalid_dtype_int32_013)
{
    FusedSgdTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape tensorShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape gradScaleShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .InputShapes({&tensorShape, &tensorShape, &tensorShape, &gradScaleShape})
                      .OutputShapes({&tensorShape, &tensorShape, &tensorShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"weight_decay", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                  {"momentum", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                  {"dampening", Ops::NN::AnyValue::CreateFrom(0.0f)},
                                  {"nesterov", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"is_first_step", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
