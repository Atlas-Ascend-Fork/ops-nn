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
#include "../../../op_host/apply_fused_ema_adam_tiling.h"
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

class ApplyFusedEmaAdamTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ApplyFusedEmaAdamTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ApplyFusedEmaAdamTiling TearDown" << std::endl; }
};

TEST_F(ApplyFusedEmaAdamTiling, test_tiling_float32_mode_0_bias_0)
{
    gert::StorageShape gradShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape varShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape mShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape vShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape sShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stepShape = {{1}, {1}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    struct ApplyFusedEmaAdamCompileInfo {};
    ApplyFusedEmaAdamCompileInfo compile_info;

    std::string op_type("ApplyFusedEmaAdam");
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
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gradShape, &varShape, &mShape, &vShape, &sShape, &stepShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &sShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                                  {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"bias_correction", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 102);
}

TEST_F(ApplyFusedEmaAdamTiling, test_tiling_float16_mode_1_bias_0)
{
    gert::StorageShape gradShape = {{6400, 42}, {6400, 42}};
    gert::StorageShape varShape = {{6400, 42}, {6400, 42}};
    gert::StorageShape mShape = {{6400, 42}, {6400, 42}};
    gert::StorageShape vShape = {{6400, 42}, {6400, 42}};
    gert::StorageShape sShape = {{6400, 42}, {6400, 42}};
    gert::StorageShape stepShape = {{2, 8}, {2, 8}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    struct ApplyFusedEmaAdamCompileInfo {};
    ApplyFusedEmaAdamCompileInfo compile_info;

    std::string op_type("ApplyFusedEmaAdam");
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
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gradShape, &varShape, &mShape, &vShape, &sShape, &stepShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &sShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                                  {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"bias_correction", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 101);
}

TEST_F(ApplyFusedEmaAdamTiling, test_tiling_bfloat16_mode_0_bias_1)
{
    gert::StorageShape gradShape = {{42, 42, 42, 137}, {42, 42, 42, 137}};
    gert::StorageShape varShape = {{42, 42, 42, 137}, {42, 42, 42, 137}};
    gert::StorageShape mShape = {{42, 42, 42, 137}, {42, 42, 42, 137}};
    gert::StorageShape vShape = {{42, 42, 42, 137}, {42, 42, 42, 137}};
    gert::StorageShape sShape = {{42, 42, 42, 137}, {42, 42, 42, 137}};
    gert::StorageShape stepShape = {{2, 2}, {2, 2}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    struct ApplyFusedEmaAdamCompileInfo {};
    ApplyFusedEmaAdamCompileInfo compile_info;

    std::string op_type("ApplyFusedEmaAdam");
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
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gradShape, &varShape, &mShape, &vShape, &sShape, &stepShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &sShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                                  {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"bias_correction", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100);
}

namespace {
// 错误码整改负向测试公共构造：与正向用例相同的平台信息准备流程
struct ApplyFusedEmaAdamTilingNegativeEnv {
    fe::PlatFormInfos platform_info;
    struct ApplyFusedEmaAdamCompileInfo {};
    ApplyFusedEmaAdamCompileInfo compile_info;
    std::string op_type;

    ApplyFusedEmaAdamTilingNegativeEnv()
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
        op_type = "ApplyFusedEmaAdam";
    }
};
} // namespace

// 负向场景覆盖说明（TilingContextFaker 能力限制，未单独构造用例）：
// 1. attrs 判空检查点（GetTilingAttr 中 GetAttrs()）：TilingContext::GetAttrs() 仅在
//    ComputeNodeInfo 缺失时返回空，而 TilingContextFaker::Build 必然构造 ComputeNodeInfo，
//    且 GetAttrs 为固定偏移计算永不为空，该检查点无法通过 faker 构造（原 null_attrs_001 用例删除）；
// 2. attrLr/attrEmaDecay/attrBeta1/attrBeta2/attrEps/attrMode/attrBiasCorrection/attrWeightDecay
//    判空检查点（GetAttrPointer 路径）：attr 指针指向 ComputeNodeInfo 中真实的 attrs 存储，
//    faker 按 NodeAttrs 声明构造后该路径不经过下标判空，无法构造空指针
//    （原 null_attr_pointer_002 用例删除）；
// 3. 输入 shape 判空检查点（CheckInputShape 中 GetInputShape(0)-(5)）：与输入 desc 判空同受
//    inputs_num_ 越界约束，且 CheckInputDtype 先于 CheckInputShape 执行，输入实例数置 0 时必然
//    先在 CheckInputDtype 的 desc 判空检查点提前返回，由场景1（null_input_desc_003）一并覆盖
//    （原 null_input_shape_004 用例删除，其断言 GetInputShape(0) 为空已并入场景1）。

// 负向场景1：必选输入的实例数全 0（IrInstanceNum 输入实例数全 0），
// ComputeNodeInfo 的 inputs_num_ 变为 0，GetInputDesc(0)/GetInputShape(0) 按下标越界规则返回
// 空指针，首个触达的整改检查点为 CheckInputDtype 中 param_name="dtypeInput"（EZ0037 首报），
// 其后序 CheckInputShape 的 shapeInput 判空检查点因前置检查已返回而一并覆盖（见断言）
TEST_F(ApplyFusedEmaAdamTiling, apply_fused_ema_adam_tiling_null_input_desc_003)
{
    ApplyFusedEmaAdamTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape varShape = {{300, 4, 2}, {300, 4, 2}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    // 输入实例数全 0 使 ComputeNodeInfo 的 inputs_num_ 为 0，GetInputDesc(0)/GetInputShape(0)
    // 均返回空指针；仍传入 4 个输出 shape 保持输出链存在，使 compile/platform 等扩展链的
    // 槽位按 inputs_num_+outputs_num_ 对齐；attrs 保持有效，保证首错确为 desc 判空
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({0, 0, 0, 0, 0, 0}, {1, 1, 1, 1})
                      .OutputShapes({&varShape, &varShape, &varShape, &varShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                                  {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"bias_correction", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    ASSERT_EQ(tiling_context->GetInputDesc(0), nullptr);
    ASSERT_EQ(tiling_context->GetInputShape(0), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景2：GetWorkspaceSizes 返回 nullptr（workspace 容量为 0）。
// 整改点：apply_fused_ema_adam_tiling.cpp Tiling4ApplyFusedEmaAdam 中 workspaces 判空，
// param_name=workspaces，失败时应上报 EZ0037 首错并返回 ge::GRAPH_FAILED。
// 构造方式：Workspace 传入容量为 0 的 ContinuousVector，GetWorkspaceSizes(1) 内部
// SetSize(1) 因 1 > capacity_(0) 失败返回 nullptr（见 tiling_context.h GetWorkspaceSizes）；
// 其余与正向用例一致，保证 GetTilingAttr/CheckInputDtype/CheckInputShape/TilingCompute 均成功后
// 走到 workspaces 判空检查。
TEST_F(ApplyFusedEmaAdamTiling, apply_fused_ema_adam_tiling_null_workspace_005)
{
    ApplyFusedEmaAdamTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape gradShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape varShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape mShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape vShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape sShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stepShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    // 容量为 0 的 workspace，GetWorkspaceSizes(1) 中 SetSize(1) 越界失败返回空指针
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(0);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gradShape, &varShape, &mShape, &vShape, &sShape, &stepShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &sShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                                  {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"bias_correction", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    ASSERT_EQ(tiling_context->GetWorkspaceSizes(1), nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// 负向场景6：tiling 阶段平台核数为 0（SoCInfo 去掉 ai_core_cnt，GetCoreNumAiv() 返回 0），
// 其余与正向用例一致，保证 GetTilingAttr/CheckInputDtype/CheckInputShape 均成功后
// 在 TilingCompute 中触发 ReportTilingFailure(nodeName, "Fail to get core num.")（EZ1008 首报），
// 该调用同时覆盖本文件内 ReportTilingFailure 辅助函数的 EZ1008 规范上报路径
TEST_F(ApplyFusedEmaAdamTiling, apply_fused_ema_adam_tiling_core_num_zero_006)
{
    ApplyFusedEmaAdamTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(env.op_type.c_str())->tiling;

    gert::StorageShape gradShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape varShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape mShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape vShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape sShape = {{300, 4, 2}, {300, 4, 2}};
    gert::StorageShape stepShape = {{1}, {1}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gradShape, &varShape, &mShape, &vShape, &sShape, &stepShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &sShape})
                      .CompileInfo(&env.compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                                  {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                                  {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"bias_correction", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // GetCoreNumAiv 读取 SoCInfo.ai_core_cnt，擦除后返回 0，
    // 触发 TilingCompute 中 coreNum<=0 的 ReportTilingFailure（EZ1008）
    // （参照 fused_sgd_tiling_core_num_zero_007 先例：擦除后跳过 SetCoreNumByCoreType）
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    string compile_info_string = R"({"hardware_info": {"CORE_NUM": 48, "UB_SIZE": 196608}})";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    soc_infos.erase("ai_core_cnt");
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
