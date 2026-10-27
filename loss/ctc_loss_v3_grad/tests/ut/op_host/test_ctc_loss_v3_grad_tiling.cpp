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
 * \file test_ctc_loss_v3_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/ctc_loss_v3_grad_tiling.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "array_ops.h"
#include "ut_op_util.h"

using namespace std;
using namespace ge;

class CTCLossV3GradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CTCLossV3GradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CTCLossV3GradTiling TearDown" << std::endl; }
};

void TestCTCLossV3GradTiling(gert::StorageShape& gradOutShape, gert::StorageShape& logProbsShape,
                             gert::StorageShape& targetsShape, gert::StorageShape& inputLengthsShape,
                             gert::StorageShape& targetLengthsShape, gert::StorageShape& lossShape,
                             gert::StorageShape& logAlphaShape, gert::StorageShape& gradShape,
                             std::vector<std::pair<std::string, Ops::NN::AnyValue>>& AttrList,
                             ge::DataType gradDataType, ge::DataType indicesDataType, uint64_t expectTilingKey)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string COMPILE_INFO_STRING_910B = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 40}})";
    GetPlatFormInfos(COMPILE_INFO_STRING_910B.c_str(), socInfos, aicoreSpec, intrinsics);

    // Platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // Compile info
    optiling::CTCLossV3GradCompileInfo compileInfo;

    std::string op_type("CTCLossV3Grad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // TilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(COMPILE_INFO_STRING_910B.c_str()),
                                     reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);

    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("CTCLossV3Grad")
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&gradOutShape, &logProbsShape, &targetsShape, &inputLengthsShape,
                                    &targetLengthsShape, &lossShape, &logAlphaShape})
                      .OutputShapes({&gradShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<string>("mean")},
                                  {"zero_infinity", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    auto realTilingKey = tilingContext->GetTilingKey();
    ASSERT_EQ(realTilingKey, expectTilingKey);
}

TEST_F(CTCLossV3GradTiling, ctc_loss_v3_grad_tilingkey_0)
{
    std::cout << "run case: "
              << "ctc_loss_v3_grad_tilingkey_0" << std::endl;
    gert::StorageShape gradOutShape = {{1}, {1}};
    gert::StorageShape logProbsShape = {{32, 1, 1024}, {32, 1, 1024}};
    gert::StorageShape targetsShape = {{1, 32}, {1, 32}};
    gert::StorageShape inputLengthsShape = {{1}, {1}};
    gert::StorageShape targetLengthsShape = {{1}, {1}};
    gert::StorageShape lossShape = {{1}, {1}};
    gert::StorageShape logAlphaShape = {{1, 32, 65}, {1, 32, 65}};
    gert::StorageShape gradShape = {{32, 1, 1024}, {32, 1, 1024}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {};
    TestCTCLossV3GradTiling(gradOutShape, logProbsShape, targetsShape, inputLengthsShape, targetLengthsShape, lossShape,
                            logAlphaShape, gradShape, attrList, ge::DT_FLOAT, ge::DT_INT64, 0);
}

namespace {
// 错误码整改负向测试公共构造：与正向用例相同的平台信息初始化及 tilingParse 流程，用于填充 compileInfo
struct CTCLossV3GradTilingNegativeEnv {
    fe::PlatFormInfos platformInfo;
    optiling::CTCLossV3GradCompileInfo compileInfo;
    std::string opType;
    std::string compileInfoString;

    CTCLossV3GradTilingNegativeEnv()
    {
        map<string, string> socInfos;
        map<string, string> aicoreSpec;
        map<string, string> intrinsics;
        compileInfoString = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 40}})";
        GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
        platformInfo.Init();
        opType = "CTCLossV3Grad";
        auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
        if (opImpl == nullptr || opImpl->tiling_parse == nullptr) {
            return;
        }
        // tilingParseFunc simulate，填充 compileInfo 的 coreNum/sysWorkspaceSize/ubSizePlatForm
        auto kernelHolder = gert::KernelRunContextFaker()
                                .KernelIONum(2, 1)
                                .Inputs({const_cast<char*>(compileInfoString.c_str()),
                                         reinterpret_cast<void*>(&platformInfo)})
                                .Outputs({&compileInfo})
                                .Build();
        kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
        kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
        kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",
                                                                                               aicoreSpec);
        kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
            "AICoreintrinsicDtypeMap", intrinsics);
        opImpl->tiling_parse(kernelHolder.GetContext<gert::KernelContext>());
    }
};
} // namespace

// 负向场景1：必选输入 grad_out 的实例数置 0（IrInstanceNum 输入实例数全 0），
// ComputeNodeInfo 的 inputs_num_ 变为 0，GetInputShape(0) 按下标越界规则返回空指针，
// 首个触达的整改检查点为 CheckShapeInfoForN 中 param_name="gradOutShape"（EZ0037 首报），
// 其后序的 inputLengthsShape/targetLengthsShape/lossShape/logProbsShape/targetsShape/logAlphaShape
// 等输入 shape 判空检查点因前置检查已返回而一并覆盖（见断言）
TEST_F(CTCLossV3GradTiling, ctc_loss_v3_grad_tiling_null_input_shape_001)
{
    CTCLossV3GradTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling;

    gert::StorageShape gradShape = {{32, 1, 1024}, {32, 1, 1024}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    // 输入实例数全 0 使 ComputeNodeInfo 的 inputs_num_ 为 0，GetInputShape(0) 返回空指针；
    // 仍传入 1 个输出 shape 保持输出链存在，使 compile/platform 等扩展链的槽位按声明序对齐
    auto holder = gert::TilingContextFaker()
                      .SetOpType("CTCLossV3Grad")
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({0, 0, 0, 0, 0, 0, 0}, {1})
                      .OutputShapes({&gradShape})
                      .CompileInfo(&env.compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platformInfo))
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<string>("mean")},
                                  {"zero_infinity", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    ASSERT_EQ(tilingContext->GetInputShape(0), nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}

// 负向场景2：输出 grad 的实例数置 0（IrInstanceNum 输出实例数为 0），
// ComputeNodeInfo 的 outputs_num_ 变为 0，GetOutputShape(0) 按下标越界规则返回空指针，
// 对应整改检查点 param_name="gradShape"（CheckShapeInfoForCT 末个检查点），
// 整改后应上报 EZ0037 首错并返回失败
TEST_F(CTCLossV3GradTiling, ctc_loss_v3_grad_tiling_null_output_shape_002)
{
    CTCLossV3GradTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling;

    gert::StorageShape gradOutShape = {{1}, {1}};
    gert::StorageShape logProbsShape = {{32, 1, 1024}, {32, 1, 1024}};
    gert::StorageShape targetsShape = {{1, 32}, {1, 32}};
    gert::StorageShape inputLengthsShape = {{1}, {1}};
    gert::StorageShape targetLengthsShape = {{1}, {1}};
    gert::StorageShape lossShape = {{1}, {1}};
    gert::StorageShape logAlphaShape = {{1, 32, 65}, {1, 32, 65}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    // 输出实例数置 0 使 ComputeNodeInfo 的 outputs_num_ 为 0，GetOutputShape(0) 返回空指针；
    // 输入实例数保持 7 个且 shape 全部合法（gradShape 判空位于 CheckShapeInfoForCT 末尾，
    // 前序输入 shape 检查须全部通过），compile/platform 等扩展链槽位按 inputs_num_+outputs_num_ 对齐
    auto holder = gert::TilingContextFaker()
                      .SetOpType("CTCLossV3Grad")
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1}, {0})
                      .InputShapes({&gradOutShape, &logProbsShape, &targetsShape, &inputLengthsShape,
                                    &targetLengthsShape, &lossShape, &logAlphaShape})
                      .CompileInfo(&env.compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<string>("mean")},
                                  {"zero_infinity", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    ASSERT_EQ(tilingContext->GetOutputShape(0), nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}

// 负向场景覆盖说明（TilingContextFaker 能力限制，未单独构造用例）：
// 1. attrs 判空检查点（ctc_loss_v3_grad_tiling.cpp Init 中 GetAttrs()）：TilingContext::GetAttrs()
//    仅在 ComputeNodeInfo 缺失时返回空，而 TilingContextFaker::Build 必然构造 ComputeNodeInfo，
//    且 GetAttrs 为固定偏移计算永不为空，该检查点无法通过 faker 构造（原 null_attrs_003 用例删除）；
// 2. blankPtr/zeroInfinityPtr 判空检查点（GetAttrPointer 路径）：attr 指针指向 ComputeNodeInfo 中
//    真实的 attrs 存储，faker 按 NodeAttrs 声明构造后该路径不经过下标访问，无法构造空指针；
// 3. inputLengthsShape/targetLengthsShape/lossShape/logProbsShape/targetsShape/logAlphaShape 判空
//    检查点：与 gradOutShape 同受 inputs_num_ 越界约束，GetInputShape(0) 为空时必然先在
//    CheckShapeInfoForN 的 gradOutShape 检查点提前返回，由场景1（null_input_shape_001）一并覆盖。

// 负向场景4：TilingParse 阶段 compileInfo 为空（GetCompiledInfo 返回 nullptr），
// 对应整改检查点 param_name="compileInfo"（TilingPrepare4CTCLossV3Grad），
// 整改后应上报 EZ0037 首错并返回失败
TEST_F(CTCLossV3GradTiling, ctc_loss_v3_grad_tiling_parse_null_compile_info_004)
{
    CTCLossV3GradTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling_parse;

    // Outputs 放 nullptr，使 GetCompiledInfo<CTCLossV3GradCompileInfo>() 返回空指针
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(env.compileInfoString.c_str()),
                                     reinterpret_cast<void*>(&env.platformInfo)})
                            .Outputs({nullptr})
                            .Build();

    ASSERT_EQ(
        kernelHolder.GetContext<gert::TilingParseContext>()->GetCompiledInfo<optiling::CTCLossV3GradCompileInfo>(),
        nullptr);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_FAILED);
}

// 负向场景5：TilingParse 阶段 platformInfo 为空（GetPlatformInfo 返回 nullptr），
// 对应整改检查点 param_name="platformInfo"（TilingPrepare4CTCLossV3Grad），
// 整改后应上报 EZ0037 首错并返回失败
TEST_F(CTCLossV3GradTiling, ctc_loss_v3_grad_tiling_parse_null_platform_info_005)
{
    CTCLossV3GradTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling_parse;

    // CANN 的 OpTilingParseContextBuilder 要求 PlatformInfo 非空（传 nullptr 时 Build 直接返回空 context），
    // 参照 in_training_reduce_v2 的先例：先正常构建，再向输入链位置 1（platform 槽位）注入空值，
    // 使 GetPlatformInfo() 返回空指针；Outputs 传有效 compileInfo 以先通过 compileInfo 检查点
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(env.compileInfoString.c_str()),
                                     reinterpret_cast<void*>(&env.platformInfo)})
                            .Outputs({&env.compileInfo})
                            .Build();

    auto* kernelContext = kernelHolder.GetContext<gert::KernelContext>();
    ASSERT_NE(kernelContext, nullptr);
    // TilingParseContext 输入链布局：位置 0 = CompiledJson，位置 1 = PlatformInfo
    auto* platformSlot = kernelContext->MutableInput(1);
    ASSERT_NE(platformSlot, nullptr);
    platformSlot->Set(nullptr, nullptr);

    ASSERT_EQ(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tilingParseFunc(kernelContext), ge::GRAPH_FAILED);
}

// 负向场景6：TilingParse 阶段 context 为空（tilingParseFunc 直接收 nullptr），
// 对应整改检查点 ReportTilingFailure("CtcLossV3Grad", "context is nullptr.")（EZ1008 首报，
// ctc_loss_v3_grad_tiling.cpp 的 TilingPrepare4CTCLossV3Grad），整改后应返回失败；
// 该调用同时覆盖本文件内 ReportTilingFailure 辅助函数的 EZ1008 规范上报路径
TEST_F(CTCLossV3GradTiling, ctc_loss_v3_grad_tiling_parse_null_context_006)
{
    CTCLossV3GradTilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling_parse;

    EXPECT_EQ(tilingParseFunc(nullptr), ge::GRAPH_FAILED);
}
