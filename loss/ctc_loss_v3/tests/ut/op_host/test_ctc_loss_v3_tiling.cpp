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
 * \file test_ctc_loss_v3_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/ctc_loss_v3_tiling.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "array_ops.h"

using namespace std;
using namespace ge;

class CTCLossV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CTCLossV3Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CTCLossV3Tiling TearDown" << std::endl; }
};

template <typename T>
void SetConstInput(size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
                   std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>>& const_tensors)
{
    std::unique_ptr<uint8_t[]> input_tensor_holder = std::unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor({{data_size}, {data_size}},         // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        dtype,                              // dt
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

void TestCTCLossV3Tiling(gert::StorageShape& logProbsShape, gert::StorageShape& targetsShape,
                         gert::StorageShape& inputLengthsShape, gert::StorageShape& targetLengthsShape,
                         gert::StorageShape& lossShape, gert::StorageShape& logAlphaShape,
                         std::vector<std::pair<std::string, Ops::NN::AnyValue>>& AttrList, ge::DataType gradDataType,
                         ge::DataType indicesDataType, uint64_t expectTilingKey)
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
    optiling::CTCLossV3CompileInfo compileInfo;

    std::string op_type("CTCLossV3");
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
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> const_tensors;
    int64_t shape_data[1] = {1};
    SetConstInput(3, DT_INT64, shape_data, 1, const_tensors);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("CTCLossV3")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&logProbsShape, &targetsShape, &inputLengthsShape, &targetLengthsShape})
                      .OutputShapes({&lossShape, &logAlphaShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, gradDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<string>("mean")},
                                  {"zero_infinity", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .ConstInput(const_tensors)
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

TEST_F(CTCLossV3Tiling, ctc_loss_v3_tilingkey_0)
{
    std::cout << "run case: "
              << "ctc_loss_v3_tilingkey_0" << std::endl;
    gert::StorageShape logProbsShape = {{32, 1, 1024}, {32, 1, 1024}};
    gert::StorageShape targetsShape = {{1, 32}, {1, 32}};
    gert::StorageShape inputLengthsShape = {{1}, {1}};
    gert::StorageShape targetLengthsShape = {{1}, {1}};
    gert::StorageShape lossShape = {{1}, {1}};
    gert::StorageShape logAlphaShape = {{1, 32, 65}, {1, 32, 65}};
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrList = {};
    TestCTCLossV3Tiling(logProbsShape, targetsShape, inputLengthsShape, targetLengthsShape, lossShape, logAlphaShape,
                        attrList, ge::DT_FLOAT, ge::DT_INT64, 0);
}

namespace {
// 错误码整改负向测试公共构造：与正向用例相同的平台信息初始化及 tilingParse 流程，用于填充 compileInfo
struct CTCLossV3TilingNegativeEnv {
    fe::PlatFormInfos platformInfo;
    optiling::CTCLossV3CompileInfo compileInfo;
    std::string opType;
    std::string compileInfoString;

    CTCLossV3TilingNegativeEnv()
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
        opType = "CTCLossV3";
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

// 负向场景1：必选输入 input_lengths 的实例数置 0（IrInstanceNum 第 3 个输入实例数为 0），
// ComputeNodeInfo 的 inputs_num_ 变为 2，GetInputDesc(2)/GetInputShape(2) 按下标越界规则返回空指针，
// 首个触达的整改检查点为 CheckDtypeInfo 中 param_name="inputLengthsDesc"（EZ0037 首报），
// 其后序的 inputLengthsShape 等 shape 判空检查点因前置检查已返回而一并覆盖（见断言）
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_null_input_desc_003)
{
    CTCLossV3TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling;

    gert::StorageShape logProbsShape = {{32, 1, 1024}, {32, 1, 1024}};
    gert::StorageShape targetsShape = {{1, 32}, {1, 32}};
    gert::StorageShape lossShape = {{1}, {1}};
    gert::StorageShape logAlphaShape = {{1, 32, 65}, {1, 32, 65}};

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    // 将第 3 个输入 input_lengths 的实例数置 0，inputs_num_ 由 4 变为 2：
    // GetInputDesc(2)/GetInputShape(2) 均按 ComputeNodeInfo::GetInputsNum() 越界规则返回空指针；
    // 保留 2 个输出实例，使 compile/platform 等扩展链槽位按 inputs_num_+outputs_num_ 对齐；
    // 前两个输入 desc 的 dtype 保持合法，保证首错确为 inputLengthsDesc 判空而非 dtype 校验
    auto holder = gert::TilingContextFaker()
                      .SetOpType("CTCLossV3")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 0, 0}, {1, 1})
                      .InputShapes({&logProbsShape, &targetsShape})
                      .OutputShapes({&lossShape, &logAlphaShape})
                      .CompileInfo(&env.compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<string>("mean")},
                                  {"zero_infinity", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    ASSERT_EQ(tilingContext->GetInputDesc(2), nullptr);
    ASSERT_EQ(tilingContext->GetInputShape(2), nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}

// 负向场景覆盖说明（TilingContextFaker 能力限制，未单独构造用例）：
// 1. attrs 判空检查点（ctc_loss_v3_tiling.cpp 中 GetAttrs()）：TilingContext::GetAttrs() 仅在
//    ComputeNodeInfo 缺失时返回空，而 TilingContextFaker::Build 必然构造 ComputeNodeInfo，
//    且 GetAttrs 为固定偏移计算永不为空，该检查点无法通过 faker 构造；
// 2. blankPtr/zeroInfinityPtr 判空检查点（GetAttrPointer 路径）：attr 指针指向 ComputeNodeInfo 中
//    真实的 attrs 存储，faker 按 NodeAttrs 声明构造后越界访问才可能为空，而该路径不经过下标访问，
//    无法构造空指针，与 attrs 判空同属不可构造项；
// 3. lossShape/logAlphaShape 输出判空检查点：CheckDtypeInfo 中 GetOutputDesc(0)/GetOutputDesc(1)
//    在输出 shape 判空之前无判空直接解引用，若通过 IrInstanceNum 置输出实例数为 0 构造输出空指针，
//    tiling 函数在 CheckDtypeInfo 阶段即段错误，故该组检查点不可覆盖；
// 4. inputLengthsShape/targetLengthsShape 判空检查点：与 inputLengthsDesc 同受 inputs_num_ 越界
//    约束，GetInputShape(2) 为空时必然先在 CheckDtypeInfo 的 inputLengthsDesc 检查点提前返回，
//    由场景1（null_input_desc_003）一并覆盖（见上方对 GetInputShape 的断言）。

// 负向场景4：TilingParse 阶段 compileInfo 为空（GetCompiledInfo 返回 nullptr），
// 对应整改检查点 param_name="compileInfo"（TilingPrepare4CTCLossV3），
// 整改后应上报 EZ0037 首错并返回失败
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_parse_null_compile_info_004)
{
    CTCLossV3TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling_parse;

    // Outputs 放 nullptr，使 GetCompiledInfo<CTCLossV3CompileInfo>() 返回空指针
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(env.compileInfoString.c_str()),
                                     reinterpret_cast<void*>(&env.platformInfo)})
                            .Outputs({nullptr})
                            .Build();

    ASSERT_EQ(kernelHolder.GetContext<gert::TilingParseContext>()->GetCompiledInfo<optiling::CTCLossV3CompileInfo>(),
              nullptr);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_FAILED);
}

// 负向场景5：TilingParse 阶段 platformInfo 为空（GetPlatformInfo 返回 nullptr），
// 对应整改检查点 param_name="platformInfo"（TilingPrepare4CTCLossV3），
// 整改后应上报 EZ0037 首错并返回失败
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_parse_null_platform_info_005)
{
    CTCLossV3TilingNegativeEnv env;
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

// 负向场景6：TilingParse 阶段平台核数为 0（SoCInfo 去掉 ai_core_cnt，GetCoreNumAiv() 返回 0），
// 对应整改检查点 ReportTilingFailure(nodeName, "Failed to get core num.")（EZ1008 首报，
// ctc_loss_v3_tiling.cpp 的 TilingPrepare4CTCLossV3），整改后应返回失败
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_parse_core_num_zero_006)
{
    CTCLossV3TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling_parse;

    // 使用独立的平台信息对象，避免 env 构造时已填充的 SoCInfo/AICoreSpec 残留干扰
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(env.compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    // GetCoreNumAiv 读取 SoCInfo.ai_core_cnt，去掉后返回 0 -> compileInfo->coreNum <= 0 分支
    // （参照 max_pool3d_grad 用例先例：擦除后跳过 SetCoreNumByCoreType）
    socInfos.erase("ai_core_cnt");

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(env.compileInfoString.c_str()),
                                     reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&env.compileInfo})
                            .Build();

    auto* pf = kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo();
    ASSERT_TRUE(pf->Init());
    pf->SetPlatformRes("SoCInfo", socInfos);
    pf->SetPlatformRes("AICoreSpec", aicoreSpec);
    pf->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_FAILED);
}

// 负向场景7：TilingParse 阶段平台 UB 大小为 0（AICoreSpec 去掉 ub_size，GetCoreMemSize(UB) 返回 0），
// 对应整改检查点 ReportTilingFailure(nodeName, "Failed to get ub size.")（EZ1008 首报，
// ctc_loss_v3_tiling.cpp 的 TilingPrepare4CTCLossV3），整改后应返回失败
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_parse_ub_size_zero_007)
{
    CTCLossV3TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling_parse;

    // 使用独立的平台信息对象，避免 env 构造时已填充的 SoCInfo/AICoreSpec 残留干扰
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(env.compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    // JSON 的 UB_SIZE 经 GetPlatFormInfos 映射为 AICoreSpec.ub_size，
    // 去掉后 GetCoreMemSize(UB) 返回 0 -> compileInfo->ubSizePlatForm <= 0 分支
    aicoreSpec.erase("ub_size");

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(env.compileInfoString.c_str()),
                                     reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&env.compileInfo})
                            .Build();

    auto* pf = kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo();
    ASSERT_TRUE(pf->Init());
    pf->SetPlatformRes("SoCInfo", socInfos);
    pf->SetPlatformRes("AICoreSpec", aicoreSpec);
    pf->SetCoreNumByCoreType("AICore");
    pf->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_FAILED);
}

// 负向场景8：TilingParse 阶段 context 为空（tilingParseFunc 直接收 nullptr），
// 对应整改检查点 ReportTilingFailure(OP_NAME, "context is nullptr.")（EZ1008 首报，
// ctc_loss_v3_tiling.cpp 的 TilingPrepare4CTCLossV3），整改后应返回失败
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_parse_null_context_008)
{
    CTCLossV3TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling_parse;

    EXPECT_EQ(tilingParseFunc(nullptr), ge::GRAPH_FAILED);
}

// 负向场景9-15：CheckDtypeInfo 中各 dtype 校验分支（EZ 上报）。
// CheckDtypeInfo 先于 CheckShapeInfo 执行（见 CTCLossV3Tiling::Init），故仅将对应输入/输出的
// dtype 置为非法值、其余保持合法，即可在目标分支首报并返回失败。
// 逐场景覆盖：log_probs / targets / input_lengths / target_lengths / neg_log_likelihood /
// log_alpha 的非法 dtype 分支，以及 log_probs/loss/log_alpha 三者 dtype 不一致分支

namespace {
// dtype 负向公共构造：log_probs/targets/input_lengths/target_lengths 四输入 + loss/log_alpha 两输出，
// dtypes 由各场景传入，shapes/attrs 与正向用例一致保证前置检查通过
void TestCTCLossV3DtypeNegative(ge::DataType logProbsDtype, ge::DataType targetsDtype, ge::DataType inputLengthsDtype,
                                ge::DataType targetLengthsDtype, ge::DataType lossDtype, ge::DataType logAlphaDtype)
{
    CTCLossV3TilingNegativeEnv env;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(env.opType.c_str())->tiling;

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
    auto holder = gert::TilingContextFaker()
                      .SetOpType("CTCLossV3")
                      .NodeIoNum(4, 2)
                      .InputShapes({&logProbsShape, &targetsShape, &inputLengthsShape, &targetLengthsShape})
                      .OutputShapes({&lossShape, &logAlphaShape})
                      .CompileInfo(&env.compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&env.platformInfo))
                      .NodeInputTd(0, logProbsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, targetsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputLengthsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, targetLengthsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, lossDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, logAlphaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"blank", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<string>("mean")},
                                  {"zero_infinity", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}
} // namespace

// 负向场景9：log_probs 的 dtype 非法（DT_INT32，不在 fp16/bf16/fp32 支持列表），
// 对应整改检查点 OP_LOGE_FOR_INVALID_DTYPE(nodeName, "log_probs", ...)（CheckDtypeInfo）
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_invalid_dtype_log_probs_009)
{
    TestCTCLossV3DtypeNegative(ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT);
}

// 负向场景10：targets 的 dtype 非法（DT_FLOAT，不在 int32/int64 支持列表），
// 对应整改检查点 OP_LOGE_FOR_INVALID_DTYPE(nodeName, "targets", ...)（CheckDtypeInfo）
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_invalid_dtype_targets_010)
{
    TestCTCLossV3DtypeNegative(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT);
}

// 负向场景11：input_lengths 的 dtype 非法（DT_FLOAT，不在 int32/int64 支持列表），
// 对应整改检查点 OP_LOGE_FOR_INVALID_DTYPE(nodeName, "input_lengths", ...)（CheckDtypeInfo）
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_invalid_dtype_input_lengths_011)
{
    TestCTCLossV3DtypeNegative(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT);
}

// 负向场景12：target_lengths 的 dtype 非法（DT_FLOAT，不在 int32/int64 支持列表），
// 对应整改检查点 OP_LOGE_FOR_INVALID_DTYPE(nodeName, "target_lengths", ...)（CheckDtypeInfo）
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_invalid_dtype_target_lengths_012)
{
    TestCTCLossV3DtypeNegative(ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT);
}

// 负向场景13：输出 neg_log_likelihood 的 dtype 非法（DT_INT32，不在 fp16/bf16/fp32 支持列表），
// 对应整改检查点 OP_LOGE_FOR_INVALID_DTYPE(nodeName, "neg_log_likelihood", ...)（CheckDtypeInfo）
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_invalid_dtype_loss_013)
{
    TestCTCLossV3DtypeNegative(ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_FLOAT);
}

// 负向场景14：输出 log_alpha 的 dtype 非法（DT_INT32，不在 fp16/bf16/fp32 支持列表），
// 对应整改检查点 OP_LOGE_FOR_INVALID_DTYPE(nodeName, "log_alpha", ...)（CheckDtypeInfo）
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_invalid_dtype_log_alpha_014)
{
    TestCTCLossV3DtypeNegative(ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT32);
}

// 负向场景15：log_probs/loss/log_alpha 三者 dtype 不一致（log_probs 为 fp16，loss/log_alpha 为 fp32），
// 对应整改检查点 OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, "log_probs, neg_log_likelihood and
// log_alpha", ...)（CheckDtypeInfo）
TEST_F(CTCLossV3Tiling, ctc_loss_v3_tiling_dtype_mismatch_outputs_015)
{
    TestCTCLossV3DtypeNegative(ge::DT_FLOAT16, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT);
}

// 其余 EZ1008 触发点不可构造说明（平台 faker 能力限制）：
// sysWorkspaceSize 检查点：GetLibApiWorkSpaceSize() 为芯片级常量（不读取 SoCInfo/AICoreSpec 中
// faker 可控字段，正向用例 JSON 不含任何 workspace 字段仍返回正值），无法构造 <=0 场景。
