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
 * \file test_group_norm_v2_tiling.cpp
 * \brief
 */

#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "exe_graph/runtime/kernel_context.h"
#include "tiling_context_faker.h"
#include "exe_graph/runtime/tiling_data.h"
#include "exe_graph/runtime/continuous_vector.h"
#include <iostream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "array_ops.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../op_host/arch35/group_norm_v2_tiling_arch35.h"
using namespace ge;
using namespace ut_util;
using namespace std;

class GroupNormV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupNormV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupNormV2Tiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = tiling_data->GetDataSize() - sizeof(int64_t); i < tiling_data->GetDataSize();
         i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                         map<string, string>& socVersion)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false,
                           "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true,
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    // platform info
    platFormInfo.Init();
}

TEST_F(GroupNormV2Tiling, GroupNormV2_tiling_0)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("GroupNormV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    // compile info
    optiling::GroupNormV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;
    compile_info.blockSize = 32;
    compile_info.vectorLength = 256;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{36, 48, 24, 1}, {36, 48, 24, 1}};
    gert::StorageShape gamma = {{48}, {48}};
    gert::StorageShape beta = {{48}, {48}};
    gert::StorageShape y = {{36, 48, 24, 1}, {36, 48, 24, 1}};
    gert::StorageShape mean = {{36, 6}, {36, 6}};
    gert::StorageShape rstd = {{36, 6}, {36, 6}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &gamma, &beta})
                      .OutputShapes({&y, &mean, &rstd})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    // workspaces nullptr return failed
    std::cout << "test>> holder.GetContext end" << std::endl;

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1110);
}

TEST_F(GroupNormV2Tiling, GroupNormV2_tiling_1)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("GroupNormV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    // compile info
    optiling::GroupNormV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262114;
    compile_info.blockSize = 32;
    compile_info.vectorLength = 256;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1, 128, 64, 64}, {1, 128, 64, 64}};
    gert::StorageShape gamma = {{128}, {128}};
    gert::StorageShape beta = {{128}, {128}};
    gert::StorageShape y = {{1, 128, 64, 64}, {1, 128, 64, 64}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &gamma, &beta})
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    // workspaces nullptr return failed
    std::cout << "test>> holder.GetContext end" << std::endl;

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1111);
}

TEST_F(GroupNormV2Tiling, GroupNormV2_tiling_2)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("GroupNormV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    // compile info
    optiling::GroupNormV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262114;
    compile_info.blockSize = 32;
    compile_info.vectorLength = 256;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1, 5120, 32, 32}, {1, 5120, 32, 32}};
    gert::StorageShape gamma = {{5120}, {5120}};
    gert::StorageShape beta = {{5120}, {5120}};
    gert::StorageShape y = {{1, 5120, 32, 32}, {1, 5120, 32, 32}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &gamma, &beta})
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    // workspaces nullptr return failed
    std::cout << "test>> holder.GetContext end" << std::endl;

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1120);
}

// Dump key fields of GroupNormV2TilingData from the raw tiling buffer.
// Flat layout (all int64, epsilon float @112), total 160 bytes:
// 0:numGroups 1:hwNum 2:elemNum 3:shapeC 4:shapeD 5:realCoreNum 6:numPerCore 7:numLastCore 8:processSize
// 9:loopNum 10:loopTail 11:innerLoopNum 12:innerLoopTail 13:tilingKey 14:eps(float) 15:parallelN
// 16:ubSize 17:dichotomyAddPower 18:dichotomyAddK 19:dichotomyAddLastNum
static void DumpTilingData(gert::TilingContext* ctx, const char* tag)
{
    auto raw = ctx->GetRawTilingData();
    auto fields = reinterpret_cast<const int64_t*>(raw->GetData());
    std::cout << "[" << tag << "] tilingKey=" << fields[13] << " parallelN=" << fields[15] << " loopNum=" << fields[9]
              << " loopTail=" << fields[10] << " processSize=" << fields[8] << " innerLoopNum=" << fields[11]
              << " dichotomyAddPower=" << fields[17] << " dichotomyAddK=" << fields[18]
              << " dichotomyAddLastNum=" << fields[19] << " ubSize=" << fields[16] << std::endl;
}

// Large channel (C > 4096) with R axis not full-loadable => TILINGKEY_WELFORD_GENERALIZED_MIX_TYPE(1121).
// Inside SetUbTiling4WelfordGeneralized the while-loop shrinks count and recomputes gamma/beta chunk sizes;
// the UB remainder deducted after the loop must be the final chunk sizes, otherwise parallelN is
// under-allocated (regression guard for issue #9394: parallelN 21392 -> 21408, kernel welford passes 50 -> 49).
TEST_F(GroupNormV2Tiling, GroupNormV2_tiling_welford_generalized_ub_budget)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("GroupNormV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    // compile info
    optiling::GroupNormV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262114;
    compile_info.blockSize = 32;
    compile_info.vectorLength = 256;

    // x = {1, 16384, 64, 1} fp16, gamma/beta = {16384} fp32, num_groups = 1
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1, 16384, 64, 1}, {1, 16384, 64, 1}};
    gert::StorageShape gamma = {{16384}, {16384}};
    gert::StorageShape beta = {{16384}, {16384}};
    gert::StorageShape y = {{1, 16384, 64, 1}, {1, 16384, 64, 1}};
    gert::StorageShape mean = {{1, 1}, {1, 1}};
    gert::StorageShape rstd = {{1, 1}, {1, 1}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &gamma, &beta})
                      .OutputShapes({&y, &mean, &rstd})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1121);
    DumpTilingData(tiling_context, "welford_generalized_ub_budget");
    // parallelN is fully determined by the compile_info above (ubSize=262114, blockSize=32) and this
    // shape: 21360 uses the gamma/beta chunk sizes at the FINAL shrunk count (2016B each) minus the fixed
    // UB_RESERVED_SIZE, while the shadowing bug deducted the pre-shrink sizes (2048B each) and produced 21392.
    auto rawTiling = tiling_context->GetRawTilingData();
    auto fields = reinterpret_cast<const int64_t*>(rawTiling->GetData());
    EXPECT_EQ(fields[15], 21360); // parallelN
    EXPECT_EQ(fields[9], 33);     // loopNum
}

// C <= 4096 with R axis not full-loadable => TILINGKEY_WELFORD_PERF(1100): gamma/beta stay fully
// resident in UB and the reserved margin is deducted in the isLargeChannel == false branch.
TEST_F(GroupNormV2Tiling, GroupNormV2_tiling_welford_perf_budget)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("GroupNormV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    optiling::GroupNormV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262114;
    compile_info.blockSize = 32;
    compile_info.vectorLength = 256;

    // x = {1, 128, 128, 128} fp16, gamma/beta = {128} fp16, num_groups = 32 => shapeD = 4, hwNum = 16384
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1, 128, 128, 128}, {1, 128, 128, 128}};
    gert::StorageShape gamma = {{128}, {128}};
    gert::StorageShape beta = {{128}, {128}};
    gert::StorageShape y = {{1, 128, 128, 128}, {1, 128, 128, 128}};
    gert::StorageShape mean = {{1, 32}, {1, 32}};
    gert::StorageShape rstd = {{1, 32}, {1, 32}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &gamma, &beta})
                      .OutputShapes({&y, &mean, &rstd})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1100);
    DumpTilingData(tiling_context, "welford_perf_budget");
}

// num_groups attached with a mismatched attr type must fail gracefully instead of dereferencing
// a null attr pointer (regression guard for the numGroups null-check fix).
TEST_F(GroupNormV2Tiling, GroupNormV2_tiling_num_groups_invalid_type)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("GroupNormV2");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    optiling::GroupNormV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262114;
    compile_info.blockSize = 32;
    compile_info.vectorLength = 256;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1, 128, 64, 64}, {1, 128, 64, 64}};
    gert::StorageShape gamma = {{128}, {128}};
    gert::StorageShape beta = {{128}, {128}};
    gert::StorageShape y = {{1, 128, 64, 64}, {1, 128, 64, 64}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &gamma, &beta})
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      // num_groups attached as float (type mismatch): GetAttrPointer<int64_t> -> nullptr
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<float>(32.0f)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    std::cout << "[num_groups_invalid_type] calling tiling_func with float-typed num_groups..." << std::endl;
    auto ret = tiling_func(tiling_context);
    std::cout << "[num_groups_invalid_type] tiling_func returned " << ret << " (expected GRAPH_FAILED)" << std::endl;
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

// gamma input desc present but storage shape missing: tiling must fail explicitly (via CheckInputParams)
// instead of letting an unvalidated shape query flow into the UB size computation.
TEST_F(GroupNormV2Tiling, GroupNormV2_tiling_gamma_shape_null)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("GroupNormV2");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    optiling::GroupNormV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262114;
    compile_info.blockSize = 32;
    compile_info.vectorLength = 256;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1, 128, 64, 64}, {1, 128, 64, 64}};
    gert::StorageShape beta = {{128}, {128}};
    gert::StorageShape y = {{1, 128, 64, 64}, {1, 128, 64, 64}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, nullptr, &beta}) // gamma shape = nullptr
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    std::cout << "[gamma_shape_null] calling tiling_func with gamma storage shape nullptr..." << std::endl;
    auto ret = tiling_func(tiling_context);
    std::cout << "[gamma_shape_null] tiling_func returned " << ret << " (expected GRAPH_FAILED)" << std::endl;
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST_F(GroupNormV2Tiling, GroupNormV2_tiling_3)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("GroupNormV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    // compile info
    optiling::GroupNormV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;
    compile_info.blockSize = 32;
    compile_info.vectorLength = 256;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{36, 48, 24, 1}, {36, 48, 24, 1}};
    gert::StorageShape gamma = {{48}, {48}};
    gert::StorageShape beta = {{48}, {48}};
    gert::StorageShape y = {{36, 48, 24, 1}, {36, 48, 24, 1}};
    gert::StorageShape mean = {{36, 6}, {36, 6}};
    gert::StorageShape rstd = {{36, 6}, {36, 6}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &gamma, &beta})
                      .OutputShapes({&y, &mean, &rstd})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(6)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    // workspaces nullptr return failed
    std::cout << "test>> holder.GetContext end" << std::endl;

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1111);
}

namespace optiling {
ge::graphStatus SetTilingData(gert::TilingContext*);
}
namespace {
class GroupNormV2MetadataError : public testing::TestWithParam<int> {};
TEST_P(GroupNormV2MetadataError, MetadataGuardFailsBeforeByteBudgetAndReportsOnce)
{
    gert::StorageShape x = {{2, 4, 2}, {2, 4, 2}};
    gert::StorageShape a = {{4}, {4}};
    auto f = gert::TilingContextFaker();
    f.SetOpType("GroupNormV2").NodeIoNum(3, 3).IrInstanceNum({1, 1, 1}, {1, 1, 1});
    for (int i = 0; i < 3; ++i) {
        f.NodeInputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        f.NodeOutputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    f.InputShapes(std::vector<gert::StorageShape*>{&x, &a, &a})
        .OutputShapes(std::vector<gert::StorageShape*>{&x, &a, &a});
    fe::PlatFormInfos platform;
    platform.Init();
    optiling::GroupNormV2CompileInfo compileInfo{};
    auto data = gert::TilingData::CreateCap(10240);
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);
    f.CompileInfo(&compileInfo)
        .PlatformInfo(&platform)
        .TilingData(data.get())
        .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
    auto h = f.Build();
    auto* c = h.GetContext<gert::TilingContext>();
    ASSERT_NE(c, nullptr);
    const int index = GetParam();
    ASSERT_NE(c->GetInputDesc(index), nullptr);
    ASSERT_NE(c->GetInputShape(index), nullptr);
    auto* chain = reinterpret_cast<gert::KernelContext*>(c)->MutableInput(index);
    ASSERT_NE(chain, nullptr);
    void*& slot = chain->GetValue<void*>();
    void* saved = slot;
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    (void)manager.GetRawErrorMessages();
    slot = nullptr;
    EXPECT_EQ(c->GetInputShape(index), nullptr);
    const auto status = optiling::SetTilingData(c);
    slot = saved;
    const auto errors = manager.GetRawErrorMessages();
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    const auto& fields = errors[0].args_map;
    ASSERT_EQ(fields.count("opName"), 1U);
    EXPECT_EQ(fields.at("opName"), c->GetNodeName());
    ASSERT_EQ(fields.count("reason"), 1U);
    EXPECT_NE(fields.at("reason").find("shape metadata is missing"), std::string::npos);
    EXPECT_NE(fields.at("reason").find(index == 1 ? "gamma" : "beta"), std::string::npos);
}
INSTANTIATE_TEST_SUITE_P(GammaBeta, GroupNormV2MetadataError, testing::Values(1, 2));
} // namespace
