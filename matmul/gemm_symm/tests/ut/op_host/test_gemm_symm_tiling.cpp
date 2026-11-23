/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <initializer_list>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_kernel/arch35/gemm_symm_tiling_data.h"
#include "../../../op_kernel/arch35/gemm_symm_tiling_key.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"

using namespace std;

namespace {

struct GemmSymmTilingParam {
    string caseName;
    initializer_list<int64_t> aShape;
    initializer_list<int64_t> bShape;
    initializer_list<int64_t> cShape;
    initializer_list<int64_t> yShape;
    bool hasC{true};
    float alpha{1.0F};
    float beta{1.0F};
    bool transA{false};
    bool transB{false};
    string symmType{"full"};
    ge::DataType dtype{ge::DT_BF16};
    ge::graphStatus expectStatus{ge::GRAPH_SUCCESS};
    uint32_t expectM{0};
    uint32_t expectK{0};
    uint32_t expectBatch{1};
    uint32_t expectUsedCore{1};
    int64_t cubeMathType{1};
};

class GemmSymmTilingTest : public testing::TestWithParam<GemmSymmTilingParam> {};

static const string kAscend950CompileInfo = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown",
    "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false,
    "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "socVersion": "Ascend950",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 64}
})";

void RunTilingCase(const GemmSymmTilingParam& param)
{
    gert::StorageShape aShape = {param.aShape, param.aShape};
    gert::StorageShape bShape = {param.bShape, param.bShape};
    gert::StorageShape cShape = {param.cShape, param.cShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    struct GemmSymmUtCompileInfo {
    } compileInfo;

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion;
    GetPlatFormInfos(kAscend950CompileInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSymm"), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSymm")->tiling;

    auto tilingData = gert::TilingData::CreateCap(2048);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(16);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    vector<void*> inputShapes{&aShape, &bShape};
    if (param.hasC) {
        inputShapes.push_back(&cShape);
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType("GemmSymm")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, param.hasC ? 1 : 0})
                      .InputShapes(inputShapes)
                      .OutputShapes({&yShape})
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(param.alpha)},
                                  {"beta", Ops::NN::AnyValue::CreateFrom<float>(param.beta)},
                                  {"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(param.transA)},
                                  {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(param.transB)},
                                  {"symm_type", Ops::NN::AnyValue::CreateFrom<std::string>(param.symmType)},
                                  {"op_impl_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(param.cubeMathType)}})
                      .NodeInputTd(0, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();
    if (param.hasC) {
        holder = gert::TilingContextFaker()
                     .SetOpType("GemmSymm")
                     .NodeIoNum(3, 1)
                     .IrInstanceNum({1, 1, 1})
                     .InputShapes({&aShape, &bShape, &cShape})
                     .OutputShapes({&yShape})
                     .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(param.alpha)},
                                 {"beta", Ops::NN::AnyValue::CreateFrom<float>(param.beta)},
                                 {"transpose_a", Ops::NN::AnyValue::CreateFrom<bool>(param.transA)},
                                 {"transpose_b", Ops::NN::AnyValue::CreateFrom<bool>(param.transB)},
                                 {"symm_type", Ops::NN::AnyValue::CreateFrom<std::string>(param.symmType)},
                                 {"op_impl_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(param.cubeMathType)}})
                     .NodeInputTd(0, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(1, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeInputTd(2, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                     .CompileInfo(&compileInfo)
                     .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                     .TilingData(tilingData.get())
                     .Workspace(workspace)
                     .Build();
    }

    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context->GetPlatformInfo(), nullptr);
    context->GetPlatformInfo()->SetPlatformRes("version", socVersion);
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetWorkspaceSizes(1)[0] = 123;

    EXPECT_EQ(tilingFunc(context), param.expectStatus) << param.caseName;
    if (param.expectStatus != ge::GRAPH_SUCCESS) {
        return;
    }

    EXPECT_EQ(context->GetWorkspaceSizes(1)[0], 0U);
    const bool highPrec = param.cubeMathType == 4 && param.hasC && param.beta != 0.0F;
    EXPECT_EQ(context->GetTilingKey(),
              GET_TPL_TILING_KEY(param.transA ? GEMM_SYMM_TRANS_TRUE : GEMM_SYMM_TRANS_FALSE,
                                 param.transB ? GEMM_SYMM_TRANS_TRUE : GEMM_SYMM_TRANS_FALSE,
                                 highPrec ? GEMM_SYMM_HIGH_PREC_TRUE : GEMM_SYMM_HIGH_PREC_FALSE));
    ASSERT_GE(context->GetRawTilingData()->GetDataSize(), sizeof(GemmSymmTilingData));
    const auto* data = reinterpret_cast<const GemmSymmTilingData*>(context->GetRawTilingData()->GetData());
    EXPECT_EQ(data->m, param.expectM);
    EXPECT_EQ(data->k, param.expectK);
    EXPECT_EQ(data->batch, param.expectBatch);
    EXPECT_EQ(data->usedCoreNum, param.expectUsedCore);
    EXPECT_EQ(context->GetBlockDim(), param.expectUsedCore);
    EXPECT_EQ(data->transA, param.transA ? 1 : 0);
    EXPECT_EQ(data->transB, param.transB ? 1 : 0);
    EXPECT_EQ(data->hasC, param.hasC ? 1 : 0);
    EXPECT_FLOAT_EQ(data->alpha, param.alpha);
    EXPECT_FLOAT_EQ(data->beta, param.hasC ? param.beta : 0.0F);
}

static const GemmSymmTilingParam kCases[] = {
    {"bf16_2d",
     {256, 64},
     {64, 256},
     {256, 256},
     {256, 256},
     true,
     2.0F,
     0.5F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_SUCCESS,
     256,
     64,
     1,
     1},
    {"bf16_batch",
     {2, 128, 32},
     {2, 32, 128},
     {2, 128, 128},
     {2, 128, 128},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_SUCCESS,
     128,
     32,
     2,
     1},
    {"transpose_both",
     {64, 256},
     {256, 64},
     {256, 256},
     {256, 256},
     true,
     1.0F,
     1.0F,
     true,
     true,
     "full",
     ge::DT_BF16,
     ge::GRAPH_SUCCESS,
     256,
     64,
     1,
     1},
    {"two_tile_rows",
     {512, 128},
     {128, 512},
     {512, 512},
     {512, 512},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_SUCCESS,
     512,
     128,
     1,
     3},
    {"beta0_without_c",
     {32, 16},
     {16, 32},
     {},
     {32, 32},
     false,
     1.0F,
     0.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_SUCCESS,
     32,
     16,
     1,
     1},
    {"k0_keeps_c",
     {32, 0},
     {0, 32},
     {32, 32},
     {32, 32},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_SUCCESS,
     32,
     0,
     1,
     1},
    {"high_prec_192",
     {256, 64},
     {64, 256},
     {256, 256},
     {256, 256},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_SUCCESS,
     256,
     64,
     1,
     3,
     4},
    {"high_prec_without_c",
     {256, 64},
     {64, 256},
     {},
     {256, 256},
     false,
     1.0F,
     0.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_SUCCESS,
     256,
     64,
     1,
     1,
     4},
    {"high_prec_beta0",
     {256, 64},
     {64, 256},
     {256, 256},
     {256, 256},
     true,
     1.0F,
     0.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_SUCCESS,
     256,
     64,
     1,
     1,
     4},
    {"fp16_2d",
     {32, 16},
     {16, 32},
     {32, 32},
     {32, 32},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_FLOAT16,
     ge::GRAPH_SUCCESS,
     32,
     16,
     1,
     1},
    {"reject_fp32",
     {32, 16},
     {16, 32},
     {32, 32},
     {1, 1},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_FLOAT,
     ge::GRAPH_FAILED,
     0,
     0,
     1,
     1},
    {"reject_missing_c",
     {32, 16},
     {16, 32},
     {},
     {1, 1},
     false,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_FAILED,
     0,
     0,
     1,
     1},
    {"reject_shape",
     {32, 16},
     {16, 31},
     {32, 32},
     {1, 1},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_FAILED,
     0,
     0,
     1,
     1},
    {"reject_b_batch",
     {2, 32, 16},
     {1, 16, 32},
     {2, 32, 32},
     {2, 32, 32},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_FAILED,
     0,
     0,
     1,
     1},
    {"reject_y_batch",
     {2, 32, 16},
     {2, 16, 32},
     {2, 32, 32},
     {1, 32, 32},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_FAILED,
     0,
     0,
     1,
     1},
    {"reject_c_shape",
     {32, 16},
     {16, 32},
     {32, 16},
     {32, 32},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_FAILED,
     0,
     0,
     1,
     1},
    {"reject_c_batch",
     {2, 32, 16},
     {2, 16, 32},
     {1, 32, 32},
     {2, 32, 32},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "full",
     ge::DT_BF16,
     ge::GRAPH_FAILED,
     0,
     0,
     1,
     1},
    {"reject_symm_type",
     {32, 16},
     {16, 32},
     {32, 32},
     {1, 1},
     true,
     1.0F,
     1.0F,
     false,
     false,
     "upper",
     ge::DT_BF16,
     ge::GRAPH_FAILED,
     0,
     0,
     1,
     1},
};

INSTANTIATE_TEST_SUITE_P(GemmSymm, GemmSymmTilingTest, testing::ValuesIn(kCases));

TEST_P(GemmSymmTilingTest, tiling) { RunTilingCase(GetParam()); }

} // namespace
