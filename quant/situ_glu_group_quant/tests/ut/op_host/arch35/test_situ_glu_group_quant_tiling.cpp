/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/situ_glu_group_quant_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;

class SituGluGroupQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SituGluGroupQuantTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "SituGluGroupQuantTilingTest TearDown" << std::endl; }
};

// attrs follow the def registration order: beta, linear_beta,
// activate_left, axis, quant_mode, dst_type, dst_type_max, round_mode
static void ExecuteTestCase(ge::DataType xDtype, ge::DataType yDtype, ge::DataType y_scaleDtype,
                            gert::StorageShape xShape, gert::StorageShape yShape, gert::StorageShape y_scaleShape,
                            gert::StorageShape tokensShape, float beta, float linearBeta,
                            ge::graphStatus status = ge::GRAPH_SUCCESS, bool tokensAbsent = false, int64_t axis = -1,
                            int64_t quantMode = 0, int64_t dstType = 36, float dstTypeMax = 0.0f,
                            const std::string& roundMode = "rint", int64_t groupListType = 1,
                            int64_t* outRowsPerBatch = nullptr)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
            "CORE_NUM": 64}
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::SituGluGroupQuantCompileInfo compile_info;

    std::string op_type("SituGluGroupQuant");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::TilingContextFaker holderBuilder;
    holderBuilder.SetOpType(op_type)
        .NodeIoNum(3, 2)
        .IrInstanceNum(tokensAbsent ? std::vector<uint32_t>{1, 0, 0} : std::vector<uint32_t>{1, 0, 1})
        .InputShapes(tokensAbsent ? std::vector<void*>{&xShape} : std::vector<void*>{&xShape, &tokensShape})
        .OutputShapes({&yShape, &y_scaleShape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, y_scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"beta", Ops::NN::AnyValue::CreateFrom<float>(beta)},
                    {"linear_beta", Ops::NN::AnyValue::CreateFrom<float>(linearBeta)},
                    {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                    {"quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(quantMode)},
                    {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dstType)},
                    {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(dstTypeMax)},
                    {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>(roundMode)},
                    {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(groupListType)}})
        .TilingData(param.get())
        .Workspace(ws_size);
    if (!tokensAbsent) {
        holderBuilder.NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = holderBuilder.Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), status);
    if (outRowsPerBatch != nullptr) {
        auto* tilingData = tiling_context->GetTilingData<SituGluGroupQuantTilingData>();
        ASSERT_NE(tilingData, nullptr);
        *outRowsPerBatch = tilingData->rowsPerBatch;
    }
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_bf16_basic)
{
    gert::StorageShape xShape = {{8, 128, 4096}, {8, 128, 4096}};
    gert::StorageShape yShape = {{8, 128, 2048}, {8, 128, 2048}};
    gert::StorageShape y_scaleShape = {{8, 128, 32, 2}, {8, 128, 32, 2}};
    gert::StorageShape tokensShape = {{4}, {4}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_fp16)
{
    gert::StorageShape xShape = {{4, 64, 2048}, {4, 64, 2048}};
    gert::StorageShape yShape = {{4, 64, 1024}, {4, 64, 1024}};
    gert::StorageShape y_scaleShape = {{4, 64, 16, 2}, {4, 64, 16, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_epr_3072_boundary)
{
    // EPR = 3072 = MAX_TILE_COLS, exactly at the whole-row chunk cap
    gert::StorageShape xShape = {{4, 6144}, {4, 6144}};
    gert::StorageShape yShape = {{4, 3072}, {4, 3072}};
    gert::StorageShape y_scaleShape = {{4, 48, 2}, {4, 48, 2}};
    gert::StorageShape tokensShape = {{4}, {4}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_zero_beta)
{
    gert::StorageShape xShape = {{8, 128, 8192}, {8, 128, 8192}};
    gert::StorageShape yShape = {{8, 128, 4096}, {8, 128, 4096}};
    gert::StorageShape y_scaleShape = {{8, 128, 64, 2}, {8, 128, 64, 2}};
    gert::StorageShape tokensShape = {{4}, {4}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    0.0f, 2.0f, ge::GRAPH_FAILED);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_linear_disabled_negative_linearBeta)
{
    // linear_beta <= 0 is now a legal disabled mode (up passthrough, TPL
    // dimension): tiling must succeed and select the disabled tiling key.
    gert::StorageShape xShape = {{8, 128, 4096}, {8, 128, 4096}};
    gert::StorageShape yShape = {{8, 128, 2048}, {8, 128, 2048}};
    gert::StorageShape y_scaleShape = {{8, 128, 32, 2}, {8, 128, 32, 2}};
    gert::StorageShape tokensShape = {{4}, {4}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, -1.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_linear_disabled_zero_linearBeta)
{
    // linear_beta = 0 (the new default) disables the soft-clamp: legal.
    gert::StorageShape xShape = {{8, 128, 4096}, {8, 128, 4096}};
    gert::StorageShape yShape = {{8, 128, 2048}, {8, 128, 2048}};
    gert::StorageShape y_scaleShape = {{8, 128, 32, 2}, {8, 128, 32, 2}};
    gert::StorageShape tokensShape = {{4}, {4}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 0.0f, ge::GRAPH_SUCCESS);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_odd_gc_k_batch_engaged)
{
    // odd-gc K-batching gate: EPR = 160 -> groupCount = 5, scaleColNum = 6.
    // The batch is allowed now (per-row Compact scale copy-out in the kernel);
    // assert the tiling succeeds AND K-batching actually engages
    // (rowsPerBatch > 1) instead of degenerating to the per-row cycle.
    // rows=512: K is rows-aware (load-balance cap = rows / (cores*2), mock
    // CORE_NUM=64 -> cap 4), so batching engages only with enough rows.
    gert::StorageShape xShape = {{512, 320}, {512, 320}};
    gert::StorageShape yShape = {{512, 160}, {512, 160}};
    gert::StorageShape y_scaleShape = {{512, 3, 2}, {512, 3, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    int64_t rowsPerBatch = -1;
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_SUCCESS, false, -1, 0, 36, 0.0f, "rint", 1, &rowsPerBatch);
    EXPECT_GT(rowsPerBatch, 1);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_odd_last_dim)
{
    gert::StorageShape xShape = {{4, 64, 1023}, {4, 64, 1023}};
    gert::StorageShape yShape = {{4, 64, 512}, {4, 64, 512}};
    gert::StorageShape y_scaleShape = {{4, 64, 8, 2}, {4, 64, 8, 2}};
    gert::StorageShape tokensShape = {{4}, {4}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_1d_shape)
{
    gert::StorageShape xShape = {{256}, {256}};
    gert::StorageShape yShape = {{128}, {128}};
    gert::StorageShape y_scaleShape = {{2, 2}, {2, 2}};
    gert::StorageShape tokensShape = {{1}, {1}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 1.0f, ge::GRAPH_SUCCESS);
}
// recipes caliber: no empty-tensor rejection — a zero-size leading dim passes
// and degenerates to totalRows=0 (blockDim clamps to 1 in the class, validRows=0)
TEST_F(SituGluGroupQuantTilingTest, test_tiling_empty_leading_dim_passes)
{
    gert::StorageShape xShape = {{0, 256}, {0, 256}};
    gert::StorageShape yShape = {{0, 128}, {0, 128}};
    gert::StorageShape y_scaleShape = {{0, 2, 2}, {0, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_SUCCESS);
}

// recipes caliber: no group_index emptiness check — an empty tokens tensor
// passes (groupIndexCount=0, kernel validRows=0, nothing is processed)
TEST_F(SituGluGroupQuantTilingTest, test_tiling_empty_group_index_passes)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{0}, {0}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_SUCCESS);
}
// EPR = 4096 > 3072 (MAX_TILE_COLS): the kernel CopyIn moves the whole row into
// a chunk-sized buffer, so tiling fail-fasts (owner decision, fix-forward of the
// upstream granularity bug)
TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_epr_above_3072)
{
    gert::StorageShape xShape = {{4, 8192}, {4, 8192}};
    gert::StorageShape yShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y_scaleShape = {{4, 64, 2}, {4, 64, 2}};
    gert::StorageShape tokensShape = {{4}, {4}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED);
}
// Wrong output dtype (y = e5m2 instead of the fixed e4m3fn): rejected by
// ValidateOutput (early-error defense; also rejected by def combination /
// simplifiedKey one layer later)
TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_wrong_y_dtype)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED);
}

// Optional group_index NOT instantiated ({1, 0}): whole-tensor mode, tiling
// succeeds with fullRowsMode=1 (all rows valid, no tokens validation)
TEST_F(SituGluGroupQuantTilingTest, test_tiling_optional_absent_full_mode)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape;
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_SUCCESS, true);
}

// Optional group_index instantiated: behaves identically to the (formerly
// required) grouped semantics
TEST_F(SituGluGroupQuantTilingTest, test_tiling_optional_present_same_as_grouped)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_SUCCESS, false);
}

// ---- Extension attrs (situ_mx_quant caliber): default values pass ----
TEST_F(SituGluGroupQuantTilingTest, test_tiling_ext_attrs_default_pass)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_SUCCESS, false);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_axis_not_last)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED, false, /*axis=*/0);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_quant_mode_nonzero)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED, false, -1, /*quantMode=*/1);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_dst_type_not_e4m3fn)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED, false, -1, 0, /*dstType=*/40);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_dst_type_out_of_enum)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED, false, -1, 0, /*dstType=*/99);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_dst_type_max_negative)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED, false, -1, 0, 36, /*dstTypeMax=*/-1.0f);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_round_mode_not_rint)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED, false, -1, 0, 36, 0.0f, /*roundMode=*/"floor");
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_round_mode_invalid_string)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED, false, -1, 0, 36, 0.0f, /*roundMode=*/"nearest");
}

// ---- group_list_type: 0=cumsum (reserved, rejected), 1=count (only supported) ----
TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_group_list_type_cumsum_reserved)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED, false, -1, 0, 36, 0.0f, "rint", /*groupListType=*/0);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_group_list_type_out_of_range)
{
    gert::StorageShape xShape = {{8, 256}, {8, 256}};
    gert::StorageShape yShape = {{8, 128}, {8, 128}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED, false, -1, 0, 36, 0.0f, "rint", /*groupListType=*/2);
}

// ---- EPR lower bound / 32-alignment guards (owner decision, onboard-proven
// failure modes inherited from recipes: EPR=64 leaves y_scale uninitialized,
// EPR=129 faults the AIV in the non-32B-aligned CopyIn else-branch) ----
TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_epr_below_128)
{
    // x last dim 128 -> EPR=64 < 128
    gert::StorageShape xShape = {{8, 128}, {8, 128}};
    gert::StorageShape yShape = {{8, 64}, {8, 64}};
    gert::StorageShape y_scaleShape = {{8, 1, 2}, {8, 1, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_epr_96_below_128)
{
    // multiple of 32 but still below the groupCount>=4 floor
    gert::StorageShape xShape = {{8, 192}, {8, 192}};
    gert::StorageShape yShape = {{8, 96}, {8, 96}};
    gert::StorageShape y_scaleShape = {{8, 2, 2}, {8, 2, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED);
}

TEST_F(SituGluGroupQuantTilingTest, test_tiling_error_epr_not_multiple_of_32)
{
    // x last dim 258 -> EPR=129, non-32B-aligned CopyIn else-branch
    gert::StorageShape xShape = {{8, 258}, {8, 258}};
    gert::StorageShape yShape = {{8, 129}, {8, 129}};
    gert::StorageShape y_scaleShape = {{8, 3, 2}, {8, 3, 2}};
    gert::StorageShape tokensShape = {{2}, {2}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, xShape, yShape, y_scaleShape, tokensShape,
                    1.0f, 2.0f, ge::GRAPH_FAILED);
}
