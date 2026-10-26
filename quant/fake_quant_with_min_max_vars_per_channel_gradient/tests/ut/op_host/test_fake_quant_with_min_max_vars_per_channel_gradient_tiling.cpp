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
 * \file test_fake_quant_with_min_max_vars_per_channel_gradient_tiling.cpp
 * \brief op_host tiling UT for FakeQuantWithMinMaxVarsPerChannelGradient (arch35).
 *
 * Covers:
 *   - tiling_rows_split_rank2   : totalRows > 1 -> split rows (splitMode=0)
 *   - tiling_single_row_dchunks : totalRows == 1 -> split d-chunks (splitMode=1)
 *   - tiling_num_bits_16        : boundary num_bits=16 (quantMax=65535)
 *   - tiling_narrow_range_true  : narrow_range=true (quantMin=1)
 *   - reject: min length != channel C
 *   - reject: gradients shape != x shape
 *   - reject: x rank > 8
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/fake_quant_with_min_max_vars_per_channel_gradient_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class FakeQuantWithMinMaxVarsPerChannelGradientTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FakeQuantWithMinMaxVarsPerChannelGradientTiling SetUp" << std::endl; }
    static void TearDownTestCase()
    {
        std::cout << "FakeQuantWithMinMaxVarsPerChannelGradientTiling TearDown" << std::endl;
    }
};

static const char* kCompileInfoStr = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                      "Intrinsic_fix_pipe_l0c2out": false,
                      "Intrinsic_data_move_l12ub": true,
                      "Intrinsic_data_move_l0c2ub": true,
                      "Intrinsic_data_move_out2l1_nd2nz": false,
                      "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                      "CORE_NUM": 48}
                      })";

// Per-case tiling context setup. 4 inputs (gradients, x, min, max), 3 outputs.
#define SETUP_TILING_CTX(gradShapeV, xShapeV, mnShapeV, mxShapeV, bpxShapeV, bpmnShapeV, bpmxShapeV, numBitsVal,       \
                         narrowRangeVal)                                                                               \
    fe::PlatFormInfos platform_info;                                                                                   \
    map<string, string> soc_infos;                                                                                     \
    map<string, string> aicore_spec;                                                                                   \
    map<string, string> intrinsics;                                                                                    \
    GetPlatFormInfos(kCompileInfoStr, soc_infos, aicore_spec, intrinsics);                                             \
    platform_info.Init();                                                                                              \
                                                                                                                       \
    std::string op_type("FakeQuantWithMinMaxVarsPerChannelGradient");                                                  \
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);                                \
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;                         \
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;             \
                                                                                                                       \
    optiling::FakeQuantWithMinMaxVarsPerChannelGradientCompileInfo compile_info;                                       \
    auto kernel_holder = gert::KernelRunContextFaker()                                                                 \
                             .KernelIONum(2, 1)                                                                        \
                             .Inputs({const_cast<char*>(kCompileInfoStr), reinterpret_cast<void*>(&platform_info)})    \
                             .Outputs({&compile_info})                                                                 \
                             .Build();                                                                                 \
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());                      \
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);     \
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec",              \
                                                                                            aicore_spec);              \
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");           \
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", \
                                                                                            intrinsics);               \
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);                  \
                                                                                                                       \
    auto param = gert::TilingData::CreateCap(4096);                                                                    \
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);                                          \
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());                              \
    ASSERT_NE(param, nullptr);                                                                                         \
    auto holder = gert::TilingContextFaker()                                                                           \
                      .NodeIoNum(4, 3)                                                                                 \
                      .IrInstanceNum({1, 1, 1, 1})                                                                     \
                      .InputShapes({&(gradShapeV), &(xShapeV), &(mnShapeV), &(mxShapeV)})                              \
                      .OutputShapes({&(bpxShapeV), &(bpmnShapeV), &(bpmxShapeV)})                                      \
                      .CompileInfo(&compile_info)                                                                      \
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))                                           \
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                      \
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                      \
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                      \
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                      \
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                     \
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                     \
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)                                     \
                      .NodeAttrs({                                                                                     \
                          {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(numBitsVal)},                            \
                          {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(narrowRangeVal)},                       \
                      })                                                                                               \
                      .TilingData(param.get())                                                                         \
                      .Workspace(ws_size)                                                                              \
                      .Build();                                                                                        \
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();                                    \
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);                                                             \
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);                                           \
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);                                      \
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");                                                 \
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics)

// TilingData is serialized in field declaration order; read it as 32-bit words:
// 0 usedCoreNum 1 rowsPerCore 2 tailRows 3 totalRows 4 channelNum 5 tileLen 6 quantMin
// 7 quantMax 8 dTileLen 9 numDChunks 10 splitMode 11 chunksPerCore 12 tailChunks
static const uint32_t* ExtractTilingWords(gert::TilingContext* ctx)
{
    auto* raw = ctx->GetRawTilingData();
    EXPECT_NE(raw, nullptr);
    EXPECT_GE(raw->GetDataSize(), 13U * sizeof(uint32_t));
    return reinterpret_cast<const uint32_t*>(raw->GetData());
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTiling, tiling_rows_split_rank2)
{
    gert::StorageShape grad_shape = {{8, 16}, {8, 16}};
    gert::StorageShape x_shape = {{8, 16}, {8, 16}};
    gert::StorageShape mn_shape = {{16}, {16}};
    gert::StorageShape mx_shape = {{16}, {16}};
    gert::StorageShape bpx_shape = {{8, 16}, {8, 16}};
    gert::StorageShape bpmn_shape = {{16}, {16}};
    gert::StorageShape bpmx_shape = {{16}, {16}};

    SETUP_TILING_CTX(grad_shape, x_shape, mn_shape, mx_shape, bpx_shape, bpmn_shape, bpmx_shape, 8, false);
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    const uint32_t* w = ExtractTilingWords(tiling_context);
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w[4], 16U);                                                  // channelNum
    EXPECT_EQ(w[3], 8U);                                                   // totalRows
    EXPECT_EQ(w[10], 0U);                                                  // splitMode=0 (rows)
    EXPECT_EQ(w[9], 1U);                                                   // numDChunks
    EXPECT_EQ(w[8], 64U);                                                  // dTileLen = ceil(16, 64)
    EXPECT_EQ(static_cast<int32_t>(w[6]), 0);                              // quantMin (narrow_range=false)
    EXPECT_EQ(static_cast<int32_t>(w[7]), 255);                            // quantMax = 2^8 - 1
    EXPECT_EQ(w[0], static_cast<uint32_t>(tiling_context->GetBlockDim())); // usedCoreNum
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTiling, tiling_single_row_dchunks)
{
    gert::StorageShape grad_shape = {{16}, {16}};
    gert::StorageShape x_shape = {{16}, {16}};
    gert::StorageShape mn_shape = {{16}, {16}};
    gert::StorageShape mx_shape = {{16}, {16}};
    gert::StorageShape bpx_shape = {{16}, {16}};
    gert::StorageShape bpmn_shape = {{16}, {16}};
    gert::StorageShape bpmx_shape = {{16}, {16}};

    SETUP_TILING_CTX(grad_shape, x_shape, mn_shape, mx_shape, bpx_shape, bpmn_shape, bpmx_shape, 8, false);
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    const uint32_t* w = ExtractTilingWords(tiling_context);
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w[4], 16U); // channelNum
    EXPECT_EQ(w[3], 1U);  // totalRows
    EXPECT_EQ(w[10], 1U); // splitMode=1 (d-chunks)
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTiling, tiling_num_bits_16_narrow_range_true)
{
    gert::StorageShape grad_shape = {{4, 8}, {4, 8}};
    gert::StorageShape x_shape = {{4, 8}, {4, 8}};
    gert::StorageShape mn_shape = {{8}, {8}};
    gert::StorageShape mx_shape = {{8}, {8}};
    gert::StorageShape bpx_shape = {{4, 8}, {4, 8}};
    gert::StorageShape bpmn_shape = {{8}, {8}};
    gert::StorageShape bpmx_shape = {{8}, {8}};

    SETUP_TILING_CTX(grad_shape, x_shape, mn_shape, mx_shape, bpx_shape, bpmn_shape, bpmx_shape, 16, true);
    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    const uint32_t* w = ExtractTilingWords(tiling_context);
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(static_cast<int32_t>(w[6]), 1);     // quantMin (narrow_range=true)
    EXPECT_EQ(static_cast<int32_t>(w[7]), 65535); // quantMax = 2^16 - 1
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTiling, tiling_reject_min_length_mismatch)
{
    gert::StorageShape grad_shape = {{8, 16}, {8, 16}};
    gert::StorageShape x_shape = {{8, 16}, {8, 16}};
    gert::StorageShape mn_shape = {{8}, {8}}; // != C(16)
    gert::StorageShape mx_shape = {{16}, {16}};
    gert::StorageShape bpx_shape = {{8, 16}, {8, 16}};
    gert::StorageShape bpmn_shape = {{16}, {16}};
    gert::StorageShape bpmx_shape = {{16}, {16}};

    SETUP_TILING_CTX(grad_shape, x_shape, mn_shape, mx_shape, bpx_shape, bpmn_shape, bpmx_shape, 8, false);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTiling, tiling_reject_gradients_shape_mismatch)
{
    gert::StorageShape grad_shape = {{8, 15}, {8, 15}}; // != x
    gert::StorageShape x_shape = {{8, 16}, {8, 16}};
    gert::StorageShape mn_shape = {{16}, {16}};
    gert::StorageShape mx_shape = {{16}, {16}};
    gert::StorageShape bpx_shape = {{8, 16}, {8, 16}};
    gert::StorageShape bpmn_shape = {{16}, {16}};
    gert::StorageShape bpmx_shape = {{16}, {16}};

    SETUP_TILING_CTX(grad_shape, x_shape, mn_shape, mx_shape, bpx_shape, bpmn_shape, bpmx_shape, 8, false);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTiling, tiling_reject_rank_over_8)
{
    gert::StorageShape grad_shape = {{1, 1, 1, 1, 1, 1, 1, 1, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 4}};
    gert::StorageShape x_shape = {{1, 1, 1, 1, 1, 1, 1, 1, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 4}};
    gert::StorageShape mn_shape = {{4}, {4}};
    gert::StorageShape mx_shape = {{4}, {4}};
    gert::StorageShape bpx_shape = {{1, 1, 1, 1, 1, 1, 1, 1, 4}, {1, 1, 1, 1, 1, 1, 1, 1, 4}};
    gert::StorageShape bpmn_shape = {{4}, {4}};
    gert::StorageShape bpmx_shape = {{4}, {4}};

    SETUP_TILING_CTX(grad_shape, x_shape, mn_shape, mx_shape, bpx_shape, bpmn_shape, bpmx_shape, 8, false);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
