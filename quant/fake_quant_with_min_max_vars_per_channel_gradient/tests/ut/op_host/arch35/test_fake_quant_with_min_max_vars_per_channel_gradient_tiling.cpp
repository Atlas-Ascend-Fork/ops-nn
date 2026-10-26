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
 * \brief FakeQuantWithMinMaxVarsPerChannelGradient Host 侧 Tiling 单元测试（arch35 / Ascend 950）
 *
 * 覆盖（期望值按 op_host/..._tiling.cpp 公式手工推导，桩参数 coreNum=56、UB=262144）：
 *   - 多行场景（split_mode=0，按行多核切分）：rowsPerCore/tailRows/usedCoreNum 与
 *     workspace = 16MB + 2*usedCoreNum*ceil(D,64)*4 逐字段验证。
 *   - 单行场景（split_mode=1，按 d-chunk 多核切分）：numDChunks/chunksPerCore/tailChunks，
 *     workspace 仅系统部分（无跨核归约）。
 *   - 空 tensor（numel==0）：MODE=EMPTY 退化分支（单核、workspace=16MB、totalRows=0）。
 *   - 属性：num_bits / narrow_range 对 quantMin/quantMax 的影响（16bit + narrow → 1/65535）。
 *   - 异常：num_bits 越界（>16）与 rank-0 输入 → GRAPH_FAILED。
 *
 * 说明：TilingKey 由 GET_TPL_TILING_KEY 按值列表序号编码生成（NORMAL=0 / EMPTY=1），
 * 与模板机制绑定，本用例不硬编码断言其整型值，聚焦 TilingData 字段与 blockDim/workspace。
 */

#include <gtest/gtest.h>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/fake_quant_with_min_max_vars_per_channel_gradient_tiling.h"
#include "../../../../op_kernel/arch35/fake_quant_with_min_max_vars_per_channel_gradient_tilingdata.h"

using namespace std;
using namespace ge;

namespace {
constexpr char kOpType[] = "FakeQuantWithMinMaxVarsPerChannelGradient";

// 桩 CompileInfo：本算子 Host Tiling 从 CompileInfo 读取 totalCoreNum / ubSizePlatForm。
// 取自 950PR 实测值：AIV 核数 56、UB 256KB。
constexpr int32_t kCoreNum = 56;
constexpr uint64_t kUbSize = 262144;
constexpr uint64_t kSysWorkspace = 16u * 1024u * 1024u;

optiling::FakeQuantWithMinMaxVarsPerChannelGradientCompileInfo g_compileInfo = {kCoreNum, kUbSize};

gert::TilingContextPara MakePara(const gert::StorageShape& xShape, const gert::StorageShape& mmShape,
                                 const std::vector<gert::TilingContextPara::OpAttr>& attrs)
{
    return gert::TilingContextPara(kOpType,
                                   {
                                       {xShape, ge::DT_FLOAT, ge::FORMAT_ND},  // gradients
                                       {xShape, ge::DT_FLOAT, ge::FORMAT_ND},  // x
                                       {mmShape, ge::DT_FLOAT, ge::FORMAT_ND}, // min
                                       {mmShape, ge::DT_FLOAT, ge::FORMAT_ND}, // max
                                   },
                                   {
                                       {xShape, ge::DT_FLOAT, ge::FORMAT_ND},  // backprops_wrt_x
                                       {mmShape, ge::DT_FLOAT, ge::FORMAT_ND}, // backprops_wrt_min
                                       {mmShape, ge::DT_FLOAT, ge::FORMAT_ND}, // backprops_wrt_max
                                   },
                                   attrs, &g_compileInfo);
}

// 正常用例统一入口：跑 tiling + 逐字段断言 TilingData / blockDim / workspace
void RunTilingCase(const gert::TilingContextPara& para, const FakeQuantWithMinMaxVarsPerChannelGradientTilingData& exp,
                   size_t expBlockNum, int64_t expWorkspace)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_GE(info.tilingDataSize, sizeof(FakeQuantWithMinMaxVarsPerChannelGradientTilingData));
    const auto* td = reinterpret_cast<const FakeQuantWithMinMaxVarsPerChannelGradientTilingData*>(
        info.tilingData.get());
    EXPECT_EQ(td->usedCoreNum, exp.usedCoreNum);
    EXPECT_EQ(td->rowsPerCore, exp.rowsPerCore);
    EXPECT_EQ(td->tailRows, exp.tailRows);
    EXPECT_EQ(td->totalRows, exp.totalRows);
    EXPECT_EQ(td->channelNum, exp.channelNum);
    EXPECT_EQ(td->tileLen, exp.tileLen);
    EXPECT_EQ(td->quantMin, exp.quantMin);
    EXPECT_EQ(td->quantMax, exp.quantMax);
    EXPECT_EQ(td->dTileLen, exp.dTileLen);
    EXPECT_EQ(td->numDChunks, exp.numDChunks);
    EXPECT_EQ(td->splitMode, exp.splitMode);
    EXPECT_EQ(td->chunksPerCore, exp.chunksPerCore);
    EXPECT_EQ(td->tailChunks, exp.tailChunks);
    EXPECT_EQ(info.blockNum, expBlockNum);
    ASSERT_EQ(info.workspaceSizes.size(), 1u);
    EXPECT_EQ(info.workspaceSizes[0], expWorkspace);
}

std::vector<gert::TilingContextPara::OpAttr> AttrsNumBits8()
{
    return {{"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
            {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}};
}
} // namespace

class FakeQuantWithMinMaxVarsPerChannelGradientTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FakeQuantWithMinMaxVarsPerChannelGradientTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FakeQuantWithMinMaxVarsPerChannelGradientTilingTest TearDown" << std::endl;
    }
};

// 多行场景：x=(512,2)，按行切 56 核（dTileLen 被封顶到 ceil(D,64)=64，单 d-chunk）
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTilingTest, tiling_rows_split_2d)
{
    auto para = MakePara({{512, 2}, {512, 2}}, {{2}, {2}}, AttrsNumBits8());
    FakeQuantWithMinMaxVarsPerChannelGradientTilingData exp;
    exp.usedCoreNum = 56;
    exp.rowsPerCore = 9; // 512 / 56
    exp.tailRows = 8;    // 512 % 56
    exp.totalRows = 512;
    exp.channelNum = 2;
    exp.tileLen = 64;
    exp.quantMin = 0;
    exp.quantMax = 255;
    exp.dTileLen = 64;
    exp.numDChunks = 1;
    exp.splitMode = 0;
    exp.chunksPerCore = 1;
    exp.tailChunks = 0;
    // workspace = 16MB + 2 * 56 * ceil(2,64) * 4 = 16MB + 28672
    RunTilingCase(para, exp, 56, static_cast<int64_t>(kSysWorkspace) + 2 * 56 * 64 * 4);
}

// 单行场景：x=(1,300000)，按 d-chunk 切 43 核（dTileLen=7040=(256K-8K)/36 对 64 向下对齐）
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTilingTest, tiling_dchunk_split_1row)
{
    auto para = MakePara({{1, 300000}, {1, 300000}}, {{300000}, {300000}}, AttrsNumBits8());
    FakeQuantWithMinMaxVarsPerChannelGradientTilingData exp;
    exp.usedCoreNum = 43; // numDChunks = ceil(300000/7040) = 43 < 56
    exp.rowsPerCore = 1;
    exp.tailRows = 0;
    exp.totalRows = 1;
    exp.channelNum = 300000;
    exp.tileLen = 7040;
    exp.quantMin = 0;
    exp.quantMax = 255;
    exp.dTileLen = 7040;
    exp.numDChunks = 43;
    exp.splitMode = 1;
    exp.chunksPerCore = 1; // 43 / 43
    exp.tailChunks = 0;    // 43 % 43
    // split_mode=1 无跨核归约 workspace，仅系统部分
    RunTilingCase(para, exp, 43, static_cast<int64_t>(kSysWorkspace));
}

// 单行单块：x=(1,3)，numDChunks=1 → 单核
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTilingTest, tiling_dchunk_split_single_core)
{
    auto para = MakePara({{1, 3}, {1, 3}}, {{3}, {3}}, AttrsNumBits8());
    FakeQuantWithMinMaxVarsPerChannelGradientTilingData exp;
    exp.usedCoreNum = 1;
    exp.rowsPerCore = 1;
    exp.tailRows = 0;
    exp.totalRows = 1;
    exp.channelNum = 3;
    exp.tileLen = 64;
    exp.quantMin = 0;
    exp.quantMax = 255;
    exp.dTileLen = 64;
    exp.numDChunks = 1;
    exp.splitMode = 1;
    exp.chunksPerCore = 1;
    exp.tailChunks = 0;
    RunTilingCase(para, exp, 1, static_cast<int64_t>(kSysWorkspace));
}

// 空 tensor：x=(0,2) → MODE=EMPTY 退化分支（单核、totalRows=0、仅系统 workspace）
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTilingTest, tiling_empty_tensor)
{
    auto para = MakePara({{0, 2}, {0, 2}}, {{2}, {2}}, AttrsNumBits8());
    FakeQuantWithMinMaxVarsPerChannelGradientTilingData exp;
    exp.usedCoreNum = 1;
    exp.rowsPerCore = 0;
    exp.tailRows = 0;
    exp.totalRows = 0;
    exp.channelNum = 2;
    exp.tileLen = 64;
    exp.quantMin = 0;
    exp.quantMax = 255;
    exp.dTileLen = 64;
    exp.numDChunks = 1;
    exp.splitMode = 0;
    exp.chunksPerCore = 0;
    exp.tailChunks = 0;
    RunTilingCase(para, exp, 1, static_cast<int64_t>(kSysWorkspace));
}

// 属性：16bit + narrow_range → quantMin=1, quantMax=65535
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTilingTest, tiling_attrs_16bit_narrow)
{
    auto para = MakePara({{64, 4}, {64, 4}}, {{4}, {4}},
                         {{"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(16)},
                          {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(true)}});
    FakeQuantWithMinMaxVarsPerChannelGradientTilingData exp;
    exp.usedCoreNum = 56;
    exp.rowsPerCore = 1; // 64 / 56
    exp.tailRows = 8;    // 64 % 56
    exp.totalRows = 64;
    exp.channelNum = 4;
    exp.tileLen = 64;
    exp.quantMin = 1;
    exp.quantMax = 65535;
    exp.dTileLen = 64;
    exp.numDChunks = 1;
    exp.splitMode = 0;
    exp.chunksPerCore = 1;
    exp.tailChunks = 0;
    RunTilingCase(para, exp, 56, static_cast<int64_t>(kSysWorkspace) + 2 * 56 * 64 * 4);
}

// 异常：num_bits 越界（20 > 16）→ GRAPH_FAILED
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTilingTest, tiling_invalid_num_bits)
{
    auto para = MakePara({{8, 8}, {8, 8}}, {{8}, {8}},
                         {{"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(20)},
                          {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 异常：rank-0 输入 → GRAPH_FAILED
TEST_F(FakeQuantWithMinMaxVarsPerChannelGradientTilingTest, tiling_rank0_invalid)
{
    auto para = MakePara({{}, {}}, {{}, {}}, AttrsNumBits8());
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
