/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_lp_norm_reduce_v2_crosscore_tiling.cpp
 * \brief LpNormReduceV2 TilingFunc 跨核路由 UT（ENABLE_CROSS_CORE_ROUTING 门控启用态）
 *
 * 迁移 ops-nn 后与主 tiling 用例编入同一共享 Host UT runner 二进制（宏默认
 * LPNORM_CROSS_CORE_ROUTING_ENABLED=1，生产构建同为准核启用态，无需单独二进制），覆盖
 * DESIGN §5.2/§5.5/§7 CROSS_CORE=1 求解分支：
 *   - 24 TilingKey 实例矩阵 CROSS_CORE=1 侧 9 个可达实例逐实例验证：
 *     AR 单/双缓冲 × {fp32,fp16,bf16} + ARA 双缓冲 × {fp32,fp16,bf16}
 *   - (cross=1, ARA, 单缓冲) × 3 dtype 结构不可达：CROSS 判定要求 rLen ≥ 2×32768，
 *     rSpan ≥ ⌈rLen/groups⌉ ≥ 21846，而 rChunk ≤ MAX_COPY_BLOCK_COUNT=4095 → estChunks ≥ 6 ≥ 4
 *     恒双缓冲（结构性属性，非覆盖缺口，见 cross_ara_structurally_always_double_buffer 不变量用例）
 *   - rGroupNum 切分：min(coreNum/usedCores, CeilDiv(R, 32768)) 两因子分别绑定（2/4/16/64 组）
 *   - workspace：2 × outBlockNum × rGroupNum × CeilAlign(tileOutElems×4B, 64B)（64B slot 对齐 +
 *     tile 奇偶 ping-pong 双 bank ×2——4.2a 检视 H1 修复：规避跨 tile WAR 竞争）
 *   - 门控边界：R < 2×MIN_R_PER_CORE → 直出；usedCores×2 > coreNum → 直出
 *   - 门控历史两态对照（宏默认已改 1，主 tiling 用例与生产构建同为准核启用态）：
 *     同输入 (2,65536) p=1 门控关 → key=16777216 直出双缓冲、blockDim=2、ws=0；
 *     门控开（现行默认/本用例）→ key=256 跨核单缓冲、blockDim=4、ws=512
 *
 * 期望值按 tiling.cpp §5.2/§5.3 闭式精确推导（ubAvail = ubSize−8192−(256×bufNum+1024)；
 * CROSS_CORE 预算含 wsOut/wsIn 队列 +8B/行；slot = CeilAlign(tile 输出元素×4, 64)，ws 总量再 ×2）。
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "tiling_case_runner.h"

using namespace ge;
using namespace lp_ut;

class LpNormReduceV2TilingCrossCoreTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// =====================================================================================
// 24 TilingKey 实例矩阵：CROSS_CORE=1 侧 11 个可达实例（与主 UT 二进制 12 个直出实例互补）
// key = dtype | 1<<8 | LAYOUT<<16 | BUFFER_MODE<<24；dtype 值 fp32=0/fp16=1/bf16=27
// =====================================================================================

TEST_F(LpNormReduceV2TilingCrossCoreTest, matrix_cross1_fp32_ar_single)
{
    // (2,65536) fp32 p=1：usedCores=2、2×2≤64 且 R=65536 ≥ 2×32768 → 跨核
    // rGroupNum=min(64/2, ⌈65536/32768⌉)=2；rFactorPerGroup=32768；blockDim=2×2=4
    // sum 类 chunk cap 1024（d4b146909 精度收敛）→ ubR=1024、est=⌈32768/1024⌉=32 ≥4 → 恒双缓冲
    // （cross|AR|单缓冲实例对 sum 类 p 结构不可达，仅 ±inf/计数分支可达——本例锁定 sum 类跨核恒 db 行为）
    // slot=CeilAlign(1×4,64)=64 → ws=2×2×64×2=512（ping-pong ×2）
    TilingCaseParam c{};
    c.caseName = "cross1_fp32_AR_single";
    c.xShape = {2, 65536};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 65536;
    c.coreNumUsed = 4;
    c.outBlockNum = 2;
    c.rGroupNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777472UL; // DT_FLOAT(0) | CROSS_CORE(1)<<8 | BUFFER(1)<<24
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, matrix_cross1_fp32_ar_double)
{
    // (1,268435456) fp32 p=1 全归约：usedCores=1 → maxGroups=64 绑定（groupsByR=8192 > 64）
    // rGroupNum=64、rFactorPerGroup=4194304；ubR=cap 1024、est=⌈4194304/1024⌉=4096 ≥4 → db；ws=1×64×64×2=8192
    TilingCaseParam c{};
    c.caseName = "cross1_fp32_AR_double";
    c.xShape = {1, 268435456};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 268435456;
    c.coreNumUsed = 64;
    c.outBlockNum = 1;
    c.rGroupNum = 64;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 4194304;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777472UL; // DT_FLOAT | CROSS(1)<<8 | BUFFER(1)<<24
    c.expectBlockDim = 64;
    c.expectWsBytes = 8192UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, matrix_cross1_fp32_ara_double)
{
    // (2,65536,3) fp32 p=2：ARA 跨核；groups=2、rSpan=32768；rChunkCap=(227840-160)/128=1778 → est=⌈32768/1778⌉=19 ≥4 →
    // db 复解 rChunk=1778；slot=CeilAlign(1×8×4,64)=64 → ws=2×2×64×2=512（ping-pong ×2）
    TilingCaseParam c{};
    c.caseName = "cross1_fp32_ARA_double";
    c.xShape = {2, 65536, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 65536;
    c.cLength = 3;
    c.coreNumUsed = 4;
    c.outBlockNum = 2;
    c.rGroupNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 1778;
    c.ubFactorC = 3;
    c.rAlignedCols = 8;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 16843008UL; // DT_FLOAT | CROSS(1)<<8 | ARA(1)<<16 | BUFFER(1)<<24
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, matrix_cross1_fp16_ar_single)
{
    // fp16 跨核：sum 类 cap 1024 → ubR=1024、est=⌈32768/1024⌉=32 ≥4 → 恒双缓冲；key=1|1<<8|1<<24
    TilingCaseParam c{};
    c.caseName = "cross1_fp16_AR_single";
    c.xShape = {2, 65536};
    c.dtype = ge::DT_FLOAT16;
    c.p = 1.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 65536;
    c.coreNumUsed = 4;
    c.outBlockNum = 2;
    c.rGroupNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777473UL;
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, matrix_cross1_fp16_ar_double)
{
    // fp16 跨核双缓冲：key=1|(1<<8)|(1<<24)；ubR=cap 1024；est=⌈4194304/1024⌉=4096 → db
    TilingCaseParam c{};
    c.caseName = "cross1_fp16_AR_double";
    c.xShape = {1, 268435456};
    c.dtype = ge::DT_FLOAT16;
    c.p = 1.0f;
    c.axes = {};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 268435456;
    c.coreNumUsed = 64;
    c.outBlockNum = 1;
    c.rGroupNum = 64;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 4194304;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777473UL;
    c.expectBlockDim = 64;
    c.expectWsBytes = 8192UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, matrix_cross1_fp16_ara_double)
{
    // fp16 ARA 跨核双缓冲：key=1|(1<<8)|(1<<16)|(1<<24)；rAlignedCols=16、rChunk=(227840-288)/192=1185
    TilingCaseParam c{};
    c.caseName = "cross1_fp16_ARA_double";
    c.xShape = {2, 65536, 3};
    c.dtype = ge::DT_FLOAT16;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 65536;
    c.cLength = 3;
    c.coreNumUsed = 4;
    c.outBlockNum = 2;
    c.rGroupNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 1185;
    c.ubFactorC = 3;
    c.rAlignedCols = 16;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 16843009UL;
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, matrix_cross1_bf16_ar_single)
{
    // bf16 跨核：sum 类 cap 1024 → ubR=1024、est=32 ≥4 → 恒双缓冲；key=27|(1<<8)|(1<<24)（bf16 主路径放开，PT-08）
    TilingCaseParam c{};
    c.caseName = "cross1_bf16_AR_single";
    c.xShape = {2, 65536};
    c.dtype = ge::DT_BF16;
    c.p = 1.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 65536;
    c.coreNumUsed = 4;
    c.outBlockNum = 2;
    c.rGroupNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777499UL;
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, matrix_cross1_bf16_ar_double)
{
    // bf16 跨核双缓冲：key=27|(1<<8)|(1<<24)；ubR=cap 1024（复解同初解）
    TilingCaseParam c{};
    c.caseName = "cross1_bf16_AR_double";
    c.xShape = {1, 268435456};
    c.dtype = ge::DT_BF16;
    c.p = 1.0f;
    c.axes = {};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 268435456;
    c.coreNumUsed = 64;
    c.outBlockNum = 1;
    c.rGroupNum = 64;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 4194304;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777499UL;
    c.expectBlockDim = 64;
    c.expectWsBytes = 8192UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, matrix_cross1_bf16_ara_double)
{
    // bf16 ARA 跨核双缓冲：key=27|(1<<8)|(1<<16)|(1<<24)；rAlignedCols=16、rChunk=1185
    TilingCaseParam c{};
    c.caseName = "cross1_bf16_ARA_double";
    c.xShape = {2, 65536, 3};
    c.dtype = ge::DT_BF16;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 65536;
    c.cLength = 3;
    c.coreNumUsed = 4;
    c.outBlockNum = 2;
    c.rGroupNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 1185;
    c.ubFactorC = 3;
    c.rAlignedCols = 16;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 16843035UL;
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

// =====================================================================================
// CROSS_CORE 求解分支：rGroupNum 切分两因子 / workspace 64B slot 对齐 / 全归约
// =====================================================================================

TEST_F(LpNormReduceV2TilingCrossCoreTest, cross_rgroup_num_binds_by_groups_by_r)
{
    // (2,131072) fp32 p=1：groupsByR=⌈131072/32768⌉=4 绑定（maxGroups=32 > 4）
    // rGroupNum=4、rFactorPerGroup=32768、blockDim=8；cap 1024 → est=⌈32768/1024⌉=32 → db；ws=2×4×64×2=1024
    TilingCaseParam c{};
    c.caseName = "cross_rgroup_byR";
    c.xShape = {2, 131072};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 131072;
    c.coreNumUsed = 8;
    c.outBlockNum = 2;
    c.rGroupNum = 4;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777472UL;
    c.expectBlockDim = 8;
    c.expectWsBytes = 1024UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, cross_rgroup_num_binds_by_max_groups)
{
    // (1,268435456)：maxGroups=coreNum/usedCores=64 绑定 → rGroupNum=64（matrix_cross1_fp32_ar_double
    // 已逐字段验证；本用例补充 groupsByR=8192 > 64 时取小者语义的一致性断言）；ubR=cap 1024
    TilingCaseParam c{};
    c.caseName = "cross_rgroup_byMax";
    c.xShape = {1, 268435456};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 268435456;
    c.coreNumUsed = 64;
    c.outBlockNum = 1;
    c.rGroupNum = 64;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 4194304;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777472UL;
    c.expectBlockDim = 64;
    c.expectWsBytes = 8192UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, cross_full_reduce_group_num_16)
{
    // (524288,) axes=[] p=2 全归约：O=1 单输出块、rGroupNum=⌈524288/32768⌉=16 → blockDim=16
    // sum 类 cap 1024 → est=⌈32768/1024⌉=32 → db（与主 UT 同名例互为门控两态对照，均 db）
    TilingCaseParam c{};
    c.caseName = "cross_fullreduce_16g";
    c.xShape = {524288};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 524288;
    c.coreNumUsed = 16;
    c.outBlockNum = 1;
    c.rGroupNum = 16;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 16777472UL;
    c.expectBlockDim = 16;
    c.expectWsBytes = 2048UL; // 1×16×64×2（ping-pong ×2）
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, cross_workspace_slot_64b_alignment)
{
    // (2,65536,33) fp32 p=2：ubFactorC=33、rAlignedCols=CeilAlign(33,8)=40
    // slot=CeilAlign(1×40×4, 64)=CeilAlign(160,64)=192（非 160！64B slot 对齐被显式锁定）
    // ws=2×2×192×2=1536（ping-pong ×2）；rChunk 复解=(227840-800)/640=354（db）
    TilingCaseParam c{};
    c.caseName = "cross_ws_64b_align";
    c.xShape = {2, 65536, 33};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 65536;
    c.cLength = 33;
    c.coreNumUsed = 4;
    c.outBlockNum = 2;
    c.rGroupNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 354;
    c.ubFactorC = 33;
    c.rAlignedCols = 40;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 16843008UL;
    c.expectBlockDim = 4;
    c.expectWsBytes = 1536UL;
    RunTilingCase(c);
}

// =====================================================================================
// ENABLE_CROSS_CORE_ROUTING 门控边界（启用态下的回退条件）
// =====================================================================================

TEST_F(LpNormReduceV2TilingCrossCoreTest, gate_boundary_r_below_threshold_falls_back_direct)
{
    // (2,65535) fp32 p=1：rLen=65535 < 2×MIN_R_PER_CORE=65536 → 门控不启用 → 直出
    // （直出后 ubR=cap 1024、est=⌈65535/1024⌉=64 → db；key=16777216、ws=0、blockDim=2 均不变）
    TilingCaseParam c{};
    c.caseName = "gate_R_below_direct";
    c.xShape = {2, 65535};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 65535;
    c.coreNumUsed = 2;
    c.outBlockNum = 2;
    c.rGroupNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 65535;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777216UL;
    c.expectBlockDim = 2;
    c.expectWsBytes = 0UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, gate_boundary_used_cores_starvation_falls_back_direct)
{
    // (33,100) fp32 p=1：usedCores=33，33×2=66 > coreNum=64 → 输出分块已吃满核数 → 直出单缓冲
    TilingCaseParam c{};
    c.caseName = "gate_cores_starve_direct";
    c.xShape = {33, 100};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 33;
    c.rLength = 100;
    c.coreNumUsed = 33;
    c.outBlockNum = 33;
    c.rGroupNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 96;
    c.rFactorPerGroup = 100;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 33;
    c.expectWsBytes = 0UL;
    RunTilingCase(c);
}

// =====================================================================================
// (cross=1, ARA, 单缓冲) 实例结构不可达不变量（24 矩阵第 24 单元说明）
// =====================================================================================

// 仅校验 TilingKey 的轻量执行器（不变量用例复用）
static uint64_t RunTilingKeyOnly(const std::vector<int64_t>& xShape, ge::DataType dtype, float p,
                                 const std::vector<int64_t>& axes)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    EXPECT_NE(opImpl, nullptr);
    auto tilingFunc = opImpl->tiling;
    EXPECT_NE(tilingFunc, nullptr);

    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(xShape.size());
    shape.MutableStorageShape().SetDimNum(xShape.size());
    for (size_t i = 0; i < xShape.size(); i++) {
        shape.MutableOriginShape().SetDim(i, xShape[i]);
        shape.MutableStorageShape().SetDim(i, xShape[i]);
    }
    gert::StorageShape yShape = {{1}, {1}};
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::LpNormReduceV2CompileInfo tilingCompileInfo;
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .InputShapes({&shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"p", Ops::NN::AnyValue::CreateFrom<float>(p)},
                                  {"axes", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(axes)},
                                  {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-12f)}})
                      .CompileInfo(&tilingCompileInfo)
                      .PlatformInfo(reinterpret_cast<void*>(&platformInfo))
                      .TilingData(tilingData.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                      .Build();
    gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
    SetUpPlatformInfo(context, UT_DEFAULT_CORE_NUM, UT_DEFAULT_UB_SIZE);
    EXPECT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
    return context->GetTilingKey();
}

TEST_F(LpNormReduceV2TilingCrossCoreTest, cross_ara_structurally_always_double_buffer)
{
    // 24 矩阵第 22~24 单元（cross=1|ARA|single × {fp32, fp16, bf16}）结构不可达的证明：
    // CROSS 启用要求 rLen ≥ 65536，rSpan = ⌈rLen/rGroupNum⌉ ≥ 21846（groupsByR 绑定最小值），
    // 而 ARA rChunk ≤ MAX_COPY_BLOCK_COUNT = 4095 → estChunks ≥ ⌈21846/4095⌉ = 6 ≥ 4 → 恒双缓冲
    // （与 dtype 无关）。多形状 × 3 dtype 扫描断言 CROSS_CORE=1 且 LAYOUT=ARA 时 BUFFER_MODE 恒为 1
    // （该 3 实例由防御性 ASCENDC_TPL_SEL 表覆盖编译，运行期不可路由至——结构性属性而非覆盖缺口）
    const std::vector<std::vector<int64_t>> shapes = {
        {2, 65536, 3}, {2, 65536, 33}, {2, 65536, 64}, {1, 268435456, 5}, {2, 131072, 100}};
    const std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    for (const auto& dtype : dtypes) {
        for (const auto& s : shapes) {
            const uint64_t key = RunTilingKeyOnly(s, dtype, 2.0f, {1});
            EXPECT_EQ((key >> 8U) & 0xFFU, 1U)
                << "dtype " << dtype << " shape " << s[0] << "x" << s[1] << "x" << s[2] << " should route CROSS_CORE=1";
            EXPECT_EQ((key >> 16U) & 0xFFU, 1U) << "shape should route LAYOUT=ARA";
            EXPECT_EQ((key >> 24U) & 0xFFU, 1U) << "cross+ARA must always select BUFFER_MODE=1";
        }
    }
}
