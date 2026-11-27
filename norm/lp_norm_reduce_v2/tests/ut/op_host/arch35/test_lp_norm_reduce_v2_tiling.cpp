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
 * \file test_lp_norm_reduce_v2_tiling.cpp
 * \brief LpNormReduceV2 arch35 TilingFunc UT（迭代一核心路径 + 迭代二分档矩阵）
 *
 * 覆盖（DESIGN §5 / spec / PLAN §2.1）：
 *   - AR 主路径 ubR/rowsTile 两步闭式求解验证（fp32/fp16 × 单/双缓冲 × R 大/R 小/大 O 小 R）
 *   - 迭代二分档矩阵：LAYOUT（AR/ARA）、CROSS_CORE（直出/跨核 rGroup 切分/workspace）、
 *     BUFFER_MODE（每核 chunk 数 ≥4 启用，PT-02/PT-11）、bf16 主路径放开（PT-08）
 *   - 迭代二 A2 新增：24 TilingKey 实例选择矩阵（CROSS_CORE=0 12 实例，crosscore 二进制覆盖
 *     CROSS_CORE=1 侧）、ARA 布局求解分支（bTile/ubFactorC/alignedCols/pad 分支路由）、
 *     BUFFER_MODE 判据边界（estChunks=3/4）、BR_ZERO 计数分支预算（+5B/元素）、
 *     tmpBuf/fixedExtra 入账 pin（256×bufNum+1024 与 E×4B 口径由期望值精确锁定）、
 *     PT-15 fp32 计数口径边界（N=2^24+1）、PT-17 rSegNum==0 退化（全归约轴 size=1 消除）
 *   - TilingKey 选择（key = dtype | CROSS_CORE<<8 | LAYOUT<<16 | BUFFER_MODE<<24；
 *     dtype 值 = DT_FLOAT(0)/DT_FLOAT16(1)/DT_BF16(27)）
 *   - TilingData 字段合法性（边界：缓冲不溢出 UB、tile 数为正）
 *   - 空输出 O==0 / R==0 快速路径（§5.6）
 *   - 合轴：size=1 消除、相邻 R 合并、负索引、全归约（§4.1）
 *   - pBranch 六分支 + ±inf 别名 + NaN 拒绝（§1.6）
 *   - 错误分支：dtype/rank/numel/axes 越界重复（正索引与负索引归一化后重复）
 *   - P2 段表路径（迭代三）：非连续多轴段表字段/key/分片 + 中段 R=0 归零路由（§4.3）
 *   - 迭代三 A2 新增：P2 全覆盖（A/R 交替四段满表、8 维混合消除 3 R 段、全 1 维退化、
 *     负索引段表、fp16/bf16 dtype-key、p=+inf 分支组合、oTile=512 上限档位），
 *     段表逐字段断言（segSize/segStride/aSize/aStride 后缀积口径 + 尾部槽位清零）、
 *     TilingData ABI 钉死（sizeof=328 → opParaSize=336）、三条快速路径
 *     DeclareWorkspace 失败传播（无 workspace 注入防御分支）
 *
 * 期望值按 SolveUbTilingAR/ARA（§5.3 BLK-2 闭式）精确推导：
 *   ubAvail = ubSize − 8192 − (256×bufNum + 1024)（fixedExtra 显式入账，回填-8）；
 *   perElemBytes = bufNum×typeSize + 8 + (ZERO 分支 +5)（8 = fp32Buf 4 + tmpBuf E×4B 口径 4）；
 *   AR outPerRowBytes = 8 + typeSize + (CROSS_CORE +8)；ARA outPerBatchBytes = rAlignedCols × 同式
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include "tiling_case_runner.h"

using namespace ge;
using namespace lp_ut;

class LpNormReduceV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// ============================ AR 主路径：两步闭式求解 ============================

TEST_F(LpNormReduceV2TilingTest, ar_main_fp32_double_buffer_rmsnorm_like)
{
    // shape (4096,4096) axes=[1]：O=4096 R=4096，每核 64 行×4096 → 双缓冲
    // sum 类 cap 1024 → ubR=1024；perRowBytes=1024*16+12=16396 → rowsTile=227840/16396=13
    TilingCaseParam c{};
    c.caseName = "ar_fp32_double_4096x4096";
    c.xShape = {4096, 4096};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.coreNum = UT_DEFAULT_CORE_NUM;
    c.ubSize = UT_DEFAULT_UB_SIZE;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 4096;
    c.rLength = 4096;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 64;
    c.rowsTile = 13;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 4096;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.powTimes = 1;
    c.zeroOutputFlag = 0;
    c.expectTilingKey = 16777216UL; // DT_FLOAT(0) | BUFFER_MODE=1 << 24
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ar_main_fp32_single_buffer_small_shape)
{
    // shape (7,37) axes=[1] p=1：每核 1 行×37 → 单缓冲；ubR=FloorAlign(37,8)=32
    TilingCaseParam c{};
    c.caseName = "ar_fp32_single_7x37";
    c.xShape = {7, 37};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 7;
    c.rLength = 37;
    c.coreNumUsed = 7;
    c.outBlockNum = 7;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 32;
    c.rFactorPerGroup = 37;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 0UL; // DT_FLOAT(0) | BUFFER_MODE=0
    c.expectBlockDim = 7;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ar_main_fp32_large_o_small_r_output_tile_loop)
{
    // shape (200000,100) axes=[1]：O=2M、R=100（BLK-1 大 O 小 R 场景）
    // bFactor=CeilDiv(200000,64)=3125；ubR=FloorAlign(100,8)=96（R=100 < cap 不受影响）；perRowBytes=96*16+12=1548
    // rowsTile=227840/1548=147 → outTileNum=CeilDiv(3125,147)=22（逐 tile 回写）
    TilingCaseParam c{};
    c.caseName = "ar_fp32_largeO_smallR";
    c.xShape = {200000, 100};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 200000;
    c.rLength = 100;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.bFactor = 3125;
    c.rowsTile = 147;
    c.ubFactorR = 96;
    c.rFactorPerGroup = 100;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 16777216UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ar_main_fp16_big_r_single_row_deep_reduce)
{
    // shape (2,1000000) axes=[1] fp16：R=1e6 单行深度归约；跨核路由已恢复默认开启
    // （issue_20261104_ge_ws_write_fault.md 修复，kernel 经 GetSysWorkSpacePtr 取真实 ws 基址）
    // → rGroup=min(64/2,⌈1e6/32768⌉)=31；usedCores=2×31=62；rPerGroup=⌈1e6/31⌉=32259
    // sum 类 cap 1024 → ubR=1024、est=⌈32259/1024⌉=32 → 恒双缓冲；ws=2×31×64×2=7936（ping-pong ×2，H1）
    TilingCaseParam c{};
    c.caseName = "ar_fp16_bigR_direct_db";
    c.xShape = {2, 1000000};
    c.dtype = ge::DT_FLOAT16;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 1000000;
    c.coreNumUsed = 62;
    c.outBlockNum = 2;
    c.rGroupNum = 31;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 32259;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 16777473UL; // DT_FLOAT16(1) | CROSS_CORE=1<<8 | AR | BUFFER_MODE=1
    c.expectBlockDim = 62;
    c.expectWsBytes = 7936UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ar_main_fp16_small_single_buffer_key)
{
    // fp16 单缓冲：key = DT_FLOAT16(1) | 0
    TilingCaseParam c{};
    c.caseName = "ar_fp16_single_2x3";
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT16;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 3;
    c.coreNumUsed = 2;
    c.outBlockNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 3; // rLength < ubBlockElems(16)：chunk 即整行
    c.rFactorPerGroup = 3;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 1UL;
    c.expectBlockDim = 2;
    RunTilingCase(c);
}

// ============================ 快速路径（§5.6） ============================

TEST_F(LpNormReduceV2TilingTest, rank_zero_scalar_degenerates_to_1x1)
{
    // rank=0 → EnsureNotScalar {1}：全维消除 → rSegNum=0 退化 (B=1,R=1,C=1) 单核直出
    TilingCaseParam c{};
    c.caseName = "rank0_scalar";
    c.xShape = {};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 1;
    c.coreNumUsed = 1;
    c.outBlockNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1;
    c.rFactorPerGroup = 1;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 0; // 全 A 退化（无 R 段）
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 1;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, empty_output_fast_path_o_zero)
{
    // spec boundary: x=[0,4] axes=[1] → O==0 空输出，blockDim=1 空跑
    TilingCaseParam c{};
    c.caseName = "empty_output_O0";
    c.xShape = {0, 4};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 0;
    c.rLength = 4;
    c.coreNumUsed = 1;
    c.outBlockNum = 1;
    c.bFactor = 0;
    c.rowsTile = 1;
    c.ubFactorR = 1;
    c.rFactorPerGroup = 0;
    c.zeroOutputFlag = 0;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 1;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, empty_full_reduce_fast_path_r_zero)
{
    // spec boundary: x=[0,4] axes=[] → R==0 且 O>0 → zeroOutputFlag=1 输出全 0
    TilingCaseParam c{};
    c.caseName = "empty_full_reduce_R0";
    c.xShape = {0, 4};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 0;
    c.coreNumUsed = 1;
    c.outBlockNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1;
    c.rFactorPerGroup = 0;
    c.zeroOutputFlag = 1;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 1;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, empty_output_fp16_tiling_key)
{
    // O==0 快速路径 fp16：key = DT_FLOAT16(1)
    TilingCaseParam c{};
    c.caseName = "empty_output_fp16";
    c.xShape = {0, 4};
    c.dtype = ge::DT_FLOAT16;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 0;
    c.rLength = 4;
    c.coreNumUsed = 1;
    c.outBlockNum = 1;
    c.bFactor = 0;
    c.rowsTile = 1;
    c.ubFactorR = 1;
    c.rFactorPerGroup = 0;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 1UL;
    c.expectBlockDim = 1;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, empty_output_bf16_tiling_key)
{
    // O==0 快速路径 bf16：key = DT_BF16(27)（迭代二 bf16 主路径已放开，快速路径仍走 cross=0/AR/单缓冲最小实例）
    TilingCaseParam c{};
    c.caseName = "empty_output_bf16";
    c.xShape = {0, 4};
    c.dtype = ge::DT_BF16;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 0;
    c.rLength = 4;
    c.coreNumUsed = 1;
    c.outBlockNum = 1;
    c.bFactor = 0;
    c.rowsTile = 1;
    c.ubFactorR = 1;
    c.rFactorPerGroup = 0;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 27UL;
    c.expectBlockDim = 1;
    RunTilingCase(c);
}

// ============================ 合轴（§4.1：负索引 / size=1 消除 / 相邻合并 / 全归约） ============================

TEST_F(LpNormReduceV2TilingTest, merge_axes_negative_index)
{
    // axes=[-1] → 归一化为 1：b=4, r=8
    TilingCaseParam c{};
    c.caseName = "merge_negative_index";
    c.xShape = {4, 8};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {-1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 4;
    c.rLength = 8;
    c.coreNumUsed = 4;
    c.outBlockNum = 4;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 8;
    c.rFactorPerGroup = 8;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 4;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, merge_axes_trailing_size_one_dims)
{
    // {2,3,1,1} axes=[1]：尾随 size=1 维消除 → b=2, r=3
    TilingCaseParam c{};
    c.caseName = "merge_trailing_ones";
    c.xShape = {2, 3, 1, 1};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 3;
    c.coreNumUsed = 2;
    c.outBlockNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 3;
    c.rFactorPerGroup = 3;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 2;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, merge_axes_reduce_dim_size_one_eliminated)
{
    // {2,1,3} axes=[1]：归约轴 size=1 消除 → rSegNum=0 退化 (B=6, R=1, C=1)（常量归约）
    TilingCaseParam c{};
    c.caseName = "merge_reduce_dim_one";
    c.xShape = {2, 1, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 6;
    c.rLength = 1;
    c.coreNumUsed = 6;
    c.outBlockNum = 6;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1;
    c.rFactorPerGroup = 1;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 0; // 全 A 退化（R 段全被 size=1 消除）
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 6;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, merge_axes_adjacent_reduce_merge)
{
    // {2,3,4} axes=[1,2]：相邻 R 维合并 → b=2, r=12；ubR=FloorAlign(12,8)=8
    TilingCaseParam c{};
    c.caseName = "merge_adjacent_R";
    c.xShape = {2, 3, 4};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1, 2};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 12;
    c.coreNumUsed = 2;
    c.outBlockNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 8;
    c.rFactorPerGroup = 12;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 2;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, merge_axes_full_reduce)
{
    // {2,3} axes=[]：全归约 → b=1, r=6（O=1 单核）
    TilingCaseParam c{};
    c.caseName = "merge_full_reduce";
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 6;
    c.coreNumUsed = 1;
    c.outBlockNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 6;
    c.rFactorPerGroup = 6;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 1;
    RunTilingCase(c);
}

// ============================ pBranch 判定（§1.6 判定优先级） ============================

static void RunPBranchCase(const std::string& name, float p, int32_t expectBranch, int32_t expectPowTimes,
                           ge::graphStatus expectStatus = ge::GRAPH_SUCCESS)
{
    TilingCaseParam c{};
    c.caseName = name;
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = p;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = expectStatus;
    if (expectStatus == ge::GRAPH_SUCCESS) {
        // {2,3} fp32 p 无关的公共期望值
        c.bLength = 2;
        c.rLength = 3;
        c.coreNumUsed = 2;
        c.outBlockNum = 2;
        c.bFactor = 1;
        c.rowsTile = 1;
        c.ubFactorR = 3;
        c.rFactorPerGroup = 3;
        c.expectTilingKey = 0UL;
        c.expectBlockDim = 2;
    }
    c.pBranch = expectBranch;
    c.powTimes = expectPowTimes;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, p_branch_pos_inf)
{
    RunPBranchCase("p_pos_inf", std::numeric_limits<float>::infinity(), LPNORM_P_BRANCH_POS_INF, 1);
}

TEST_F(LpNormReduceV2TilingTest, p_branch_pos_inf_alias)
{
    // spec 别名 2147483647.0（fp32 属性存储舍入为 2^31）→ +inf 分支
    RunPBranchCase("p_pos_inf_alias", 2147483647.0f, LPNORM_P_BRANCH_POS_INF, 1);
}

TEST_F(LpNormReduceV2TilingTest, p_branch_neg_inf)
{
    RunPBranchCase("p_neg_inf", -std::numeric_limits<float>::infinity(), LPNORM_P_BRANCH_NEG_INF, 1);
}

TEST_F(LpNormReduceV2TilingTest, p_branch_neg_inf_alias)
{
    RunPBranchCase("p_neg_inf_alias", -2147483648.0f, LPNORM_P_BRANCH_NEG_INF, 1);
}

TEST_F(LpNormReduceV2TilingTest, p_branch_zero_count) { RunPBranchCase("p_zero", 0.0f, LPNORM_P_BRANCH_ZERO, 1); }

TEST_F(LpNormReduceV2TilingTest, p_branch_one_abs_sum) { RunPBranchCase("p_one", 1.0f, LPNORM_P_BRANCH_ONE, 1); }

TEST_F(LpNormReduceV2TilingTest, p_branch_int_pow_lower_bound)
{
    RunPBranchCase("p_two", 2.0f, LPNORM_P_BRANCH_INT_POW, 1);
}

TEST_F(LpNormReduceV2TilingTest, p_branch_int_pow_upper_bound)
{
    // p=15（连乘上界）：powTimes = 14
    RunPBranchCase("p_fifteen", 15.0f, LPNORM_P_BRANCH_INT_POW, 14);
}

TEST_F(LpNormReduceV2TilingTest, p_branch_int_pow_above_bound_to_exp)
{
    // p=16 > 15 → EXP 路径
    RunPBranchCase("p_sixteen", 16.0f, LPNORM_P_BRANCH_EXP, 1);
}

TEST_F(LpNormReduceV2TilingTest, p_branch_float_exp)
{
    // spec boundary: p=0.5 → exp(p·ln|x|) 路径
    RunPBranchCase("p_half", 0.5f, LPNORM_P_BRANCH_EXP, 1);
}

TEST_F(LpNormReduceV2TilingTest, p_branch_negative_float_exp)
{
    // spec extreme: p=-1 → EXP 路径（含零输入时 exp(+inf) 极限语义）
    RunPBranchCase("p_neg_one", -1.0f, LPNORM_P_BRANCH_EXP, 1);
}

// 黑盒路由 lpnorm_l2_017: p=NaN 拒收（op_host 原型校验 attribute_value_out_of_range）→ 本测试用例
// LpNormReduceV2TilingTest.p_branch_nan_rejected（real 链路编译期拒收被隔离 OPP IGNORE_INFER_ERROR=1
// 抑制不可达，拒收语义由本 UT 直测，st_real_result.json 路由 ut_host）
TEST_F(LpNormReduceV2TilingTest, p_branch_nan_rejected)
{
    // p 为 NaN → attribute_value_out_of_range（DESIGN §8）
    RunPBranchCase("p_nan", std::numeric_limits<float>::quiet_NaN(), LPNORM_P_BRANCH_EXP, 1, ge::GRAPH_FAILED);
}

// ============================ 错误分支（§8 错误处理） ============================

// 黑盒路由 lpnorm_l2_018: x dtype=int32 拒收（InferDataType 仅支持 FLOAT/FLOAT16/BFLOAT16）→
// 本测试用例 LpNormReduceV2TilingTest.dtype_not_supported（ut_host 路由依据，同上）
TEST_F(LpNormReduceV2TilingTest, dtype_not_supported)
{
    TilingCaseParam c{};
    c.caseName = "dtype_int32_rejected";
    c.xShape = {2, 3};
    c.dtype = ge::DT_INT32;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, bf16_main_path_enabled_in_iteration2)
{
    // 迭代二放开 bf16 主路径（PT-08：CAST_RINT RN-even 回写验证通过）；路由同 fp32/fp16 AR 直出
    // shape (7,37) axes=[1] p=2：ubR=FloorAlign(37,16)=32、每核 1 行 → est=1×2=2 chunk → 单缓冲
    TilingCaseParam c{};
    c.caseName = "bf16_main_enabled";
    c.xShape = {7, 37};
    c.dtype = ge::DT_BF16;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 7;
    c.rLength = 37;
    c.coreNumUsed = 7;
    c.outBlockNum = 7;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 32;
    c.rFactorPerGroup = 37;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 27UL; // DT_BF16(27) | CROSS_CORE=0 | LAYOUT=AR | BUFFER_MODE=0
    c.expectBlockDim = 7;
    RunTilingCase(c);
}

// 黑盒路由 lpnorm_l2_019: rank=9>8 拒收（InferShape 校验，tiling 侧同分支）→
// 本测试用例 LpNormReduceV2TilingTest.rank_exceed_eight（ut_host 路由依据，同上）
TEST_F(LpNormReduceV2TilingTest, rank_exceed_eight)
{
    TilingCaseParam c{};
    c.caseName = "rank9_rejected";
    c.xShape = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, numel_exceed_limit)
{
    // numel(x) ≤ 2^31-1（spec global_constraints）
    TilingCaseParam c{};
    c.caseName = "numel_overflow_rejected";
    c.xShape = {1073741825, 2}; // numel = 2147483650 > 2147483647
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

// 黑盒路由 lpnorm_l2_015: axes 越界拒收（tiling 侧同分支）→
// 本测试用例 LpNormReduceV2TilingTest.axes_out_of_range（ut_host 路由依据，同上）
TEST_F(LpNormReduceV2TilingTest, axes_out_of_range)
{
    TilingCaseParam c{};
    c.caseName = "axes_out_of_range";
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {2};
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, axes_negative_out_of_range)
{
    TilingCaseParam c{};
    c.caseName = "axes_neg_out_of_range";
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {-3};
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

// 黑盒路由 lpnorm_l2_016: axes 元素重复拒收（tiling 侧同分支）→
// 本测试用例 LpNormReduceV2TilingTest.axes_duplicate（ut_host 路由依据，同上）
TEST_F(LpNormReduceV2TilingTest, axes_duplicate)
{
    TilingCaseParam c{};
    c.caseName = "axes_duplicate";
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0, 0};
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, axes_size_exceed_rank)
{
    TilingCaseParam c{};
    c.caseName = "axes_size_exceed_rank";
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0, 1, 2};
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, axes_negative_duplicate_after_normalize)
{
    // 负索引归一化后重复：[-2, 0] → [0, 0]（spec machine_constraint 元素不得重复，
    // 归一化后集合判定，BuildMergedSegments axesSet 分支）
    TilingCaseParam c{};
    c.caseName = "axes_neg_dup";
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {-2, 0};
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, declare_workspace_null_rejected_o_zero_fast_path)
{
    // O==0 快速路径 workspace 声明失败传播：不注入 workspace → GetWorkspaceSizes 返回空
    // → DeclareWorkspace OP_CHECK_NULL 防御分支（internal_error 口径，DESIGN §8）
    TilingCaseParam c{};
    c.caseName = "ws_null_o_zero";
    c.xShape = {0, 4};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_FAILED;
    c.withWorkspace = false;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, declare_workspace_null_rejected_r_zero_fast_path)
{
    // R==0 置零路径 workspace 声明失败传播：(2,0,3) axes=[1] → rTotal=0、oTotal=6
    TilingCaseParam c{};
    c.caseName = "ws_null_r_zero";
    c.xShape = {2, 0, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_FAILED;
    c.withWorkspace = false;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, declare_workspace_null_rejected_p2_segment_path)
{
    // P2 段表路径 workspace 声明失败传播：rSegNum=2 走 P2 分档后 DeclareWorkspace 防御分支
    TilingCaseParam c{};
    c.caseName = "ws_null_p2";
    c.xShape = {4, 32, 16, 70};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0, 2};
    c.expectStatus = ge::GRAPH_FAILED;
    c.withWorkspace = false;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ara_layout_direct_mode_routing)
{
    // 迭代二 ARA 路径放开（PT-09：Pattern::Reduce::RA + 分支 pad 值 + bTile×ubFactorC tile 循环）
    // shape (8,64,70) axes=[1] → 合轴 (B=8, R=64, C=70)：usedCores=8、R=64 < 2×32768 → 直出
    // ubFactorC=min(64,70)=64、rAlignedCols=64；rChunk=min(64, cap)=64（单 chunk）；
    // est = 1 批tile × 2 列tile × 1 chunk = 2 <4 → 单缓冲
    TilingCaseParam c{};
    c.caseName = "ara_direct_8x64x70";
    c.xShape = {8, 64, 70};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 8;
    c.rLength = 64;
    c.cLength = 70;
    c.coreNumUsed = 8;
    c.outBlockNum = 8;
    c.rGroupNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 64; // rChunk（RA 归约行数）
    c.ubFactorC = 64;
    c.rAlignedCols = 64;
    c.rFactorPerGroup = 64;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65536UL; // DT_FLOAT(0) | CROSS_CORE=0 | LAYOUT=ARA(1)<<16 | BUFFER_MODE=0
    c.expectBlockDim = 8;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ara_layout_cross_core_routing)
{
    // shape (2,65536,3) axes=[1] → 合轴 (B=2, R=65536, C=3)：ARA 布局分档判定矩阵用例
    // （跨核路由已恢复默认开启 → 真跨核；groups=min(32,⌈65536/32768⌉)=2、usedCores=2×2=4）
    // rChunkCap=(227840-160)/128=1778 → est=⌈32768/1778⌉=19 ≥4 → db；rChunk 复解=1778；ws=2×2×64×2=512（ping-pong
    // ×2，H1）
    TilingCaseParam c{};
    c.caseName = "ara_2x65536x3_gated_direct";
    c.xShape = {2, 65536, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.keepdim = false;
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

TEST_F(LpNormReduceV2TilingTest, ar_cross_core_small_o_big_r_routing)
{
    // shape (2,65536) axes=[1] fp32：小 O 大 R 分档判定矩阵用例（跨核路由已恢复默认开启）
    // groups=min(32,⌈65536/32768⌉)=2、usedCores=2×2=4；rPerGroup=32768；sum 类 cap 1024 → ubR=1024、est=⌈32768/1024⌉=32
    // → 恒双缓冲
    // （cross|AR|single 对 sum 类结构不可达）；ws=2×2×64×2=512（ping-pong ×2，H1）
    TilingCaseParam c{};
    c.caseName = "ar_smallo_bigr_gated_direct";
    c.xShape = {2, 65536};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.keepdim = false;
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
    c.expectTilingKey = 16777472UL; // DT_FLOAT(0) | CROSS_CORE=1<<8 | AR | BUFFER_MODE=1
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ar_cross_core_full_reduce_routing)
{
    // shape (524288,) axes=[] 全归约（跨核路由已恢复默认开启 → 真跨核）
    // groups=min(64,⌈524288/32768⌉)=16、usedCores=1×16=16；rPerGroup=32768
    // sum 类 cap 1024 → ubR=1024、est=⌈32768/1024⌉=32 → 恒双缓冲；ws=1×16×64×2=2048（ping-pong ×2，H1）
    TilingCaseParam c{};
    c.caseName = "ar_fullreduce_gated_direct";
    c.xShape = {524288};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {};
    c.keepdim = false;
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
    c.expectTilingKey = 16777472UL; // DT_FLOAT | CROSS_CORE=1<<8 | AR | BUFFER_MODE=1
    c.expectBlockDim = 16;
    c.expectWsBytes = 2048UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, multi_r_segment_p2_tiling_table)
{
    // 非连续多轴（axes=[0,2] → 两段 R）P2 段表路径（迭代三，DESIGN §4.3/PT-13）：
    // 段序列 [R:4][A:32][R:16][A:70]：O=2240、rTotal=64、rStride={35840,70}、aStride={1120,1}
    // key=(fp32,0,0,0)；outBlockNum=min(2240,64)=64、bFactor=35、chunk=64（<8192 单 chunk）、oTile=35
    TilingCaseParam c{};
    c.caseName = "multi_seg_p2_table";
    c.xShape = {4, 32, 16, 70};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0, 2};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2240;
    c.rLength = 64;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 35;
    c.rowsTile = 35;
    c.bTile = 1;
    c.ubFactorR = 64;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 64;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 2;
    c.aSegNum = 2;
    c.expectSegSize = {4, 16};
    c.expectSegStride = {35840, 70};
    c.expectASize = {32, 70};
    c.expectAStride = {1120, 1};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, multi_r_segment_p2_pt13_seg13_shape)
{
    // PT-13 seg13 同款：(4,64,32,16) axes=[1,3] → [A:4][R:64][A:32][R:16]：
    // O=128、rTotal=1024、rStride={512,1}、aStride={32768,16}；chunk=1024、oTile=min(512,2)=2
    TilingCaseParam c{};
    c.caseName = "p2_pt13_seg13";
    c.xShape = {4, 64, 32, 16};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1, 3};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 128;
    c.rLength = 1024;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 2;
    c.rowsTile = 2;
    c.bTile = 1;
    c.ubFactorR = 1024;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 1024;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 2;
    c.aSegNum = 2;
    c.expectSegSize = {64, 16};
    c.expectSegStride = {512, 1};
    c.expectASize = {4, 32};
    c.expectAStride = {32768, 16};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, multi_r_segment_p2_pt13_seg03_a_merge)
{
    // PT-13 seg03 同款：(4,64,32,16) axes=[0,3] → [R:4][A:2048][R:16]（中间双 A 段合并）：
    // O=2048、rTotal=64、aSegNum=1（相邻 A 合并验证）；bFactor=CeilDiv(2048,64)=32、oTile=32
    TilingCaseParam c{};
    c.caseName = "p2_pt13_seg03";
    c.xShape = {4, 64, 32, 16};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0, 3};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2048;
    c.rLength = 64;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 32;
    c.rowsTile = 32;
    c.bTile = 1;
    c.ubFactorR = 64;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 64;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 2;
    c.aSegNum = 1;
    c.expectSegSize = {4, 16};
    c.expectSegStride = {32768, 1};
    c.expectASize = {2048};
    c.expectAStride = {16};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, multi_r_segment_p2_big_rtotal_chunked)
{
    // rTotal > chunk 上限（8192）分片：shape (4,4096,3,128) axes=[1,3] → [A:4][R:4096][A:3][R:128]
    // O=12、rTotal=524288、chunk=8192（rFactorPerGroup=rTotal）；bFactor=1、oTile=1
    TilingCaseParam c{};
    c.caseName = "p2_big_rtotal_chunked";
    c.xShape = {4, 4096, 3, 128};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1, 3};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 12;
    c.rLength = 524288;
    c.cLength = 1;
    c.coreNumUsed = 12;
    c.outBlockNum = 12;
    c.rGroupNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 8192;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 524288;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 2;
    c.aSegNum = 2;
    c.expectSegSize = {4096, 128};
    c.expectSegStride = {384, 1};
    c.expectASize = {4, 3};
    c.expectAStride = {1572864, 128};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 12;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, multi_r_segment_p2_middle_zero_routed_zero_output)
{
    // P2 + 中段 R size=0：(4,64,0,16) axes=[0,2] → [R:4][A:64][R:0][A:16]，rTotal=0
    // → R==0 置零路径（O=oTotal=1024 展平 bLength，zeroOutputFlag=1，segNum 归一 1）
    TilingCaseParam c{};
    c.caseName = "p2_zero_rseg_zero_output";
    c.xShape = {4, 64, 0, 16};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0, 2};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1024;
    c.rLength = 0;
    c.cLength = 1;
    c.coreNumUsed = 1;
    c.outBlockNum = 1;
    c.rGroupNum = 1;
    c.bFactor = 1024;
    c.rowsTile = 1024; // SolveUbTilingAR(253952,1,4,8,1,1024)：perRowBytes=8*12+12=108 → cap 2263 → min(bFactor)=1024
    c.bTile = 1;
    c.ubFactorR = 1;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 0;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.zeroOutputFlag = 1;
    c.segNum = 1;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 1;
    RunTilingCase(c);
}

// =====================================================================================
// 迭代三 A2：P2 分支全覆盖（BuildMergedSegments 段表生成：A/R 交替四段 / 8 维混合消除 /
// 全 1 维退化 / 负索引 / dtype-key 矩阵 / p 分支路由 / oTile 上限档位）
// 期望段表全部按 DESIGN §4.1 消除→合并→后缀积口径手工推导，逐字段断言（含尾部清零）
// =====================================================================================

TEST_F(LpNormReduceV2TilingTest, p2_alternating_four_segments_rank8_max_table)
{
    // rank=8 无 size=1 维、axes=[1,3,5,7] → [A:2][R:3][A:4][R:5][A:6][R:7][A:8][R:9]
    // 段数上限形态：rSegNum=4、aSegNum=4（rank≤8 交替段上界，PT-13 §3.2.1 满表）
    // O=2×4×6×8=384、rTotal=3×5×7×9=945；stride=段后全积：R={60480,3024,72,1}、A={181440,15120,504,9}
    // chunk=min(945,8192)=945 → ubFactorR=CeilAlign(945,8)=952（尾部 pad 对齐档位）；oTile=min(512,6)=6
    TilingCaseParam c{};
    c.caseName = "p2_alt4seg_rank8";
    c.xShape = {2, 3, 4, 5, 6, 7, 8, 9};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1, 3, 5, 7};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 384;
    c.rLength = 945;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 6;
    c.rowsTile = 6;
    c.bTile = 1;
    c.ubFactorR = 952;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 945;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 4;
    c.aSegNum = 4;
    c.expectSegSize = {3, 5, 7, 9};
    c.expectSegStride = {60480, 3024, 72, 1};
    c.expectASize = {2, 4, 6, 8};
    c.expectAStride = {181440, 15120, 504, 9};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, p2_rank8_mixed_elimination_three_r_segments)
{
    // 8 维混合消除：(2,3,4,1,5,6,7,8) axes=[0,1,4,6]：dim0/dim1 同 R 合并、dim3 size=1 消除
    // → [R:6][A:4][R:5][A:6][R:7][A:8]：3 R 段 × 3 A 段（消除与合并不改变 R-A 交替语义）
    // O=4×6×8=192、rTotal=6×5×7=210；chunk=210 → ubFactorR=CeilAlign(210,8)=216
    TilingCaseParam c{};
    c.caseName = "p2_8d_elim_3rseg";
    c.xShape = {2, 3, 4, 1, 5, 6, 7, 8};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0, 1, 4, 6};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 192;
    c.rLength = 210;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 3;
    c.rowsTile = 3;
    c.bTile = 1;
    c.ubFactorR = 216;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 210;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 3;
    c.aSegNum = 3;
    c.expectSegSize = {6, 5, 7};
    c.expectSegStride = {6720, 336, 8};
    c.expectASize = {4, 6, 8};
    c.expectAStride = {1680, 56, 1};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, p2_all_ones_dims_full_elimination_degenerate)
{
    // 全 1 维：(1,1,1,1) axes=[1,2] → 所有维 size=1 消除 → segCnt=0 → rSegNum==0 退化
    // （PT-17 §2.3.1 同条款、O=1 特例：B=O=1、R=1 恒等归约走 AR 单核；segNum=0 真实段数下发）
    TilingCaseParam c{};
    c.caseName = "p2_all_ones_degenerate";
    c.xShape = {1, 1, 1, 1};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1, 2};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 1;
    c.cLength = 1;
    c.coreNumUsed = 1;
    c.outBlockNum = 1;
    c.rGroupNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 1;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 1;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 0;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 1;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, p2_negative_index_axes_normalized_to_segment_table)
{
    // 负索引 P2：(2,5,3,7,4) axes=[-5,-3] → 归一化 [0,2]；7×4 相邻 A 合并 → [R:2][A:5][R:3][A:28]
    // O=140 → outBlockNum=64、bFactor=CeilDiv(140,64)=3、usedCores=47（输出块不整除分档）
    // chunk=min(6,8192)=6 → ubFactorR=CeilAlign(6,8)=8（跨 block 对齐 pad 档位）
    TilingCaseParam c{};
    c.caseName = "p2_neg_index_segtable";
    c.xShape = {2, 5, 3, 7, 4};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {-5, -3};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 140;
    c.rLength = 6;
    c.cLength = 1;
    c.coreNumUsed = 47;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 3;
    c.rowsTile = 3;
    c.bTile = 1;
    c.ubFactorR = 8;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 6;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 2;
    c.aSegNum = 2;
    c.expectSegSize = {2, 3};
    c.expectSegStride = {420, 28};
    c.expectASize = {5, 28};
    c.expectAStride = {84, 1};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 47;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, p2_fp16_dtype_tiling_key_routes_seg_table)
{
    // P2 dtype-key 矩阵 fp16：key=DT_FLOAT16(1)|(0,0,0)；ubBlockElems=16 → chunk 对齐 CeilAlign(64,16)=64
    TilingCaseParam c{};
    c.caseName = "p2_fp16_key";
    c.xShape = {4, 32, 16, 70};
    c.dtype = ge::DT_FLOAT16;
    c.p = 2.0f;
    c.axes = {0, 2};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2240;
    c.rLength = 64;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 35;
    c.rowsTile = 35;
    c.bTile = 1;
    c.ubFactorR = 64;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 64;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 2;
    c.aSegNum = 2;
    c.expectSegSize = {4, 16};
    c.expectSegStride = {35840, 70};
    c.expectASize = {32, 70};
    c.expectAStride = {1120, 1};
    c.expectTilingKey = 1UL; // DT_FLOAT16(1) | CROSS_CORE=0 | LAYOUT=0 | BUFFER_MODE=0
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, p2_bf16_dtype_tiling_key_routes_seg_table)
{
    // P2 dtype-key 矩阵 bf16：key=DT_BF16(27)；段表同 fp32 同形（dtype 不影响段表几何）
    TilingCaseParam c{};
    c.caseName = "p2_bf16_key";
    c.xShape = {4, 32, 16, 70};
    c.dtype = ge::DT_BF16;
    c.p = 2.0f;
    c.axes = {0, 2};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2240;
    c.rLength = 64;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 35;
    c.rowsTile = 35;
    c.bTile = 1;
    c.ubFactorR = 64;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 64;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 2;
    c.aSegNum = 2;
    c.expectSegSize = {4, 16};
    c.expectSegStride = {35840, 70};
    c.expectASize = {32, 70};
    c.expectAStride = {1120, 1};
    c.expectTilingKey = 27UL; // DT_BF16(27)
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, p2_pos_inf_branch_routed_through_seg_table)
{
    // P2 × p 分支组合：p=+inf → pBranch=POS_INF（max|x| 合并）；pVal=+inf 运行期下发
    // （P2 与 pBranch 正交：段表几何不受 p 影响，kernel 侧按 pBranch 选 Add/Max 合并）
    TilingCaseParam c{};
    c.caseName = "p2_posinf_branch";
    c.xShape = {4, 64, 32, 16};
    c.dtype = ge::DT_FLOAT;
    c.p = std::numeric_limits<float>::infinity();
    c.axes = {1, 3};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 128;
    c.rLength = 1024;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 2;
    c.rowsTile = 2;
    c.bTile = 1;
    c.ubFactorR = 1024;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 1024;
    c.pBranch = LPNORM_P_BRANCH_POS_INF;
    c.segNum = 2;
    c.aSegNum = 2;
    c.expectSegSize = {64, 16};
    c.expectSegStride = {512, 1};
    c.expectASize = {4, 32};
    c.expectAStride = {32768, 16};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, p2_large_output_o_tile_capped_to_512)
{
    // P2 oTile 上限档位：O=131072 → bFactor=CeilDiv(131072,64)=2048 > 512 → oTile 钳位 512
    // （rowsTile=512 回写批量上限，DESIGN §4.3 P2_O_TILE_CAP；chunk=min(15,8192)=15 → 16 对齐）
    TilingCaseParam c{};
    c.caseName = "p2_ouput_tile_512cap";
    c.xShape = {2, 3, 65536, 5};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1, 3};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 131072;
    c.rLength = 15;
    c.cLength = 1;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.rGroupNum = 1;
    c.bFactor = 2048;
    c.rowsTile = 512;
    c.bTile = 1;
    c.ubFactorR = 16;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 15;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 2;
    c.aSegNum = 2;
    c.expectSegSize = {3, 5};
    c.expectSegStride = {327680, 1};
    c.expectASize = {2, 65536};
    c.expectAStride = {983040, 5};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, platform_core_num_zero_internal_error)
{
    // coreNum==0 → internal_error（DESIGN §8）
    TilingCaseParam c{};
    c.caseName = "coreNum_zero";
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.coreNum = 0;
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, platform_ub_size_zero_internal_error)
{
    // ubSize==0 → internal_error（DESIGN §8）
    TilingCaseParam c{};
    c.caseName = "ubSize_zero";
    c.xShape = {2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.ubSize = 0;
    c.expectStatus = ge::GRAPH_FAILED;
    RunTilingCase(c);
}

// ============================ 注册冒烟（P1：def 注册与属性） ============================

TEST_F(LpNormReduceV2TilingTest, tiling_data_abi_pins_op_para_size_336)
{
    // TilingData ABI 钉死（迭代三 v1.3 全字段）：sizeof=328 → 已安装 kernel 二进制 JSON 元数据
    // opParaSize = sizeof + 8（ContinuousVector 头） = 336（build_out/custom_opp .../LpNormReduceV2_*.json
    // 实测值一致）；字段偏移锁定防止插队导致 FE 下发偏移漂移（aSegNum/aSize[4]/aStride[4] 为
    // 迭代三 P2 新增字段，op_host memset 全量下发与 kernel GET_TILING_DATA_WITH_STRUCT 读取强依赖此布局）
    EXPECT_EQ(sizeof(LpNormReduceV2TilingData), 328U);
    EXPECT_EQ(offsetof(LpNormReduceV2TilingData, segSize), 128U);
    EXPECT_EQ(offsetof(LpNormReduceV2TilingData, segStride), 192U);
    EXPECT_EQ(offsetof(LpNormReduceV2TilingData, aSegNum), 256U);
    EXPECT_EQ(offsetof(LpNormReduceV2TilingData, aSize), 264U);
    EXPECT_EQ(offsetof(LpNormReduceV2TilingData, aStride), 296U);
    // opParaSize 口径：sizeof(TilingData) + 8B ContinuousVector 头（capacity+size）
    EXPECT_EQ(sizeof(LpNormReduceV2TilingData) + 8U, 336U);
}

TEST_F(LpNormReduceV2TilingTest, op_impl_registry_registration_smoke)
{
    // IMPL_OP_OPTILING / IMPL_OP_INFERSHAPE 注册完整性（含 tiling_parse 占位）。
    // 注：infer_datatype 按 nn 仓拆分口径由 op_graph/lp_norm_reduce_v2_graph_infer.cpp 的
    // IMPL_OP 注册链承载，其注册与推导由 tests/ut/op_graph（共享 op_graph UT runner，
    // 链接 GRAPH_PLUGIN 含 graph_infer.cpp）验证；op_host UT runner 不链接该翻译单元。
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    ASSERT_NE(opImpl, nullptr);
    EXPECT_NE(opImpl->infer_shape, nullptr);
    EXPECT_NE(opImpl->tiling, nullptr);
    EXPECT_NE(opImpl->tiling_parse, nullptr);
}

// =====================================================================================
// 迭代二 A2：24 TilingKey 实例选择矩阵——CROSS_CORE=0 侧 12 实例
// （CROSS_CORE=1 侧 9 个可达实例 + (cross=1,ARA,single)×3 dtype 结构不可达证明见
//   test_lp_norm_reduce_v2_crosscore_tiling.cpp——迁移 ops-nn 后与主 tiling 用例编入
//   同一共享 Host UT runner 二进制，宏默认 LPNORM_CROSS_CORE_ROUTING_ENABLED=1）
// key = dtype | CROSS_CORE<<8 | LAYOUT<<16 | BUFFER_MODE<<24；dtype 值 fp32=0/fp16=1/bf16=27
// =====================================================================================

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_fp32_ar_single)
{
    // fp32|AR|单缓冲 → key=0：每核 1 行×37，est=1×CeilDiv(37,32)=2 chunk <4
    TilingCaseParam c{};
    c.caseName = "matrix_fp32_AR_single";
    c.xShape = {7, 37};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 7;
    c.rLength = 37;
    c.coreNumUsed = 7;
    c.outBlockNum = 7;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 32;
    c.rFactorPerGroup = 37;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 7;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_fp32_ar_double)
{
    // fp32|AR：跨核路由默认开启 → groups=2、usedCores=4；rPerGroup=32768
    // sum 类 cap 1024 → ubR=1024、est=⌈32768/1024⌉=32 → 恒双缓冲（24 矩阵 cross|AR|single 对 sum 类 p 不可达）
    // ws=2×2×64×2=512（ping-pong ×2，H1）；key=DT_FLOAT|CROSS(1)<<8|AR|BUFFER(1)<<24
    TilingCaseParam c{};
    c.caseName = "matrix_fp32_AR_double";
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
    c.expectTilingKey = 16777472UL;
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_fp32_ara_single)
{
    // fp32|ARA|单缓冲 → key=1<<16：(8,64,70) C=70>1 → ubFactorC=min(64,70)=64（C 维 tile 截断）
    TilingCaseParam c{};
    c.caseName = "matrix_fp32_ARA_single";
    c.xShape = {8, 64, 70};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 8;
    c.rLength = 64;
    c.cLength = 70;
    c.coreNumUsed = 8;
    c.outBlockNum = 8;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 64;
    c.ubFactorC = 64;
    c.rAlignedCols = 64;
    c.rFactorPerGroup = 64;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65536UL;
    c.expectBlockDim = 8;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_fp32_ara_double)
{
    // fp32|ARA：跨核默认开启 → groups=2、usedCores=4；rChunk 复解=1778 → est=⌈32768/1778⌉=19 ≥4 → db
    TilingCaseParam c{};
    c.caseName = "matrix_fp32_ARA_double";
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

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_fp16_ar_single)
{
    // fp16|AR|单缓冲 → key=1：ubBlockElems=16 → ubR=FloorAlign(37,16)=32
    TilingCaseParam c{};
    c.caseName = "matrix_fp16_AR_single";
    c.xShape = {7, 37};
    c.dtype = ge::DT_FLOAT16;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 7;
    c.rLength = 37;
    c.coreNumUsed = 7;
    c.outBlockNum = 7;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 32;
    c.rFactorPerGroup = 37;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 1UL;
    c.expectBlockDim = 7;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_fp16_ar_double)
{
    // fp16|AR|双缓冲 → key=1|(1<<24)：sum 类 cap 1024 → ubR=1024、perRowBytes=1024*12+12=12300 →
    // rowsTile=227840/12300=18
    TilingCaseParam c{};
    c.caseName = "matrix_fp16_AR_double";
    c.xShape = {4096, 4096};
    c.dtype = ge::DT_FLOAT16;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 4096;
    c.rLength = 4096;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.bFactor = 64;
    c.rowsTile = 18;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 4096;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 16777217UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_fp16_ara_single)
{
    // fp16|ARA|单缓冲 → key=1|(1<<16)：rAlignedCols=64、rChunk=min(64,381)=64，est=2 <4
    TilingCaseParam c{};
    c.caseName = "matrix_fp16_ARA_single";
    c.xShape = {8, 64, 70};
    c.dtype = ge::DT_FLOAT16;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 8;
    c.rLength = 64;
    c.cLength = 70;
    c.coreNumUsed = 8;
    c.outBlockNum = 8;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 64;
    c.ubFactorC = 64;
    c.rAlignedCols = 64;
    c.rFactorPerGroup = 64;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65537UL;
    c.expectBlockDim = 8;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_fp16_ara_double)
{
    // fp16|ARA：跨核默认开启 → groups=2、usedCores=4；rAlignedCols=CeilAlign(3,16)=16；rChunk 复解=1270
    TilingCaseParam c{};
    c.caseName = "matrix_fp16_ARA_double";
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
    c.expectTilingKey = 16843009UL; // DT_FLOAT16 | CROSS(1)<<8 | ARA(1)<<16 | BUFFER(1)<<24
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_bf16_ar_single)
{
    // bf16|AR|单缓冲 → key=27（bf16 迭代二主路径放开，PT-08）
    TilingCaseParam c{};
    c.caseName = "matrix_bf16_AR_single";
    c.xShape = {7, 37};
    c.dtype = ge::DT_BF16;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 7;
    c.rLength = 37;
    c.coreNumUsed = 7;
    c.outBlockNum = 7;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 32;
    c.rFactorPerGroup = 37;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 27UL;
    c.expectBlockDim = 7;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_bf16_ar_double)
{
    // bf16|AR|双缓冲 → key=27|(1<<24)：sum 类 cap 1024 → ubR=1024、rowsTile=18（同 fp16 口径）
    TilingCaseParam c{};
    c.caseName = "matrix_bf16_AR_double";
    c.xShape = {4096, 4096};
    c.dtype = ge::DT_BF16;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 4096;
    c.rLength = 4096;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.bFactor = 64;
    c.rowsTile = 18;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 4096;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 16777243UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_bf16_ara_single)
{
    // bf16|ARA|单缓冲 → key=27|(1<<16)
    TilingCaseParam c{};
    c.caseName = "matrix_bf16_ARA_single";
    c.xShape = {8, 64, 70};
    c.dtype = ge::DT_BF16;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 8;
    c.rLength = 64;
    c.cLength = 70;
    c.coreNumUsed = 8;
    c.outBlockNum = 8;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 64;
    c.ubFactorC = 64;
    c.rAlignedCols = 64;
    c.rFactorPerGroup = 64;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65563UL;
    c.expectBlockDim = 8;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, matrix_cross0_bf16_ara_double)
{
    // bf16|ARA：跨核默认开启 → groups=2、usedCores=4；rAlignedCols=16、rChunk 复解=1270
    TilingCaseParam c{};
    c.caseName = "matrix_bf16_ARA_double";
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
    c.expectTilingKey = 16843035UL; // DT_BF16(27) | CROSS(1)<<8 | ARA(1)<<16 | BUFFER(1)<<24
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

// =====================================================================================
// 迭代二 A2：ARA 布局求解分支（bTile / ubFactorC / alignedCols 对齐 / pad 分支路由）
// =====================================================================================

TEST_F(LpNormReduceV2TilingTest, ara_btile_multi_batch_per_tile)
{
    // (10000,64,3)：bLen>coreNum → bFactor=CeilDiv(10000,64)=157；bTile=min(2481,157)=157（>1）
    // est=1×1×1=1 chunk <4 → 单缓冲；key=fp32|ARA|single
    TilingCaseParam c{};
    c.caseName = "ara_btile_157";
    c.xShape = {10000, 64, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 10000;
    c.rLength = 64;
    c.cLength = 3;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.bFactor = 157;
    c.rowsTile = 1;
    c.bTile = 157;
    c.ubFactorR = 64;
    c.ubFactorC = 3;
    c.rAlignedCols = 8;
    c.rFactorPerGroup = 64;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65536UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ara_ub_factor_c_clamped_to_64)
{
    // (4,32,100)：cLength=100 > ARA_C_TILE_MAX=64 → ubFactorC=64（列 tile 数=CeilDiv(100,64)=2）
    TilingCaseParam c{};
    c.caseName = "ara_ubfc_clamp64";
    c.xShape = {4, 32, 100};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 4;
    c.rLength = 32;
    c.cLength = 100;
    c.coreNumUsed = 4;
    c.outBlockNum = 4;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 32;
    c.ubFactorC = 64;
    c.rAlignedCols = 64;
    c.rFactorPerGroup = 32;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65536UL;
    c.expectBlockDim = 4;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ara_aligned_cols_pad_width_fp16)
{
    // (4,32,5) fp16：alignedCols=CeilAlign(ubFactorC=5, 32B/2B=16)=16 → pad 宽度 11 列（pad 值按分支）
    TilingCaseParam c{};
    c.caseName = "ara_alignedcols_fp16";
    c.xShape = {4, 32, 5};
    c.dtype = ge::DT_FLOAT16;
    c.p = 2.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 4;
    c.rLength = 32;
    c.cLength = 5;
    c.coreNumUsed = 4;
    c.outBlockNum = 4;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 32;
    c.ubFactorC = 5;
    c.rAlignedCols = 16;
    c.rFactorPerGroup = 32;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65537UL;
    c.expectBlockDim = 4;
    RunTilingCase(c);
}

// ARA pad 值分支路由（DESIGN §5.3：sum 类填 0 / +inf 填 -inf / -inf 填 +inf；
// tiling 侧可观测对象为 pBranch/pVal，驱动 kernel 侧 pad 值选择）
static void RunAraPBranchCase(const std::string& name, float p, int32_t expectBranch)
{
    TilingCaseParam c{};
    c.caseName = name;
    c.xShape = {8, 64, 70};
    c.dtype = ge::DT_FLOAT;
    c.p = p;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 8;
    c.rLength = 64;
    c.cLength = 70;
    c.coreNumUsed = 8;
    c.outBlockNum = 8;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 64;
    c.ubFactorC = 64;
    c.rAlignedCols = 64;
    c.rFactorPerGroup = 64;
    c.pBranch = expectBranch;
    c.expectTilingKey = 65536UL;
    c.expectBlockDim = 8;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, ara_pad_branch_pos_inf_routes_max)
{
    // p=+inf → BR_POS_INF（kernel pad=-INFINITY，Add/Max 合并矩阵 §1.6）
    RunAraPBranchCase("ara_pad_pos_inf", std::numeric_limits<float>::infinity(), LPNORM_P_BRANCH_POS_INF);
}

TEST_F(LpNormReduceV2TilingTest, ara_pad_branch_neg_inf_routes_min)
{
    // p=-inf → BR_NEG_INF（kernel pad=+INFINITY）
    RunAraPBranchCase("ara_pad_neg_inf", -std::numeric_limits<float>::infinity(), LPNORM_P_BRANCH_NEG_INF);
}

TEST_F(LpNormReduceV2TilingTest, ara_pad_branch_zero_routes_sum)
{
    // p=0 → BR_ZERO（kernel pad=0，计数分支）
    RunAraPBranchCase("ara_pad_zero", 0.0f, LPNORM_P_BRANCH_ZERO);
}

TEST_F(LpNormReduceV2TilingTest, ara_pad_branch_float_exp_routes_sum)
{
    // p=0.5 → BR_EXP（kernel pad=0，超越函数分支）
    RunAraPBranchCase("ara_pad_exp", 0.5f, LPNORM_P_BRANCH_EXP);
}

TEST_F(LpNormReduceV2TilingTest, ara_prefix_reduce_b_equals_one)
{
    // (4,8) axes=[0]：R 段居首 → B=1 合法（PT-17：ARA 退化单批）；b=1, r=4, c=8
    TilingCaseParam c{};
    c.caseName = "ara_prefix_B1";
    c.xShape = {4, 8};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 4;
    c.cLength = 8;
    c.coreNumUsed = 1;
    c.outBlockNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 4;
    c.ubFactorC = 8;
    c.rAlignedCols = 8;
    c.rFactorPerGroup = 4;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65536UL;
    c.expectBlockDim = 1;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, merge_axes_8d_deep_shape)
{
    // PT-17 8 维深形状：(2,3,1,4,5,1,2,3) axes=[3] → size=1 消除 + 前后 A 维分别合并
    // → B=2×3=6、R=4、C=5×2×3=30 → ARA；alignedCols=CeilAlign(30,8)=32
    TilingCaseParam c{};
    c.caseName = "merge_8d_deep";
    c.xShape = {2, 3, 1, 4, 5, 1, 2, 3};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {3};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 6;
    c.rLength = 4;
    c.cLength = 30;
    c.coreNumUsed = 6;
    c.outBlockNum = 6;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 4;
    c.ubFactorC = 30;
    c.rAlignedCols = 32;
    c.rFactorPerGroup = 4;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65536UL;
    c.expectBlockDim = 6;
    RunTilingCase(c);
}

// =====================================================================================
// 迭代二 A2：BUFFER_MODE 判据边界（PT-02/PT-11：每核 chunk 数 ≥4 启用双缓冲）
// =====================================================================================

TEST_F(LpNormReduceV2TilingTest, buffer_mode_boundary_three_chunks_single)
{
    // (2,61104) fp32 p=1：sum 类 cap 1024 → ubR=1024、est=⌈61104/1024⌉=60 ≥4 → 双缓冲
    // （cap 1024 前旧边界例 est=3 单缓冲已失效——sum 类大 R 恒 db；单缓冲仅 ±inf/计数/小 R 可达）
    TilingCaseParam c{};
    c.caseName = "buffer_boundary_est3_single";
    c.xShape = {2, 61104};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 61104;
    c.coreNumUsed = 2;
    c.outBlockNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1024;
    c.rFactorPerGroup = 61104;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777216UL;
    c.expectBlockDim = 2;
    RunTilingCase(c);
}

// =====================================================================================
// 迭代二 A2：回填修正点——BR_ZERO 计数预算 / tmpBuf E×4B / fixedExtra 入账
// =====================================================================================

TEST_F(LpNormReduceV2TilingTest, br_zero_budget_reduces_ub_r)
{
    // (2,65536) fp32 p=0：BR_ZERO 预算 +5B/元素（ones+mask 辅助块，PT-05/PT-12）
    // 跨核默认开启 → groups=2、usedCores=4；rPerGroup=32768；单缓冲 perElem=12+5=17
    // → ubR=FloorAlign(ubRCap,8)=13416；est=⌈32768/13416⌉=3 <4 → 单缓冲（p=0 计数分支不受 1024 cap）
    // （对照 p=1 同形状恒双缓冲：+5B 项与 +8 项（fp32Buf 4 + tmpBuf E×4B 口径 4）仍被锁定）
    TilingCaseParam c{};
    c.caseName = "brzero_budget_ubR";
    c.xShape = {2, 65536};
    c.dtype = ge::DT_FLOAT;
    c.p = 0.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 65536;
    c.coreNumUsed = 4;
    c.outBlockNum = 2;
    c.rGroupNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 13416;
    c.rFactorPerGroup = 32768;
    c.pBranch = LPNORM_P_BRANCH_ZERO;
    c.expectTilingKey = 256UL; // DT_FLOAT | CROSS(1)<<8 | AR | 单缓冲
    c.expectBlockDim = 4;
    c.expectWsBytes = 512UL;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, fixed_extra_accounting_pins_rows_tile)
{
    // (64000,48) fp32 p=1：单缓冲 perRow=48×12+12=588；rowsTile=227776/588=387
    // （fixedExtra=256×1+1024+kahanBuf 8KB+pwBuf 8KB 全额入账口径逐值锁定）
    TilingCaseParam c{};
    c.caseName = "fixedExtra_single_buf";
    c.xShape = {64000, 48};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 64000;
    c.rLength = 48;
    c.coreNumUsed = 64;
    c.outBlockNum = 64;
    c.bFactor = 1000;
    c.rowsTile = 387;
    c.ubFactorR = 48;
    c.rFactorPerGroup = 48;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 64;
    RunTilingCase(c);
}

// =====================================================================================
// 迭代二 A2：PT-15 fp32 计数口径边界（N=2^24+1：golden 维持 fp32 求和口径，§1.10/R4）
// =====================================================================================

TEST_F(LpNormReduceV2TilingTest, pt15_count_boundary_n_2p24_plus_1)
{
    // (1,16777217) fp32 p=0：N=2^24+1 计数上界边界（fp32 表示域外首个值；ST golden 按 fp32 口径）
    // 跨核默认开启 → 全归约 groups=min(64,⌈16777217/32768⌉上限 64)=64、usedCores=64
    // rPerGroup=⌈16777217/64⌉=262145；p=0 计数分支不受 1024 cap → ubR=FloorAlign((227840-20)/21,8)=10848
    // ws=1×64×64×2=8192（ping-pong ×2，H1）；key=DT_FLOAT|CROSS(1)<<8|AR|BUFFER(1)<<24
    TilingCaseParam c{};
    c.caseName = "pt15_count_2p24p1";
    c.xShape = {1, 16777217};
    c.dtype = ge::DT_FLOAT;
    c.p = 0.0f;
    c.axes = {};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 1;
    c.rLength = 16777217;
    c.coreNumUsed = 64;
    c.outBlockNum = 1;
    c.rGroupNum = 64;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 10848;
    c.rFactorPerGroup = 262145;
    c.pBranch = LPNORM_P_BRANCH_ZERO;
    c.expectTilingKey = 16777472UL; // DT_FLOAT | CROSS(1)<<8 | AR | BUFFER(1)<<24
    c.expectBlockDim = 64;
    c.expectWsBytes = 8192UL;
    RunTilingCase(c);
}

// =====================================================================================
// 迭代二 A2：PT-17 rSegNum==0 退化（归约轴全部 size=1 被消除 → R=1 恒等归约）
// =====================================================================================

TEST_F(LpNormReduceV2TilingTest, pt17_all_reduce_axes_size_one_degenerate)
{
    // (1,2,1,3,1,1,4,1) axes=[0,2,4,5,7]：归约轴全 size=1 消除 → rSegNum==0 退化
    // → R=1 恒等归约、B=2×3×4=24、C=1 → AR；ubR=rSpan=1（<ubBlockElems 短行整段）；segNum=0（真实段数）
    TilingCaseParam c{};
    c.caseName = "pt17_rseg0_degenerate";
    c.xShape = {1, 2, 1, 3, 1, 1, 4, 1};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0, 2, 4, 5, 7};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 24;
    c.rLength = 1;
    c.coreNumUsed = 24;
    c.outBlockNum = 24;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 1;
    c.rFactorPerGroup = 1;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 0;
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 24;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, pt17_mixed_elimination_left_two_r_segments)
{
    // (1,2,1,3,1,1,4,1) axes=[0,1,2,6]：混合消除后残留 [R:2][A:3][R:4] 两段 R → P2 段表（迭代三接入）
    // （PT-17 case11：rSegNum=2 精确探测，不误入 P1）O=3、rTotal=8、chunk=8（单 chunk）、oTile=1
    TilingCaseParam c{};
    c.caseName = "pt17_p2_mixed_table";
    c.xShape = {1, 2, 1, 3, 1, 1, 4, 1};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {0, 1, 2, 6};
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 3;
    c.rLength = 8;
    c.cLength = 1;
    c.coreNumUsed = 3;
    c.outBlockNum = 3;
    c.rGroupNum = 1;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 8;
    c.ubFactorC = 1;
    c.rAlignedCols = 1;
    c.rFactorPerGroup = 8;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.segNum = 2;
    c.aSegNum = 1;
    c.expectSegSize = {2, 4};
    c.expectSegStride = {12, 1};
    c.expectASize = {3};
    c.expectAStride = {4};
    c.expectTilingKey = 0UL;
    c.expectBlockDim = 3;
    RunTilingCase(c);
}

// =====================================================================================
// 迭代二 A2：病态平台防御钳位（SolveUbTilingAR/ARA 下界防御，ubAvail < 单块预算时不崩溃）
// =====================================================================================

TEST_F(LpNormReduceV2TilingTest, defensive_tiny_ub_ar_solver_clamps)
{
    // ubSize=9216 → ubAvail=-256 < 单行块预算：ubRCap<1 → 钳位 1；FloorAlign(1,8)=0 → 钳位 ubBlockElems=8
    // 验证求解器输出恒为正值、不越界不崩溃（防御代码路径；预算断言关闭）
    TilingCaseParam c{};
    c.caseName = "defensive_tiny_ub_AR";
    c.xShape = {7, 37};
    c.dtype = ge::DT_FLOAT;
    c.p = 1.0f;
    c.axes = {1};
    c.ubSize = 9216; // < UB_FIXED_RESERVE+fixedExtra+单块（病理平台）
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 7;
    c.rLength = 37;
    c.coreNumUsed = 7;
    c.outBlockNum = 7;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.ubFactorR = 8; // FloorAlign 钳位下界 = ubBlockElems(fp32)=8
    c.rFactorPerGroup = 37;
    c.pBranch = LPNORM_P_BRANCH_ONE;
    c.expectTilingKey = 16777216UL; // est=5 ≥4 → db（钳位后按双缓冲复解）
    c.expectBlockDim = 7;
    c.checkUbBudget = false;
    RunTilingCase(c);
}

TEST_F(LpNormReduceV2TilingTest, defensive_tiny_ub_ara_solver_clamps)
{
    // ubSize=9216 ARA：rChunkCap<1 → 钳位 1；bTileCap<0 → max(,1)=1；rChunk=1
    TilingCaseParam c{};
    c.caseName = "defensive_tiny_ub_ARA";
    c.xShape = {2, 3, 4};
    c.dtype = ge::DT_FLOAT;
    c.p = 2.0f;
    c.axes = {1};
    c.ubSize = 9216;
    c.expectStatus = ge::GRAPH_SUCCESS;
    c.bLength = 2;
    c.rLength = 3;
    c.cLength = 4;
    c.coreNumUsed = 2;
    c.outBlockNum = 2;
    c.bFactor = 1;
    c.rowsTile = 1;
    c.bTile = 1;
    c.ubFactorR = 1; // rChunk 钳位下界
    c.ubFactorC = 4;
    c.rAlignedCols = 8;
    c.rFactorPerGroup = 3;
    c.pBranch = LPNORM_P_BRANCH_INT_POW;
    c.expectTilingKey = 65536UL; // est=1×1×3=3 <4 → 单缓冲
    c.expectBlockDim = 2;
    c.checkUbBudget = false;
    RunTilingCase(c);
}

// =====================================================================================
// 迭代二 A2：TilingParse 回调冒烟（IMPL_OP_OPTILING 注册完整性，覆盖 Parse 函数体）
// =====================================================================================

TEST_F(LpNormReduceV2TilingTest, tiling_parse_callback_returns_success)
{
    // TilingParse<LpNormReduceV2CompileInfo> 注册的解析回调为占位实现：入图场景依赖，
    // UT 直调验证注册回调可被调用且返回 SUCCESS（覆盖 tiling.cpp TilingParseForLpNormReduceV2）
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    EXPECT_EQ(opImpl->tiling_parse(nullptr), ge::GRAPH_SUCCESS);
}
