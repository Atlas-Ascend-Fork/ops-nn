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
 * \file lp_norm_reduce_v2_tiling_data.h
 * \brief LpNormReduceV2 TilingData 结构体定义（一次定形，避免 ABI 变动；P2 字段迭代三启用）
 *
 * ✅ 必须使用标准 C++ struct 定义 TilingData
 * ❌ 禁止使用废弃的 BEGIN_TILING_DATA_DEF 宏
 */

#ifndef _LP_NORM_REDUCE_V2_TILING_DATA_H_
#define _LP_NORM_REDUCE_V2_TILING_DATA_H_

// p 分支编号（DESIGN §1.6 判定优先级：+inf 别名 > -inf 别名 > 0 > 1 > 整数[2,15] > 其他有限浮点）
constexpr int32_t LPNORM_P_BRANCH_POS_INF = 0; // p=+inf（含别名 2147483647.0）→ max|x|
constexpr int32_t LPNORM_P_BRANCH_NEG_INF = 1; // p=-inf（含别名 -2147483648.0）→ min|x|
constexpr int32_t LPNORM_P_BRANCH_ZERO = 2;    // p=0 → 非零元素计数
constexpr int32_t LPNORM_P_BRANCH_ONE = 3;     // p=1 → sum|x|
constexpr int32_t LPNORM_P_BRANCH_INT_POW = 4; // 整数 p∈[2,15] → 连乘 sum|x|^p
constexpr int32_t LPNORM_P_BRANCH_EXP = 5;     // 其他有限浮点 → exp(p·ln|x|)

// CROSS_CORE workspace slot 对齐步进（字节）：host 侧容量计算（tiling.cpp wsBytes）与 kernel 侧
// slot 步进（lp_norm_reduce_v2.h slotStrideElems_）的共享 ABI 常量（GEN-4.2：消除跨文件裸字面量 64）
constexpr int64_t WS_SLOT_ALIGN_BYTES = 64;

struct LpNormReduceV2TilingData {
    int64_t bLength = 0;                       // 合轴后 R 段之前的 A 维积（AR 行数上界 / ARA 批次数）
    int64_t rLength = 0;                       // 每输出元素的归约元素总数
    int64_t cLength = 1;                       // 合轴后 R 段之后的 A 维积（AR 时 =1）
    int64_t coreNum = 1;                       // blockDim（实际使用核数）
    int64_t outBlockNum = 1;                   // 输出分块数
    int64_t rGroupNum = 1;                     // R 跨核分组数（CROSS_CORE=0 时 =1）
    int64_t bFactor = 0;                       // 每输出块的行数（AR）/ 批数（ARA）
    int64_t rowsTile = 1;                      // AR：输出 tile 行数（BLK-1 输出 tile 外层循环步长）
    int64_t bTile = 1;                         // ARA：输出 tile 批数（AR 无效）
    int64_t ubFactorR = 1;                     // AR：每行 chunk 有效长度；ARA：RA 归约行数
    int64_t ubFactorC = 1;                     // ARA：C 向 tile 大小（AR 无效）
    int64_t rAlignedCols = 1;                  // ARA：alignedCols（AR 无效）
    int64_t rFactorPerGroup = 0;               // 每个 rGroup 的 R 工作量（CROSS_CORE=0 时 =rLength）
    int32_t pBranch = LPNORM_P_BRANCH_INT_POW; // p 分支编号（0..5，运行期 switch 分发）
    float pVal = 2.0f;                         // p 数值（EXP 分支 Muls 系数）
    int32_t powTimes = 1;                      // INT_POW 分支连乘次数（p-1）
    int32_t zeroOutputFlag = 0;                // R==0 置零路径标志（§5.6）
    int32_t segNum = 1;         // P2：R 段数（P1 =1；0=合轴后无 R 段退化——迭代三启用）
    int64_t segSize[8] = {0};   // P2：各 R 段大小（rank≤8 → 段数上限 4，数组冗余）
    int64_t segStride[8] = {0}; // P2：各 R 段元素步进（该段之后所有维度积）
    int32_t aSegNum = 0;        // P2：A 段数（输出混合基数，上限 4；迭代三 v1.3 新增）
    int64_t aSize[4] = {0};     // P2：各 A 段大小（aSize[0] 为输出最外层基数，PT-13 §3.2）
    int64_t aStride[4] = {0};   // P2：各 A 段元素步进（输出坐标 → GM 基地址分解）
};
#endif // _LP_NORM_REDUCE_V2_TILING_DATA_H_
