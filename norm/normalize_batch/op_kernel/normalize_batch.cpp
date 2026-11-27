/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Ascend C kernel entry point for NormalizeBatch
//   (arch35: ascend950 / ascend960dt / ascend960pr / ascend350).
//   核函数签名：
//     template <bool isAllFeatures, bool isBigD>
//     __global__ __aicore__ void normalize_batch(GM_ADDR input_x, GM_ADDR seq_len, GM_ADDR output_y,
//                                                GM_ADDR workspace, GM_ADDR tiling);
//   - 形参顺序 = 算子原型 tensor 定义顺序（input_x, seq_len, output_y）+ 末尾固定
//     workspace + tiling；normalize_type / epsilon 均为 attr，经 tiling data 下发。
//   - 模板参数为 TilingKey 双 bool 轴（isAllFeatures / isBigD）；
//     dtype 走编译期 DTYPE_INPUT_X 实例化（当前仅 float32，宏名按输入名 input_x 生成）。
//   - isAllFeatures：normalize_type == "all_features"（同一 n 的所有 c 共享统计量）；
//   - isBigD：单遍数据（行/平面）放不进 UB → d 分段两遍。
//   - 无跨核归约（batch 维切核、核间零依赖），workspace 形参保留但未使用。

#include "kernel_operator.h"                      // Ascend C kernel framework (AscendC:: namespace)
#include "arch35/normalize_batch_tiling_struct.h" // NormalizeBatchTilingData
#include "arch35/normalize_batch_struct.h"        // TPL 声明（isAllFeatures / isBigD 双 bool 轴）
#include "arch35/normalize_batch_kernel.h"        // kernel 模板类（四实例合一）

// kernel 入口：每个 AIV 核执行一次；模板参数为 TilingKey 双 bool 轴
// （语义详见文件头说明）。
template <bool isAllFeatures, bool isBigD>
__global__ __aicore__ void normalize_batch(GM_ADDR input_x, GM_ADDR seq_len, GM_ADDR output_y, GM_ADDR workspace,
                                           GM_ADDR tiling)
{
    // 不走框架 tiling 自动注册，tiling data 由 GET_TILING_DATA_WITH_STRUCT 手工读取
    REGISTER_NONE_TILING;

    // 仅在 AIV（AI Vector）核上运行
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TPipe 统一在入口申请，经 Init 指针传入模板类。
    AscendC::TPipe pipe;

    // workspace 保留形参：本算子 batch 维切核、核间零依赖，无 workspace 需求。
    (void)workspace;

    GET_TILING_DATA_WITH_STRUCT(NormalizeBatchTilingData, tilingData, tiling);
    // dtype 宏名由框架按 REG_OP 输入名生成：输入名 input_x → DTYPE_INPUT_X
    NsNormalizeBatch::NormalizeBatchKernel<DTYPE_INPUT_X, isAllFeatures, isBigD> op;
    op.Init(input_x, seq_len, output_y, &tilingData, &pipe);
    op.Process();
}
