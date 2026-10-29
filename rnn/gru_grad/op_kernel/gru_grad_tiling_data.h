/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad_tiling_data.h
 * \brief GRU 反向 TilingData —— 基线(arch22/KFC)/优化栈(arch35/基础 API) 公用单一声明。
 *
 * - 基线 kernel 只消费前段字段与四路 TCubeTiling；deprecated 的四个 CutBatchTiling
 *   （早期 concat [x|h] 方案遗留）保留为字段布局不变量。优化栈 kernel 全量消费。
 * - host 两套 tiling 实现（arch22/arch35，按 TILING_DIR 构建 SoC 选源）写同一 struct。
 * - TCubeTiling 按编译侧选头（三棵头文件布局一致）：dav_3510 kernel 用 asc 树
 *   adv_api/kernel_tiling.h；其余 kernel 用 kernel_tiling/kernel_tiling.h
 *   （__CCE_ARCH__ 兜底 bisheng 直调）；host 侧用 tiling/tiling_api.h。
 * - 优化栈 kernel 经 GruGradLoadTilingData 字节循环拷贝消费（dav_3510 无
 *   GET_TILING_DATA_WITH_STRUCT）；基线走框架宏（同为字节 memcpy 语义）。
 */
#ifndef _GRU_GRAD_TILING_DATA_H_
#define _GRU_GRAD_TILING_DATA_H_

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "adv_api/kernel_tiling.h"
#elif defined(__CCE_AICORE__) || defined(__NPU_ARCH__) || defined(__CCE_ARCH__)
#include "kernel_tiling/kernel_tiling.h"
#else
#include "tiling/tiling_api.h"
#endif

struct CutBatchTiling {
    int64_t taskNum = 0;
    int64_t copyMLines = 0;
    int64_t copyMLinesTail = 0;
    int64_t nLoop = 0;
    int64_t copyNLength = 0;
    int64_t copyNLengthTail = 0;
    int64_t splitTaskPerCore = 0;
    int64_t splitPreCore = 0;
};

struct GruGradTilingData {
    int64_t ubSize = 0;
    int64_t timeStep = 0;
    int64_t batch = 0;
    int64_t inputSize = 0;
    int64_t hiddenSize = 0;
    int64_t isBias = 0;
    int64_t isSeqLength = 0;
    int64_t totalSteps = 0; // sum(batch_sizes) 当不定长; = timeStep * batch 当定长

    int64_t singleCoreM = 0;
    int64_t singleCoreMTail = 0;
    int64_t singleCoreN = 0;
    int64_t singleCoreNTail = 0;
    int64_t baseN = 0;
    int64_t baseM = 0;
    int64_t mCnt = 0;
    int64_t nCnt = 0;

    int64_t singleCoreReduceN = 0;
    int64_t singleCoreReduceNTail = 0;
    int64_t baseReduceN = 0;
    int64_t nReduceCnt = 0;
    int64_t maxReduceNumOnce = 0;
    int64_t reduceBlockSize = 0;

    int64_t direction = 0;

    int64_t inputSizeAligned = 0;
    int64_t hiddenSizeAligned = 0;
    int64_t oneLineAligned = 0;
    CutBatchTiling dxhInputTiling;  // deprecated: 早期 concat [x|h] 方案遗留，kernel 未消费
    CutBatchTiling dxhHiddenTiling; // deprecated
    CutBatchTiling xhInputTiling;   // deprecated
    CutBatchTiling xhHiddenTiling;  // deprecated

    TCubeTiling dwIhMMParam;
    TCubeTiling dwHhMMParam;
    TCubeTiling dgateMMParam;
    TCubeTiling dxMMParam;
    int64_t wNzEnabled = 0; // 巨型 shape w_hh NZ 像预暂存段使能（arch35 消费；host 按域设 1, kernel 复核）
    int64_t s1Enabled = 0; // 巨型 shape dgate K 维切分+跨簇部分和规约使能（arch35 消费；依赖 wNz 域）
};

#endif // _GRU_GRAD_TILING_DATA_H_
