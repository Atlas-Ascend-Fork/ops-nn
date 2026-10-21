/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file anti_hi_f4_quant_tilingdata.h
 * \brief
 */

#ifndef OPS_NN_ANTI_HI_F4_QUANT_TILINGDATA_H
#define OPS_NN_ANTI_HI_F4_QUANT_TILINGDATA_H

#include <cstdint>

struct AntiHiF4QuantNotTailTilingData {
    int64_t tilingKey{0};
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t dstType{27};
    int64_t xDimNeg1{0};
    int64_t xDimNeg2{0};
    int64_t xDimNeg3{1};
    int64_t splitBlockColSize{0};
    int64_t splitBlockRowSize{0};
    int64_t tailBlockRowSize{0};
    int64_t tailBlockColSize{0};
    int64_t splitBlockColNum{0};
    int64_t splitBlockRowNum{0};
    int64_t headCoreLoopNum{0};
    int64_t tailCoreLoopNum{0};
    int64_t headCoreNum{0};
    int64_t totalTaskNum{0};
    int64_t ubBlockNum{0};
    int64_t taskNumPerBatch{0};
};

struct AntiHiF4QuantTailTilingData {
    int64_t totalCoreNum{0}; // 硬件可用 AIV 总核数
    int64_t usedCoreNum{0};  // 实际使用的核数min(totalCoreNum, ceil(groupNum/HIFP4_PER_CORE_MIN_GROUP_NUM))
    int64_t tilingKey{0};    // 模板选择键：GET_TPL_TILING_KEY(quantAxis)，固定为 TPL_QUANT_TAIL_AXIS=0

    int64_t groupNum{0}; // 单核分配的 group 总数（满载核），每 group = 64 个 fp4 元素 = 32 字节 x
    int64_t alignGroupNum{0};       // 单次 UB 处理的 group 数，对齐到 8 的倍数
    int64_t rowPerHeadCore{0};      // 满载核的 UB 循环次数CeilDiv(blockFormer, alignGroupNum)
    int64_t rowPerTailCore{0};      // 尾核的 UB 循环次数CeilDiv(blockTail, alignGroupNum)
    int64_t multiRowNumHeadCore{0}; // 满载核尾次 UB 处理的 group 数
    int64_t multiRowNumTailCore{0}; // 尾核尾次 UB 处理的 group 数

    // Compute 阶段循环参数（类比 DynamicHiF4Quant 的 innerLoop*）
    int64_t scaleLoopEle{0};   // 满载 UB 块的 ExtractScale+Dequant 循环次数CeilDiv(alignGroupNum, 8)
    int64_t scaleLoopTimes{0}; // 满载核尾块的循环次数
    int64_t scaleLoopTail{0};  // 尾核尾块的循环次数
    int64_t outLoopEle{0};     // 满载 UB 块的 CopyOut 循环次数CeilDiv(alignGroupNum, 2)
    int64_t outLoopTimes{0};   // 满载核尾块的 CopyOut 循环次数
    int64_t outLoopTail{0};    // 尾核尾块的 CopyOut 循环次数

    int64_t dstType{27}; // 目标输出类型: DT_BF16=27, DT_FLOAT16=1
};
#endif // OPS_NN_ANTI_HI_F4_QUANT_TILINGDATA_H
