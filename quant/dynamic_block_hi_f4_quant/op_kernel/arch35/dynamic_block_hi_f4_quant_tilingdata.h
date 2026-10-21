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
 * \file dynamic_block_hi_f4_quant_tilingdata.h
 * \brief TilingData structure for DynamicBlockHiF4Quant kernel on ascend960dt
 */

#ifndef OPS_NN_DYNAMIC_BLOCK_HI_F4_QUANT_TILINGDATA_H
#define OPS_NN_DYNAMIC_BLOCK_HI_F4_QUANT_TILINGDATA_H

#include <cstdint>

struct DynamicBlockHiF4QuantTilingData {
    int64_t tilingKey;
    int64_t totalCoreNum;       // 总核数
    int64_t usedCoreNum;        // 实际使用的核数
    int64_t ubSize;             // UB大小
    int64_t roundMode;          // 数据类型转换的模式
    int64_t scaleAlg;           // 缩放算法选择：0-默认(阈值为4/2)，1-阈值3.75/1.875
    int64_t blockSizeRow;       // 基本块的高
    int64_t blockSizeCol;       // 基本块的宽
    int64_t rowNum;             // 行数
    int64_t colNum;             // 列数
    int64_t rowBlockLoopNum;    // 行方向基本块的数量
    int64_t colBlockLoopNum;    // 列方向基本块的数量
    int64_t colTileNum;         // 列方向尾块长度
    int64_t normalCoreNum;      // 头核的数量
    int64_t normalCoreBlockNum; // 头核处理基本块的数量
    int64_t tailCoreBlockNum;   // 尾核处理基本块的数量
};

#endif // OPS_NN_DYNAMIC_BLOCK_HI_F4_QUANT_TILINGDATA_H
