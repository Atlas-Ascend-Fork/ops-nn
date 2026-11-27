/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Tiling data structure shared between host-side tiling and device-side
//   kernel for NormalizeBatch on arch35 (ascend950 / ascend960dt / ascend960pr / ascend350).
//   Host 侧填充（normalize_batch_tiling_arch35.cpp）与 kernel 侧读取
//   （GET_TILING_DATA_WITH_STRUCT）看到同一份结构体布局。
//
// 切分模型：
//   - batch 单元 = per_feature 的 (n, c) 行 / all_features 的 n 平面，
//     按 usedCoreNum 均分（normalCoreProcessNum / 尾核 tailCoreProcessNum）；
//   - seq_len 为运行时 GM 数据，统计长度 L 在核内读取后动态钳制到 [1, d]；
//   - rowStride = CeilAlign(d, VL)：UB 行距，保证行内 LoadAlign 整 VL 读取
//     不越过行缓冲末尾（掩码保证语义正确，行距保证读安全）。

#ifndef OPS_NORM_NORMALIZE_BATCH_TILING_STRUCT_H_
#define OPS_NORM_NORMALIZE_BATCH_TILING_STRUCT_H_

#include <cstdint>

struct NormalizeBatchTilingData {
    // 静态形状
    int64_t nDim = 0; // n
    int64_t cDim = 0; // c
    int64_t dDim = 0; // d

    // batch 维多核切分（大小核均衡）
    int32_t usedCoreNum = 0;          // 实际使用核数 = min(batch, coreNum)
    int64_t normalCoreProcessNum = 0; // 每核 batch 单元数
    int64_t tailCoreProcessNum = 0;   // 尾核 batch 单元数（∈ [1, normal]）

    // UB 切分
    int64_t rowStride = 0;     // UB 行距（元素）= CeilAlign(d, VL)，VL=64（fp32 256B）
    int64_t eachLoopDNum = 0;  // BigD 模板 d 分段元素数（VL 对齐，≥ VL）
    int64_t dataBufElems = 0;  // 主数据 buffer 元素数（SmallD: rowStride 或 c*rowStride；BigD: eachLoopDNum）
    int64_t seqLenBufSize = 0; // seq_len buffer 字节（32B 对齐）
    int64_t scalarBufSize = 0; // 标量槽字节（sum/sumsq 两槽 × 32B）

    // 算子自定义字段
    float epsilon = 0.0f; // attr epsilon（默认 1e-5）：y = (x-mean)/(std+eps)
};

#endif // OPS_NORM_NORMALIZE_BATCH_TILING_STRUCT_H_
