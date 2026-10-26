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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_tilingdata.h
 * \brief FakeQuantWithMinMaxVarsPerChannelGradient TilingData (arch35)
 *
 * 新写法：POD 结构体，host/kernel 共享，无 BEGIN/END_TILING_DATA_DEF 与 REGISTER_TILING_DATA_CLASS。
 * 字段顺序与旧宏注册版本保持完全一致，保证 host 写入 / kernel 读取的二进制布局不变。
 */
#ifndef FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_TILINGDATA_H_
#define FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_TILINGDATA_H_

#include <cstdint>

struct FakeQuantWithMinMaxVarsPerChannelGradientTilingData {
    uint32_t usedCoreNum = 0; // 实际活跃核数（空 tensor 模板下恒为 1）
    uint32_t rowsPerCore = 0;
    uint32_t tailRows = 0;
    uint32_t totalRows = 0;
    uint32_t channelNum = 0; // = x 最后一维（空 tensor 时可为 0）
    uint32_t tileLen = 0;    // legacy 字段，与 dTileLen 相同
    int32_t quantMin = 0;
    int32_t quantMax = 0;
    uint32_t dTileLen = 0;
    uint32_t numDChunks = 0;
    uint32_t splitMode = 0; // 0 = split rows, 1 = split d-chunks
    uint32_t chunksPerCore = 0;
    uint32_t tailChunks = 0;
};

#endif // FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_TILINGDATA_H_
