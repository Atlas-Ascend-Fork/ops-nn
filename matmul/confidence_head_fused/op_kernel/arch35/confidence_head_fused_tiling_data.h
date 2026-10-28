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
 * \file confidence_head_fused_tiling_data.h
 * \brief TilingData struct for ConfidenceHeadFused, registered in kernel directory.
 *        Host side fills this struct via memcpy_s, kernel side reads via GET_TILING_DATA_WITH_STRUCT.
 */

#ifndef CONFIDENCE_HEAD_FUSED_TILING_DATA_H
#define CONFIDENCE_HEAD_FUSED_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace CHF {

#pragma pack(push, 8)
struct ConfidenceHeadFusedCubeTiling {
    uint32_t baseN = 0;       // L0 baseN
    uint32_t baseK = 0;       // L0 baseK
    uint32_t kL1 = 0;         // Feature K tile in L1
    uint32_t l1BufferNum = 4; // L1 buffer stages
};
#pragma pack(pop)

#pragma pack(push, 8)
struct ConfidenceHeadFusedParams {
    uint32_t n = 0;           // Flattened row count N or B*N
    uint32_t kEmbed = 0;      // hidden K dimension
    uint32_t kMarkov = 0;     // markov_emb K dimension
    uint32_t usedCoreNum = 0; // AIC core count
    uint8_t hasMarkov = 0;    // whether markov_emb input exists
    uint8_t hasBias = 0;
    uint8_t hasLowLimit = 0;
    uint8_t hasHighLimit = 0;
    uint32_t biasN = 0; // Bias length and rows per batch
    float lowLimit = 0.0f;
    float highLimit = 0.0f;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct ConfidenceHeadFusedTilingData {
    ConfidenceHeadFusedParams params;
    ConfidenceHeadFusedCubeTiling cubeTiling;
};
#pragma pack(pop)

} // namespace CHF
#endif // CONFIDENCE_HEAD_FUSED_TILING_DATA_H
