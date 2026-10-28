/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)

#include "confidence_head_fused_kernel.h"
#include "confidence_head_fused_kernel_tiling_key.h"

using CHF::ConfidenceHeadFusedTilingData;

template <int8_t TILING_KEY>
__global__ __aicore__ void confidence_head_fused(GM_ADDR hidden, GM_ADDR weight, GM_ADDR markovEmb, GM_ADDR temperature,
                                                 GM_ADDR bias, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ConfidenceHeadFusedTilingData);
    GET_TILING_DATA_WITH_STRUCT(ConfidenceHeadFusedTilingData, tilingData, tiling);
    CHF::ConfidenceHeadFusedKernel<TILING_KEY, DTYPE_HIDDEN, DTYPE_WEIGHT, float>(
        hidden, weight, markovEmb, temperature, bias, out, workspace, &tilingData);
}

#endif
