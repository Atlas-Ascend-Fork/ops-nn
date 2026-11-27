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
 * \file foreach_log2_flat.cpp
 * \brief Flat RegBase kernel entry for ForeachLog2 (arch35 / Ascend950). Ascend350 keeps the
 *        original SIMT entry in foreach_log2.cpp; this file is bound to ascend950 only.
 */

#include "foreach_log2_flat_regbase.h"
#include "foreach_log2_tiling_key.h"
#include "../../foreach_utils/arch35/foreach_flat_tiling_data.h"

using namespace ForeachLog2;

enum class ForeachLog2TilingKey : uint32_t {
    TILING_KEY_FLOAT = FOREACH_LOG2_TPL_SCH_MODE_FLOAT,
    TILING_KEY_FLOAT16 = FOREACH_LOG2_TPL_SCH_MODE_FLOAT16,
    TILING_KEY_BF16 = FOREACH_LOG2_TPL_SCH_MODE_BF16,
};

template <uint32_t schMode>
__global__ __aicore__ void foreach_log2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachFlatTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachFlatTilingData, tilingDataIn, tiling);
    (void)workspace;

    TPipe pipe;
    if constexpr (schMode == static_cast<uint32_t>(ForeachLog2TilingKey::TILING_KEY_FLOAT)) {
        ForeachLog2::ForeachLog2Flat<float, ForeachFlatTilingData> op;
        op.Init(&tilingDataIn, &pipe);
        op.Process(x, y);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachLog2TilingKey::TILING_KEY_FLOAT16)) {
        ForeachLog2::ForeachLog2Flat<half, ForeachFlatTilingData> op;
        op.Init(&tilingDataIn, &pipe);
        op.Process(x, y);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachLog2TilingKey::TILING_KEY_BF16)) {
        ForeachLog2::ForeachLog2Flat<bfloat16_t, ForeachFlatTilingData> op;
        op.Init(&tilingDataIn, &pipe);
        op.Process(x, y);
    }
    pipe.Destroy();
}
