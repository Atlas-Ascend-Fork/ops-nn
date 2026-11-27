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
 * \file foreach_sign.cpp
 * \brief Kernel entry for foreach_sign (arch35 / ascend950, RegBase path)
 */

#include "foreach_sign_regbase.h"

using namespace ForeachSign;

template <typename T>
__aicore__ inline void ForeachSignImpl(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling, TPipe* tPipe)
{
    GET_TILING_DATA_WITH_STRUCT(ForeachSoloTilingDataRegbase, tilingDataIn, tiling);
    const ForeachSoloTilingDataRegbase* __restrict tilingData = &tilingDataIn;
    ForeachSignRegbase<T, ForeachSoloTilingDataRegbase> op;
    op.Init(inputs, outputs, workspace, tilingData, tPipe);
    AscendC::WaitPreTaskEnd();
    op.Process();
}

#if defined(__DAV_VEC__)
extern "C" __global__ __aicore__ void foreach_sign(GM_ADDR inputs, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipeOp;
    if (TILING_KEY_IS(FOREACH_TILING_KEY_HALF)) {
        ForeachSignImpl<half>(inputs, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_FLOAT)) {
        ForeachSignImpl<float>(inputs, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_BF16)) {
        ForeachSignImpl<bfloat16_t>(inputs, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_INT)) {
        ForeachSignImpl<int32_t>(inputs, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_INT8)) {
        ForeachSignImpl<int8_t>(inputs, outputs, workspace, tiling, &pipeOp);
    } else if (TILING_KEY_IS(FOREACH_TILING_KEY_INT64)) {
        ForeachSignImpl<int64_t>(inputs, outputs, workspace, tiling, &pipeOp);
    }
    pipeOp.Destroy();
}
#endif
