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
 * \file foreach_erfc.cpp
 * \brief Kernel entry point for foreach_erfc operator (arch35 / Ascend950).
 */

#include "foreach_erfc_regbase.h"
#include "foreach_erfc_scalar_tiling.h"
#include "foreach_erfc_tiling_key.h"
#include "simt_api/math_functions.h"

using namespace AscendC;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void ErfcScalarListVF(GM_ADDR x, GM_ADDR y, uint32_t count)
{
    auto inputWords = reinterpret_cast<__gm__ uint64_t*>(x);
    auto outputWords = reinterpret_cast<__gm__ uint64_t*>(y);
    auto inputs = inputWords + (inputWords[0] >> 3);
    auto outputs = outputWords + (outputWords[0] >> 3);
    for (uint32_t i = Simt::GetThreadIdx(); i < count; i += Simt::GetThreadNum()) {
        const float value = static_cast<float>(*reinterpret_cast<__gm__ T*>(inputs[i]));
        *reinterpret_cast<__gm__ T*>(outputs[i]) = static_cast<T>(erfcf(value));
    }
}

template <typename T>
__aicore__ inline void ForeachErfcImpl(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, TPipe* pipe)
{
    GET_TILING_DATA_WITH_STRUCT(ForeachFlatTilingData, tilingDataIn, tiling);
    const ForeachFlatTilingData* __restrict tilingData = &tilingDataIn;
    ForeachErfc::ForeachErfcRegbase<T, ForeachFlatTilingData> op;
    op.Init(x, y, tilingData, pipe);
    op.Process();
}

template <uint32_t schMode>
__global__ __aicore__ void foreach_erfc(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachFlatTilingData);
    GET_TILING_DATA_MEMBER(ForeachFlatTilingData, tileElems, scalarHeader, tiling);
    if ((scalarHeader & FOREACH_ERFC_SCALAR_FLAG) != 0) {
        const uint32_t count = scalarHeader & ~FOREACH_ERFC_SCALAR_FLAG;
        if constexpr (schMode == FOREACH_ERFC_TPL_SCH_MODE_FP16) {
            Simt::VF_CALL<ErfcScalarListVF<half>>(Simt::Dim3(32), x, y, count);
        } else if constexpr (schMode == FOREACH_ERFC_TPL_SCH_MODE_FP32) {
            Simt::VF_CALL<ErfcScalarListVF<float>>(Simt::Dim3(32), x, y, count);
        } else if constexpr (schMode == FOREACH_ERFC_TPL_SCH_MODE_BF16) {
            Simt::VF_CALL<ErfcScalarListVF<bfloat16_t>>(Simt::Dim3(32), x, y, count);
        }
        return;
    }
    TPipe pipeOp;
    if constexpr (schMode == FOREACH_ERFC_TPL_SCH_MODE_FP16) {
        ForeachErfcImpl<half>(x, y, tiling, &pipeOp);
    } else if constexpr (schMode == FOREACH_ERFC_TPL_SCH_MODE_FP32) {
        ForeachErfcImpl<float>(x, y, tiling, &pipeOp);
    } else if constexpr (schMode == FOREACH_ERFC_TPL_SCH_MODE_BF16) {
        ForeachErfcImpl<bfloat16_t>(x, y, tiling, &pipeOp);
    }
    pipeOp.Destroy();
}
