/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "foreach_sigmoid_regbase.h"
#include "foreach_sigmoid_tiling_key.h"
#include "../../foreach_utils/arch35/foreach_flat_tiling_data.h"

#include "simt_api/math_functions.h"
using namespace AscendC;
namespace ForeachSigmoidScalars {
template <typename T>
__simt_callee__ inline __gm__ T* TensorAddress(GM_ADDR list, uint32_t index)
{
    auto words = reinterpret_cast<__gm__ uint64_t*>(list);
    auto pointers = words + (words[0] >> 3);
    return reinterpret_cast<__gm__ T*>(pointers[index]);
}
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void Run(GM_ADDR input, GM_ADDR output, uint32_t count)
{
    for (uint32_t index = Simt::GetThreadIdx(); index < count; index += Simt::GetThreadNum()) {
        const float value = static_cast<float>(*TensorAddress<T>(input, index));
        *TensorAddress<T>(output, index) = static_cast<T>(1.0f / (1.0f + expf(-value)));
    }
}
} // namespace ForeachSigmoidScalars

template <uint32_t schMode>
__global__ __aicore__ void foreach_sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachFlatTilingData);
    (void)workspace;
    if constexpr (schMode == FOREACH_SIGMOID_SCH_MODE_SCALARS) {
        // Avoid copying the complete list/core tiling arrays for a scalar VF.
        GET_TILING_DATA_MEMBER(ForeachFlatTilingData, tileElems, scalarCount, tiling);
        Simt::VF_CALL<ForeachSigmoidScalars::Run<DTYPE_X>>(Simt::Dim3(32), x, y, scalarCount);
    } else {
        GET_TILING_DATA_WITH_STRUCT(ForeachFlatTilingData, td, tiling);
        TPipe pipe;
        ForeachSigmoid::FlatKernel<DTYPE_X, ForeachFlatTilingData> op;
        op.Init(&td, &pipe);
        op.Process(x, y);
        pipe.Destroy();
    }
}
