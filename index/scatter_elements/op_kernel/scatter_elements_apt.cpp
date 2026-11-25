/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/scatter_elements_input_copy.h"
#include "../scatter_elements_v2/arch35/scatter_elements_v2_multi.h"
#include "../scatter_elements_v2/arch35/scatter_elements_v2_deterministic.h"
#include "../scatter_elements_v2/arch35/scatter_elements_v2_small_shape.h"
#include "../scatter_elements_v2/arch35/scatter_elements_with_sorted.h"

namespace {
using namespace ScatterElements;

// Use the V2 key layout. The compiler needs literal keys during precompilation.
#define SCATTER_KEY_IMPL(P, W, I, R, D) P##0##W##I##R##D
#define SCATTER_KEY(P, W, I, R, D) SCATTER_KEY_IMPL(P, W, I, R, D)
#if ORIG_DTYPE_INDICES == DT_INT64
#define SCATTER_INDEX 1
#else
#define SCATTER_INDEX 0
#endif

#if ORIG_DTYPE_DATA == DT_INT8 || ORIG_DTYPE_DATA == DT_UINT8 || ORIG_DTYPE_DATA == DT_BOOL
#define SCATTER_WIDTH 01
#elif ORIG_DTYPE_DATA == DT_INT16 || ORIG_DTYPE_DATA == DT_FLOAT16 || ORIG_DTYPE_DATA == DT_BF16
#define SCATTER_WIDTH 02
#elif ORIG_DTYPE_DATA == DT_INT64 || ORIG_DTYPE_DATA == DT_DOUBLE
#define SCATTER_WIDTH 08
#else
#define SCATTER_WIDTH 04
#endif

#if ORIG_DTYPE_DATA == DT_FLOAT
#define SCATTER_DTYPE 00
#elif ORIG_DTYPE_DATA == DT_FLOAT16
#define SCATTER_DTYPE 01
#elif ORIG_DTYPE_DATA == DT_INT8
#define SCATTER_DTYPE 02
#elif ORIG_DTYPE_DATA == DT_INT32
#define SCATTER_DTYPE 03
#elif ORIG_DTYPE_DATA == DT_UINT8
#define SCATTER_DTYPE 04
#elif ORIG_DTYPE_DATA == DT_INT16
#define SCATTER_DTYPE 06
#elif ORIG_DTYPE_DATA == DT_INT64
#define SCATTER_DTYPE 09
#elif ORIG_DTYPE_DATA == DT_BOOL
#define SCATTER_DTYPE 12
#elif ORIG_DTYPE_DATA == DT_BF16
#define SCATTER_DTYPE 27
#endif

#define SCATTER_COPY_ONLY 4000000

#if ORIG_DTYPE_DATA == DT_BOOL
using Data = uint8_t;
using AddCast = half;
#else
using Data = DTYPE_DATA;
using AddCast = std::conditional_t<IsSameType<Data, int8_t>::value || IsSameType<Data, int16_t>::value, int32_t,
                                   uint32_t>;
#endif
using Index = DTYPE_INDICES;
using Payload = std::conditional_t<
    sizeof(Data) == 1, int8_t,
    std::conditional_t<sizeof(Data) == 2, int16_t, std::conditional_t<sizeof(Data) == 4, int32_t, int64_t>>>;

template <typename T, typename Offset, uint32_t REDU>
__aicore__ inline void ScatterTensor(TPipe& pipe, const ScatterElementsV2AscTilingData& tilingData, GM_ADDR y,
                                     GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace)
{
    if constexpr (REDU == REDU_ADD && (IsSameType<T, float>::value || IsSameType<T, half>::value ||
                                       IsSameType<T, bfloat16_t>::value || IsSameType<T, int64_t>::value)) {
        if (tilingData.isDeterministic != 0) {
            if (DeterministicAddTraits<T>::FIXED_FULL_WARP || tilingData.enableAdaptiveReduce != 0) {
                KernelScatterElementsDeterm<T, Index, Offset, REDU, 1, true> op(&tilingData, &pipe);
                op.Init(y, indices, updates, y);
                op.Process();
            } else {
                KernelScatterElementsDeterm<T, Index, Offset, REDU, 1> op(&tilingData, &pipe);
                op.Init(y, indices, updates, y);
                op.Process();
            }
            return;
        }
    }
    if constexpr (REDU == REDU_NONE) {
        if (tilingData.isDeterministic != 0) {
            KernelScatterElementsDeterm<T, Index, Offset, REDU, 1> op(&tilingData, &pipe);
            op.Init(y, indices, updates, y);
            op.Process();
            return;
        }
    }
    using Cast = std::conditional_t<REDU == REDU_ADD, AddCast, uint32_t>;
    KernelScatterElements<T, Index, Offset, Cast, REDU, 1> op(pipe);
    op.Init(y, indices, updates, y, workspace, &tilingData);
    op.Process();
}
} // namespace

extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR data, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TILING_KEY_IS(SCATTER_COPY_ONLY);
    TILING_KEY_IS(SCATTER_KEY(3, 0, SCATTER_INDEX, 0, SCATTER_WIDTH));
    TILING_KEY_IS(SCATTER_KEY(1, 0, SCATTER_INDEX, 0, SCATTER_WIDTH));
    TILING_KEY_IS(SCATTER_KEY(1, 1, SCATTER_INDEX, 0, SCATTER_WIDTH));
    TILING_KEY_IS(SCATTER_KEY(2, 0, SCATTER_INDEX, 0, SCATTER_WIDTH));
    TILING_KEY_IS(SCATTER_KEY(2, 1, SCATTER_INDEX, 0, SCATTER_WIDTH));
#ifdef SCATTER_DTYPE
    TILING_KEY_IS(SCATTER_KEY(1, 0, SCATTER_INDEX, 1, SCATTER_DTYPE));
    TILING_KEY_IS(SCATTER_KEY(1, 1, SCATTER_INDEX, 1, SCATTER_DTYPE));
#if ORIG_DTYPE_DATA != DT_BOOL
    TILING_KEY_IS(SCATTER_KEY(1, 0, SCATTER_INDEX, 2, SCATTER_DTYPE));
    TILING_KEY_IS(SCATTER_KEY(1, 1, SCATTER_INDEX, 2, SCATTER_DTYPE));
#endif
#if ORIG_DTYPE_DATA != DT_UINT8 && ORIG_DTYPE_DATA != DT_INT64 && ORIG_DTYPE_DATA != DT_BOOL
    TILING_KEY_IS(SCATTER_KEY(2, 0, SCATTER_INDEX, 1, SCATTER_DTYPE));
    TILING_KEY_IS(SCATTER_KEY(2, 1, SCATTER_INDEX, 1, SCATTER_DTYPE));
#endif
#endif

    TPipe pipe;
#if TILING_KEY_VAR == SCATTER_COPY_ONLY
    GET_TILING_DATA_WITH_STRUCT(ScatterElementsCopyTilingData, copyTiling, tiling);
    CopyScatterInput(pipe, data, y, copyTiling.dataBytes, false);
#else
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
#if TILING_KEY_VAR / 1000000 == 3
    GET_TILING_DATA_WITH_STRUCT(ScatterElementsV2SmallShapeTilingData, smallTiling, tiling);
    CopyScatterInput(pipe, data, y, smallTiling.dataBytes, true);
    LaunchScatterSmallShapeNone<Payload, Index>(y, indices, updates, &smallTiling, tiling);
#else
    GET_TILING_DATA_WITH_STRUCT(ScatterElementsV2AscTilingData, computeTiling, tiling);
    CopyScatterInput(pipe, data, y, static_cast<uint64_t>(computeTiling.dataAxis) * sizeof(Data), true);
    constexpr uint32_t reduction = (TILING_KEY_VAR / 100) % 10;
    using Offset = std::conditional_t<(TILING_KEY_VAR / 10000) % 10 == 0, uint32_t, uint64_t>;
    using T = std::conditional_t<reduction == REDU_NONE, Payload, Data>;
    if constexpr (TILING_KEY_VAR / 1000000 == 2) {
        constexpr uint32_t mode = reduction == REDU_NONE ? REDU_LAST : REDU_ADD;
        KernelScatterElementsWithSorted<T, Index, mode> op(&computeTiling, &pipe);
        op.Init(y, indices, updates, y, userWorkspace);
        op.Process();
    } else {
        ScatterTensor<T, Offset, reduction>(pipe, computeTiling, y, indices, updates, userWorkspace);
    }
#endif
#endif
}
