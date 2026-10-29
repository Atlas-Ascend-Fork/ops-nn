/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ELEMENTS_V2_SMALL_SHAPE_H_
#define SCATTER_ELEMENTS_V2_SMALL_SHAPE_H_

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "scatter_elements_v2_small_shape_tiling.h"

namespace ScatterElements {
constexpr uint32_t SMALL_SHAPE_MAX_THREADS = 1024;

template <typename T, typename IndexT, uint32_t Rank, uint32_t Dim, bool SingleRow = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(SMALL_SHAPE_MAX_THREADS) inline void ScatterSmallShapeNone(
    __gm__ T* var, __gm__ const IndexT* indices, __gm__ const T* updates, uint32_t count, uint32_t dataStride,
    uint32_t indicesStride, uint32_t updatesStride)
{
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        if constexpr (Rank == 1) {
            var[static_cast<uint32_t>(indices[i])] = updates[i];
        } else {
            uint32_t row = SingleRow ? 0 : i / indicesStride;
            uint32_t col = i - row * indicesStride;
            uint32_t target;
            if constexpr (Dim == 0) {
                target = static_cast<uint32_t>(indices[i]) * dataStride + col;
            } else {
                target = row * dataStride + static_cast<uint32_t>(indices[i]);
            }
            var[target] = updates[row * updatesStride + col];
        }
    }
}

template <typename T, typename IndexT>
__aicore__ inline void LaunchScatterSmallShapeNone(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR tiling)
{
    auto hdr = reinterpret_cast<__gm__ const ScatterElementsV2SmallShapeTiling*>(tiling);
    auto dst = reinterpret_cast<__gm__ T*>(var);
    auto idx = reinterpret_cast<__gm__ const IndexT*>(indices);
    auto src = reinterpret_cast<__gm__ const T*>(updates);
    if (hdr->rank == 1) {
        asc_vf_call<ScatterSmallShapeNone<T, IndexT, 1, 0>>(dim3(hdr->threads), dst, idx, src, hdr->allAxis, 1, 1, 1);
    } else if (hdr->allAxis <= hdr->indicesStride && hdr->dim == 0) {
        asc_vf_call<ScatterSmallShapeNone<T, IndexT, 2, 0, true>>(
            dim3(hdr->threads), dst, idx, src, hdr->allAxis, hdr->dataStride, hdr->indicesStride, hdr->updatesStride);
    } else if (hdr->allAxis <= hdr->indicesStride) {
        asc_vf_call<ScatterSmallShapeNone<T, IndexT, 2, 1, true>>(
            dim3(hdr->threads), dst, idx, src, hdr->allAxis, hdr->dataStride, hdr->indicesStride, hdr->updatesStride);
    } else if (hdr->dim == 0) {
        asc_vf_call<ScatterSmallShapeNone<T, IndexT, 2, 0>>(dim3(hdr->threads), dst, idx, src, hdr->allAxis,
                                                            hdr->dataStride, hdr->indicesStride, hdr->updatesStride);
    } else {
        asc_vf_call<ScatterSmallShapeNone<T, IndexT, 2, 1>>(dim3(hdr->threads), dst, idx, src, hdr->allAxis,
                                                            hdr->dataStride, hdr->indicesStride, hdr->updatesStride);
    }
}
} // namespace ScatterElements

#endif
