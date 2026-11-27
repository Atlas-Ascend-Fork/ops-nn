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
 * \file foreach_asin_simt.h
 * \brief SIMT kernel implementation for foreach_asin
 */

#ifndef FOREACH_ASIN_SIMT_H
#define FOREACH_ASIN_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "foreach_asin_tiling_data.h"
#include "foreach_asin_tiling_key.h"
#include "simt_api/math_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace NsForeachAsin {
using namespace AscendC;

constexpr int TENSOR_PTR_SHIFT = 3;

constexpr uint32_t THREAD_NUM = 512;
constexpr int64_t UNROLL_COUNT = 4;

constexpr int64_t FLAT_SEG_MAX_ELEMENTS = 8;

/**
 * \brief 获取 TensorList 中第 idx 个 tensor 的 GM 地址
 */
template <typename T>
__simt_callee__ inline __gm__ T* SimtGetTensorAddr(GM_ADDR tensorListPtr, int64_t idx)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> TENSOR_PTR_SHIFT);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

/**
 * @brief 对单个元素计算 asin，支持类型提升
 * @tparam T 数据类型（float/half/bfloat16_t）
 */
template <typename T>
__simt_callee__ inline T ComputeAsin(T val)
{
    if constexpr (std::is_same<T, float>::value) {
        return asinf(val);
    } else if constexpr (std::is_same<T, half>::value) {
        float fVal = __half2float(val);
        float result = asinf(fVal);
        return __float2half(result);
    } else {
        // bfloat16_t
        float fVal = __bfloat162float(val);
        float result = asinf(fVal);
        return __float2bfloat16(result);
    }
}

/**
 * @brief 单个 tensor 的一段连续元素：每线程先发 UNROLL_COUNT 个独立 load，再统一计算/写回
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ForeachAsinTensorSegSimt(int32_t tensorId,
                                                                                     int64_t localStart, int64_t count,
                                                                                     GM_ADDR xList, GM_ADDR yList)
{
    __gm__ T* xData = SimtGetTensorAddr<T>(xList, tensorId);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, tensorId);

    const int64_t stride = static_cast<int64_t>(Simt::GetThreadNum());
    const int64_t end = localStart + count;
    const int64_t unrollEnd = end - static_cast<int64_t>(UNROLL_COUNT - 1) * stride;
    int64_t idx = localStart + static_cast<int64_t>(Simt::GetThreadIdx());
    for (; idx < unrollEnd; idx += UNROLL_COUNT * stride) {
        T val[UNROLL_COUNT];
#pragma unroll
        for (int64_t k = 0; k < UNROLL_COUNT; k++) {
            val[k] = xData[idx + k * stride];
        }
#pragma unroll
        for (int64_t k = 0; k < UNROLL_COUNT; k++) {
            yData[idx + k * stride] = ComputeAsin<T>(val[k]);
        }
    }
    for (; idx < end; idx += stride) {
        yData[idx] = ComputeAsin<T>(xData[idx]);
    }
}

/**
 * @brief 小段 flat VF：一个 VF 覆盖本核区间内所有 tensor，线程用前缀表二分定位自己的 tensor
 */
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ForeachAsinFlatSegSimt(int64_t coreStart, int64_t coreEnd,
                                                                                   GM_ADDR xList, GM_ADDR yList,
                                                                                   GM_ADDR tiling)
{
    __gm__ const ForeachAsinTilingData* tilingGm = reinterpret_cast<__gm__ const ForeachAsinTilingData*>(tiling);
    const int32_t tensorNum = tilingGm->tensorNum;
    const int64_t stride = static_cast<int64_t>(Simt::GetThreadNum());
    for (int64_t flat = coreStart + static_cast<int64_t>(Simt::GetThreadIdx()); flat < coreEnd; flat += stride) {
        int32_t lo = 0;
        int32_t hi = tensorNum - 1;
        while (lo < hi) {
            const int32_t mid = (lo + hi + 1) >> 1;
            if (tilingGm->tensorCumulativeOffset[mid] <= flat) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        const int64_t localIdx = flat - tilingGm->tensorCumulativeOffset[lo];
        __gm__ T* xData = SimtGetTensorAddr<T>(xList, lo);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, lo);
        yData[localIdx] = ComputeAsin<T>(xData[localIdx]);
    }
}

/**
 * @brief Process 函数：按 totalElements 统一分核，小段一次 flat VF，大段逐 tensor 发 U 展开 VF
 */
template <typename T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, const ForeachAsinTilingData* tilingData)
{
    const int64_t totalElements = tilingData->totalElements;
    if (totalElements <= 0) {
        return;
    }

    const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
    if (blockIdx >= tilingData->needCoreNum) {
        return;
    }
    const int64_t coreStart = static_cast<int64_t>(blockIdx) * tilingData->perCoreElements;
    if (coreStart >= totalElements) {
        return;
    }
    int64_t coreEnd;
    if (blockIdx == tilingData->needCoreNum - 1) {
        coreEnd = coreStart + tilingData->lastCoreElements;
    } else {
        coreEnd = coreStart + tilingData->perCoreElements;
    }
    if (coreEnd > totalElements) {
        coreEnd = totalElements;
    }
    if (coreStart >= coreEnd) {
        return;
    }

    if (coreEnd - coreStart <= FLAT_SEG_MAX_ELEMENTS) {
        Simt::VF_CALL<ForeachAsinFlatSegSimt<T>>(Simt::Dim3(THREAD_NUM), coreStart, coreEnd, x, y, tiling);
        return;
    }

    int32_t lo = 0;
    int32_t hi = tilingData->tensorNum - 1;
    while (lo < hi) {
        const int32_t mid = (lo + hi + 1) >> 1;
        if (tilingData->tensorCumulativeOffset[mid] <= coreStart) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    for (int32_t t = lo; t < tilingData->tensorNum; t++) {
        const int64_t tensorStart = tilingData->tensorCumulativeOffset[t];
        if (tensorStart >= coreEnd) {
            break;
        }
        const int64_t tensorEnd = tilingData->tensorCumulativeOffset[t + 1];
        const int64_t segStart = (coreStart > tensorStart) ? coreStart : tensorStart;
        const int64_t segEnd = (coreEnd < tensorEnd) ? coreEnd : tensorEnd;
        if (segStart < segEnd) {
            Simt::VF_CALL<ForeachAsinTensorSegSimt<T>>(Simt::Dim3(THREAD_NUM), t, segStart - tensorStart,
                                                       segEnd - segStart, x, y);
        }
    }
}

} // namespace NsForeachAsin

#endif // FOREACH_ASIN_SIMT_H
