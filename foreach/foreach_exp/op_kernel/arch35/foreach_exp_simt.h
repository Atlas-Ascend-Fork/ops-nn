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
 * \file foreach_exp_simt.h
 * \brief SIMT kernel implementation for foreach_exp
 */

#ifndef FOREACH_EXP_SIMT_H
#define FOREACH_EXP_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "foreach_exp_tiling_data.h"
#include "foreach_exp_tiling_key.h"

namespace NsForeachExp {

constexpr int TENSOR_PTR_SHIFT = 3;

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

/**
 * \brief Get the GM address of the idx-th tensor in a TensorList
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
 * \brief Promote half/bf16 to float32 for precision
 */
template <typename T>
__simt_callee__ inline float PromoteToFloat(T val)
{
    return static_cast<float>(val);
}

template <>
__simt_callee__ inline float PromoteToFloat<half>(half val)
{
    return static_cast<float>(val);
}

template <>
__simt_callee__ inline float PromoteToFloat<bfloat16_t>(bfloat16_t val)
{
    return static_cast<float>(val);
}

template <>
__simt_callee__ inline float PromoteToFloat<float>(float val)
{
    return val;
}

template <typename X_T, typename Y_T>
__simt_callee__ inline void ComputeExp(__gm__ X_T* xData, __gm__ Y_T* yData, int64_t idx)
{
    X_T xVal = xData[idx];
    float xFloat = PromoteToFloat<X_T>(xVal);
    float yFloat = expf(xFloat);
    yData[idx] = static_cast<Y_T>(yFloat);
}

/**
 * \brief 最大 i 满足 prefix[i] <= flat（跳过空 tensor）
 */
__simt_callee__ inline int32_t FindTensorId(__gm__ int64_t* prefix, int32_t tensorCount, int64_t flat)
{
    int32_t lo = 0;
    int32_t hi = tensorCount - 1;
    while (lo < hi) {
        int32_t mid = (lo + hi + 1) >> 1;
        if (prefix[mid] <= flat) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

/**
 * \brief block-cyclic tile：block b 处理 tile b, b+blockDim, ...，一轮下来所有 block
 *        合起来覆盖一段连续窗口；tile 落在一张 tensor 内时走无 mapping 快速路径。
 */
template <typename X_T, typename Y_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpForeachExpTiled(int64_t totalElements,
                                                                              int32_t tensorCount, int32_t tileElements,
                                                                              GM_ADDR xList, GM_ADDR yList,
                                                                              GM_ADDR prefixAddr)
{
    if (tensorCount <= 0 || tileElements <= 0) {
        return;
    }
    __gm__ int64_t* prefix = reinterpret_cast<__gm__ int64_t*>(prefixAddr);
    const int64_t blockNum = static_cast<int64_t>(AscendC::Simt::GetBlockNum());
    const int64_t blockIdx = static_cast<int64_t>(AscendC::Simt::GetBlockIdx());
    const int64_t threadIdx = static_cast<int64_t>(AscendC::Simt::GetThreadIdx());
    const int64_t threadStride = static_cast<int64_t>(THREAD_NUM);
    const int64_t tileElems = static_cast<int64_t>(tileElements);
    const int64_t numTiles = (totalElements + tileElems - 1) / tileElems;
    for (int64_t tile = blockIdx; tile < numTiles; tile += blockNum) {
        int64_t tileBegin = tile * tileElems;
        int64_t tileEnd = tileBegin + tileElems;
        if (tileEnd > totalElements) {
            tileEnd = totalElements;
        }
        if (tileBegin >= tileEnd) {
            continue;
        }

        const int32_t tileTensor = FindTensorId(prefix, tensorCount, tileBegin);
        const int64_t tileTensorEnd = prefix[tileTensor + 1];
        if (tileEnd <= tileTensorEnd) {
            __gm__ X_T* xData = SimtGetTensorAddr<X_T>(xList, tileTensor);
            __gm__ Y_T* yData = SimtGetTensorAddr<Y_T>(yList, tileTensor);
            const int64_t localBegin = tileBegin - prefix[tileTensor];
            const int64_t tileLen = tileEnd - tileBegin;
            for (int64_t i = threadIdx; i < tileLen; i += threadStride) {
                ComputeExp<X_T, Y_T>(xData, yData, localBegin + i);
            }
            continue;
        }

        int64_t off = tileBegin + threadIdx;
        if (off >= tileEnd) {
            continue;
        }
        int32_t tensorId = FindTensorId(prefix, tensorCount, off);
        int64_t tensorBegin = prefix[tensorId];
        int64_t tensorEnd = prefix[tensorId + 1];
        __gm__ X_T* xData = SimtGetTensorAddr<X_T>(xList, tensorId);
        __gm__ Y_T* yData = SimtGetTensorAddr<Y_T>(yList, tensorId);
        while (off < tileEnd) {
            while (off >= tensorEnd) {
                ++tensorId;
                tensorBegin = tensorEnd;
                tensorEnd = prefix[tensorId + 1];
                xData = SimtGetTensorAddr<X_T>(xList, tensorId);
                yData = SimtGetTensorAddr<Y_T>(yList, tensorId);
            }
            ComputeExp<X_T, Y_T>(xData, yData, off - tensorBegin);
            off += threadStride;
        }
    }
}

/**
 * \brief Process entry: 所有列表统一走自适应 block-cyclic tile
 */
template <typename X_T, typename Y_T>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR y, const ForeachExpTilingData* tilingGm, GM_ADDR tilingRaw)
{
    const int64_t totalElements = tilingGm->totalElements;
    if (totalElements <= 0) {
        return;
    }
    AscendC::Simt::VF_CALL<OpForeachExpTiled<X_T, Y_T>>(AscendC::Simt::Dim3(THREAD_NUM), totalElements,
                                                        tilingGm->tensorCount, tilingGm->tileElements, x, y, tilingRaw);
}

} // namespace NsForeachExp

#endif // FOREACH_EXP_SIMT_H
