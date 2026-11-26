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
 * \file foreach_maximum_scalar_list_simt.h
 * \brief SIMT kernel implementation for foreach_maximum_scalar_list
 */

#ifndef FOREACH_MAXIMUM_SCALAR_LIST_SIMT_H
#define FOREACH_MAXIMUM_SCALAR_LIST_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_functions.h"
#include "foreach_maximum_scalar_list_tiling_data.h"
#include "foreach_maximum_scalar_list_tiling_key.h"
#include <type_traits>

namespace NsForeachMaximumScalarList {

constexpr int TENSOR_PTR_SHIFT = 3;

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

template <typename T>
__simt_callee__ inline __gm__ T* SimtGetTensorAddr(GM_ADDR tensorListPtr, int64_t idx)
{
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorListPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> TENSOR_PTR_SHIFT);
    return reinterpret_cast<__gm__ T*>(*(tensorPtr + idx));
}

template <typename T>
struct ComputeType {
    using type = T;
};
template <>
struct ComputeType<half> {
    using type = float;
};
template <>
struct ComputeType<bfloat16_t> {
    using type = float;
};
template <>
struct ComputeType<int32_t> {
    using type = int64_t;
};

__simt_callee__ inline float SimtMaximum(float a, float b)
{
    float m = AscendC::Simt::Max(a, b);
    float r = (a != a) ? a : m;
    r = (b != b) ? b : r;
    r = (a == 0.0f && b == 0.0f) ? (a + b) : r;
    return r;
}

__simt_callee__ inline int64_t SimtMaximum(int64_t a, int64_t b) { return (a >= b) ? a : b; }

template <typename IDX_T>
struct MakeUnsigned;
template <>
struct MakeUnsigned<int32_t> {
    using type = uint32_t;
};
template <>
struct MakeUnsigned<int64_t> {
    using type = uint64_t;
};

constexpr int64_t SMALL_MAP_MAX_AVG = 8;
// TensorMap assigns a whole tensor to one SIMT lane and walks its elements
// serially; it is only beneficial when every tensor is tiny. A single large
// tensor (e.g. [2048, 0, ..., 0]) must not be pushed through it, otherwise the
// work degenerates to one serial lane while the other lanes idle.
constexpr int64_t SMALL_MAP_MAX_ELEMS = 8;

template <typename T, typename S>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpForeachMaximumScalarListTensorMap(
    int32_t tensorCount, GM_ADDR xList, GM_ADDR scalars, GM_ADDR yList, GM_ADDR tiling)
{
    using C = typename ComputeType<T>::type;
    __gm__ ForeachMaximumScalarListTilingData* tilingGm = reinterpret_cast<__gm__ ForeachMaximumScalarListTilingData*>(
        tiling);
    __gm__ S* scalarsGm = reinterpret_cast<__gm__ S*>(scalars);
    uint64_t tid = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
                                         AscendC::Simt::GetThreadIdx());
    uint64_t stride = static_cast<uint64_t>(AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());
    for (int32_t t = static_cast<int32_t>(tid); t < tensorCount; t += static_cast<int32_t>(stride)) {
        int64_t cnt = tilingGm->tensorElements[t];
        if (cnt <= 0) {
            continue;
        }
        __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);
        C sVal = static_cast<C>(scalarsGm[t]);
        for (int64_t k = 0; k < cnt; ++k) {
            C xVal = static_cast<C>(xData[k]);
            yData[k] = static_cast<T>(SimtMaximum(xVal, sVal));
        }
    }
}

template <typename T, typename S, typename IDX_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpForeachMaximumScalarListPerTensor(
    int32_t tensorId, IDX_T count, GM_ADDR xList, GM_ADDR scalars, GM_ADDR yList)
{
    using C = typename ComputeType<T>::type;
    using UIDX_T = typename MakeUnsigned<IDX_T>::type;
    __gm__ T* xData = SimtGetTensorAddr<T>(xList, tensorId);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, tensorId);
    __gm__ S* scalarsGm = reinterpret_cast<__gm__ S*>(scalars);
    C sVal = static_cast<C>(scalarsGm[tensorId]);
    UIDX_T stride = static_cast<UIDX_T>(AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());
    UIDX_T start = static_cast<UIDX_T>(AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
                                       AscendC::Simt::GetThreadIdx());
    UIDX_T cnt = static_cast<UIDX_T>(count);
    if constexpr (std::is_same<T, float>::value) {
        UIDX_T vecCnt = cnt >> 2;
        __gm__ float4* x4 = reinterpret_cast<__gm__ float4*>(xData);
        __gm__ float4* y4 = reinterpret_cast<__gm__ float4*>(yData);
        for (UIDX_T i = start; i < vecCnt; i += stride) {
            float4 v = x4[i];
            v[0] = SimtMaximum(v[0], sVal);
            v[1] = SimtMaximum(v[1], sVal);
            v[2] = SimtMaximum(v[2], sVal);
            v[3] = SimtMaximum(v[3], sVal);
            y4[i] = v;
        }
        for (UIDX_T idx = (vecCnt << 2) + start; idx < cnt; idx += stride) {
            C xVal = static_cast<C>(xData[idx]);
            yData[idx] = static_cast<T>(SimtMaximum(xVal, sVal));
        }
        return;
    }
    for (UIDX_T idx = start; idx < cnt; idx += stride) {
        C xVal = static_cast<C>(xData[idx]);
        yData[idx] = static_cast<T>(SimtMaximum(xVal, sVal));
    }
}

constexpr uint32_t PACKED_SCALAR_THREAD_NUM = 4;

template <typename T, typename S>
__simt_vf__ __aicore__ LAUNCH_BOUND(PACKED_SCALAR_THREAD_NUM) inline void OpForeachMaximumScalarListPackedScalar(
    int32_t firstTensorId, int32_t tensorCount, GM_ADDR xList, GM_ADDR scalars, GM_ADDR yList)
{
    using C = typename ComputeType<T>::type;
    int32_t tensorId = firstTensorId + static_cast<int32_t>(AscendC::Simt::GetThreadIdx());
    if (tensorId < tensorCount) {
        __gm__ T* xData = SimtGetTensorAddr<T>(xList, tensorId);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, tensorId);
        __gm__ S* scalarsGm = reinterpret_cast<__gm__ S*>(scalars);
        C xVal = static_cast<C>(xData[0]);
        C sVal = static_cast<C>(scalarsGm[tensorId]);
        yData[0] = static_cast<T>(SimtMaximum(xVal, sVal));
    }
}

template <typename T, typename S>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR scalars, GM_ADDR y,
                               const ForeachMaximumScalarListTilingData* tilingData, GM_ADDR tiling)
{
    if (tilingData->tensorCount <= 0 || tilingData->totalElements <= 0) {
        return;
    }
    int64_t blockNum = tilingData->needCoreNum > 0 ? tilingData->needCoreNum : 1;
    int64_t perBlock = (tilingData->totalElements + blockNum - 1) / blockNum;
    uint32_t threads = static_cast<uint32_t>(((perBlock + 31) / 32) * 32);
    if (threads < 32) {
        threads = 32;
    }
    if (threads > THREAD_NUM) {
        threads = THREAD_NUM;
    }
    bool allScalar = true;
    int64_t maxElems = 0;
    for (int32_t t = 0; t < tilingData->tensorCount; ++t) {
        const int64_t count = tilingData->tensorElements[t];
        if (count != 1) {
            allScalar = false;
        }
        if (count > maxElems) {
            maxElems = count;
        }
    }
    if (allScalar) {
        const int32_t stride = static_cast<int32_t>(blockNum) * static_cast<int32_t>(PACKED_SCALAR_THREAD_NUM);
        for (int32_t firstTensorId =
                 static_cast<int32_t>(AscendC::GetBlockIdx()) * static_cast<int32_t>(PACKED_SCALAR_THREAD_NUM);
             firstTensorId < tilingData->tensorCount; firstTensorId += stride) {
            AscendC::Simt::VF_CALL<OpForeachMaximumScalarListPackedScalar<T, S>>(
                AscendC::Simt::Dim3(PACKED_SCALAR_THREAD_NUM), firstTensorId, tilingData->tensorCount, x, scalars, y);
        }
        return;
    }
    if (maxElems <= SMALL_MAP_MAX_ELEMS &&
        tilingData->totalElements <= static_cast<int64_t>(tilingData->tensorCount) * SMALL_MAP_MAX_AVG) {
        AscendC::Simt::VF_CALL<OpForeachMaximumScalarListTensorMap<T, S>>(
            AscendC::Simt::Dim3(threads), tilingData->tensorCount, x, scalars, y, tiling);
        return;
    }
    for (int32_t tensorId = 0; tensorId < tilingData->tensorCount; ++tensorId) {
        int64_t count = tilingData->tensorElements[tensorId];
        if (count <= 0) {
            continue;
        }
        uint32_t perThreads = static_cast<uint32_t>(((count + 31) / 32) * 32);
        if (perThreads < 32) {
            perThreads = 32;
        }
        if (perThreads > THREAD_NUM) {
            perThreads = THREAD_NUM;
        }
        if (count <= static_cast<int64_t>(2147483647)) {
            AscendC::Simt::VF_CALL<OpForeachMaximumScalarListPerTensor<T, S, int32_t>>(
                AscendC::Simt::Dim3(perThreads), tensorId, static_cast<int32_t>(count), x, scalars, y);
        } else {
            AscendC::Simt::VF_CALL<OpForeachMaximumScalarListPerTensor<T, S, int64_t>>(AscendC::Simt::Dim3(perThreads),
                                                                                       tensorId, count, x, scalars, y);
        }
    }
}

} // namespace NsForeachMaximumScalarList

#endif // FOREACH_MAXIMUM_SCALAR_LIST_SIMT_H
