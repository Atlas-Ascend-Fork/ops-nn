/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BN3D_TRAINING_REDUCE_SHORT_ROWS_H_
#define BN3D_TRAINING_REDUCE_SHORT_ROWS_H_

#include "bn3d_training_reduce_recover_sum.h"
#include "bn3d_training_reduce_tiling_data.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"

namespace BN3DTrainingReduceOps {

struct CompensatedValue {
    float hi;
    float lo;

    __simt_callee__ inline void Add(float x)
    {
        float next = hi + x;
        // TwoSum keeps the lost low bits, including after cancellation. Infinite
        // square sums remain infinite; NaN input still propagates through hi.
        float delta = next - hi;
        float error = (hi - (next - delta)) + (x - delta);
        lo += (__float_as_uint(next) & FLOAT_INF_BITS) == FLOAT_INF_BITS ? 0.0f : error;
        hi = next;
    }
};

struct RowPartial {
    CompensatedValue sum;
    CompensatedValue squares;
};
static_assert(sizeof(RowPartial) == ROW_PARTIAL_BYTES, "Host and kernel partial sizes must agree");

template <typename T>
__simt_callee__ inline float ReadRowValue(__gm__ T* x, uint32_t i)
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return __uint_as_float(static_cast<uint32_t>(reinterpret_cast<__gm__ uint16_t*>(x)[i]) << 16U);
    } else {
        return static_cast<float>(x[i]);
    }
}

template <typename T>
__simt_callee__ inline void StoreRowResult(__gm__ T* x, __gm__ float* sums, __gm__ float* squares, RowPartial value,
                                           uint32_t c, uint32_t n, uint32_t channels, uint32_t spatial)
{
    float sum = value.sum.hi + value.sum.lo;
    if constexpr (!AscendC::IsSameType<T, half>::value) {
        if ((__float_as_uint(sum) & FLOAT_INF_BITS) == FLOAT_INF_BITS) {
            WideSum wide{};
            for (uint32_t row = 0; row < n; ++row) {
                uint32_t base = (row * channels + c) * spatial;
                for (uint32_t k = 0; k < spatial; ++k) {
                    wide.AddBits(__float_as_uint(ReadRowValue(x, base + k)));
                }
            }
            sum = __uint_as_float(wide.RoundBits());
        }
    }
    sums[c] = sum;
    squares[c] = value.squares.hi + value.squares.lo;
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(ROW_THREADS) inline void ReduceRows(__gm__ T* x, __gm__ float* sums,
                                                                        __gm__ float* squares,
                                                                        __gm__ RowPartial* partials, uint32_t n,
                                                                        uint32_t channels, uint32_t spatial,
                                                                        uint32_t parts, uint32_t grid)
{
    uint32_t tid = AscendC::Simt::GetBlockIdx() * ROW_THREADS + AscendC::Simt::GetThreadIdx();
    for (uint32_t i = tid; i < parts * channels; i += grid * ROW_THREADS) {
        uint32_t part = i / channels;
        uint32_t c = i - part * channels;
        RowPartial value{};
        if (channels <= ROW_WARP) {
            uint32_t firstRow = part / spatial;
            uint32_t k = part - firstRow * spatial;
            uint32_t rowStep = parts / spatial;
            for (uint32_t row = firstRow; row < n; row += rowStep) {
                float v = ReadRowValue(x, (row * channels + c) * spatial + k);
                value.sum.Add(v);
                value.squares.Add(v * v);
            }
        } else {
            const uint32_t rowsPerPart = (n + parts - 1) / parts;
            const uint32_t firstRow = part * rowsPerPart;
            const uint32_t lastRow = firstRow + rowsPerPart < n ? firstRow + rowsPerPart : n;
            if (spatial == 1) {
                for (uint32_t row = firstRow; row < lastRow; ++row) {
                    float v = ReadRowValue(x, row * channels + c);
                    value.sum.Add(v);
                    value.squares.Add(v * v);
                }
            } else {
                for (uint32_t row = firstRow; row < lastRow; ++row) {
                    uint32_t base = (row * channels + c) * spatial;
                    for (uint32_t k = 0; k < spatial; ++k) {
                        float v = ReadRowValue(x, base + k);
                        value.sum.Add(v);
                        value.squares.Add(v * v);
                    }
                }
            }
        }
        if (parts == 1) {
            StoreRowResult(x, sums, squares, value, c, n, channels, spatial);
        } else {
            partials[i].sum.hi = value.sum.hi;
            partials[i].sum.lo = value.sum.lo;
            partials[i].squares.hi = value.squares.hi;
            partials[i].squares.lo = value.squares.lo;
        }
    }
}

__simt_callee__ inline void AddPartial(RowPartial& dst, float sh, float sl, float qh, float ql)
{
    dst.sum.Add(sh);
    dst.sum.Add(sl);
    dst.squares.Add(qh);
    dst.squares.Add(ql);
}

__simt_callee__ inline void FoldWarp(RowPartial& value, uint32_t width)
{
    for (uint32_t span = width / 2; span > 0; span /= 2) {
        float sh = asc_shfl_down(value.sum.hi, span, width);
        float sl = asc_shfl_down(value.sum.lo, span, width);
        float qh = asc_shfl_down(value.squares.hi, span, width);
        float ql = asc_shfl_down(value.squares.lo, span, width);
        AddPartial(value, sh, sl, qh, ql);
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(ROW_THREADS) inline void MergeRowPartitions(
    __gm__ T* x, __gm__ float* sums, __gm__ float* squares, __gm__ RowPartial* partials, __ubuf__ RowPartial* warpSums,
    uint32_t n, uint32_t channels, uint32_t spatial, uint32_t parts, uint32_t grid)
{
    const uint32_t thread = AscendC::Simt::GetThreadIdx();
    const uint32_t block = AscendC::Simt::GetBlockIdx();
    const uint32_t tid = block * ROW_THREADS + thread;
    const bool narrow = channels <= ROW_WARP;
    // Narrow channels: the entire block merges one long column of partials.
    // Wide channels: distribute adjacent channel outputs across a warp.
    const uint32_t first = narrow ? block : tid;
    const uint32_t stride = narrow ? grid : grid * ROW_THREADS;
    for (uint32_t c = first; c < channels; c += stride) {
        RowPartial value{};
        for (uint32_t p = narrow ? thread : 0U; p < parts; p += narrow ? ROW_THREADS : 1U) {
            uint32_t i = p * channels + c;
            AddPartial(value, partials[i].sum.hi, partials[i].sum.lo, partials[i].squares.hi, partials[i].squares.lo);
        }
        if (narrow) {
            FoldWarp(value, ROW_WARP);
            if (thread % ROW_WARP == 0) {
                uint32_t w = thread / ROW_WARP;
                warpSums[w].sum.hi = value.sum.hi;
                warpSums[w].sum.lo = value.sum.lo;
                warpSums[w].squares.hi = value.squares.hi;
                warpSums[w].squares.lo = value.squares.lo;
            }
            asc_syncthreads();
            constexpr uint32_t warps = ROW_THREADS / ROW_WARP;
            if (thread < warps) {
                value.sum.hi = warpSums[thread].sum.hi;
                value.sum.lo = warpSums[thread].sum.lo;
                value.squares.hi = warpSums[thread].squares.hi;
                value.squares.lo = warpSums[thread].squares.lo;
                FoldWarp(value, warps);
            }
            asc_syncthreads();
        }
        if (!narrow || thread == 0) {
            StoreRowResult(x, sums, squares, value, c, n, channels, spatial);
        }
    }
}

template <typename T>
__aicore__ inline void RunShortRows(GM_ADDR x, GM_ADDR sum, GM_ADDR squares, GM_ADDR workspace,
                                    const BN3DTrainingReduceDenseChannelTilingData& d)
{
    AscendC::Simt::VF_CALL<ReduceRows<T>>(
        AscendC::Simt::Dim3(ROW_THREADS), (__gm__ T*)x, (__gm__ float*)sum, (__gm__ float*)squares,
        (__gm__ RowPartial*)workspace, static_cast<uint32_t>(d.numN), static_cast<uint32_t>(d.numC),
        static_cast<uint32_t>(d.numR0), static_cast<uint32_t>(d.rowPartitions), static_cast<uint32_t>(d.usedCoreNum));
    if (d.rowPartitions > 1) {
        AscendC::SyncAll<true>();
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> buffer;
        pipe.InitBuffer(buffer, ROW_MERGE_UB_BYTES);
        auto warpSums = (__ubuf__ RowPartial*)buffer.Get<float>().GetPhyAddr();
        AscendC::Simt::VF_CALL<MergeRowPartitions<T>>(
            AscendC::Simt::Dim3(ROW_THREADS), (__gm__ T*)x, (__gm__ float*)sum, (__gm__ float*)squares,
            (__gm__ RowPartial*)workspace, warpSums, static_cast<uint32_t>(d.numN), static_cast<uint32_t>(d.numC),
            static_cast<uint32_t>(d.numR0), static_cast<uint32_t>(d.rowPartitions),
            static_cast<uint32_t>(d.usedCoreNum));
    }
}
} // namespace BN3DTrainingReduceOps
#endif
