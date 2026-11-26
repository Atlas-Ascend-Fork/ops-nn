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
 * \file ascend_requant_flat_route.h
 * \brief Flat 1-D data-path route ids, budget constants and the structural req_scale classifier.
 *
 * The classifier depends only on the broadcast shapes and the derived strides of x / req_scale,
 * so host tiling and kernel share one definition of "scalar / periodic / contiguous-suffix / legacy".
 */

#ifndef ASCEND_REQUANT_FLAT_ROUTE_H
#define ASCEND_REQUANT_FLAT_ROUTE_H

#include <cstdint>

// This header is compiled by the host tiling translation unit as well, where the kernel-side
// function annotations are not defined: provide the no-op fallback so the shared helpers below can
// be called from both sides.
#ifndef __host_aicore__
#define __host_aicore__
#endif

// Route stored in AscendRequantTilingData::flat.route and consumed by the kernel entry.
constexpr int64_t ASCEND_REQUANT_ROUTE_SCALAR = 0;   // req_scale holds exactly one element
constexpr int64_t ASCEND_REQUANT_ROUTE_PERIODIC = 1; // flattening period L divides the vector length
constexpr int64_t ASCEND_REQUANT_ROUTE_CONTIG = 2;   // req_scale support is a contiguous x suffix
constexpr int64_t ASCEND_REQUANT_ROUTE_LEGACY = 3;   // keep the baseline ND data path

// int32 vector length of one vector register: VECTOR_REG_WIDTH (256 B) / sizeof(int32_t) on arch35.
constexpr int64_t ASCEND_REQUANT_FLAT_VEC_ELEMS = 64;
// Flat tiles are expressed in elements and aligned to this many elements (256 B of int32 input).
constexpr int64_t ASCEND_REQUANT_FLAT_ALIGN_ELEMS = 64;
// Declared lower bound of the flat tile extent whenever the element count allows it.
constexpr int64_t ASCEND_REQUANT_FLAT_MIN_TILE_ELEMS = 4096;
// A contiguous req_scale period shorter than this cannot keep the declared minimum tile extent.
constexpr int64_t ASCEND_REQUANT_FLAT_MIN_CONTIG_PERIOD = ASCEND_REQUANT_FLAT_MIN_TILE_ELEMS;
// Real per-element byte list of the flat data path.
constexpr int64_t ASCEND_REQUANT_FLAT_X_BYTES = 4;
constexpr int64_t ASCEND_REQUANT_FLAT_Y_BYTES = 1;
constexpr int64_t ASCEND_REQUANT_FLAT_S_BYTES = 8;
// UB kept out of the tile-extent equation: init staging, vector-store tail slack, buffer alignment.
constexpr int64_t ASCEND_REQUANT_FLAT_UB_RESERVE_BYTES = 2048;
// DataCopyPad moves whole 32 B blocks: both the GM and the UB address must be 32 B aligned.
constexpr int64_t ASCEND_REQUANT_FLAT_MTE_ALIGN_BYTES = 32;
// UB bytes actually allocated for the one-shot req_scale staging / pattern / tail slack.
constexpr int64_t ASCEND_REQUANT_FLAT_SCALE_INIT_BYTES = 512;
constexpr int64_t ASCEND_REQUANT_FLAT_PATTERN_BYTES = 256;
constexpr int64_t ASCEND_REQUANT_FLAT_BUF_SLACK_BYTES = 32;

// Element-aligned tile extent helpers shared by the flat solver.
inline int64_t AscendRequantFlatAlignDown(int64_t value)
{
    return (value / ASCEND_REQUANT_FLAT_ALIGN_ELEMS) * ASCEND_REQUANT_FLAT_ALIGN_ELEMS;
}

inline int64_t AscendRequantFlatAlignUp(int64_t value)
{
    return (value + ASCEND_REQUANT_FLAT_ALIGN_ELEMS - 1) / ASCEND_REQUANT_FLAT_ALIGN_ELEMS *
           ASCEND_REQUANT_FLAT_ALIGN_ELEMS;
}

inline int64_t AscendRequantFlatGcd(int64_t a, int64_t b)
{
    if (a < 0)
        a = -a;
    if (b < 0)
        b = -b;
    while (b != 0) {
        int64_t rest = a % b;
        a = b;
        b = rest;
    }
    return a;
}

/*!
 * \brief Period of the req_scale value along the contiguous flat x index, capped at cap.
 *
 * Dim d of the broadcast shape walks the flat index with step xShape[d] * xFlatStride[d]; only dims
 * whose req_scale stride is non-zero change the selected scale element, so the period is their lcm.
 * A period above cap is reported as cap + 1 (the caller only needs "too long for one vector register").
 */
inline int64_t AscendRequantFlatScalePeriod(const int64_t* xShape, const int64_t* sShape, int64_t rank, int64_t cap)
{
    int64_t xFlatStride[8] = {0};
    int64_t sStride[8] = {0};
    int64_t xInner = 1;
    int64_t sInner = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
        xFlatStride[d] = xInner;
        sStride[d] = (sShape[d] > 1) ? sInner : 0;
        xInner *= xShape[d];
        sInner *= sShape[d];
    }
    int64_t period = 1;
    for (int64_t d = 0; d < rank; ++d) {
        if (sStride[d] == 0)
            continue;
        int64_t term = xShape[d] * xFlatStride[d];
        if (term > cap || period > cap)
            return cap + 1;
        int64_t next = period / AscendRequantFlatGcd(period, term) * term;
        if (next > cap)
            return cap + 1;
        period = next;
    }
    return period;
}

/*!
 * \brief req_scale element selected for flat element `flat` under the broadcast map.
 *
 * Same map as the baseline CalcInputOffset(coord(flat)) with the req_scale strides; with
 * flat < one scale period the result is a valid index into the dense flat req_scale prefix
 * (staged once in UB), which is what the PERIODIC pattern expansion needs.
 */
__host_aicore__ inline int64_t AscendRequantFlatScaleIndex(int64_t flat, const int64_t* broShape,
                                                           const int64_t* sStride, int64_t rank)
{
    int64_t idx = 0;
    for (int64_t d = rank - 1; d >= 0; --d) {
        int64_t dim = broShape[d];
        if (dim <= 1)
            continue;
        idx += (flat % dim) * sStride[d];
        flat /= dim;
    }
    return idx;
}

/*!
 * \brief Length of the contiguous x suffix that req_scale covers, or 0 when the support has holes.
 *
 * The req_scale of the flat path must be readable as one contiguous block per period: every x dim
 * the scale does not broadcast over must be a suffix of the broadcast shape.
 */
inline int64_t AscendRequantFlatContigPeriod(const int64_t* xShape, const int64_t* sShape, int64_t rank)
{
    int64_t suffix = rank;
    while (suffix > 0 && xShape[suffix - 1] > 1 && sShape[suffix - 1] == xShape[suffix - 1]) {
        --suffix;
    }
    for (int64_t d = 0; d < suffix; ++d) {
        if (sShape[d] != 1)
            return 0;
    }
    int64_t period = 1;
    for (int64_t d = suffix; d < rank; ++d)
        period *= xShape[d];
    return period;
}

#endif // ASCEND_REQUANT_FLAT_ROUTE_H
