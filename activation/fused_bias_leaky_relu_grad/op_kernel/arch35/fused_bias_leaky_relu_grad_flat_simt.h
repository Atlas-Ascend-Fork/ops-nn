/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "kernel_operator.h"
#include "fused_bias_leaky_relu_grad_tiling_struct.h"

namespace FusedBiasLeakyReluGradFlat {
using namespace AscendC;
constexpr uint32_t THREADS = 512;

template <typename T, int RANK, bool GRAD_CONTIGUOUS, bool FEATURE_CONTIGUOUS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void Run(__gm__ T* grad, __gm__ T* features, __gm__ T* output,
                                                             const __gm__ FusedBiasLeakyReluGradTilingData<RANK>* td)
{
    uint32_t shape[RANK];
    uint32_t gradStride[RANK];
    uint32_t featureStride[RANK];
    uint32_t outputStride[RANK];
#pragma unroll
    for (int d = 0; d < RANK; ++d) {
        shape[d] = static_cast<uint32_t>(td->max_bro_shape[d]);
        gradStride[d] = static_cast<uint32_t>(td->input_strides[0][d]);
        featureStride[d] = static_cast<uint32_t>(td->input_strides[1][d]);
        outputStride[d] = static_cast<uint32_t>(td->output_strides[0][d]);
    }
    // A single contiguous varying slab has the exact index (i / innerRepeat) % elements.
    // Metadata classification runs once per thread, outside the element loop.
    uint32_t gradInner = 1, featureInner = 1, gradElements = 1, featureElements = 1;
    bool gradSeen = false, featureSeen = false, gradGap = false, featureGap = false;
    bool gradSlab = true, featureSlab = true;
#pragma unroll
    for (int d = RANK - 1; d >= 0; --d) {
        if (gradStride[d] != 0) {
            gradSlab = gradSlab && !gradGap;
            gradSeen = true;
            gradElements *= shape[d];
        } else if (!gradSeen) {
            gradInner *= shape[d];
        } else if (shape[d] > 1) {
            gradGap = true;
        }
        if (featureStride[d] != 0) {
            featureSlab = featureSlab && !featureGap;
            featureSeen = true;
            featureElements *= shape[d];
        } else if (!featureSeen) {
            featureInner *= shape[d];
        } else if (shape[d] > 1) {
            featureGap = true;
        }
    }
    const uint32_t count = td->flat_count;
    const float negativeSlope = td->negative_slope;
    const float scale = td->scale;
    const uint32_t step = Simt::GetThreadNum() * Simt::GetBlockNum();
    for (uint32_t index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index += step) {
        uint32_t g = GRAD_CONTIGUOUS ? index : 0;
        uint32_t f = FEATURE_CONTIGUOUS ? index : 0;
        if constexpr (!GRAD_CONTIGUOUS) {
            if (gradSlab)
                g = gradElements == 1 ? 0 : (index / gradInner) % gradElements;
        }
        if constexpr (!FEATURE_CONTIGUOUS) {
            if (featureSlab)
                f = featureElements == 1 ? 0 : (index / featureInner) % featureElements;
        }
#pragma unroll
        for (int d = 0; d < RANK; ++d) {
            const bool needGrad = !GRAD_CONTIGUOUS && !gradSlab && gradStride[d] != 0;
            const bool needFeature = !FEATURE_CONTIGUOUS && !featureSlab && featureStride[d] != 0;
            if (needGrad || needFeature) {
                const uint32_t coord = (index / outputStride[d]) % shape[d];
                if (needGrad)
                    g += coord * gradStride[d];
                if (needFeature)
                    f += coord * featureStride[d];
            }
        }
        const float gradValue = static_cast<float>(grad[g]);
        const float featureValue = static_cast<float>(features[f]);
        const float coef = featureValue > 0.0f ? 1.0f : negativeSlope;
        const float product = gradValue * coef;
        output[index] = static_cast<T>(product * scale);
    }
}

template <typename T, int RANK>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void RunTwoSlab(
    __gm__ T* grad, __gm__ T* features, __gm__ T* output, const __gm__ FusedBiasLeakyReluGradTilingData<RANK>* td)
{
    const uint32_t count = td->flat_count;
    const uint32_t gf = td->grad_contiguous, ff = td->feature_contiguous;
    const uint32_t gi0 = static_cast<uint32_t>(td->input_strides[0][0]);
    const uint32_t ge0 = static_cast<uint32_t>(td->input_strides[0][1]);
    const uint32_t gi1 = static_cast<uint32_t>(td->input_strides[0][2]);
    const uint32_t ge1 = static_cast<uint32_t>(td->input_strides[0][3]);
    const uint32_t fi0 = static_cast<uint32_t>(td->input_strides[1][0]);
    const uint32_t fe0 = static_cast<uint32_t>(td->input_strides[1][1]);
    const uint32_t fi1 = static_cast<uint32_t>(td->input_strides[1][2]);
    const uint32_t fe1 = static_cast<uint32_t>(td->input_strides[1][3]);
    const float negativeSlope = td->negative_slope;
    const float scale = td->scale;
    const uint32_t step = Simt::GetThreadNum() * Simt::GetBlockNum();
    for (uint32_t index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
         index += step) {
        uint32_t g = 0, f = 0;
        if (gf & (1U << 10)) {
            g = index;
        } else if (gf & 0xffU) {
            g = index / gi0;
            if (!(gf & (1U << 8)))
                g %= ge0;
            if ((gf & 0xffU) == 2) {
                uint32_t upper = index / gi1;
                if (!(gf & (1U << 9)))
                    upper %= ge1;
                g += upper * ge0;
            }
        }
        if (ff & (1U << 10)) {
            f = index;
        } else if (ff & 0xffU) {
            f = index / fi0;
            if (!(ff & (1U << 8)))
                f %= fe0;
            if ((ff & 0xffU) == 2) {
                uint32_t upper = index / fi1;
                if (!(ff & (1U << 9)))
                    upper %= fe1;
                f += upper * fe0;
            }
        }
        const float gradValue = static_cast<float>(grad[g]);
        const float featureValue = static_cast<float>(features[f]);
        const float coef = featureValue > 0.0f ? 1.0f : negativeSlope;
        const float product = gradValue * coef;
        output[index] = static_cast<T>(product * scale);
    }
}

template <typename T, int RANK>
__aicore__ inline void Launch(GM_ADDR grad, GM_ADDR features, GM_ADDR output, GM_ADDR tiling,
                              const FusedBiasLeakyReluGradTilingData<RANK>& td)
{
    auto g = reinterpret_cast<__gm__ T*>(grad);
    auto f = reinterpret_cast<__gm__ T*>(features);
    auto o = reinterpret_cast<__gm__ T*>(output);
    auto info = reinterpret_cast<const __gm__ FusedBiasLeakyReluGradTilingData<RANK>*>(tiling);
    if (td.use_flat_simt == 2) {
        Simt::VF_CALL<RunTwoSlab<T, RANK>>(Simt::Dim3(THREADS), g, f, o, info);
        return;
    }
    if (td.grad_contiguous) {
        Simt::VF_CALL<Run<T, RANK, true, false>>(Simt::Dim3(THREADS), g, f, o, info);
    } else if (td.feature_contiguous) {
        Simt::VF_CALL<Run<T, RANK, false, true>>(Simt::Dim3(THREADS), g, f, o, info);
    } else {
        Simt::VF_CALL<Run<T, RANK, false, false>>(Simt::Dim3(THREADS), g, f, o, info);
    }
}
} // namespace FusedBiasLeakyReluGradFlat
