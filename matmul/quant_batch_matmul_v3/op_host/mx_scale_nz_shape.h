/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstddef>
#include <cstdint>

namespace QuantBatchMatmul {
// GroupedMatmul ScaleB NZ: [batch..., N1, ceil(K/64), 16, 2].
// Logical view is always [batch..., Kg, N, 2], independent of B transpose.
constexpr int64_t MX_SCALE_MIN_RANK = 3;
constexpr int64_t MX_SCALE_MAX_RANK = 7;
constexpr int64_t MX_GROUP_K = 64;
constexpr int64_t NZ_C0 = 16;
constexpr int64_t MX_SCALE_PAIR = 2;
template <typename Shape>
inline bool CheckMxScaleNzShape(const Shape& view, const Shape& storage, int64_t n, int64_t k)
{
    if (n <= 0 || k <= 0 || view.GetDimNum() < MX_SCALE_MIN_RANK || view.GetDimNum() > MX_SCALE_MAX_RANK ||
        storage.GetDimNum() != view.GetDimNum() + 1) {
        return false;
    }
    const size_t batchDims = view.GetDimNum() - static_cast<size_t>(MX_SCALE_MIN_RANK);
    for (size_t i = 0; i < batchDims; ++i) {
        if (view.GetDim(i) <= 0 || storage.GetDim(i) != view.GetDim(i)) {
            return false;
        }
    }
    const int64_t kg = k / MX_GROUP_K + (k % MX_GROUP_K != 0 ? 1 : 0);
    const int64_t n1 = n / NZ_C0 + (n % NZ_C0 != 0 ? 1 : 0);
    return view.GetDim(batchDims) == kg && view.GetDim(batchDims + 1) == n &&
           view.GetDim(batchDims + 2) == MX_SCALE_PAIR && storage.GetDim(batchDims) == n1 &&
           storage.GetDim(batchDims + 1) == kg && storage.GetDim(batchDims + 2) == NZ_C0 &&
           storage.GetDim(batchDims + 3) == MX_SCALE_PAIR;
}
} // namespace QuantBatchMatmul
