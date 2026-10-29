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
template <typename Shape>
inline bool CheckMxScaleNzShape(const Shape& view, const Shape& storage, int64_t n, int64_t k)
{
    if (n <= 0 || k <= 0 || view.GetDimNum() < 3 || view.GetDimNum() > 7 ||
        storage.GetDimNum() != view.GetDimNum() + 1) {
        return false;
    }
    const size_t batchDims = view.GetDimNum() - 3;
    for (size_t i = 0; i < batchDims; ++i) {
        if (view.GetDim(i) <= 0 || storage.GetDim(i) != view.GetDim(i)) {
            return false;
        }
    }
    const int64_t kg = k / 64 + (k % 64 != 0);
    const int64_t n1 = n / 16 + (n % 16 != 0);
    return view.GetDim(batchDims) == kg && view.GetDim(batchDims + 1) == n && view.GetDim(batchDims + 2) == 2 &&
           storage.GetDim(batchDims) == n1 && storage.GetDim(batchDims + 1) == kg &&
           storage.GetDim(batchDims + 2) == 16 && storage.GetDim(batchDims + 3) == 2;
}
} // namespace QuantBatchMatmul
