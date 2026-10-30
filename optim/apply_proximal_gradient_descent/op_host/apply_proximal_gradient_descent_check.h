/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APPLY_PROXIMAL_GRADIENT_DESCENT_CHECK_H
#define APPLY_PROXIMAL_GRADIENT_DESCENT_CHECK_H

#include <limits>
#include "exe_graph/runtime/shape.h"
#include "log/log.h"

namespace ApplyProximalGradientDescentCheck {
constexpr size_t INPUT_COUNT = 5;
constexpr const char* INPUT_NAMES[] = {"var", "alpha", "l1", "l2", "delta"};

inline ge::graphStatus ReportInferenceFailure(const char* opName, const char* reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason);
    return ge::GRAPH_FAILED;
}

inline bool IsUnknownRank(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

inline bool IsValidShape(const gert::Shape& shape, bool dynamic)
{
    if (dynamic && IsUnknownRank(shape)) {
        return true;
    }
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0 && !(dynamic && shape.GetDim(i) == ge::UNKNOWN_DIM)) {
            return false;
        }
    }
    return true;
}

inline bool IsScalar(const gert::Shape& shape, bool dynamic)
{
    return (dynamic && IsUnknownRank(shape)) || shape.GetDimNum() == 0 ||
           (shape.GetDimNum() == 1 && (shape.GetDim(0) == 1 || (dynamic && shape.GetDim(0) == ge::UNKNOWN_DIM)));
}

inline bool ShapesMatch(const gert::Shape& lhs, const gert::Shape& rhs, bool dynamic)
{
    if (dynamic && (IsUnknownRank(lhs) || IsUnknownRank(rhs))) {
        return true;
    }
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i) &&
            !(dynamic && (lhs.GetDim(i) == ge::UNKNOWN_DIM || rhs.GetDim(i) == ge::UNKNOWN_DIM))) {
            return false;
        }
    }
    return true;
}

inline bool GetElementCount(const gert::Shape& shape, int64_t& count)
{
    if (!IsValidShape(shape, false)) {
        return false;
    }
    // Find empty dimensions before multiplying, including shapes such as [INT64_MAX, 2, 0].
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) == 0) {
            count = 0;
            return true;
        }
    }
    int64_t result = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (result > std::numeric_limits<int64_t>::max() / shape.GetDim(i)) {
            return false;
        }
        result *= shape.GetDim(i);
    }
    count = result;
    return true;
}
} // namespace ApplyProximalGradientDescentCheck
#endif
