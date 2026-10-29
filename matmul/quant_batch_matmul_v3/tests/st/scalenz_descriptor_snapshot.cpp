/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include "opdev/op_arg_def.h"

// Test-only accessor: the public ACL getters do not expose OriginalShape/OriginalFormat.
// Build against the same opdev headers/libraries as the ACLNN package under test.
extern "C" uint64_t QbmmScaleDescriptorSnapshot(const aclTensor* tensor, int64_t* values, uint64_t capacity)
{
    constexpr uint64_t MAX_SNAPSHOT_SIZE = 128;
    if (tensor == nullptr || values == nullptr || capacity < MAX_SNAPSHOT_SIZE) {
        return 0;
    }
    uint64_t count = 0;
    values[count++] = static_cast<int64_t>(tensor->GetDataType());
    values[count++] = static_cast<int64_t>(tensor->GetViewFormat());
    values[count++] = static_cast<int64_t>(tensor->GetStorageFormat());
    values[count++] = static_cast<int64_t>(tensor->GetOriginalFormat());
    values[count++] = tensor->GetViewOffset();
    for (const auto* shape : {&tensor->GetViewShape(), &tensor->GetStorageShape(), &tensor->GetOriginalShape()}) {
        values[count++] = shape->GetDimNum();
        for (size_t i = 0; i < shape->GetDimNum(); ++i) {
            values[count++] = shape->GetDim(i);
        }
    }
    const auto rank = tensor->GetViewShape().GetDimNum();
    values[count++] = rank;
    for (size_t i = 0; i < rank; ++i) {
        values[count++] = tensor->GetViewStrides()[i];
    }
    return count;
}
