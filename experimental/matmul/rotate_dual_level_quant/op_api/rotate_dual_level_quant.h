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

#include <tuple>
#include "opdev/op_executor.h"

namespace l0op {
std::tuple<aclTensor*, aclTensor*, aclTensor*> RotateDualLevelQuant(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, const aclTensor* groupList, int64_t axis,
    const char* roundMode, int64_t scaleAlg, float dstTypeMax, bool trans, int64_t dstType, int64_t groupListType,
    int64_t level0BlockSize, int64_t level1BlockSize, aclTensor* yOut, aclTensor* level0ScaleOut,
    aclTensor* level1ScaleOut, aclOpExecutor* executor);
} // namespace l0op
