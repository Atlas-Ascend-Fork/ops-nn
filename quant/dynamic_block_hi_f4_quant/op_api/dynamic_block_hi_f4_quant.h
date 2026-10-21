/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_DYNAMIC_BLOCK_HI_F4_QUANT_H_
#define OP_API_INC_LEVEL0_DYNAMIC_BLOCK_HI_F4_QUANT_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {
constexpr size_t DYNAMIC_BLOCK_HI_F4_QUANT_OUT_NUM = 3;

const std::array<aclTensor*, DYNAMIC_BLOCK_HI_F4_QUANT_OUT_NUM> DynamicBlockHiF4Quant(
    const aclTensor* x, const char* roundMode, int64_t scaleAlg, const aclTensor* yOut, const aclTensor* scale1Out,
    const aclTensor* scale2Out, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_DYNAMIC_BLOCK_HI_F4_QUANT_H_
