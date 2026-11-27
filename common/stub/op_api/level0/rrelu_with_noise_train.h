/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_RRELU_WITH_NOISE_TRAIN_H_
#define OP_API_INC_LEVEL0_RRELU_WITH_NOISE_TRAIN_H_

#include "opdev/op_executor.h"

namespace l0op {
// L0 实现位于 ops-math 仓 random/rrelu_with_noise/op_api（随包合并后由 libopapi 提供），
// 此处仅为 ops-nn 独立构建/UT 提供声明与桩。
// lower/upper/seed/offset 以属性形式下发（与 OpDef 一致），seed/offset 值域 [0, 2^63-1]。
const aclTensor* RreluWithNoiseTrain(const aclTensor* x, int64_t seed, int64_t offset, float lower, float upper,
                                     const aclTensor** noise, aclOpExecutor* executor);
} // namespace l0op
#endif // OP_API_INC_LEVEL0_RRELU_WITH_NOISE_TRAIN_H_
