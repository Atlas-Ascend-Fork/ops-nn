/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "anti_hi_f4_quant.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AntiHiF4Quant);

const aclTensor* AntiHiF4Quant(const aclTensor* x, const aclTensor* scale, int64_t dstType, aclOpExecutor* executor)
{
    L0_DFX(AntiHiF4Quant, x, scale, dstType);
    auto out = executor->AllocTensor(x->GetViewShape(), op::DataType(dstType));
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc output tensor failed.");
        return nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AntiHiF4Quant, OP_INPUT(x, scale), OP_OUTPUT(out), OP_ATTR(dstType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AntiHiF4Quant launch kernel failed.");
        return nullptr;
    }
    return out;
}

} // namespace l0op
