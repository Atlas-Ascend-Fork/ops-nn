/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "confidence_head_fused.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ConfidenceHeadFused);

const aclTensor* ConfidenceHeadFused(const aclTensor* hidden, const aclTensor* weight, const aclTensor* markovEmb,
                                     const aclTensor* temperature, const aclTensor* bias, float clampLow,
                                     float clampHigh, aclOpExecutor* executor)
{
    L0_DFX(ConfidenceHeadFused, hidden, weight, markovEmb, temperature, bias, clampLow, clampHigh);
    auto outputShape = hidden->GetViewShape();
    outputShape.SetDimNum(outputShape.GetDimNum() - 1);
    auto out = executor->AllocTensor(outputShape, DataType::DT_FLOAT, Format::FORMAT_ND);
    if (out == nullptr) {
        return nullptr;
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ConfidenceHeadFused, OP_INPUT(hidden, weight, markovEmb, temperature, bias),
                                           OP_OUTPUT(out), OP_ATTR(clampLow, clampHigh));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID, "ConfidenceHeadFused launch failed.");
        return nullptr;
    }
    return out;
}

} // namespace l0op
