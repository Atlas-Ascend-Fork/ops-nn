/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_dynamic_hi_f4_quant.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(GroupedDynamicHiF4Quant);

const std::array<aclTensor*, GROUPED_DYNAMIC_HI_F4_QUANT_OUT_NUM> GroupedDynamicHiF4Quant(
    const aclTensor* x, const aclTensor* groupIndex, int64_t scaleAlg, const aclTensor* yOut, const aclTensor* scaleOut,
    aclOpExecutor* executor)
{
    L0_DFX(GroupedDynamicHiF4Quant, x, groupIndex, yOut, scaleOut);
    auto y = executor->AllocTensor(yOut->GetViewShape(), yOut->GetDataType(), yOut->GetViewFormat());
    auto scale = executor->AllocTensor(scaleOut->GetViewShape(), scaleOut->GetDataType(), scaleOut->GetViewFormat());
    if (y == nullptr || scale == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc output tensor failed.");
        return {nullptr, nullptr};
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedDynamicHiF4Quant, OP_INPUT(x, groupIndex), OP_OUTPUT(y, scale),
                                           OP_ATTR(scaleAlg));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GroupedDynamicHiF4Quant launch kernel failed.");
        return {nullptr, nullptr};
    }
    return {y, scale};
}

} // namespace l0op
