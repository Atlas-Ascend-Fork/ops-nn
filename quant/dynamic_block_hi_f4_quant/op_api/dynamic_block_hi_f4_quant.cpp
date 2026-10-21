/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dynamic_block_hi_f4_quant.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(DynamicBlockHiF4Quant);

const std::array<aclTensor*, DYNAMIC_BLOCK_HI_F4_QUANT_OUT_NUM> DynamicBlockHiF4Quant(
    const aclTensor* x, const char* roundMode, int64_t scaleAlg, const aclTensor* yOut, const aclTensor* scale1Out,
    const aclTensor* scale2Out, aclOpExecutor* executor)
{
    L0_DFX(DynamicBlockHiF4Quant, x, yOut, scale1Out, scale2Out);
    auto y = executor->AllocTensor(yOut->GetViewShape(), yOut->GetDataType(), yOut->GetViewFormat());
    auto scale1 = executor->AllocTensor(scale1Out->GetViewShape(), scale1Out->GetDataType(),
                                        scale1Out->GetViewFormat());
    auto scale2 = executor->AllocTensor(scale2Out->GetViewShape(), scale2Out->GetDataType(),
                                        scale2Out->GetViewFormat());
    if (y == nullptr || scale1 == nullptr || scale2 == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc output tensor failed.");
        return {nullptr, nullptr, nullptr};
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(DynamicBlockHiF4Quant, OP_INPUT(x), OP_OUTPUT(y, scale1, scale2),
                                           OP_ATTR(roundMode, scaleAlg));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "DynamicBlockHiF4Quant launch kernel failed.");
        return {nullptr, nullptr, nullptr};
    }
    return {y, scale1, scale2};
}

} // namespace l0op
