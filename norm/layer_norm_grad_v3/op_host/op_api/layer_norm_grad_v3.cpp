/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_grad_v3.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(LayerNormGradV3);

const std::array<aclTensor*, GRAD_V3_OUT_NUM> LayerNormGradV3(const aclTensor* gradOut, const aclTensor* input,
                                                              const aclTensor* rstd, const aclTensor* mean,
                                                              const aclTensor* weight, const aclBoolArray* outputMask,
                                                              const DataType gradWeightType, aclOpExecutor* executor)
{
    L0_DFX(LayerNormGradV3, gradOut, input, rstd, mean, weight, outputMask, gradWeightType);
    auto gradInputOut = executor->AllocTensor(input->GetViewShape(), input->GetDataType(), Format::FORMAT_ND);
    auto gradWeight = executor->AllocTensor(weight->GetViewShape(), gradWeightType, Format::FORMAT_ND);
    auto gradBias = executor->AllocTensor(weight->GetViewShape(), gradWeightType, Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(LayerNormGradV3, OP_INPUT(gradOut, input, rstd, mean, weight),
                                           OP_OUTPUT(gradInputOut, gradWeight, gradBias), OP_ATTR(outputMask));
    if (ret != ACL_SUCCESS) {
        // 下层已在根因点首报, 此处只记录日志并透传状态, 不再追加EZ9999
        OP_LOGE_WITHOUT_REPORT(ret,
                               "LayerNormGradV3: ADD_TO_LAUNCHER_LIST_AICORE failed during "
                               "task-list construction, ret=%d. See the first error reported by the "
                               "failing stage.",
                               static_cast<int32_t>(ret));
        return std::array<aclTensor*, GRAD_V3_OUT_NUM>{nullptr, nullptr, nullptr};
    }

    return {gradInputOut, gradWeight, gradBias};
}

} // namespace l0op
