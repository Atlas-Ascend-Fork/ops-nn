/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turbo_quant.cpp
 * \brief level0 API of TurboQuant
 */
#include "turbo_quant.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(TurboQuant);

std::tuple<aclTensor*, aclTensor*> TurboQuant(const aclTensor* latent, const aclTensor* centroids,
                                              aclOpExecutor* executor)
{
    L0_DFX(TurboQuant, latent, centroids);
    const auto& latentView = latent->GetViewShape();
    const int64_t numTokens = latentView.GetDim(0);
    const int64_t headDim = latentView.GetDim(1);

    // Two nibbles per element: y is [numTokens, headDim / 2] uint8.
    op::Shape yShape;
    yShape.AppendDim(numTokens);
    yShape.AppendDim(headDim / 2);
    auto yOut = executor->AllocTensor(yShape, op::DataType::DT_UINT8);
    if (yOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TurboQuant alloc y tensor failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }

    op::Shape scaleShape;
    scaleShape.AppendDim(numTokens);
    auto scaleOut = executor->AllocTensor(scaleShape, op::DataType::DT_FLOAT16);
    if (scaleOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TurboQuant alloc scale tensor failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(TurboQuant, OP_INPUT(latent, centroids), OP_OUTPUT(yOut, scaleOut));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TurboQuant launch kernel failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(yOut, scaleOut);
}

} // namespace l0op
