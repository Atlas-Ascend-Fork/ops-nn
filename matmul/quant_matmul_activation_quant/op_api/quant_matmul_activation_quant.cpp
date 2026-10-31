/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "quant_matmul_activation_quant.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantMatmulActivationQuant);

const std::array<aclTensor*, QUANT_MATMUL_ACTIVATION_QUANT_OUT_NUM> QuantMatmulActivationQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    bool transposeX1, bool transposeX2, int64_t groupSize, const char* activationType, int64_t y_dtype,
    const char* quantMode, const char* roundMode, int64_t scaleAlg, double dstTypeMax, aclOpExecutor* executor)
{
    if (x1 == nullptr || x2 == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "x1, x2 and executor must not be null.");
        return {nullptr, nullptr};
    }

    L0_DFX(QuantMatmulActivationQuant, x1, x2, bias, x1Scale, x2Scale, transposeX1, transposeX2, groupSize,
           activationType, y_dtype, quantMode, roundMode, scaleAlg, dstTypeMax);

    auto yOut = executor->AllocTensor(static_cast<op::DataType>(y_dtype), Format::FORMAT_ND, Format::FORMAT_ND);
    auto yScaleOut = executor->AllocTensor(op::DataType::DT_FLOAT8_E8M0, Format::FORMAT_ND, Format::FORMAT_ND);

    if (yOut == nullptr || yScaleOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to allocate output tensors.");
        return {yOut, yScaleOut};
    }

    // 运行QuantMatmulActivationQuant算子的InferShape函数，推导输出shape
    auto ret = INFER_SHAPE(QuantMatmulActivationQuant, OP_INPUT(x1, x2, bias, x1Scale, x2Scale),
                           OP_OUTPUT(yOut, yScaleOut),
                           OP_ATTR(transposeX1, transposeX2, groupSize, activationType, y_dtype, quantMode, roundMode,
                                   scaleAlg, dstTypeMax));

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed for QuantMatmulActivationQuant.");
        return {nullptr, nullptr};
    }

    OP_LOGD("Resolved L0 transpose attributes: transposeX1=%s, transposeX2=%s", transposeX1 ? "true" : "false",
            transposeX2 ? "true" : "false");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(QuantMatmulActivationQuant, OP_INPUT(x1, x2, bias, x1Scale, x2Scale),
                                      OP_OUTPUT(yOut, yScaleOut),
                                      OP_ATTR(transposeX1, transposeX2, groupSize, activationType, y_dtype, quantMode,
                                              roundMode, scaleAlg, static_cast<float>(dstTypeMax)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to add QuantMatmulActivationQuant to the AICore launcher list.");
        return {nullptr, nullptr};
    }
    return {yOut, yScaleOut};
}

} // namespace l0op
