/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_batch_mat_mul.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TransposeBatchMatMul);

const aclTensor* TransposeBatchMatMul(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                      const aclTensor* scale, const aclIntArray* permX1, const aclIntArray* permX2,
                                      const aclIntArray* permY, const bool enableHf32, const int32_t batchSplitFactor,
                                      aclOpExecutor* executor)
{
    L0_DFX(TransposeBatchMatMul, x1, x2, bias, scale, permX1, permX2, permY, enableHf32, batchSplitFactor);
    auto outType = x1->GetDataType();
    if (scale != nullptr) {
        outType = DataType::DT_INT8;
    }
    auto out = executor->AllocTensor(outType, Format::FORMAT_ND, Format::FORMAT_ND);
    OP_CHECK_NULL(out, return nullptr);
    auto ret = INFER_SHAPE(TransposeBatchMatMul, OP_INPUT(x1, x2, bias, scale), OP_OUTPUT(out),
                           OP_ATTR(permX1, permX2, permY, enableHf32, batchSplitFactor));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "TransposeBatchMatMul InferShape failed.");
    uint32_t execMode = enableHf32 ? static_cast<uint32_t>(OpExecMode::OP_EXEC_MODE_HF32) : 0U;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(TransposeBatchMatMul, OP_INPUT(x1, x2, bias, scale), OP_OUTPUT(out),
                                      OP_ATTR(permX1, permX2, permY, enableHf32, batchSplitFactor), OP_MODE(execMode));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "TransposeBatchMatMul ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
};
} // namespace l0op
