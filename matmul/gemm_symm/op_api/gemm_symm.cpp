/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gemm_symm.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

namespace l0op {
OP_TYPE_REGISTER(GemmSymm);

const aclTensor* GemmSymm(const aclTensor* a, const aclTensor* b, const aclTensor* c, const aclTensor* y, float alpha,
                          float beta, bool transposeA, bool transposeB, const char* fillMode, int8_t cubeMathType,
                          aclOpExecutor* executor)
{
    L0_DFX(GemmSymm, a, b, c, y, alpha, beta, transposeA, transposeB, cubeMathType);
    const int64_t opImplMode = cubeMathType;
    const char* symmType = fillMode == nullptr ? "full" : fillMode;
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(GemmSymm, OP_INPUT(a, b, c), OP_OUTPUT(y),
                                           OP_ATTR(alpha, beta, transposeA, transposeB, symmType, opImplMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "ADD_TO_LAUNCHER_LIST_AICORE GemmSymm failed.");
        return nullptr;
    }
    return y;
}
} // namespace l0op
