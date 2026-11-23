/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GEMM_SYMM_H
#define OP_API_INC_GEMM_SYMM_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Y = alpha * (A @ B) + beta * C. The caller guarantees that A @ B is symmetric.
 * k == 0 skips the matmul and computes Y = beta * C.
 * c may be null when beta is 0.
 * cubeMathType 4 keeps the fp32 product through Fixpipe. Other values cast it to fp16 or bf16 in Fixpipe.
 */
ACLNN_API aclnnStatus aclnnGemmSymmGetWorkspaceSize(const aclTensor* a, const aclTensor* b, const aclTensor* c,
                                                    aclTensor* y, const aclScalar* alphaOptional,
                                                    const aclScalar* betaOptional, bool transposeA, bool transposeB,
                                                    const char* fillMode, int8_t cubeMathType, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnGemmSymm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
