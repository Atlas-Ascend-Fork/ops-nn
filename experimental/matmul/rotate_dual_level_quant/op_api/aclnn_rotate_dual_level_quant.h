/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_ROTATE_DUAL_LEVEL_QUANT_H
#define ACLNN_ROTATE_DUAL_LEVEL_QUANT_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Rotate, optionally clamp, then quantize with shared FP32/512 and E8M0/32 scales on Ascend950.
 * @domain aclnn_ops_infer
 * @param [in] x ND FP16/BF16 input, rank 1..7; grouped mode requires [M,N].
 * @param [in] rotation Same dtype as x, [K,K]/[N/K,K,K], or grouped [E,K,K]/[E,N/K,K,K].
 * K is 32/64/128/256, N must be divisible by K. Grouped rotation cannot be empty.
 * @param [in] alpha Optional ND BF16 [1]. Clamp is enabled only for a value in (0,1).
 * @param [in] groupList Optional ND INT64 [E], 1 <= E <= 1024. Describes consecutive M-axis groups.
 * Empty groups are allowed and still consume a rotation index. Uncovered output rows are preserved.
 * @param [in] groupListType 0: nonnegative nondecreasing cumsum, final value <= M;
 * 1: nonnegative counts whose sum <= M. Must be 0/1 even when groupList is null.
 * The caller must supply valid device-side group values; host checks metadata only.
 * @param [in] axis -1 or the positive last-dimension index.
 * @param [in] roundMode rint/round/floor; null means rint. FP8 accepts rint only.
 * @param [in] scaleAlg FP4: 0/2; FP8: 0/1. Controls level1 MX quantization.
 * @param [in] dstTypeMax [6,12] for scaleAlg=2, otherwise 0. Does not replace the level0 dtype maximum.
 * @param [in] trans Must be false.
 * @param [in] level0BlockSize First-level quantization block size; only 512 is supported.
 * @param [in] level1BlockSize Second-level MX quantization block size; only 32 is supported.
 * The corresponding Graph defaults are 512 and 32. ACLNN callers must pass both scalar values explicitly.
 * @param [out] yOut Preallocated ND FP4_E2M1/FP8_E4M3FN/FP8_E5M2, same shape as x.
 * @param [out] level0ScaleOut Preallocated ND FP32 [...,ceil(N/512)].
 * @param [out] level1ScaleOut Preallocated ND E8M0 [...,ceil(N/64),2], paired scales with padding.
 * Inputs and outputs may be noncontiguous. In grouped mode, uncovered rows of ALL three outputs,
 * including their scale padding, retain their initial contents. Valid empty input is a no-op.
 * @param [out] workspaceSize Required device workspace size in bytes.
 * @param [out] executor Execution plan; consumed by the second API.
 * @return ACLNN_SUCCESS when the execution plan is created, including a valid empty no-op;
 * ACLNN_ERR_RUNTIME_ERROR when the platform is not Ascend950.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnRotateDualLevelQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* rotation, const aclTensor* alpha, const aclTensor* groupList,
    int64_t groupListType, int64_t axis, char* roundMode, int64_t scaleAlg, double dstTypeMax, bool trans,
    int64_t level0BlockSize, int64_t level1BlockSize, aclTensor* yOut, aclTensor* level0ScaleOut,
    aclTensor* level1ScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Execute the plan produced by aclnnRotateDualLevelQuantGetWorkspaceSize.
 * @param [in] workspace Device workspace, may be null when workspaceSize is zero.
 * @param [in] workspaceSize Allocated workspace size in bytes.
 * @param [in] executor Plan returned by the first API; cannot be reused after execution.
 * @param [in] stream ACL stream.
 * @return aclnnStatus.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnRotateDualLevelQuant(void* workspace, uint64_t workspaceSize,
                                                                             aclOpExecutor* executor,
                                                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // ACLNN_ROTATE_DUAL_LEVEL_QUANT_H
