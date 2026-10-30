/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_INPLACE_CROSS_ENTROPY_SUMEXP_AND_INDEX_LOGIT_H_
#define OP_API_INC_INPLACE_CROSS_ENTROPY_SUMEXP_AND_INDEX_LOGIT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

// L2 第 1 段接口：计算 workspaceSize，并生成承载本次计算的 executor。
// vocabParallelLogits 既是输入也是输出（exp_logits 原地写回），故为非 const。
// 出参个数比非原地版本少 1（没有独立的 expLogits）。
// 它支持非连续（沿最后一维 vocab 切片，末维 stride 必须为 1），由内部 CreateView 零拷贝透传；
// target / globalLogitsMax 若非连续由内部转连续。
ACLNN_API aclnnStatus aclnnInplaceCrossEntropySumExpAndIndexLogitGetWorkspaceSize(
    aclTensor* vocabParallelLogits, const aclTensor* target, const aclTensor* globalLogitsMax, int64_t vocabStartIndex,
    int64_t vocabEndIndex, const aclTensor* predictedLogitsOut, const aclTensor* sumExpLogitsOut,
    const aclTensor* targetOffsetOut, const aclTensor* targetMaskOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);

// L2 第 2 段接口：执行计算。
ACLNN_API aclnnStatus aclnnInplaceCrossEntropySumExpAndIndexLogit(void* workspace, uint64_t workspaceSize,
                                                                  aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_INPLACE_CROSS_ENTROPY_SUMEXP_AND_INDEX_LOGIT_H_
