/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_CONFIDENCE_HEAD_FUSED_H
#define OP_API_INC_CONFIDENCE_HEAD_FUSED_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnConfidenceHeadFused的第一段接口，计算workspace大小并生成执行器。
 * @param [in] hidden: 待投影的特征Tensor，shape为[N, K1]或[B, N, K1]，类型为FLOAT16或BFLOAT16。
 * @param [in] weight: 投影权重Tensor，shape为[K1 + K2]，数据类型与hidden一致，所有行共享。
 * @param [in] markovEmbOptional: 可选Tensor，shape为[N, K2]或[B, N, K2]，类型与hidden一致。
 * 传入nullptr时不拼接，K2为0。
 * @param [in] temperatureOptional: 可选FLOAT类型Tensor，shape为[1]。传入nullptr或其值为零时按1计算。
 * @param [in] biasOptional: 可选FLOAT类型Tensor，shape为[N]，在batch维度广播。传入nullptr时不加偏置。
 * @param [in] clampLowOptional: 可选aclScalar，指数输入的下限。传入nullptr时不做下限截断。
 * @param [in] clampHighOptional: 可选aclScalar，指数输入的上限。传入nullptr时不做上限截断。
 * @param [in] out: 输出Tensor，类型为FLOAT，shape为[N]或[B, N]，与hidden的前导维度一致。
 * @param [out] workspaceSize: 返回用户需要在NPU device侧申请的workspace大小。
 * @param [out] executor: 返回包含算子计算流程的执行器。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConfidenceHeadFusedGetWorkspaceSize(
    const aclTensor* hidden, const aclTensor* weight, const aclTensor* markovEmbOptional,
    const aclTensor* temperatureOptional, const aclTensor* biasOptional, const aclScalar* clampLowOptional,
    const aclScalar* clampHighOptional, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnConfidenceHeadFused的第二段接口，用于执行计算。
 * @param [in] workspace: 在NPU device侧申请的workspace内存地址。
 * @param [in] workspaceSize: workspace大小，由aclnnConfidenceHeadFusedGetWorkspaceSize获取。
 * @param [in] executor: 第一段接口返回的算子执行器。
 * @param [in] stream: ACL stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnConfidenceHeadFused(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_CONFIDENCE_HEAD_FUSED_H
