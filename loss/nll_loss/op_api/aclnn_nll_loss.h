/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_NLL_LOSS_H_
#define OP_API_INC_NLL_LOSS_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnNLLLoss的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 *
 * 算子功能：计算负对数似然损失值。
 *
 * @param [in] self: npu device侧的aclTensor，shape为(N,C)或者(C,)，其中N表示batch
 * size，C表示类别数。数据类型支持FLOAT、FLOAT16、BFLOAT16，支持非连续的Tensor，数据格式支持ND。
 * @param [in] target: npu device侧的aclTensor，表示真实标签，shape为(N,)或者()，其中每个元素的取值范围是[0, C-1]，
 * 取值为ignoreIndex的元素不参与损失计算。数据类型支持INT64、UINT8、INT32，支持非连续的Tensor，数据格式支持ND。
 * @param [in] weight: npu device侧的aclTensor，表示每个类别的缩放权重，shape为(C,)。GE图模式下该输入
 * 可缺省，缺省时按权重为1计算；通过本接口直调时weight指针不可为空，传空指针返回ACLNN_ERR_PARAM_NULLPTR。
 * 数据类型支持FLOAT、FLOAT16、BFLOAT16（与self一致），支持非连续的Tensor，数据格式支持ND。
 * @param [in] reduction: host侧的int64_t，指定要应用到输出的缩减，支持 0('none') | 1('mean') | 2('sum')。'none'
 * 表示不应用缩减，'mean' 表示输出为各元素损失之和除以权重之和totalWeight（即Σw_{y_n}），'sum' 表示输出将被求和。
 * @param [in] ignoreIndex: host侧的int64_t，指定一个被忽略且不影响损失计算的目标值。默认值为-100，应取
 * [0, C-1]之外且与target数据类型可比较的整数值。
 * @param [in] out: npu device侧的aclTensor，shape为(N,)（reduction为none且self为2维时）或者(1,)，
 * 数据类型与self保持一致，支持非连续的Tensor，数据格式支持ND。
 * @param [in] totalWeightOut: npu device侧的aclTensor，shape为(1,)，仅在reduction非none时输出值有效，
 * reduction为none时取值无效。数据类型与self保持一致，支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNLLLossGetWorkspaceSize(const aclTensor* self, const aclTensor* target,
                                                   const aclTensor* weight, int64_t reduction, int64_t ignoreIndex,
                                                   aclTensor* out, aclTensor* totalWeightOut, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor);

/**
 * @brief aclnnNLLLoss的第二段接口，用于执行计算。
 *
 * 算子功能：计算负对数似然损失值。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnNLLLossGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnNLLLoss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ADD_H_
