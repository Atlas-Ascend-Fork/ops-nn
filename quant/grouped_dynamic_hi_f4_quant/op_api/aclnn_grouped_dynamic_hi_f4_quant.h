/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_DYNAMIC_HI_F4_QUANT_H_
#define OP_API_INC_GROUPED_DYNAMIC_HI_F4_QUANT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedDynamicHiF4Quant的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：对分组输入Tensor按group_index进行动态量化，输出HiFloat4数据与对应的三级缩放因子。
 *
 * @param [in] x: npu device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND。
 * @param [in] group_index: npu device侧的aclTensor，数据类型支持INT32，数据格式支持ND。
 * @param [in] scale_alg: Host侧整型，指定scale的计算算法。
 * @param [out] y: npu device侧的aclTensor，数据类型为HIFLOAT4，shape与x一致，数据格式支持ND。
 * @param [out] scale: npu device侧的aclTensor，数据类型为HIFLOAT4_SCALE，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedDynamicHiF4QuantGetWorkspaceSize(const aclTensor* x, const aclTensor* group_index,
                                                                   int64_t scale_alg, aclTensor* y, aclTensor* scale,
                                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedDynamicHiF4Quant的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedDynamicHiF4Quant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_GROUPED_DYNAMIC_HI_F4_QUANT_H_
