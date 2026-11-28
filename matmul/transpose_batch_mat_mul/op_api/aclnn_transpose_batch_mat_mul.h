/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_TRANSPOSE_BATCH_MAT_MUL
#define OP_API_INC_TRANSPOSE_BATCH_MAT_MUL

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First phase of aclnnTransposeBatchMatMul. Computes the workspace size for the given computation flow.
 * @domain aclnn_ops_infer
 * Operator: fused TransposeBatchMatMul and Quant; TransposeBatchMatMul supports int8 parameters.
 * @param [in] x1: Left matmul matrix. Supported dtypes: float16, float32, bfloat16.
 * @param [in] x2: Right matmul matrix. Supported dtypes: float16, float32, bfloat16.
 * @param [in] bias: Bias. Not supported currently.
 * @param [in] scale: Quantization scale. Supported dtypes: int64, uint64.
 * @param [in] permX1: Permutation applied to the shape of x1.
 * @param [in] permX2: Permutation applied to the shape of x2.
 * @param [in] permY: Permutation applied to the shape of y.
 * @param [in] cubeMathType: Cube unit computation mode as a host integer. Supported dtype: int8.
 * @param [in] batchSplitFactor: Whether to split the shape again. Supported dtype: int32.
 * @param [out] out: Result. Supported dtypes: float32, float16, bfloat16, int8.
 * @param [out] workspaceSize: Size of the workspace to allocate on the NPU device.
 * @param [out] executor: Op executor containing the operator computation flow.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                                const aclTensor* bias, const aclTensor* scale,
                                                                const aclIntArray* permX1, const aclIntArray* permX2,
                                                                const aclIntArray* permY, int8_t cubeMathType,
                                                                const int32_t batchSplitFactor, aclTensor* out,
                                                                uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Second phase of aclnnTransposeBatchMatMul. Executes the computation.
 */
ACLNN_API aclnnStatus aclnnTransposeBatchMatMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                const aclrtStream stream);

/**
 * @brief First phase of aclnnTransposeBatchMatMulWeightNz. Computes the workspace size for the given computation flow.
 * @domain aclnn_ops_infer
 * Operator: same as aclnnTransposeBatchMatMul, but mat2 is in NZ format.
 * @param [in] x1: Left matmul matrix. Supported dtypes: float16, bfloat16. Format: ND.
 * @param [in] x2: Right matmul matrix. Supported dtypes: float16, bfloat16. Format: NZ.
 * @param [in] bias: Bias. Not supported currently.
 * @param [in] scale: Quantization scale. Supported dtypes: int64, uint64.
 * @param [in] permX1: Permutation applied to the shape of x1.
 * @param [in] permX2: Permutation applied to the shape of x2.
 * @param [in] permY: Permutation applied to the shape of y.
 * @param [in] cubeMathType: Cube unit computation mode as a host integer. Supported dtype: int8.
 * @param [in] batchSplitFactor: Whether to split the shape again. Supported dtype: int32.
 * @param [out] out: Result. Supported dtypes: float16, bfloat16. Format: ND.
 * @param [out] workspaceSize: Size of the workspace to allocate on the NPU device.
 * @param [out] executor: Op executor containing the operator computation flow.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* scale, const aclIntArray* permX1,
    const aclIntArray* permX2, const aclIntArray* permY, int8_t cubeMathType, const int32_t batchSplitFactor,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Second phase of aclnnTransposeBatchMatMulWeightNz. Executes the computation.
 * @param [in] workspace: Address of the workspace allocated on the NPU device.
 * @param [in] workspaceSize: Workspace size on the NPU device, obtained from
 * aclnnTransposeBatchMatMulWeightNzGetWorkspaceSize.
 * @param [in] executor: Op executor containing the operator computation flow.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnTransposeBatchMatMulWeightNz(void* workspace, uint64_t workspaceSize,
                                                        aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_TRANSPOSE_BATCH_MAT_MUL
