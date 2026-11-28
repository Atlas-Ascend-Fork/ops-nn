/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_TRANSPOSE_QUANT_BATCH_MAT_MUL
#define OP_API_INC_TRANSPOSE_QUANT_BATCH_MAT_MUL

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief First phase of aclnnTransposeQuantBatchMatMul. Computes the workspace size for the given computation flow.
 * @domain aclnn_ops_infer
 * Operator: TransposeQuantBatchMatMul.
 * @param [in] x1: Left matmul matrix. Supported dtypes: float8_e4m3fn, float8_e5m2, float4_e2m1 (MX),
 * hifloat8 (HIFP8), int8 (INT8 W8A8).
 * @param [in] x2: Right matmul matrix. Supported dtypes: float8_e4m3fn, float8_e5m2, float4_e2m1 (MX),
 * hifloat8 (HIFP8), int8 (INT8 W8A8).
 * @param [in] bias: Bias. Not supported currently.
 * @param [in] x1Scale: Quantization scale. Supported dtypes: float32, float8_e8m0, uint64/int64 (HIFP8).
 * @param [in] x2Scale: Quantization scale. Supported dtypes: float32, float8_e8m0, uint64/int64 (HIFP8).
 * @param [in] dtype: Output matrix dtype. Supported dtype: int32.
 * @param [in] groupSize: Group size of the output matrix. Supported dtype: int64.
 * @param [in] permX1: Transpose permutation array of x1.
 * @param [in] permX2: Transpose permutation array of x2.
 * @param [in] permY: Transpose permutation array of y.
 * @param [in] batchSplitFactor: Whether to split the shape again. Supported dtype: int32.
 * @param [out] out: Result. Supported dtypes: float16, bfloat16, hifloat8 (HIFP8).
 * @param [out] workspaceSize: Size of the workspace to allocate on the NPU device.
 * @param [out] executor: Op executor containing the operator computation flow.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    int32_t dtype, int64_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
    int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Second phase of aclnnTransposeQuantBatchMatMul. Executes the computation.
 */
ACLNN_API aclnnStatus aclnnTransposeQuantBatchMatMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     const aclrtStream stream);

/**
 * @brief First phase of aclnnTransposeQuantBatchMatMulWeightNz. Computes the workspace size.
 * @domain aclnn_ops_infer
 * Operator: TransposeQuantBatchMatMulWeightNz.
 * @param [in] x1: Left matmul matrix. Supported dtypes: float8_e4m3fn, float4_e2m1 (MX), hifloat8 (HIFP8), int8
 * (INT8 W8A8).
 * @param [in] x2: Right matmul matrix. Supported dtypes: float8_e4m3fn, float4_e2m1 (MX), hifloat8 (HIFP8), int8
 * (INT8 W8A8).
 * @param [in] bias: Bias. Not supported currently.
 * @param [in] x1Scale: Quantization scale. Supported dtypes: float8_e8m0, uint64/int64 (HIFP8), float32 (INT8 W8A8).
 * Optional in the HIFP8 scenario.
 * @param [in] x2Scale: Quantization scale. Supported dtypes: float8_e8m0, uint64/int64 (HIFP8), float32 (INT8 W8A8).
 * @param [in] dtype: Output matrix dtype. Supported dtype: int32.
 * @param [in] groupSize: Group size of the output matrix. Supported dtype: int64.
 * @param [in] permX1: Transpose permutation array of x1.
 * @param [in] permX2: Transpose permutation array of x2.
 * @param [in] permY: Transpose permutation array of y.
 * @param [in] batchSplitFactor: Whether to split the shape again. Supported dtype: int32.
 * @param [out] out: Result. Supported dtypes: float16, bfloat16, hifloat8 (HIFP8).
 * @param [out] workspaceSize: Size of the workspace to allocate on the NPU device.
 * @param [out] executor: Op executor containing the operator computation flow.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnTransposeQuantBatchMatMulWeightNzGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    int32_t dtype, int64_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
    int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Second phase of aclnnTransposeQuantBatchMatMulWeightNz. Executes the computation.
 * @param [in] workspace: Address of the workspace allocated on the NPU device.
 * @param [in] workspaceSize: Workspace size on the NPU device, obtained from
 * aclnnTransposeQuantBatchMatMulWeightNzGetWorkspaceSize.
 * @param [in] executor: Op executor containing the operator computation flow.
 * @param [in] stream: ACL stream.
 * @return aclnnStatus: Status code.
 */
ACLNN_API aclnnStatus aclnnTransposeQuantBatchMatMulWeightNz(void* workspace, uint64_t workspaceSize,
                                                             aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_TRANSPOSE_QUANT_BATCH_MAT_MUL
