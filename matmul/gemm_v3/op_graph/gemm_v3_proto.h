/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_v3_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_GEMM_V3_PROTO_H_
#define OPS_MATMUL_GEMM_V3_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Multiplies matrix "a" by matrix "b", producing "alpha * a @ b + beta * c". \n
* @par Inputs:
* Three inputs, including:
* @li a: A matrix tensor. Must be one of the following types: float32, float16,
* bfloat16, int8, int32.
* @li b: A matrix tensor. Must be one of the following types: float32, float16,
* bfloat16, int8, int32.
* @li c: An optional matrix tensor. Must be one of the following types: float32,
* float16, bfloat16, int8, int32. \n

* @par Attributes:
* @li alpha: An optional attribute, the type is float. Defaults to 1.0.
* @li beta: An optional attribute, the type is float. Defaults to 1.0.
* @li transpose_a: Optional. A bool. If True, changes the shape of "a" from
* [M, K] to [K, M] before multiplication. Defaults to "false".
* @li transpose_b: Optional. A bool. If True, changes the shape of "b" from
* [K, N] to [N, K] before multiplication. Defaults to "false".
* @li enable_hf32: Optional. A bool. If True, enable enable_hi_float_32_execution.
                   If False, do not enable enable_hi_float_32_execution.
                   Defaults to "false".
* @par Outputs:
* y: The result matrix tensor. Must be one of the following types: float32,
* float16, int8, int32. Has format [ND], the format should be equal to a.
*/

// 注册guard与canndev保持同名(OPS_PROTO_DEF_GEMMV3):防止双包安装时matrix_calculation_ops.h
// 与本头文件重复注册GemmV3原型
#ifndef OPS_PROTO_DEF_GEMMV3
#define OPS_PROTO_DEF_GEMMV3
REG_OP(GemmV3)
    .INPUT(a, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT32}))
    .INPUT(b, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT32}))
    .OPTIONAL_INPUT(c, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT32}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(beta, Float, 1.0)
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(GemmV3)
#endif
} // namespace ge

#endif // OPS_MATMUL_GEMM_V3_PROTO_H_
