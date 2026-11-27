/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Graph-mode operator prototype for NormalizeBatch (Ascend 950 / arch35).
//   公开原型（与 common/inc/op_graph/op_nn_proto_extend.h 中既有声明保持一致）：
//   - Input  input_x   : REQUIRED, {DT_FLOAT}, FORMAT_ND, shape (n, c, d)
//   - Input  seq_len   : REQUIRED, {DT_INT32}, FORMAT_ND, shape (n,)
//   - Output output_y  : REQUIRED, {DT_FLOAT}, FORMAT_ND, shape (n, c, d)
//   - Attr   normalize_type : REQUIRED String, "per_feature" | "all_features"
//   - Attr   epsilon        : OPTIONAL Float, 默认 1e-5

#ifndef OPS_NORM_NORMALIZE_BATCH_PROTO_H_
#define OPS_NORM_NORMALIZE_BATCH_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
#ifndef OPS_PROTO_DEF_NORMALIZEBATCH
#define OPS_PROTO_DEF_NORMALIZEBATCH
/**
 * @brief Performs batch normalization on variable-length sequences. \n
 * @par Inputs:
 * Two inputs
 * @li input_x: A Tensor. Support float32. shape (n, c, d).
 * @li seq_len: A Tensor. Each batch normalize data num. Support Int32. Shape (n, ). \n
 * @par Attributes:
 * @li normalize_type: Str. Support "per_feature" or "all_features".
 * @li epsilon: An optional float32, specifying the small value added to
 * std to avoid dividing by zero. Defaults to "0.00001" . \n
 * @par Outputs:
 * One outputs
 * @li output_y: A Tensor for the normalized "input_x". Support float32. shape (n, c, d).\n
 */
REG_OP(NormalizeBatch)
    .INPUT(input_x, TensorType({DT_FLOAT}))
    .INPUT(seq_len, TensorType({DT_INT32}))
    .OUTPUT(output_y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(normalize_type, String)
    .ATTR(epsilon, Float, 0.00001f)
    .OP_END_FACTORY_REG(NormalizeBatch)
#endif // OPS_PROTO_DEF_NORMALIZEBATCH
} // namespace ge

#endif // OPS_NORM_NORMALIZE_BATCH_PROTO_H_
