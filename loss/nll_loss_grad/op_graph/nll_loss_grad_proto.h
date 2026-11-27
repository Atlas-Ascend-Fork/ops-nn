/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nll_loss_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_NLL_LOSS_GRAD_OPS_H_
#define OPS_OP_PROTO_INC_NLL_LOSS_GRAD_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief The negative log likelihood loss grad.

* @par Inputs:
* @li x: A 1D, 2D or 4D tensor dtype of float32 or bfloat16 or float16 with shape (C), (N, C) or (N, C, H, W). When x
* is 4D (shape (N, C, H, W)), the H and W dimensions must be positive integers.
* @li y_grad: The gradient of the forward output, dtype of float32 or bfloat16 or float16. When reduction is "none":
* if x is 2D (shape (N, C)), y_grad is 1D with shape (N); if x is 1D (shape (C,)), y_grad is a single-element tensor
* (0D scalar or 1D with 1 element); if x is 4D (shape (N, C, H, W)), y_grad is 3D with shape (N, H, W). When
* reduction is "mean" or "sum", y_grad is a single-element tensor (0D scalar or 1D with 1 element).
* @li target: Indicates the real label. A tensor dtype of int32, int64 or uint8. When x is 2D (shape (N, C)), target
* is 1D with shape (N); when x is 1D (shape (C,)), target is 0D scalar; when x is 4D (shape (N, C, H, W)), target is
* 3D with shape (N, H, W). Each element must be in [0, C - 1] or equal to ignore_index.
* @li weight: Indicates the weight of each class. A 1D tensor dtype of float32 or bfloat16 or float16 with shape (C).
* @li total_weight: A 1D tensor dtype of float32 or bfloat16 or float16 with shape (1).

* @par Attributes:
* @li reduction: Computation method of the loss function. An optional string. Defaults to "mean"; omitting the
* attribute is equivalent to "mean" (aclnn reduction=1). Valid values are "none", "mean" and "sum" (aclnn reduction
* 0/1/2); any other value is rejected.
* @li ignore_index: Specifies a target value that is ignored and does not affect the input gradient. An optional int.
* Defaults to -100; omitting the attribute is equivalent to -100 (aclnn ignoreIndex=-100). Target elements equal to
* this value do not contribute to the input gradient.

* @par Outputs:
* x_grad: A tensor has the same shape as "x". Must be the following type: float32, bfloat16, float16.

* @par Constraints:
* @li x supports ranks 1, 2 and 4 only; other ranks are rejected. Unsupported dtypes and non-positive H/W dimensions
* of 4D inputs are rejected by the backend validation with GRAPH_FAILED.
* @li Dynamic shape is supported: -1 in a shape denotes an unknown dimension and -2 denotes unknown rank; for an
* unknown-rank input the inferred output shape is unknown rank (-2).

* @par Third-party framework compatibility
* Compatible with pytorch NLLLossGrad operator
*/
REG_OP(NLLLossGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(target, TensorType({DT_INT32, DT_INT64, DT_UINT8}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(total_weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .OP_END_FACTORY_REG(NLLLossGrad)
} // namespace ge

#endif // OPS_OP_PROTO_INC_NLL_LOSS_GRAD_OPS_H_
