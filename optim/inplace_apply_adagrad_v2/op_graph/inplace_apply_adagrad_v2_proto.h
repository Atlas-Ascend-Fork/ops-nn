/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_INPLACE_APPLY_ADAGRAD_V2_H_
#define OPS_OP_PROTO_INC_INPLACE_APPLY_ADAGRAD_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Updates "var" according to the adagradv2 scheme.
 *   accum_new = accum + grad * grad
 *   var = var - lr * grad / (sqrt(accum_new) + epsilon)
 *
 * @par Inputs:
 * @li var: A mutable tensor. Must be one of the data types defined in
 * TensorType::NumberType(). Should be from a Variable().
 * @li accum: A mutable tensor. Has the same type as "var". Should be from a
 * Variable().
 * @li lr: A tensor for the learning rate. Has the same type as "var". Should be
 * from a Variable(). Single-element tensor.
 * @li epsilon: A tensor for the numerical stability constant added outside the
 * sqrt. Has the same type as "var". Single-element tensor.
 * @li grad: A tensor for the gradient. Has the same type as "var". Should be
 * from a Variable().
 *
 * @par Attributes:
 * @li update_slots: An optional bool. Defaults to "True".
 * If "True", "accum" will be updated
 * @li use_locking: An optional bool. Defaults to "False".
 * If "True", updating of the "var" tensor is protected by a lock;
 * otherwise the behavior is undefined, but may exhibit less contention.
 *
 * @par Outputs:
 * @li var: A mutable tensor. Has the same type as input "var".
 * @li accum: A mutable tensor. Has the same type as input "accum".
 *
 * @attention Constraints:
 * The input tensors must have the same shape.
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator ApplyAdagradV2.
 *
 */
REG_OP(InplaceApplyAdagradV2)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(InplaceApplyAdagradV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_INPLACE_APPLY_ADAGRAD_V2_H_
