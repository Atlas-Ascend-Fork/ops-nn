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
 * \file relu6_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_RELU6_GRAD_H_
#define OPS_OP_PROTO_INC_RELU6_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Gradient of ReLU6. Strict open-interval semantics:
*        dx[i] = gradients[i]  if  0 < features[i] < 6
*        dx[i] = 0              otherwise (including features[i] == 0 and 6)
*        On <term>Ascend 950PR&950DT系列产品</term>, gradients and features support
*        bidirectional broadcasting: dimensions are aligned from the right and
*        each pair must be equal or contain a dimension of size 1.
*        On <term>Atlas A3系列产品</term> and <term>Atlas A2系列产品</term>, features
*        must have the same shape as gradients or have shape [1].

* @par Inputs:
* Two inputs, including:
* @li gradients: A ND tensor. Upstream gradient (dy). Must be one of the
*     following types: float16, float32, bfloat16.
* @li features: A ND tensor. Forward input or output of ReLU6; either gives
*     the same interval mask. Must have the same dtype as gradients and a
*     shape satisfying the product-specific constraints above. \n

* @par Outputs:
* backprops: A ND tensor. Has the same dtype as gradients; shape is the
*            broadcast shape under the product-specific constraints above. \n

* @par Third-party framework compatibility
* For finite gradients, matches TensorFlow Relu6Grad and PyTorch
* hardtanh_backward(min=0, max=6) at features in {0, 6}, returning 0. Unlike multiplication by a zero mask,
* the selection semantics above also return 0 for NaN/Inf gradients
* outside the open interval.
*/
#ifndef OPS_PROTO_DEF_RELU6GRAD
#define OPS_PROTO_DEF_RELU6GRAD
REG_OP(Relu6Grad)
    .INPUT(gradients, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(backprops, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(Relu6Grad)
#endif // OPS_PROTO_DEF_RELU6GRAD

} // namespace ge

#endif // OPS_OP_PROTO_INC_RELU6_GRAD_H_
