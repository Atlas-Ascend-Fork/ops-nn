/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file einsum_proto.h
 * \brief
 */
#ifndef OPS_MATMUL_EINSUM_PROTO_H_
#define OPS_MATMUL_EINSUM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Sums the product of the elements of the input operands along
* dimensions specified using a notation based on the Einstein summation
* convention. \n

* @par Inputs:
* @li x: A list of Tensors. Must be one of the following types: int32,
* float16, float32. Tensors to be summed. It's a dynamic input. \n
* @attention Constraints:
* All input tensors must have the same dtype. \n

* @par Attributes:
* @li equation: The subscripts for the Einstein summation. \n
* @li N: tensor size of input. \n

* @par Outputs:
* @li y: The result Tensor. Must be one of the following types: int32,
* float16, float32. Same dtype as the input tensors. \n

* @attention Constraints:
* Input N must be Int. \n

* @par Third-party framework compatibility
* Compatible with Tensorflow 2.x einsum operator.
*/
#ifndef OPS_PROTO_DEF_EINSUM
#define OPS_PROTO_DEF_EINSUM
REG_OP(Einsum)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(equation, String)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Einsum)
#endif
} // namespace ge

#endif // OPS_MATMUL_EINSUM_PROTO_H_
