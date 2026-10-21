/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */
/*!
 * \file dynamic_hi_f4_quant_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_DYNAMIC_HIF4_QUANT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_DYNAMIC_HIF4_QUANT_OPS_H_
#include "graph/operator_reg.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

namespace ge {
/**
* @brief Performs dynamic hifloat4 quantization on input tensor.
* Quantizes the input tensor using scaling factors of size 64.

* @par Inputs:
* @li x: An input tensor of type float16 or bfloat16.
* The shape supports at least 2 dimensions, and at most 7 dimensions.

* @par Attributes:
* @li axis: An optional int. Axis along which to quantize. Defaults to -1.
* must be -1 or -2.


* @par Outputs:
* @li y: Quantized output tensor. It has the same shape and rank as input x.
* @li mxscale: An output tensor of type FLOAT32. Shape needs to meet the following conditions: \n
* - rank(mxscale) = rank(x) + 1.
* - axis_change = -1 or axis_change = -2.
* - mxscale.shape[axis_change] = x.shape[axis_change] / 64.
* - Other dimensions match input x.


* @attention Constraints:
* @li The input x last dimension of the shape must be divisible by 2.
* @li The granularity of quantization must be 64.
* @li If the quantization axis is the -1 axis, the -1 axis of the input x must be a multiple of 64.
* @li If the quantization axis is the -2 axis, both the -1 and -2 axis of the input x must be multiples of 64.
*/
REG_OP(DynamicHiF4Quant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_HIFLOAT4}))
    .OUTPUT(scale, TensorType({Ops::NN::DT_HIFLOAT4_SCALE}))
    .ATTR(axis, Int, -1)
    .ATTR(scale_alg, Int, 0)
    .OP_END_FACTORY_REG(DynamicHiF4Quant)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_DYNAMIC_HIF4_QUANT_OPS_H_
