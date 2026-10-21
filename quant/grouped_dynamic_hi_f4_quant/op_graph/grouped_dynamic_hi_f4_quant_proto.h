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
 * \file grouped_dynamic_hi_f4_quant_proto.h
 * \brief
 */

#ifndef OPS_NN_DEV_GROUPED_DYNAMIC_HIF4_QUANT_PROTO_H
#define OPS_NN_DEV_GROUPED_DYNAMIC_HIF4_QUANT_PROTO_H

#include "graph/operator_reg.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

namespace ge {
/**
* @brief Performs grouped dynamic hifloat4 quantization on input tensor.
* Quantizes the input tensor using scaling factors of size 64.

* @par Inputs:
* @li x: An input tensor of type float16 or bfloat16, specifying the input.
* The shape only supports 2 dimensions.
* @li group_index: A tensor of type int32, specifying the index of groups.
* The shape only supports 1 dimension.

* @par Outputs:
* @li y: Quantized output tensor. It has the same shape and rank as input x.
* @li scale: An output tensor of type FLOAT32. Shape needs to meet the following conditions: \n
* - scale.shape[0] = x.shape[0] / 64 + group_index.shape[0].
* - Other dimensions match input x.

* @par Attributes:
* @li scale_alg: An optional int. Defaults to 0. Support 0, 1.
* When scale_alg=0, the Eb/Ec scale comparison thresholds are 4/2;
* when scale_alg=1, the thresholds are 3.75/1.875.

* @attention Constraints:
* @li The input x last dimension of the shape must be divisible by 2.
* @li The -1 axis of the input x must be multiples of 64.
*/
REG_OP(GroupedDynamicHiF4Quant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_HIFLOAT4}))
    .OUTPUT(scale, TensorType({Ops::NN::DT_HIFLOAT4_SCALE}))
    .ATTR(scale_alg, Int, 0)
    .OP_END_FACTORY_REG(GroupedDynamicHiF4Quant)
} // namespace ge

#endif // OPS_NN_DEV_GROUPED_DYNAMIC_HIF4_QUANT_PROTO_H
