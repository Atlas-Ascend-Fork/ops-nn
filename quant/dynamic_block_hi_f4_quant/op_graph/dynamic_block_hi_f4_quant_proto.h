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
 * \file dynamic_block_hi_f4_quant_proto.h
 * \brief Operator prototype registration for DynamicBlockHiF4Quant
 */
#ifndef OPS_NN_DYNAMIC_BLOCK_HI_F4_QUANT_PROTO_H
#define OPS_NN_DYNAMIC_BLOCK_HI_F4_QUANT_PROTO_H

#include "graph/operator_reg.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

namespace ge {

/**
* @brief Performs dynamic block HiF4 quantization on input tensor.
* Quantizes the input tensor using HiF4 three-level hierarchical scaling architecture
* (Block=64 + Sub-block=8 + Micro-block=4) along both axis -1 and axis -2.

* @par Inputs:
* @li x: An input tensor of type float16 or bfloat16.
* The shape supports at least 2 dimensions, and at most 7 dimensions.

* @par Attributes:
* @li round_mode: An optional string. Defaults to "rint".
* Support "rint", "trunc".
* @li scale_alg: An optional int. Defaults to 0.
* Support 0, 1. When scale_alg=0, the Eb/Ec scale comparison thresholds are 4/2;
* when scale_alg=1, the thresholds are 3.75/1.875.

* @par Outputs:
* @li y: Quantized output tensor of type HIFLOAT4. It has the same shape and rank as input x.
* @li scale1: An output tensor of type HIFLOAT4_SCALE. Shape needs to meet the following conditions: \n
* - rank(scale1) = rank(x) + 1.
* - scale1.shape[:-2] = x.shape[:-1].
* - scale1.shape[-2] = x.shape[-1] // 64.
* - scale1.shape[-1] = 1.
* @li scale2: An output tensor of type HIFLOAT4_SCALE. Shape needs to meet the following conditions: \n
* - rank(scale2) = rank(x) + 1.
* - scale2.shape[:-3] = x.shape[:-2].
* - scale2.shape[-3] = x.shape[-2] // 64.
* - scale2.shape[-2] = x.shape[-1].
* - scale2.shape[-1] = 1.

* @attention Constraints:
* @li The input x rank should be in [2, 7].
* @li Both the -1 axis and -2 axis of input x must be multiples of 64.
* @li The output y is HIFLOAT4 format (S1P2), every two 4-bit elements packed as one uint8.
* @li The output scale1/scale2 are HIFLOAT4_SCALE format, three-level scales packed per Block.
*/
REG_OP(DynamicBlockHiF4Quant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_HIFLOAT4}))
    .OUTPUT(scale1, TensorType({Ops::NN::DT_HIFLOAT4_SCALE}))
    .OUTPUT(scale2, TensorType({Ops::NN::DT_HIFLOAT4_SCALE}))
    .ATTR(round_mode, String, "rint")
    .ATTR(scale_alg, Int, 0)
    .OP_END_FACTORY_REG(DynamicBlockHiF4Quant)

} // namespace ge
#endif // OPS_NN_DYNAMIC_BLOCK_HI_F4_QUANT_PROTO_H
