/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\\file anti_hi_f4_quant_proto.h
 * \brief
 */

#ifndef OPS_NN_DEV_ANTI_HI_F4_QUANT_PROTO_H
#define OPS_NN_DEV_ANTI_HI_F4_QUANT_PROTO_H

#include "graph/operator_reg.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

namespace ge {
/**
* @brief Performs HiFloat4 de-quantization on input tensor.
* Converts HiFloat4 (S1P2 encoded, 4-bit) quantized data to high-precision
* floating-point format (FLOAT16 or BFLOAT16) using three-layer scaling factors.

* @par Inputs:
* @li x: An input tensor of type HiFloat4 (physical storage: UINT8).
* The shape supports at least 2 dimensions, and at most 7 dimensions.
* The last dimension must be even (S1P2 packing requirement).
* Both the last dimension and the second-last dimension must be multiples of 64 (HiF4 Block granularity).
* @li scale: An input tensor of type HiF4Scale (physical storage: UINT8),
* containing the three-layer scaling factors (Ea/Eb/Ec).
* - PER_CHANNEL: scale shape = [x[0], ..., x[n-2], ceil(x[-1]/64), 1] (n+1维, axis=-1量化).
 * - PER_HEAD: scale shape = [x[0], ..., ceil(x[-2]/64), x[-1], 1] (n+1维, axis=-2量化).

* @par Attributes:
* @li dst_type: An optional int, specifying the output data type.
* The value range is {1, 27}. 1=FLOAT16, 27=BFLOAT16. Defaults to 27.

* @par Outputs:
* @li y: De-quantized output tensor. It has the same shape as input x.
* Data type is determined by dst_type attribute (FLOAT16 or BFLOAT16).

* @attention Constraints:
* @li The last dimension of input x must be even (S1P2 packing).
* @li Both the last dimension and the second-last dimension of input x must be multiples of 64 (HiF4 Block).
* @li HiF4 has no offset: fixed zeroPointsType=NO_OFFSET.
* @li HiF4 has no sqrt: fixed sqrtMode=NO_SQRT_MODE.
*/
REG_OP(AntiHiF4Quant)
    .INPUT(x, TensorType({DT_HIFLOAT4}))
    .INPUT(scale, TensorType({Ops::NN::DT_HIFLOAT4_SCALE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(dst_type, Int, 27)
    .OP_END_FACTORY_REG(AntiHiF4Quant)
} // namespace ge

#endif // OPS_NN_DEV_ANTI_HI_F4_QUANT_PROTO_H
