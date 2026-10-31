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
 * \file mvn_v2_proto.h
 * \brief MVNV2 operator GE IR prototype definition (graph mode).
 *
 * Mean variance normalization V2:
 *   y = (x - mean) / (sqrt(var) + eps)
 *   Mean and variance are reduced along the specified axes. The output has
 *   the same shape and dtype as x.
 */
#ifndef OPS_OP_PROTO_INC_MVN_V2_H_
#define OPS_OP_PROTO_INC_MVN_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Mean Variance Normalization V2: y = (x - mean) / (std + eps).
*   Reduces mean and variance along specified axes, then normalizes.

 * @par Inputs:
 * @li x: A 4-D NCHW tensor of type float16/float32, describing the input feature map.

 * @par Attributes:
 * @li eps: An optional float32 value added to std. Defaults to 1e-9. Negative and non-finite values
 *     are accepted for compatibility and follow IEEE floating-point propagation.
 * @li axes: An optional list int, reduction axes. Elements must be in [-4, 4); negative values are
 *     normalized by adding 4. An absent or empty list selects the default axes [0, 2, 3]. Axes use
 *     set semantics: duplicates are ignored and order does not affect the result.

 * @par Outputs:
 * @li y: A 4-D NCHW tensor of the same dtype and shape as input x, describing the normalized result.
 */
#ifndef OPS_PROTO_DEF_MVNV2
#define OPS_PROTO_DEF_MVNV2

REG_OP(MVNV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(eps, Float, 1.0e-9f)
    .ATTR(axes, ListInt, {0, 2, 3})
    .OP_END_FACTORY_REG(MVNV2)

#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_MVN_V2_H_
