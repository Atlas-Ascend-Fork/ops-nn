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
 * \file lp_norm_reduce_v2_proto.h
 * \brief LpNormReduceV2 GE IR 原型定义（graph mode，带防重定义宏——仓级合并原型
 *        common/inc/op_graph/op_nn_proto_extend.h 亦含同名 REG_OP，二者由宏互斥）
 */

#ifndef OPS_NORM_LP_NORM_REDUCE_V2_PROTO_H_
#define OPS_NORM_LP_NORM_REDUCE_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes LpNormReduceV2.

 * @par Inputs:
 * x: A ND tensor of dtype float16, bfloat16, float32.
 *
 * @par Attributes:
 * @li p: An optional float, "inf" or "-inf", indicates the order of norm. Default is 2.0.
 * @li axes: ListInt, an optional attribute, indicates dimensions over which to compute the norm.
 * Default is {}, meaning all axes will be computed.
 * @li keepdim: An optional bool. If set to true, the reduced dimensions are retained in the result
 * as dimensions with size one. Default is false.
 * @li epsilon: An optional float. A value added to the denominator for numerical stability. Default is 1e-12.
 *
 * @par Outputs:
 * y: A ND tensor has the same dtype as "x". The shape of "y" is depending on "axes" and "keepdim".
 *
 * @attention Constraints:
 * @li When the attribute "p" is negative, there may be precision difference in the calculation results.
 * @li When the tensor "x" is empty and "p" < 0 or "p" is infinity, we cannot reduce the whole tensor or
 * reduce over an empty dimension.
 *
 * @par Third-party framework compatibility
 * Compatible with the Pytorch operator LpNormReduceV2.
 */
#ifndef OPS_PROTO_DEF_LPNORMREDUCEV2
#define OPS_PROTO_DEF_LPNORMREDUCEV2
REG_OP(LpNormReduceV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Float, 2.0)
    .ATTR(axes, ListInt, {})
    .ATTR(keepdim, Bool, false)
    .ATTR(epsilon, Float, 1e-12f)
    .OP_END_FACTORY_REG(LpNormReduceV2)
#endif // OPS_PROTO_DEF_LPNORMREDUCEV2
} // namespace ge

#endif // OPS_NORM_LP_NORM_REDUCE_V2_PROTO_H_
