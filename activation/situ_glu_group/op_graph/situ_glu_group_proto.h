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
 * \file situ_glu_group_proto.h
 * \brief SituGluGroup prototype (SiTU-gated linear unit with optional weight / group_index).
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SITU_GLU_GROUP_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SITU_GLU_GROUP_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief SiTU gated linear unit activation.
 *
 * @par Inputs:
 * Three inputs:
 * @li x: A tensor. Type is float32/float16/bfloat16. The dim dimension must be divisible by 2.
 * @li weight: An optional float32 tensor. Per-row scalar multiplied into the result. The element count must equal
 *   the product of the x dims before the dim axis.
 * @li group_index: An optional int64 tensor. Group token numbers whose semantics are selected by
 *   group_list_type. It must be 1D, its element values must be greater than or equal to 0, and it does not
 *   support being an empty tensor on its own. In count mode (group_list_type=1) the total is sum(group_index);
 *   in cumsum mode (group_list_type=0) only the last element group_index[-1] is used (intermediate elements do
 *   not participate and need not be monotonic). A non-negative total is legal; a zero total writes no rows.
 *   Only the first min(total, row count) rows are computed and written; the remaining rows are left untouched
 *   (contents undefined).
 *
 * @par Outputs:
 * One output:
 * @li y: A tensor. Type is float32/float16/bfloat16. The dim dimension is half of x.
 *
 * @par Attributes:
 * Five attributes:
 * @li dim: An optional int. The dimension to be split, value in [-xDim, xDim-1], default is -1.
 * @li beta: An optional float. The scale factor for the SiTU gate activation. It must be non-zero
 *   (the formula divides by beta), default is 1.0.
 * @li linear_beta: An optional float. The scale factor for the linear tanh on the up path. When <= 0, the up path is
 *   used as-is. default is 0.0.
 * @li activate_left: An optional bool. Whether the left (front) half of x is the gate. true: gate is the front half and
 *   up is the back half; false: gate is the back half and up is the front half. default is true.
 * @li group_list_type: An optional int. Semantics of group_index: 0 = cumsum mode (each element is a cumulative
 *   group end offset), 1 = count mode (each element is a per-group row count). Defaults to 0.
 *
 * @attention Constraints:
 * The dim dimension of x must be divisible by 2, the dim dimension of y must be equal to the dim dimension of x
 * divided by 2, and beta must be non-zero.
 */
REG_OP(SituGluGroup)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(dim, Int, -1)
    .ATTR(beta, Float, 1.0)
    .ATTR(linear_beta, Float, 0.0)
    .ATTR(activate_left, Bool, true)
    .ATTR(group_list_type, Int, 0)
    .OP_END_FACTORY_REG(SituGluGroup)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_SITU_GLU_GROUP_PROTO_H_
