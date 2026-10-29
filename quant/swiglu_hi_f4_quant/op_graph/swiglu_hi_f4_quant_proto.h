/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SWIGLU_HI_F4_QUANT_PROTO_H_
#define SWIGLU_HI_F4_QUANT_PROTO_H_

#include "graph/operator_reg.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

namespace ge {

REG_OP(SwigluHiF4Quant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({Ops::NN::DT_HIFLOAT4}))
    .OUTPUT(scale, TensorType({Ops::NN::DT_HIFLOAT4_SCALE}))
    .ATTR(activate_dim, Int, -1)
    .ATTR(activate_left, Bool, false)
    .ATTR(swiglu_mode, Int, 0)
    .ATTR(clamp_limit, Float, 7.0f)
    .ATTR(glu_alpha, Float, 1.702f)
    .ATTR(glu_bias, Float, 1.0f)
    .ATTR(axis, Int, -1)
    .ATTR(scale_alg, Int, 0)
    .OP_END_FACTORY_REG(SwigluHiF4Quant)

} // namespace ge

#endif // SWIGLU_HI_F4_QUANT_PROTO_H_
