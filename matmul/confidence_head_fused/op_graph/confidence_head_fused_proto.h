/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATMUL_CONFIDENCE_HEAD_FUSED_PROTO_H_
#define OPS_MATMUL_CONFIDENCE_HEAD_FUSED_PROTO_H_

#include <cstdint>
#include <limits>
#include "graph/operator_reg.h"

namespace ge {

REG_OP(ConfidenceHeadFused)
    .INPUT(hidden, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(markov_emb, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(temperature, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(out, TensorType({DT_FLOAT}))
    .ATTR(clamp_low, Float, -static_cast<float>(std::numeric_limits<int32_t>::max()))
    .ATTR(clamp_high, Float, static_cast<float>(std::numeric_limits<int32_t>::max()))
    .OP_END_FACTORY_REG(ConfidenceHeadFused)

} // namespace ge

#endif // OPS_MATMUL_CONFIDENCE_HEAD_FUSED_PROTO_H_
