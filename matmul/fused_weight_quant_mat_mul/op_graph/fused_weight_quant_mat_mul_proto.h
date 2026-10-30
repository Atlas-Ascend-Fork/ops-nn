/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_weight_quant_mat_mul_proto.h
 * \brief FusedWeightQuantMatMul 算子原型。
 *        输入布局与 WeightQuantBatchMatmulV2 严格对齐（索引 0-6 不变），保证融合 pass
 *        可直传输入、tiling/kernel 复用不受输入布局影响。
 *        属性不注册 inner_precise（融合算子固定高精度模式），fused_op_type 占索引 4。
 */

#ifndef OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_PROTO_H
#define OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
REG_OP(FusedWeightQuantMatMul)
    .INPUT(x, TensorType({DT_FLOAT16}))            // A16，仅 FP16
    .INPUT(weight, TensorType({DT_INT8, DT_INT4})) // W8/W4
    .INPUT(antiquant_scale, TensorType({DT_FLOAT, DT_UINT64}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT})) // 仅 float32
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_UINT64}))     // 输出量化，存在时 y 为 INT8
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_UINT64}))    // 仅 uint64
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT8}))
    .ATTR(transpose_x, Bool, false)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(dtype, Int, -1)
    .ATTR(fused_op_type, String, "relu") // 预留 gelu 等扩展
    .OP_END_FACTORY_REG(FusedWeightQuantMatMul)
} // namespace ge

#endif // OPS_MATMUL_FUSED_WEIGHT_QUANT_MAT_MUL_PROTO_H
