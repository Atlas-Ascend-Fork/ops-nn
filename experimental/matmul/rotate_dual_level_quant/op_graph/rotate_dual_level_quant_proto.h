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
 * \file rotate_dual_level_quant_proto.h
 * \brief
 */
#ifndef OPS_ROTATE_DUAL_LEVEL_QUANT_PROTO_H
#define OPS_ROTATE_DUAL_LEVEL_QUANT_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Performs rotation transformation on the input tensor x using rotation matrices,
 * followed by optional clamp, 512-element level0 normalization and 32-element dynamic MX quantization. \n
 *
 * @par Inputs:
 * @li x: A tensor. Input data for rotation and dual-level quantization.
 * Must be one of the following types: float16, bfloat16. The format supports ND. \n
 * @li rot: A tensor. Shared or per-column-block rotation matrices.
 * Must be one of the following types: float16, bfloat16. Has the same type as input "x".
 * The format supports ND. \n
 * @li alpha: An optional tensor. Scaling coefficient for the clamp range of each 32-element group.
 * Must be bfloat16 with shape [1]. Clamp is enabled only when the value is in (0.0, 1.0).
 * The format supports ND. \n
 * @li group_list: An optional tensor. Describes consecutive row groups along the M axis.
 * Must be int64 with shape [E]. If absent, the operator uses ungrouped mode. The format supports ND. \n
 *
 * @par Outputs:
 * @li y: A tensor. Quantized output with the same shape as input "x".
 * Must be one of the following types: float4_e2m1, float8_e4m3fn, float8_e5m2.
 * The format supports ND. \n
 * @li level0_scale: A tensor. One level0 normalization scale for each block of up to 512 elements.
 * Must be float32 with shape [*, ceilDiv(N, 512)]. The format supports ND. \n
 * @li level1_scale: A tensor. One shared MX scale for each 32-element group, stored in pairs.
 * Must be float8_e8m0 with shape [*, ceilDiv(N, 64), 2]. The format supports ND. \n
 *
 * @par Attributes:
 * @li y_dtype: An optional int. Specifies the output data type of y.
 * Supports DT_FLOAT4_E2M1, DT_FLOAT8_E4M3FN and DT_FLOAT8_E5M2. Defaults to DT_FLOAT4_E2M1. \n
 * @li axis: An optional int. Specifies the quantization axis.
 * Must be -1 or the nonnegative index of the last input dimension. Defaults to -1. \n
 * @li round_mode: An optional string. Specifies the rounding mode for y.
 * Supports "rint", "round" and "floor" for float4_e2m1; supports only "rint" for FP8 outputs.
 * Defaults to "rint". \n
 * @li scale_alg: An optional int. Specifies the level1 MX scale algorithm.
 * Supports 0 or 2 for float4_e2m1, and 0 or 1 for FP8 outputs. Defaults to 0. \n
 * @li dst_type_max: An optional float. Specifies the destination range parameter for scale_alg 2.
 * Must be a finite value in [6.0, 12.0] when scale_alg is 2, and 0.0 otherwise. Defaults to 0.0. \n
 * @li trans: An optional bool. Specifies whether to transpose. Only false is supported. Defaults to false. \n
 * @li group_list_type: An optional int. Specifies the representation of group_list.
 * 0 indicates cumulative row counts; 1 indicates the row count of each group. Defaults to 0. \n
 * @li level0_block_size: An optional int. Specifies the level0 block size.
 * Only 512 is supported. Defaults to 512. \n
 * @li level1_block_size: An optional int. Specifies the level1 MX block size.
 * Only 32 is supported. Defaults to 32. \n
 *
 * @par Constraints:
 * - Only Ascend 950PR&950DT products are supported.
 * - In ungrouped mode, x is 1-D to 7-D with shape [*, N], where * denotes all leading dimensions.
 *   M is the product of the leading dimensions, or 1 for a 1-D input.
 *   rot has shape [K, K] or [N/K, K, K].
 * - In grouped mode, x has shape [M, N], group_list has shape [E], and rot has shape [E, K, K]
 *   or [E, N/K, K, K]. E must be in [1, 1024] and N must be positive.
 * - rot must contain square matrices. K must be one of {32, 64, 128, 256}, and N must be divisible by K.
 * - At runtime, each dimension of x and the flattened M must be in [0, 2147483647].
 * - For group_list_type 0, group_list values must be nonnegative and nondecreasing,
 *   with the last value no greater than M. For group_list_type 1, values must be nonnegative
 *   and their sum must not exceed M. Empty groups are allowed.
 *   Rows outside the covered prefix retain their initial values in all three outputs.
 * - Rotation results and normalized intermediate data use bfloat16, including for float16 inputs.
 *   Optional clamp is applied to each 32-element group before level0 normalization.
 * - Both quantization levels partition the original N axis independently for each row.
 *   A level0 block may contain multiple K-wide rotation blocks. The last level0 block uses its valid elements.
 * - For finite data with a nonzero normal level0 scale, level0_scale = a / C_MAX,
 *   where a is the maximum absolute value of the rotated and optionally clamped data in the level0 block.
 *   C_MAX is 6, 448 or 57344 for float4_e2m1, float8_e4m3fn or float8_e5m2, respectively.
 *   scale_alg and dst_type_max affect level1 only and do not change this level0 denominator.
 * - Level1 applies MX quantization to the normalized data. If ceilDiv(N, 32) is odd,
 *   the last level1_scale storage byte of each row is zero padding. \n
 *
 * The alpha and group_list columns below specify their types when the optional inputs are present.
 *
 * | x       | rot     | alpha | group_list | y             | level0_scale | level1_scale |
 * |---------|---------|-------|------------|---------------|--------------|--------------|
 * | FLOAT16 | FLOAT16 | BF16  | INT64      | FLOAT4_E2M1   | FLOAT32      | FLOAT8_E8M0  |
 * | BF16    | BF16    | BF16  | INT64      | FLOAT4_E2M1   | FLOAT32      | FLOAT8_E8M0  |
 * | FLOAT16 | FLOAT16 | BF16  | INT64      | FLOAT8_E4M3FN | FLOAT32      | FLOAT8_E8M0  |
 * | BF16    | BF16    | BF16  | INT64      | FLOAT8_E4M3FN | FLOAT32      | FLOAT8_E8M0  |
 * | FLOAT16 | FLOAT16 | BF16  | INT64      | FLOAT8_E5M2   | FLOAT32      | FLOAT8_E8M0  |
 * | BF16    | BF16    | BF16  | INT64      | FLOAT8_E5M2   | FLOAT32      | FLOAT8_E8M0  |
 */
REG_OP(RotateDualLevelQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rot, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(alpha, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT4_E2M1, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(level0_scale, TensorType({DT_FLOAT}))
    .OUTPUT(level1_scale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(y_dtype, Int, DT_FLOAT4_E2M1)
    .ATTR(axis, Int, -1)
    .ATTR(round_mode, String, "rint")
    .ATTR(scale_alg, Int, 0)
    .ATTR(dst_type_max, Float, 0.0)
    .ATTR(trans, Bool, false)
    .ATTR(group_list_type, Int, 0)
    .ATTR(level0_block_size, Int, 512)
    .ATTR(level1_block_size, Int, 32)
    .OP_END_FACTORY_REG(RotateDualLevelQuant)
} // namespace ge
#endif // OPS_ROTATE_DUAL_LEVEL_QUANT_PROTO_H
