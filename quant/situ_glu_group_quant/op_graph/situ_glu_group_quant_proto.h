/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_glu_group_quant_proto.h
 * \brief SituGluGroupQuant operator prototype (GE graph mode IR registration),
 *        ported from cann-recipes-infer SituGluGroupQuant.
 */

#ifndef SITU_GLU_GROUP_QUANT_PROTO_H_
#define SITU_GLU_GROUP_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Sparse SiTU gated activation fused with MXFP8 (e4m3fn, group_size=32)
 * tail-axis dynamic quantization for grouped (MoE) rows.
 *
 * The op is bit-identical to the two small ops chained:
 * npu_situ_and_mul_sparse -> npu_dynamic_mx_quant(dst=e4m3fn).
 *
 * @par Inputs:
 * @li x: An input tensor of type float16 or bfloat16. The last dimension holds the
 * concatenated gate/up halves (2*EPR) and must be positive and even; EPR must be a multiple
 * of 32 in [128, 3072] (enforced by tiling).
 * @li weight: An optional float32 tensor of per-token scalar scales, length N
 * (= totalRows, global row index). Present: act = T(fp32(act) * w[row]) is
 * applied between activation and quantization. Absent: no scaling.
 * @li group_index: An optional int64 tensor. Present: each element is the token
 * (row) count of the corresponding expert; the kernel sums them to get the number of
 * valid rows, and rows beyond sum(group_index) are padding that stay untouched.
 * Absent: whole-tensor mode - ALL rows are processed (validRows = totalRows).
 *
 * @par Attributes:
 * @li beta: An optional float. SiTU gate-branch beta parameter. Must be greater than 0.
 * Defaults to 1.0.
 * @li linear_beta: An optional float. SiTU up-branch soft-clamp threshold
 * (up = linear_beta * tanh(up/linear_beta)). <= 0 disables the soft-clamp
 * (up passthrough); > 0 enables it (up output clamped to ±linear_beta).
 * Defaults to 0.0 (disabled).
 * @li activate_left: An optional bool. When true, gate is the first half and up the second
 * half; when false (default), gate is the second half (situ normal direction).
 * @li axis: An optional int. Quantization axis. Currently only supports -1 (last axis).
 * Defaults to -1.
 * @li quant_mode: An optional int. Quantization mode. Currently only supports 0 (mx
 * quantization). Defaults to 0.
 * @li dst_type: An optional int. Target quantization dtype: 35(FLOAT8_E5M2), 36(FLOAT8_E4M3FN),
 * 40(FLOAT4_E2M1), 41(FLOAT4_E1M2). Currently only supports 36. Defaults to 36.
 * @li dst_type_max: An optional float. Value cap of the target dtype, must be >= 0.
 * Defaults to 0.
 * @li round_mode: An optional string. Rounding mode: rint/round/floor. Currently only
 * supports "rint". Defaults to "rint".
 * @li group_list_type: An optional int. Semantics of group_index: 0 = cumsum mode
 * (each element is a group end offset; RESERVED and rejected until implemented),
 * 1 = count mode (each element is a per-group row count; current semantics).
 * Defaults to 0, so callers must explicitly pass 1 today.
 *
 * @par Outputs:
 * @li y: Quantized output, dtype float8_e4m3fn. Shape is x.shape[:-1] + [EPR].
 * @li y_scale: MX scale factors, dtype float8_e8m0. Shape is x.shape[:-1] +
 * [ceil(ceil(EPR/32)/2), 2].
 *
 * @par Constraints:
 * @li x last dimension must be positive and even; EPR = x.shape[-1] / 2.
 * @li EPR must not exceed 3072 (whole-row chunk cap).
 * @li beta must be greater than 0.
 * @li axis must be -1; quant_mode must be 0; dst_type must be 36; dst_type_max >= 0;
 * round_mode must be "rint" (single supported value each).
 * @li Empty tensors are not supported.
 *
 * @par Third-party framework compatibility
 * It is a custom operator. It has no corresponding operator in Caffe, ONNX, TensorFlow, or PyTorch.
 */
REG_OP(SituGluGroupQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT8_E4M3FN}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(beta, Float, 1.0f)
    .ATTR(linear_beta, Float, 0.0f)
    .ATTR(activate_left, Bool, false)
    .ATTR(axis, Int, -1)
    .ATTR(quant_mode, Int, 0)
    .ATTR(dst_type, Int, DT_FLOAT8_E4M3FN)
    .ATTR(dst_type_max, Float, 0.0f)
    .ATTR(round_mode, String, "rint")
    .ATTR(group_list_type, Int, 0)
    .OP_END_FACTORY_REG(SituGluGroupQuant)

} // namespace ge

#endif // SITU_GLU_GROUP_QUANT_PROTO_H_
