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
 * \file batch_norm3d_proto.h
 * \brief
 */
#ifndef OPS_NORM_BATCH_NORM3D_PROTO_H_
#define OPS_NORM_BATCH_NORM3D_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Performs batch normalization on a 5D tensor.

* The shape, format and saved-statistics descriptions below apply to
* Ascend 950PR&950DT系列产品. See the operator README for other supported products.

*@par Inputs:
*@li x: A 5D float16 or float32 tensor in NCDHW or NDHWC format.
*@li scale: A 1D float32 tensor in ND format, with one scaling factor per channel.
*@li offset: A 1D float32 tensor in ND format, with one offset per channel.
*@li mean: An optional 1D float32 tensor in ND format, with one mean per channel.
Required for inference. May be omitted in training; supplied values are not used in training.
*@li variance: An optional 1D float32 tensor in ND format, with one variance per channel.
Required for inference. May be omitted in training; supplied values are not used in training.

*@par Attributes:
*@li epsilon: An optional float32 added to the variance before taking the square root.
Defaults to 0.0001. A finite positive value is recommended for numerical stability.
*@li data_format: An optional string, NCDHW or NDHWC, matching the format of x.
Defaults to NCDHW.
*@li is_training: An optional bool selecting training or inference. Defaults to true.

*@par Outputs:
* All five outputs are required. The four statistics tensors are 1D float32 tensors
* in ND format, with one value per channel.
*@li y: Normalized x, with the same shape, data type and format as x.
*@li batch_mean: The per-channel batch mean in training; the input mean in inference.
*@li batch_variance: The unbiased batch variance in training (biased variance multiplied
by M/(M-1), where M=N*D*H*W); the input variance in inference. NaN when M=1 in training.
*@li reserve_space_1: The batch mean saved for gradient computation in training;
the input mean in inference.
*@li reserve_space_2: The reciprocal standard deviation 1/sqrt(biased variance+epsilon)
saved for gradient computation in training; the input variance in inference.

*@attention Constraints:
*@li Runtime dimensions of x must be positive. All channel tensors must have length C.
*@li In training, normalization uses the biased batch variance, not batch_variance.
*@li Invalid shapes and attributes are rejected during graph validation or tiling;
shape/type inference alone is not a validation interface.

*@par Third-party framework reference
*@li For TensorFlow, the corresponding 5D normalization interface is
tf.compat.v1.nn.fused_batch_norm. Its default epsilon and returned tensors
differ from this GEIR interface; follow the input, output and attribute
contracts above.
*/
#ifndef OPS_PROTO_DEF_BATCHNORM3D
#define OPS_PROTO_DEF_BATCHNORM3D
REG_OP(BatchNorm3D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(BatchNorm3D)
#endif // OPS_PROTO_DEF_BATCHNORM3D
} // namespace ge

#endif // OPS_NORM_BATCH_NORM3D_PROTO_H_
