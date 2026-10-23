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
 * \file threshold_grad_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_THRESHOLD_GRAD_V2_H_
#define OPS_OP_PROTO_INC_THRESHOLD_GRAD_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Thresholds each element of the input gradient with a tensor threshold.

 * @par Inputs:
 * Three inputs, including:
 * @li gradients: A tensor of type int8, uint8, int32, float16, bfloat16 or float.
 * @li features: A tensor of the same type as "gradients", broadcast compatible with "gradients".
 * @li threshold: A tensor of the same type as "gradients", broadcast compatible with "gradients". \n

 * @par Outputs:
 * backprops: A tensor with the same type as "gradients" and the broadcast shape of the three inputs.
 */
#ifndef OPS_PROTO_DEF_THRESHOLDGRADV2
#define OPS_PROTO_DEF_THRESHOLDGRADV2
REG_OP(ThresholdGradV2)
    .INPUT(gradients, TensorType({DT_INT8, DT_UINT8, DT_INT32, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(features, TensorType({DT_INT8, DT_UINT8, DT_INT32, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(threshold, TensorType({DT_INT8, DT_UINT8, DT_INT32, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(backprops, TensorType({DT_INT8, DT_UINT8, DT_INT32, DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OP_END_FACTORY_REG(ThresholdGradV2)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_THRESHOLD_GRAD_V2_H_
