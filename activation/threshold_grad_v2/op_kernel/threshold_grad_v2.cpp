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
 * \file threshold_grad_v2_apt.cpp
 * \brief threshold_grad_v2_apt
 */

#include "kernel_operator.h"
#include "arch35/threshold_grad_v2_dag.h"
#include "arch35/threshold_grad_v2_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace ThresholdGradV2Op;

template <uint64_t schMode, uint64_t dtype>
__global__ __aicore__ void threshold_grad_v2(GM_ADDR gradients, GM_ADDR features, GM_ADDR threshold, GM_ADDR backprops,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_TPL_FP16)) {
        BroadcastSch<schMode, ThresholdGradV2Dag<half>::OpDag> sch(tiling);
        sch.Process(gradients, features, threshold, backprops);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_TPL_BF16)) {
        BroadcastSch<schMode, ThresholdGradV2Dag<bfloat16_t>::OpDag> sch(tiling);
        sch.Process(gradients, features, threshold, backprops);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_TPL_FP32)) {
        BroadcastSch<schMode, ThresholdGradV2Dag<float>::OpDag> sch(tiling);
        sch.Process(gradients, features, threshold, backprops);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_TPL_INT32)) {
        BroadcastSch<schMode, ThresholdGradV2Dag<int32_t>::OpDag> sch(tiling);
        sch.Process(gradients, features, threshold, backprops);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_TPL_INT8)) {
        BroadcastSch<schMode, ThresholdGradV2Dag<int8_t>::OpDag> sch(tiling);
        sch.Process(gradients, features, threshold, backprops);
    } else if constexpr (dtype == static_cast<uint64_t>(THRESHOLD_GRAD_V2_TPL_UINT8)) {
        BroadcastSch<schMode, ThresholdGradV2Dag<uint8_t>::OpDag> sch(tiling);
        sch.Process(gradients, features, threshold, backprops);
    }
    return;
}
