/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fake_quant_with_min_max_vars_per_channel_gradient.cpp
 * \brief Kernel entry for FakeQuantWithMinMaxVarsPerChannelGradient (FP32 / arch35，新写法).
 *
 * 模板参数（与 ..._tiling_key.h 对应）：
 *   - MODE = FQPCG_TPL_MODE_NORMAL (0)：常规 regbase 模板 Kernel<float>
 *   - MODE = FQPCG_TPL_MODE_EMPTY  (1)：空 tensor 专用模板 EmptyKernel<float>
 */

#include "kernel_operator.h"

#include "arch35/fake_quant_with_min_max_vars_per_channel_gradient_tilingdata.h"
#include "arch35/fake_quant_with_min_max_vars_per_channel_gradient_tiling_key.h"

#if (__NPU_ARCH__ == 3510)
#include "arch35/fake_quant_with_min_max_vars_per_channel_gradient_regbase.h"
#include "arch35/fake_quant_with_min_max_vars_per_channel_gradient_empty.h"
#endif

template <uint64_t MODE>
__global__ __aicore__ void fake_quant_with_min_max_vars_per_channel_gradient(GM_ADDR gradients, GM_ADDR x, GM_ADDR min,
                                                                             GM_ADDR max, GM_ADDR backprops_wrt_x,
                                                                             GM_ADDR backprops_wrt_min,
                                                                             GM_ADDR backprops_wrt_max,
                                                                             GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FakeQuantWithMinMaxVarsPerChannelGradientTilingData);
    GET_TILING_DATA_WITH_STRUCT(FakeQuantWithMinMaxVarsPerChannelGradientTilingData, tilingData, tiling);

#if (__NPU_ARCH__ == 3510)
    if constexpr (MODE == FQPCG_TPL_MODE_EMPTY) {
        FakeQuantWMMVPCG::EmptyKernel<float> op;
        op.Init(backprops_wrt_min, backprops_wrt_max, &tilingData);
        op.Process();
    } else {
        FakeQuantWMMVPCG::Kernel<float> op;
        op.Init(gradients, x, min, max, backprops_wrt_x, backprops_wrt_min, backprops_wrt_max, workspace, tilingData);
        op.Process();
    }
#endif
}
