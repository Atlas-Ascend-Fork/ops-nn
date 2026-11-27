/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "rotate_dual_level_quant_tiling_data.h"
#include "blaze/rotate_dual_level_quant_blaze_kernel.h"

// Entry signature follows Graph/Host order: x, rot, alpha, group_list;
// y, level0_scale, level1_scale; workspace, tiling.
extern "C" __global__ __aicore__ void rotate_dual_level_quant(GM_ADDR x, GM_ADDR rot, GM_ADDR alpha, GM_ADDR groupList,
                                                              GM_ADDR y, GM_ADDR level0Scale, GM_ADDR level1Scale,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(RotateDualLevelQuantNS::RotateDualLevelQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(RotateDualLevelQuantNS::RotateDualLevelQuantTilingData, data, tiling);
    if (data.M == 0 || data.N == 0) {
        return;
    }
    (void)workspace; // Host reserves system workspace; intermediate results stay in per-AIV UB.
    if (TILING_KEY_IS(0)) {
        RotateDualLevelQuantNS::RotateDualLevelQuantBlazeKernel<DTYPE_X, DTYPE_Y, DTYPE_LEVEL0_SCALE,
                                                                DTYPE_LEVEL1_SCALE, false>(
            x, rot, alpha, groupList, y, level0Scale, level1Scale, data);
    } else if (TILING_KEY_IS(1)) {
        RotateDualLevelQuantNS::RotateDualLevelQuantBlazeKernel<DTYPE_X, DTYPE_Y, DTYPE_LEVEL0_SCALE,
                                                                DTYPE_LEVEL1_SCALE, true>(
            x, rot, alpha, groupList, y, level0Scale, level1Scale, data);
    }
}
