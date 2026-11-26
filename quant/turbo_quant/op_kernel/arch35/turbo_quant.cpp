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
 * \file turbo_quant.cpp
 * \brief ascend950 (arch35/regbase) kernel entry for TurboQuant. Single tiling strategy, so no
 *        tiling-key dispatch: the soc picks the binary (classic kernel for 910b/910_93, this one
 *        for ascend950).
 */
#include "turbo_quant_regbase.h"

extern "C" __global__ __aicore__ void turbo_quant(GM_ADDR latent, GM_ADDR centroids, GM_ADDR y, GM_ADDR scale,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    GET_TILING_DATA(turboData, tiling);

    TurboQuantRegbase::KernelTurboQuantRegbase op(&pipe);
    op.Init(latent, centroids, y, scale, turboData.numTokens, turboData.tokensPerCore, turboData.headDim,
            turboData.packedBytes, turboData.tokensPerBatch);
    op.Process();
}
