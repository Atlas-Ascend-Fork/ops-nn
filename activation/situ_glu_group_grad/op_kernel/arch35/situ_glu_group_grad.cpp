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
 * \file arch35/situ_glu_group_grad.cpp
 * \brief Kernel entry for SituGluGroupGrad Arch35 (Ascend 950)
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "situ_glu_group_grad_kernel.h"
#include "situ_glu_group_grad_tiling_key.h"

using namespace AscendC;

// schMode is a compile-time template argument, so the entry never branches on a numeric TilingKey at
// runtime. The x C++ type comes from DTYPE_X, which the build registers from the OpDef DataType list;
// fp16/bf16/fp32 are therefore not encoded in the TilingKey. weight / group_index may be nullptr when
// not provided; grad_weight is always a valid (possibly empty when weight is absent) output buffer.
template <uint64_t schMode = SITU_GLU_GROUP_GRAD_TPL_SCH_MODE_0>
__global__ __aicore__ void situ_glu_group_grad(GM_ADDR gradY, GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex,
                                               GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(SituGluGroupGradTilingData, tilingDataIn, tiling);
    const SituGluGroupGradTilingData* __restrict__ tilingData = &tilingDataIn;
    TPipe pipe;
    if constexpr (schMode == SITU_GLU_GROUP_GRAD_TPL_SCH_MODE_0) {
        SituGluGroupGradOp::SituGluGroupGradKernel<DTYPE_X> op;
        op.Init(gradY, x, weight, groupIndex, gradX, gradWeight, tilingData, &pipe);
        op.Process();
    }
}

#ifdef ASCENDC_CPU_DEBUG
// The CPU-debug UT links against the kernel object directly, so the instance it calls must be
// emitted explicitly; the on-device build generates the instances from the TPL SEL instead.
template __global__ __aicore__ void situ_glu_group_grad<SITU_GLU_GROUP_GRAD_TPL_SCH_MODE_0>(
    GM_ADDR gradY, GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR workspace,
    GM_ADDR tiling);
#endif
