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
 * \file situ_glu_group_quant.cpp
 * \brief Kernel entry point for SituGluGroupQuant, ported from cann-recipes-infer.
 *        Optional inputs (group_index / weight) and the attr flags
 *        (activate_left / linear_beta disable) are TPL tiling-key dimensions — the
 *        swiglu_mx_quant group_index/axisLast/roundMode caliber — dispatched
 *        with if constexpr so every kernel instance is runtime-branch-free;
 *        an absent-optional instance never touches the placeholder GM_ADDR.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/situ_glu_group_quant_tiling_key.h"
#include "arch35/situ_glu_group_quant_tiling_data.h"
#include "arch35/situ_glu_group_quant_base.h"

using namespace AscendC;
using namespace SituGluGroupQuant;
using namespace SituGluGroupQuantOp;

template <uint64_t groupIndexMode, uint64_t weightMode, uint64_t activateLeftMode, uint64_t groupListTypeMode,
          uint64_t linearBetaMode>
__global__ __aicore__ void situ_glu_group_quant(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y,
                                                GM_ADDR y_scale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SituGluGroupQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(SituGluGroupQuantTilingData, tilingData, tiling);
    TPipe pipe;
#if (__NPU_ARCH__ == 3510)
    // Save the caller's overflow mode; the per-row Compute sets saturation (0)
    // for the e4m3 quant cast. Restore on exit so the SPR does not leak out.
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif
    constexpr bool hasGroupIndex = (groupIndexMode == TPL_HAS_GROUP_INDEX);
    constexpr bool hasWeight = (weightMode == TPL_HAS_WEIGHT);
    constexpr bool kActivateLeft = (activateLeftMode == TPL_ACTIVATE_LEFT);
    constexpr bool kGroupListCount = (groupListTypeMode == TPL_GROUP_LIST_COUNT);
    constexpr bool kLinearDisabled = (linearBetaMode == TPL_LINEAR_DISABLED);
    SituGluGroupQuantKernel<DTYPE_X, hasGroupIndex, hasWeight, kActivateLeft, kGroupListCount, kLinearDisabled> op;
    op.Init(x, groupIndex, y, y_scale, weight, &tilingData, &pipe);
    op.Process();
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
