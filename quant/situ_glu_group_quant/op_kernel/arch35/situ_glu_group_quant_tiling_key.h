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
 * \file situ_glu_group_quant_tiling_key.h
 * \brief TPL tiling key template argument declarations for SituGluGroupQuant
 *        (optional-input presence + attr-flag dimensions, same caliber as
 *        swiglu_mx_quant's groupIndexType/axisLast/roundMode: compile-time
 *        isolation via if constexpr, runtime-branch-free kernel bodies).
 */

#ifndef SITU_GLU_GROUP_QUANT_TILING_KEY_H
#define SITU_GLU_GROUP_QUANT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_NO_GROUP_INDEX 0
#define TPL_HAS_GROUP_INDEX 1

#define TPL_NO_WEIGHT 0
#define TPL_HAS_WEIGHT 1

#define TPL_ACTIVATE_NOT_LEFT 0 // gate = second half (situ normal, default)
#define TPL_ACTIVATE_LEFT 1     // gate = first half (recipes original)

#define TPL_GROUP_LIST_CUMSUM 0 // group_list_type=0: cumsum mode (RESERVED, not yet implemented)
#define TPL_GROUP_LIST_COUNT 1  // group_list_type=1: count mode (only supported value currently)

#define TPL_LINEAR_ENABLED 0  // linear_beta > 0: up = linearBeta * tanh(up/linearBeta)
#define TPL_LINEAR_DISABLED 1 // linear_beta <= 0: up passthrough (no tanh soft-clamp)

namespace SituGluGroupQuantOp {
ASCENDC_TPL_ARGS_DECL(
    SituGluGroupQuant,
    ASCENDC_TPL_UINT_DECL(groupIndexMode, 2, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_HAS_GROUP_INDEX),
    ASCENDC_TPL_UINT_DECL(weightMode, 2, ASCENDC_TPL_UI_LIST, TPL_NO_WEIGHT, TPL_HAS_WEIGHT),
    ASCENDC_TPL_UINT_DECL(activateLeftMode, 2, ASCENDC_TPL_UI_LIST, TPL_ACTIVATE_NOT_LEFT, TPL_ACTIVATE_LEFT),
    ASCENDC_TPL_UINT_DECL(groupListTypeMode, 2, ASCENDC_TPL_UI_LIST, TPL_GROUP_LIST_CUMSUM, TPL_GROUP_LIST_COUNT),
    ASCENDC_TPL_UINT_DECL(linearBetaMode, 2, ASCENDC_TPL_UI_LIST, TPL_LINEAR_ENABLED, TPL_LINEAR_DISABLED));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(groupIndexMode, ASCENDC_TPL_UI_LIST, TPL_NO_GROUP_INDEX, TPL_HAS_GROUP_INDEX),
    ASCENDC_TPL_UINT_SEL(weightMode, ASCENDC_TPL_UI_LIST, TPL_NO_WEIGHT, TPL_HAS_WEIGHT),
    ASCENDC_TPL_UINT_SEL(activateLeftMode, ASCENDC_TPL_UI_LIST, TPL_ACTIVATE_NOT_LEFT, TPL_ACTIVATE_LEFT),
    ASCENDC_TPL_UINT_SEL(groupListTypeMode, ASCENDC_TPL_UI_LIST, TPL_GROUP_LIST_CUMSUM, TPL_GROUP_LIST_COUNT),
    ASCENDC_TPL_UINT_SEL(linearBetaMode, ASCENDC_TPL_UI_LIST, TPL_LINEAR_ENABLED, TPL_LINEAR_DISABLED)));
} // namespace SituGluGroupQuantOp

#endif // SITU_GLU_GROUP_QUANT_TILING_KEY_H
