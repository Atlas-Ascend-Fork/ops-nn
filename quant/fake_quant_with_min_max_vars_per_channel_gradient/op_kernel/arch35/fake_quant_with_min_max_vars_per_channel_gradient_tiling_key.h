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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_tiling_key.h
 * \brief FakeQuantWithMinMaxVarsPerChannelGradient TilingKey 模板参数（ASCENDC_TPL，无注册宏）
 *
 * 单字段 MODE：
 *   - FQPCG_TPL_MODE_NORMAL：常规路径模板（Kernel<float>，regbase 实现）
 *   - FQPCG_TPL_MODE_EMPTY ：空 tensor 专用模板（EmptyKernel<float>，输出 bp_min/bp_max=0）
 *
 * host 侧通过 GET_TPL_TILING_KEY(MODE) + SetTilingKey 下发；kernel 入口以
 * template + if constexpr 分发。GET_TPL_TILING_KEY 按 ASCENDC_TPL_UINT_DECL 值列表的
 * 序号编码 tilingKey，与 MODE 取值大小无关。
 */
#ifndef FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_TILING_KEY_H_
#define FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_TPL_DEFINED
#define FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_TPL_DEFINED

#define FQPCG_TPL_MODE_NORMAL 1
#define FQPCG_TPL_MODE_EMPTY 2

#endif

namespace FakeQuantWMMVPCG {
ASCENDC_TPL_ARGS_DECL(FakeQuantWithMinMaxVarsPerChannelGradient,
                      ASCENDC_TPL_UINT_DECL(MODE, 8, ASCENDC_TPL_UI_LIST, FQPCG_TPL_MODE_NORMAL, FQPCG_TPL_MODE_EMPTY));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(MODE, ASCENDC_TPL_UI_LIST, FQPCG_TPL_MODE_NORMAL,
                                                          FQPCG_TPL_MODE_EMPTY)));
} // namespace FakeQuantWMMVPCG

#endif // FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_TILING_KEY_H_
