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
 * \file swiglu_hi_f4_quant_tiling.cpp
 * \brief Tiling entry for SwiGLU + HiFloat4 quantization
 */

#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "swiglu_hi_f4_quant_tiling.h"

namespace optiling {

// ==================== 主 Tiling 函数 ====================

ge::graphStatus Tiling4SwigluHiF4Quant(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4SwigluHiF4Quant begin.");

    SwigluHiF4QuantRegbaseTiling tilingImpl(context);

    OP_CHECK_IF(tilingImpl.GetNpuInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to get NPU info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingImpl.ValidateInput() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Input validation failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingImpl.ParseAttrs() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to parse attributes."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingImpl.ValidateOutput() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Output validation failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingImpl.PreProcess() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Pre-process failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingImpl.CalculateTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Tiling calculation failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingImpl.FillTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to fill tiling data."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingImpl.SetParams() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Failed to set params."),
                return ge::GRAPH_FAILED);

    tilingImpl.PrintTilingData();
    OP_LOGI(context->GetNodeName(), "Tiling4SwigluHiF4Quant done.");
    return ge::GRAPH_SUCCESS;
}

// ==================== TilingPrepare ====================

ge::graphStatus TilingPrepare4SwigluHiF4Quant(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4SwigluHiF4Quant begin.");
    return ge::GRAPH_SUCCESS;
}

// ==================== 注册 Tiling 接口 ====================

IMPL_OP_OPTILING(SwigluHiF4Quant)
    .Tiling(Tiling4SwigluHiF4Quant)
    .TilingParse<SwigluHiF4QuantCompileInfo>(TilingPrepare4SwigluHiF4Quant);

} // namespace optiling
