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
 * \file quantized_batch_norm_tiling.cc
 * \brief
 */
#include <iostream>
#include <string>
#include "quantized_batch_norm_tiling.h"

namespace optiling {

// TilingParse阶段框架契约类失败统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const std::string& opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

static ge::graphStatus Tiling4QuantizedBatchNorm(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4QuantizedBatchNorm(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepare4QuantizedBatchNorm enter.");

    auto compileInfo = context->GetCompiledInfo<QuantizedBatchNormCompileInfo>();
    if (compileInfo == nullptr) {
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: compile info is null; check the SoC-specific OPP package and the "
                               "tiling parse context");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: platform info is null; check the platform info query and the "
                               "OPP/SoC installation");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNum <= 0) {
        OP_LOGD(context->GetNodeName(), "Get core num failed, core num: %u",
                static_cast<uint32_t>(compileInfo->coreNum));
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: got an invalid core num from the platform info; check the SoC "
                               "platform query and the OPP installation");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = ubSizePlatForm;
    if (compileInfo->ubSize <= 0) {
        OP_LOGD(context->GetNodeName(), "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize));
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: got an invalid UB size from the platform info; check the SoC "
                               "platform query and the OPP installation");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "TilingPrepare4QuantizedBatchNorm exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuantizedBatchNorm)
    .Tiling(Tiling4QuantizedBatchNorm)
    .TilingParse<QuantizedBatchNormCompileInfo>(TilingPrepare4QuantizedBatchNorm);

} // namespace optiling
