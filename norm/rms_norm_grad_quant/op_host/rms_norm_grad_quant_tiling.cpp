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
 * \file rms_norm_grad_quant_tiling.cpp
 * \brief RmsNormGradQuant Tiling file
 */
#include "rms_norm_grad_quant_tiling.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_util.h"
#include "tiling/tiling_api.h"

using namespace Ops::Base;

namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "RmsNormGradQuant", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

using namespace Ops::NN::OpTiling;

struct RmsNormGradQuantCompileInfo {
    uint32_t totalCoreNum;
    uint64_t totalUbSize;
};

static ge::graphStatus Tiling4RmsNormGradQuant(gert::TilingContext* context)
{
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    {
        const std::vector<const char*> msgKey = {"optype", "opskernel", "reason"};
        const std::vector<const char*> msgValue = {
            "RmsNormGradQuant", "AiCore", "the current SoC is not a Regbase-platform SoC; use a supported SoC"};
        REPORT_PREDEFINED_ERR_MSG("EZ3002", msgKey, msgValue);
    }
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingPrepare4RmsNormGradQuant(gert::TilingParseContext* context)
{
    OP_CHECK_IF(
        nullptr == context,
        ReportTilingStageError("RmsNormGradQuant",
                               "TilingPrepare: TilingParseContext is null; the tiling framework contract is broken"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4RmsNormGradQuant.");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                ReportTilingStageError(
                    context->GetNodeName(),
                    "TilingPrepare: platform info is null; check the platform info query and the OPP/SoC installation"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<RmsNormGradQuantCompileInfo>();
    OP_CHECK_IF(
        compileInfoPtr == nullptr,
        ReportTilingStageError(
            context->GetNodeName(),
            "TilingPrepare: compile info is null; check the SoC-specific OPP package and the tiling parse context"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfoPtr->totalCoreNum <= 0),
                ReportTilingStageError(
                    context->GetNodeName(),
                    "TilingPrepare: got an invalid core num " + std::to_string(compileInfoPtr->totalCoreNum) +
                        " from the platform info; check the SoC platform query and the OPP installation"),
                return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->totalUbSize);
    OP_CHECK_IF(
        (compileInfoPtr->totalUbSize <= 0),
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: got an invalid UB size " + std::to_string(compileInfoPtr->totalUbSize) +
                                   " from the platform info; check the SoC platform query and the OPP installation"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNormGradQuant)
    .Tiling(Tiling4RmsNormGradQuant)
    .TilingParse<RmsNormGradQuantCompileInfo>(TilingPrepare4RmsNormGradQuant);

} // namespace optiling
