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
 * \file layer_norm_v4_tiling.cpp
 * \brief
 */

#include "op_host/tiling_util.h"
#include "layer_norm_v4_tiling.h"

namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "LayerNormV4", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

using namespace Ops::NN::OpTiling;

static ge::graphStatus Tiling4LayerNormV4(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4CompileInfo(gert::TilingParseContext* context, LayerNormV4CompileInfo* compileInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                ReportTilingStageError(
                    context->GetNodeName(),
                    "TilingPrepare: platform info is null; check the platform info query and the OPP/SoC installation"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfo->isAscend310P = npuArch == NpuArch::DAV_2002;
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: got an invalid core num " + std::to_string(compileInfo->coreNum) +
                                   " from the platform info; check the SoC platform query and the OPP installation"),
        return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0),
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: got an invalid UB size " + std::to_string(compileInfo->ubSizePlatForm) +
                                   " from the platform info; check the SoC platform query and the OPP installation"),
        return ge::GRAPH_FAILED);
    compileInfo->isRegBase = (IsRegbaseSocVersion(context) || npuArch == NpuArch::DAV_5102) ? true : false;
    compileInfo->blockSize = Ops::Base::GetUbBlockSize(context);
    compileInfo->vectorLength = Ops::Base::GetVRegSize(context);
    OP_LOGD(context, "TilingPrepare4LayerNormV4 exit.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4LayerNormV4(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4LayerNormV4 enter.");
    auto compileInfo = context->GetCompiledInfo<LayerNormV4CompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        ReportTilingStageError(
            context->GetNodeName(),
            "TilingPrepare: compile info is null; check the SoC-specific OPP package and the tiling parse context"),
        return ge::GRAPH_FAILED);
    return TilingPrepare4CompileInfo(context, compileInfo);
}

IMPL_OP_OPTILING(LayerNormV4).Tiling(Tiling4LayerNormV4).TilingParse<LayerNormV4CompileInfo>(TilingPrepare4LayerNormV4);

} // namespace optiling
