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
 * \file rms_norm_quant_v3_tiling.cpp
 * \brief
 */
#include "norm/rms_norm_quant_v2/op_host/rms_norm_quant_v2_tiling.h"
#include "op_api/op_util.h"
#include "op_host/tiling_templates_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_impl_registry.h"

using namespace Ops::Base;
using namespace AscendC;
namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "RmsNormQuantV3", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

constexpr size_t DEST_MAX_V3 = 100;
constexpr size_t MAX_LEN_SIMPLIFIED_KEY_V3 = 256;

ge::graphStatus Tiling4RmsNormQuantV3(gert::TilingContext* context)
{
    if (Ops::NN::OpTiling::IsRegbaseSocVersion(context)) {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
    }
    {
        const std::vector<const char*> msgKey = {"optype", "opskernel", "reason"};
        const std::vector<const char*> msgValue = {
            "RmsNormQuantV3", "AiCore", "the current SoC is not a Regbase-platform SoC; use a supported SoC"};
        REPORT_PREDEFINED_ERR_MSG("EZ3002", msgKey, msgValue);
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus TilingPrepare4RmsNormQuantV3(gert::TilingParseContext* context)
{
    OP_CHECK_IF(
        nullptr == context,
        ReportTilingStageError("RmsNormQuantV3",
                               "TilingPrepare: TilingParseContext is null; the tiling framework contract is broken"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter TilingPrepare4RmsNormQuantV3.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                ReportTilingStageError(
                    context->GetNodeName(),
                    "TilingPrepare: platform info is null; check the platform info query and the OPP/SoC installation"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<RmsNormQuantV2CompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                ReportTilingStageError(
                    context->GetNodeName(),
                    "GetCompiledInfo: compile info is null; check the SoC-specific OPP package and its loading logs"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GenSimplifiedKey4RmsNormQuantV3(gert::TilingContext* context, ge::char_t* simplifiedKey)
{
    OP_CHECK_IF(
        nullptr == context,
        ReportTilingStageError("RmsNormQuantV3",
                               "GenSimplifiedKey: TilingContext is null; the tiling framework contract is broken"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter RmsNormQuantV3 genSimplifiedKey.");

    OP_CHECK_IF(nullptr == simplifiedKey,
                ReportTilingStageError(
                    context->GetNodeName(),
                    "GenSimplifiedKey: the simplified key buffer is null; check the tiling framework invocation"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context->GetInputDesc(X_INDEX) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                 "the input desc of required input x is missing in the tiling context; "
                                                 "check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context->GetInputDesc(GAMMA_INDEX) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "gamma",
                                                 "the input desc of required input gamma is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context->GetInputDesc(SCALES1_INDEX) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "scales1",
                                                 "the input desc of required input scales1 is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);

    int32_t xDtype = static_cast<int32_t>(context->GetInputDesc(X_INDEX)->GetDataType());
    int32_t gammaDtype = static_cast<int32_t>(context->GetInputDesc(GAMMA_INDEX)->GetDataType());
    int32_t scales1Dtype = static_cast<int32_t>(context->GetInputDesc(SCALES1_INDEX)->GetDataType());

    int32_t scales2Dtype = -1;
    int32_t zeroPoints1Dtype = -1;
    int32_t zeroPoints2Dtype = -1;
    int32_t betaDtype = -1;

    int32_t y1Dtype = static_cast<int32_t>(context->GetOutputDesc(Y1_INDEX)->GetDataType());

    OP_CHECK_IF(context->GetOptionalInputDesc(SCALES2_INDEX) != nullptr,
                OP_LOGW(context, "Optional input scale2 exist"),
                scales2Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(SCALES2_INDEX)->GetDataType()));
    OP_CHECK_IF(
        context->GetOptionalInputDesc(ZERO_POINTS1_INDEX) != nullptr,
        OP_LOGW(context, "Optional input zeroPoints1 exist"),
        zeroPoints1Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(ZERO_POINTS1_INDEX)->GetDataType()));
    OP_CHECK_IF(
        context->GetOptionalInputDesc(ZERO_POINTS2_INDEX) != nullptr,
        OP_LOGW(context, "Optional input zeroPoints2 exist"),
        zeroPoints2Dtype = static_cast<int32_t>(context->GetOptionalInputDesc(ZERO_POINTS2_INDEX)->GetDataType()));
    OP_CHECK_IF(context->GetOptionalInputDesc(BETA_INDEX) != nullptr, OP_LOGW(context, "Optional input beta exist"),
                betaDtype = static_cast<int32_t>(context->GetOptionalInputDesc(BETA_INDEX)->GetDataType()));

    std::string simpleKeyTemp = "";
    strcat_s(simplifiedKey, DEST_MAX_V3, "diy,");
    simpleKeyTemp.append(std::to_string(xDtype))
        .append("/")
        .append(std::to_string(gammaDtype))
        .append("/")
        .append(std::to_string(scales1Dtype))
        .append("/")
        .append(std::to_string(scales2Dtype))
        .append("/")
        .append(std::to_string(zeroPoints1Dtype))
        .append("/")
        .append(std::to_string(zeroPoints2Dtype))
        .append("/")
        .append(std::to_string(betaDtype))
        .append("/")
        .append(std::to_string(y1Dtype));
    OP_LOGD(context, "SimpleKeyTemp: %s", simpleKeyTemp.c_str());
    errno_t err = strcat_s(simplifiedKey, DEST_MAX_V3, simpleKeyTemp.c_str());
    OP_CHECK_IF((err != 0),
                ReportTilingStageError(context->GetNodeName(),
                                       "GenSimplifiedKey: strcat_s failed with error code " + std::to_string(err)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(strlen(simplifiedKey) > MAX_LEN_SIMPLIFIED_KEY_V3,
                ReportTilingStageError(
                    context->GetNodeName(),
                    "GenSimplifiedKey: the simplified key exceeds the maximum length; check the op registration"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Finish RmsNormQuantV3 genSimplifiedKey.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RmsNormQuantV3)
    .Tiling(Tiling4RmsNormQuantV3)
    .TilingParse<RmsNormQuantV2CompileInfo>(TilingPrepare4RmsNormQuantV3)
    .GenSimplifiedKey(GenSimplifiedKey4RmsNormQuantV3);
} // namespace optiling
