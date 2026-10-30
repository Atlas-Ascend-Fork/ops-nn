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
 * \file trans_quant_param_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "platform/platform_info.h"

static constexpr int INPUT_SCALE_INDEX = 0;
static constexpr int INPUT_OFFSET_INDEX = 1;
static constexpr int OUTPUT_Y_INDEX = 0;

static constexpr int GRID_SAMPLER2D_GRAD_SHAPE_LIMIT = 4;
using namespace ge;
namespace ops {

// InferShape阶段框架契约类失败统一上报EZ1014, 不再生成EZ9999内部码
static void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "TransQuantParamV2", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

static std::set<std::string> QbmmDavidSupportSoc = {"Ascend950", "Ascend350", "Ascend960DT"};

static bool IsDavidSupported()
{
    static bool isSupported = []() -> bool {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optionalInfo;
        auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
        if (ret != GRAPH_SUCCESS) {
            return false;
        }
        return QbmmDavidSupportSoc.count(platformInfo.str_info.short_soc_version) > 0;
    }();

    return isSupported;
}

static ge::graphStatus InferShape4TransQuantParamV2(gert::InferShapeContext* context)
{
    // infer shape
    OP_CHECK_IF(context == nullptr,
                ReportInferShapeStageError(
                    "TransQuantParamV2", "the InferShapeContext is null; the infer-shape framework contract is broken"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4TransQuantParamV2.");

    const gert::Shape* scaleShape = context->GetInputShape(INPUT_SCALE_INDEX);
    OP_CHECK_IF(scaleShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "scale",
                    "the input shape of required input scale is missing in the infer-shape context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);

    const gert::Shape* offsetShape = context->GetOptionalInputShape(INPUT_OFFSET_INDEX);

    gert::Shape* outYShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_IF(
        outYShape == nullptr,
        ReportInferShapeStageError(context->GetNodeName(),
                                   "the output shape container of y (output index 0) is missing in the infer-shape "
                                   "context; check the InferShapeContext construction"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "scaleShape: %s", Ops::Base::ToString(*scaleShape).c_str());
    OP_LOGD(context->GetNodeName(), "outYShape: %s", Ops::Base::ToString(*outYShape).c_str());
    if (offsetShape == nullptr) {
        OP_LOGD(context->GetNodeName(), "offsetShape is null");
    } else {
        OP_LOGD(context->GetNodeName(), "offsetShape: %s", Ops::Base::ToString(*offsetShape).c_str());
    }

    OP_LOGD(context->GetNodeName(), "now outYShape: %s", Ops::Base::ToString(*outYShape).c_str());
    OP_LOGD(context->GetNodeName(), "now scaleShape: %s", Ops::Base::ToString(*scaleShape).c_str());
    *outYShape = *scaleShape;
    OP_LOGD(context->GetNodeName(), "now outYShape: %s", Ops::Base::ToString(*outYShape).c_str());

    OP_CHECK_IF(
        Ops::Base::IsUnknownRank(*scaleShape) || (offsetShape != nullptr && Ops::Base::IsUnknownRank(*offsetShape)),
        OP_LOGI(context->GetNodeName(), "End to do InferShape4TransQuantParamV2, input is unknownrank."),
        return GRAPH_SUCCESS);

    OP_LOGD(context->GetNodeName(), "now scaleShape: %s", Ops::Base::ToString(*scaleShape).c_str());

    if (offsetShape != nullptr && (offsetShape->GetDimNum() > scaleShape->GetDimNum() ||
                                   (IsDavidSupported() && offsetShape->GetDimNum() == 1 &&
                                    scaleShape->GetDimNum() == 1 && offsetShape->GetDim(0) > scaleShape->GetDim(0)))) {
        OP_LOGD(context->GetNodeName(), "offsetShape: %s", Ops::Base::ToString(*offsetShape).c_str());
        *outYShape = *offsetShape;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4TransQuantParamV2.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4TransQuantParamV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4TransQuantParamV2");
    context->SetOutputDataType(OUTPUT_Y_INDEX, ge::DT_UINT64);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4TransQuantParamV2");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TransQuantParamV2)
    .InferShape(InferShape4TransQuantParamV2)
    .InferDataType(InferDataType4TransQuantParamV2);
} // namespace ops
