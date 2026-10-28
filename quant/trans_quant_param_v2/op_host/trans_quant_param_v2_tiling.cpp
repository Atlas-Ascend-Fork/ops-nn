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
 * \file trans_quant_param_v2_tiling.cpp
 * \brief
 */
#include "trans_quant_param_v2_tiling.h"

namespace optiling {

// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "TransQuantParamV2", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

void GetChipFeature(fe::PlatFormInfos& platform_info, const string& lable, const string& platform_info_key,
                    const string& true_value, bool& value)
{
    std::string temp_str;
    if (platform_info.GetPlatformRes(lable, platform_info_key, temp_str)) {
        OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", platform_info_key.c_str(), temp_str.c_str());
        value = (temp_str.find(true_value) != string::npos);
    } else {
        OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", platform_info_key.c_str());
        value = false;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TransQuantParamV2TilingData tiling;
    // 框架对象判空前移: context与rawTilingData任一为空时受控上报EZ1008, 避免解引用崩溃
    OP_CHECK_IF(context == nullptr || context->GetRawTilingData() == nullptr,
                ReportTilingStageError(context != nullptr ? context->GetNodeName() : "TransQuantParamV2",
                                       "TilingFunc: the tiling context or raw tiling data container is null; check "
                                       "the TilingContext construction"),
                return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        tiling.GetDataSize() > context->GetRawTilingData()->GetCapacity(),
        ReportTilingStageError(context->GetNodeName(), "the tiling data size " + std::to_string(tiling.GetDataSize()) +
                                                           " exceeds the raw tiling data capacity " +
                                                           std::to_string(context->GetRawTilingData()->GetCapacity())),
        return ge::GRAPH_FAILED);
    tiling.SetDataPtr(context->GetRawTilingData()->GetData());
    const gert::StorageShape* scaleShape = context->GetInputShape(0);
    OP_CHECK_IF(scaleShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "scale",
                                                         "the input shape of scale is missing in the tiling context; "
                                                         "check that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    uint32_t scaleLength = scaleShape->GetStorageShape().GetShapeSize();
    auto offsetShape = context->GetOptionalInputShape(1);
    uint32_t offsetLength = (offsetShape == nullptr) ? 0 : offsetShape->GetStorageShape().GetShapeSize();
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportTilingStageError(context->GetNodeName(),
                               "GetShapeAttrsInfo: GetAttrs() returned null; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    uint32_t roundMode = (attrs->GetInt(0) == nullptr) ? 0 : static_cast<uint32_t>(*(attrs->GetInt(0)));
    OP_TILING_CHECK(
        roundMode != 0 && roundMode != 1,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "round_mode", std::to_string(roundMode).c_str(),
                                              "the value of attr round_mode only supports 0 or 1"),
        return ge::GRAPH_FAILED);
    tiling.set_scaleLength(scaleLength);
    tiling.set_offsetLength(offsetLength);
    tiling.set_roundMode(roundMode);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        ReportTilingStageError(
            context->GetNodeName(),
            "GetPlatformInfo returned null; check the TilingContext construction and the platform registration"),
        return ge::GRAPH_FAILED);
    bool supportL12btBf16 = false;
    GetChipFeature(*platformInfo, "AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", "bf16", supportL12btBf16);
    auto scaleInputShape = context->GetInputShape(0);
    OP_CHECK_IF(scaleInputShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "scale",
                                                         "the input shape of scale is missing in the tiling context; "
                                                         "check that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    auto scaleOriShape = scaleInputShape->GetOriginShape();
    bool isScaleValid = (scaleOriShape.GetDimNum() == 1 && scaleOriShape.GetDim(0) > 0) ||
                        // 2 is the dimension of scale dim, shape is (1, n) at this time
                        (scaleOriShape.GetDimNum() == 2 && scaleOriShape.GetDim(0) == 1 && scaleOriShape.GetDim(1) > 0);
    OP_TILING_CHECK(!supportL12btBf16 && !isScaleValid,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "scale",
                                                          Ops::Base::ToString(scaleOriShape).c_str(),
                                                          "the shape of scale only supports (n,) and (1,n) when the "
                                                          "SoC does not support l12bt bf16; check the input layout"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspace == nullptr,
                ReportTilingStageError(context->GetNodeName(),
                                       "GetWorkspaceSizes returned null; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    context->SetBlockDim(1);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct TransQuantParamV2CompileInfo {};
static ge::graphStatus TilingParseForTransQuantParamV2(gert::TilingParseContext* /* context */)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TransQuantParamV2)
    .Tiling(TilingFunc)
    .TilingParse<TransQuantParamV2CompileInfo>(TilingParseForTransQuantParamV2);
} // namespace optiling
