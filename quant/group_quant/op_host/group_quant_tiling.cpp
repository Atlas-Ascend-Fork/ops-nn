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
 * \file group_quant_tiling.cc
 * \brief
 */
#include "group_quant_tiling.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"
#include "util/math_util.h"
#include "op_host/tiling_templates_registry.h"

using namespace gert;

namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "GroupQuant", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

using Ops::NN::Optiling::TilingBaseClass;
static const uint64_t TILING_OFFSET = 1000000000000UL;
static const size_t INPUT_INDEX_OF_X = 0;
static const size_t INPUT_INDEX_OF_SCALE = 1;
static const size_t INPUT_INDEX_OF_GROUP_INDEX = 2;
static const size_t INPUT_INDEX_OF_OFFSET = 3;
static const size_t OUTPUT_INDEX_OF_Y = 0;
static const size_t ATTR_INDEX_OF_DST_TYPE = 0;
static const size_t DIM_NUM_OF_X = 2;
static const size_t DIM_NUM_OF_SCALE = 2;
static const size_t DIM_NUM_OF_GROUP_INDEX = 1;
static const size_t DIM_NUM_OF_OFFSET = 1;
static const size_t DIM_NUM_OF_Y = 2;
static const size_t DIM_INDEX_0 = 0;
static const size_t DIM_INDEX_1 = 1;
static const int32_t DTYPE_INT8 = 2;
static const int32_t DTYPE_INT4 = 29;
static const size_t WORKSPACES_DEFAULT_SIZE_32B = 32;
static const int64_t EVEN_FACTOR = 2;

bool GroupQuantTiling::IsCapable() { return true; }
const gert::Shape g_vec_1_shape = {1};
const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus GroupQuantTiling::GetPlatformInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetPlatformInfo begin.");

    auto compileInfo = context_->GetCompileInfo<GroupQuantCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                ReportTilingStageError(
                    context_->GetNodeName(),
                    "GetPlatformInfo: compile info is null; check the SoC-specific OPP package and its loading logs"),
                return ge::GRAPH_FAILED);
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        coreNumVar = compileInfo->coreNum;
        OP_LOGD(context_->GetNodeName(), "Get core num from compile info. Core num is %ld.", coreNumVar);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNumVar = ascendcPlatform.GetCoreNumAiv();
        OP_LOGD(context_->GetNodeName(), "Get core num from platform. Core num is %ld.", coreNumVar);
    }

    OP_LOGD(context_->GetNodeName(), "GetPlatformInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::GetShapeAttrsInfo()
{
    OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo begin.");

    // check for input x
    auto inputX = context_->GetInputShape(INPUT_INDEX_OF_X);
    OP_CHECK_IF(inputX == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "x",
                                                         "the input shape/desc of x is missing in the tiling context; "
                                                         "check that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    auto inputXDesc = context_->GetInputDesc(INPUT_INDEX_OF_X);
    OP_CHECK_IF(inputXDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "x",
                                                         "the input shape/desc of x is missing in the tiling context; "
                                                         "check that the graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    auto xDtype = inputXDesc->GetDataType();
    OP_CHECK_IF((xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
                                                      "The dtype of x must be FLOAT, FLOAT16 or BF16."),
                return ge::GRAPH_FAILED);
    auto xShape = EnsureNotScalar(inputX->GetStorageShape());
    OP_CHECK_IF(
        xShape.GetDimNum() != DIM_NUM_OF_X,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "x", std::to_string(xShape.GetDimNum()).c_str(), "2"),
        return ge::GRAPH_FAILED);
    dimS = xShape.GetDim(DIM_INDEX_0);
    dimH = xShape.GetDim(DIM_INDEX_1);

    // check for input scale
    auto inputScale = context_->GetInputShape(INPUT_INDEX_OF_SCALE);
    OP_CHECK_IF(
        inputScale == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "scale",
                                                 "the input shape/desc of scale is missing in the tiling context; "
                                                 "check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto inputScaleDesc = context_->GetInputDesc(INPUT_INDEX_OF_SCALE);
    OP_CHECK_IF(
        inputScaleDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "scale",
                                                 "the input shape/desc of scale is missing in the tiling context; "
                                                 "check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto scaleDtype = inputScaleDesc->GetDataType();
    OP_CHECK_IF(
        (scaleDtype != ge::DT_FLOAT && scaleDtype != ge::DT_FLOAT16 && scaleDtype != ge::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "scale", Ops::Base::ToString(scaleDtype).c_str(),
                                              "The dtype of scale must be FLOAT, FLOAT16 or BF16."),
        return ge::GRAPH_FAILED);
    auto scaleShape = EnsureNotScalar(inputScale->GetStorageShape());
    OP_CHECK_IF(scaleShape.GetDimNum() != DIM_NUM_OF_SCALE,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "scale",
                                             std::to_string(scaleShape.GetDimNum()).c_str(), "2"),
                return ge::GRAPH_FAILED);
    dimE = scaleShape.GetDim(DIM_INDEX_0);
    OP_CHECK_IF(
        dimE < 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "scale", Ops::Base::ToString(scaleShape).c_str(),
                                              "the first dim of the scale shape must not be less than 1"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleShape.GetDim(DIM_INDEX_1) != dimH,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "scale, x",
                    (Ops::Base::ToString(scaleShape) + ", " + Ops::Base::ToString(xShape)).c_str(),
                    "the second dim of the scale shape must be the same as the second dim of the x shape"),
                return ge::GRAPH_FAILED);

    // check for input group_index
    auto inputGrpIdx = context_->GetInputShape(INPUT_INDEX_OF_GROUP_INDEX);
    OP_CHECK_IF(
        inputGrpIdx == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "group_index",
                                                 "the input shape/desc of group_index is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto inputGrpIdxDesc = context_->GetInputDesc(INPUT_INDEX_OF_GROUP_INDEX);
    OP_CHECK_IF(
        inputGrpIdxDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "group_index",
                                                 "the input shape/desc of group_index is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto grpIdxDtype = inputGrpIdxDesc->GetDataType();
    OP_CHECK_IF((grpIdxDtype != ge::DT_INT32 && grpIdxDtype != ge::DT_INT64),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "group_index",
                                                      Ops::Base::ToString(grpIdxDtype).c_str(),
                                                      "The dtype of group_index must be INT32 or INT64."),
                return ge::GRAPH_FAILED);
    auto grpIdxShape = EnsureNotScalar(inputGrpIdx->GetStorageShape());
    OP_CHECK_IF(grpIdxShape.GetDimNum() != DIM_NUM_OF_GROUP_INDEX,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "group_index",
                                             std::to_string(grpIdxShape.GetDimNum()).c_str(), "1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(grpIdxShape.GetDim(DIM_INDEX_0) != dimE,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "group_index, scale",
                    (Ops::Base::ToString(grpIdxShape) + ", " + Ops::Base::ToString(scaleShape)).c_str(),
                    "the first dim of the group_index shape must be the same as the first dim of the scale shape"),
                return ge::GRAPH_FAILED);

    // check for input offset which is an optional input
    auto inputOffset = context_->GetInputShape(INPUT_INDEX_OF_OFFSET);
    if (inputOffset == nullptr) {
        hasOffset = 0;
        OP_LOGD(context_->GetNodeName(), "Input offset is not exist.");
    } else {
        hasOffset = 1;
        auto inputOffsetDesc = context_->GetInputDesc(INPUT_INDEX_OF_OFFSET);
        OP_CHECK_IF(
            inputOffsetDesc == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "offset",
                                                     "the input shape/desc of offset is missing in the tiling context; "
                                                     "check that the graph node provides all required inputs"),
            return ge::GRAPH_FAILED);
        auto offsetDtype = inputOffsetDesc->GetDataType();
        OP_CHECK_IF(offsetDtype != scaleDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context_->GetNodeName(), "offset, scale",
                        (Ops::Base::ToString(offsetDtype) + ", " + Ops::Base::ToString(scaleDtype)).c_str(),
                        "The dtypes of offset and scale must be the same."),
                    return ge::GRAPH_FAILED);
        auto offsetShape = EnsureNotScalar(inputOffset->GetStorageShape());
        OP_CHECK_IF(offsetShape.GetDimNum() != DIM_NUM_OF_OFFSET || offsetShape.GetDim(DIM_INDEX_0) != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "offset",
                                                          Ops::Base::ToString(offsetShape).c_str(),
                                                          "offset must be a scalar or a vector with shape [1]"),
                    return ge::GRAPH_FAILED);
    }

    // check attr dst_type
    auto* attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportTilingStageError(context_->GetNodeName(),
                               "GetShapeAttrsInfo: GetAttrs() returned null; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    const int32_t* pDstType = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
    if (pDstType != nullptr) {
        int32_t dstType = *pDstType;
        OP_CHECK_IF(dstType != DTYPE_INT8 && dstType != DTYPE_INT4,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "dst_type", std::to_string(dstType).c_str(),
                        "the value of attr dst_type only supports 2 (DT_INT8) or 29 (DT_INT4)"),
                    return ge::GRAPH_FAILED);
    }

    // check for output y
    auto outputY = context_->GetOutputShape(OUTPUT_INDEX_OF_Y);
    OP_CHECK_IF(outputY == nullptr,
                ReportTilingStageError(context_->GetNodeName(),
                                       "the output shape of y is missing in the tiling context; check that InferShape "
                                       "executed and the framework propagated the output shapes"),
                return ge::GRAPH_FAILED);
    auto outputYDesc = context_->GetOutputDesc(OUTPUT_INDEX_OF_Y);
    OP_CHECK_IF(outputYDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "y",
                                                         "the output desc of y is missing in the tiling context; check "
                                                         "that the graph node provides all required outputs"),
                return ge::GRAPH_FAILED);
    auto yDtype = outputYDesc->GetDataType();
    OP_CHECK_IF(yDtype != ge::DT_INT4 && yDtype != ge::DT_INT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y", Ops::Base::ToString(yDtype).c_str(),
                                                      "The dtype of y must be INT4 or INT8."),
                return ge::GRAPH_FAILED);
    auto yShape = EnsureNotScalar(outputY->GetStorageShape());
    OP_CHECK_IF(
        yShape.GetDimNum() != DIM_NUM_OF_Y || yShape.GetDim(DIM_INDEX_0) != dimS || yShape.GetDim(DIM_INDEX_1) != dimH,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "y, x", (Ops::Base::ToString(yShape) + ", " + Ops::Base::ToString(xShape)).c_str(),
            "the shape of y must be the same as the shape of x"),
        return ge::GRAPH_FAILED);
    if (yDtype == ge::DT_INT4) {
        OP_CHECK_IF(dimH % EVEN_FACTOR != 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "y", Ops::Base::ToString(yShape).c_str(),
                        "when the dtype of y is INT4, the second dim of the y shape must be an even number"),
                    return ge::GRAPH_FAILED);
    }

    OP_LOGD(context_->GetNodeName(), "GetShapeAttrsInfo end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::DoOpTiling()
{
    OP_LOGD(context_->GetNodeName(), "DoOpTiling begin.");

    if (dimS == 0) {
        // kernel support empty tensor
        needCoreNum = 1;
    } else if (dimS < coreNumVar) {
        needCoreNum = dimS;
    } else {
        needCoreNum = coreNumVar;
    }
    OP_CHECK_IF(
        needCoreNum < 1,
        ReportTilingStageError(context_->GetNodeName(), "DoOpTiling: needCoreNum is " + std::to_string(needCoreNum) +
                                                            "; the computed core count is invalid"),
        return ge::GRAPH_FAILED);
    if (dimS % needCoreNum == 0) {
        preCoreNum = needCoreNum;
    } else {
        preCoreNum = dimS % needCoreNum;
    }
    xRowNumPreCore = Ops::Base::CeilDiv(dimS, needCoreNum);
    xRowNumPostCore = dimS / needCoreNum;

    tilingData.set_dimS(dimS);
    tilingData.set_dimE(dimE);
    tilingData.set_dimH(dimH);
    tilingData.set_hasOffset(hasOffset);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_preCoreNum(preCoreNum);
    tilingData.set_xRowNumPreCore(xRowNumPreCore);
    tilingData.set_xRowNumPostCore(xRowNumPostCore);

    OP_LOGD(context_->GetNodeName(), "DoOpTiling end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t GroupQuantTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    return tilingKey % TILING_OFFSET;
}

ge::graphStatus GroupQuantTiling::GetWorkspaceSize()
{
    OP_LOGD(context_->GetNodeName(), "GetWorkspaceSize begin.");

    workspaceSize_ = WORKSPACES_DEFAULT_SIZE_32B;

    OP_LOGD(context_->GetNodeName(), "GetWorkspaceSize end.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupQuantTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "PostTiling begin.");

    context_->SetBlockDim(tilingData.get_needCoreNum());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        workspaces == nullptr,
        ReportTilingStageError(context_->GetNodeName(), "PostTiling: GetWorkspaceSizes(1) "
                                                        "returned null; required output-slot count is 1. Check the "
                                                        "TilingContext construction, not the user device allocation"),
        return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;

    gert::TilingData* rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        rawTilingData == nullptr,
        ReportTilingStageError(context_->GetNodeName(),
                               "PostTiling: GetRawTilingData() returned null; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData.GetDataSize() > rawTilingData->GetCapacity(),
                ReportTilingStageError(context_->GetNodeName(), "PostTiling: tiling data size " +
                                                                    std::to_string(tilingData.GetDataSize()) +
                                                                    " exceeds the tiling buffer capacity " +
                                                                    std::to_string(rawTilingData->GetCapacity())),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    OP_LOGD(context_->GetNodeName(), "PostTiling end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GroupQuant(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    // 初始化算子Tiling类
    GroupQuantTiling tiling(context);
    // 执行算子tiling框架
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4GroupQuant(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4GroupQuant begin.");

    auto compileInfo = context->GetCompiledInfo<GroupQuantCompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        ReportTilingStageError(
            context->GetNodeName(),
            "TilingPrepare: compile info is null; check the SoC-specific OPP package and the tiling parse context"),
        return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                ReportTilingStageError(
                    context->GetNodeName(),
                    "TilingPrepare: platform info is null; check the platform info query and the OPP/SoC installation"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->coreNum <= 0),
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: got an invalid core num " + std::to_string(compileInfo->coreNum) +
                                   " from the platform info; check the SoC platform query and the OPP installation"),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0),
        ReportTilingStageError(context->GetNodeName(),
                               "TilingPrepare: got an invalid UB size " + std::to_string(compileInfo->ubSizePlatForm) +
                                   " from the platform info; check the SoC platform query and the OPP installation"),
        return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepare4GroupQuant end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupQuant).Tiling(Tiling4GroupQuant).TilingParse<GroupQuantCompileInfo>(TilingPrepare4GroupQuant);
} // namespace optiling
