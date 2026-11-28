/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad_tiling_base.cpp
 * \brief SwigluGroupGrad shared tiling validation (arch35, DAV_3510)
 */

#include "swiglu_group_grad_tiling_base.h"
#include <string>
#include <limits>
#include <graph/utils/type_utils.h>

namespace optiling {

ge::graphStatus GetSwigluGroupGradPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatform = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = ubSizePlatform;
    } else {
        auto compileInfoPtr = context->GetCompileInfo<SwigluGroupGradCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
        coreNum = static_cast<int64_t>(compileInfoPtr->coreNum);
        ubSize = compileInfoPtr->ubSize;
    }

    OP_CHECK_IF(coreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcDtype(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    auto inputDesc = context->GetInputDesc(GRAD_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    inputData.gradYDtype = inputDesc->GetDataType();

    if (inputData.gradYDtype != ge::DT_BF16 && inputData.gradYDtype != ge::DT_FLOAT16 &&
        inputData.gradYDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "grad_y",
                                              Ops::Base::ToString(inputData.gradYDtype),
                                              "grad_y only supports DT_BF16, DT_FLOAT16 and DT_FLOAT");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShape(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    auto gradYStorageShape = context->GetInputShape(GRAD_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYStorageShape);
    const gert::Shape& gradYShape = gradYStorageShape->GetStorageShape();

    OP_CHECK_IF(
        gradYShape.GetDimNum() < 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "grad_y",
                                                 std::to_string(gradYShape.GetDimNum()), "grad_y must be at least 1D"),
        return ge::GRAPH_FAILED);

    inputData.totalRows = 1;
    for (size_t i = 0; i < gradYShape.GetDimNum() - 1; ++i) {
        inputData.totalRows *= gradYShape.GetDim(i);
    }

    inputData.H = gradYShape.GetDim(gradYShape.GetDimNum() - 1);

    auto xStorageShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape& xShape = xStorageShape->GetStorageShape();

    OP_CHECK_IF(xShape.GetDimNum() < 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(xShape.GetDimNum()), "x must be at least 1D"),
                return ge::GRAPH_FAILED);

    int64_t xTotalRows = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; ++i) {
        xTotalRows *= xShape.GetDim(i);
    }
    inputData.dim2H = xShape.GetDim(xShape.GetDimNum() - 1);

    OP_CHECK_IF(
        xTotalRows != inputData.totalRows,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, grad_y",
                                               Ops::Base::ToString(xShape) + ", " + Ops::Base::ToString(gradYShape),
                                               "x rows must equal grad_y rows"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputData.dim2H != inputData.H * DIM_TWO,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", std::to_string(inputData.dim2H),
                                                          "x.shape[-1] must equal 2 * grad_y.shape[-1]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputData.H <= 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "grad_y", std::to_string(inputData.H),
                                                          "grad_y.shape[-1] must be greater than 0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseOptionalInputs(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    inputData.isWeight = 0;
    inputData.isYOrigin = 0;
    inputData.isGroupIndex = 0;
    inputData.groupIndexG = 0;

    auto weightShapePtr = context->GetOptionalInputShape(WEIGHT_INDEX);
    auto yOriginShapePtr = context->GetOptionalInputShape(YORIGIN_INDEX);
    auto groupIndexShapePtr = context->GetOptionalInputShape(GROUP_INDEX_INDEX);

    const gert::Shape* weightShape = (weightShapePtr != nullptr && weightShapePtr->GetStorageShape().GetDimNum() > 0) ?
                                         &weightShapePtr->GetStorageShape() :
                                         nullptr;
    const gert::Shape* yOriginShape = (yOriginShapePtr != nullptr &&
                                       yOriginShapePtr->GetStorageShape().GetDimNum() > 0) ?
                                          &yOriginShapePtr->GetStorageShape() :
                                          nullptr;
    const gert::Shape* groupIndexShape = (groupIndexShapePtr != nullptr &&
                                          groupIndexShapePtr->GetStorageShape().GetDimNum() > 0) ?
                                             &groupIndexShapePtr->GetStorageShape() :
                                             nullptr;

    if (weightShape != nullptr) {
        inputData.isWeight = 1;
        auto weightElementNum = weightShape->GetShapeSize();
        if (weightElementNum != inputData.totalRows) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "weight", std::to_string(weightElementNum).c_str(),
                "The element num of weight must be equal to the product of grad_y leading dims.");
            return ge::GRAPH_FAILED;
        }
    }

    if (yOriginShape != nullptr) {
        OP_CHECK_IF(weightShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                             "weight must be present when y_origin is present"),
                    return ge::GRAPH_FAILED);
        inputData.isYOrigin = 1;

        auto gradWeightOut = context->GetOutputShape(1);
        OP_CHECK_IF(gradWeightOut == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "grad_weight",
                                                             "grad_weight must be non-null when y_origin is present"),
                    return ge::GRAPH_FAILED);
        const gert::Shape& gradWeightShape = gradWeightOut->GetStorageShape();
        OP_CHECK_IF(gradWeightShape.GetDimNum() != weightShape->GetDimNum(),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context->GetNodeName(), "grad_weight, weight",
                        std::to_string(gradWeightShape.GetDimNum()) + ", " + std::to_string(weightShape->GetDimNum()),
                        "grad_weight rank must match weight rank"),
                    return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < static_cast<int64_t>(weightShape->GetDimNum()); ++i) {
            OP_CHECK_IF(gradWeightShape.GetDim(i) != weightShape->GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context->GetNodeName(), "grad_weight, weight",
                            Ops::Base::ToString(gradWeightShape) + ", " + Ops::Base::ToString(*weightShape),
                            "grad_weight shape must match weight shape"),
                        return ge::GRAPH_FAILED);
        }

        OP_CHECK_IF(yOriginShape->GetDimNum() < 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "y_origin",
                                                             std::to_string(yOriginShape->GetDimNum()),
                                                             "y_origin must be at least 1D"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            yOriginShape->GetDim(yOriginShape->GetDimNum() - 1) != inputData.H,
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "y_origin", std::to_string(yOriginShape->GetDim(yOriginShape->GetDimNum() - 1)),
                "y_origin.shape[-1] must equal grad_y.shape[-1]"),
            return ge::GRAPH_FAILED);
        int64_t yOriginTotalRows = 1;
        for (size_t i = 0; i < yOriginShape->GetDimNum() - 1; ++i) {
            yOriginTotalRows *= yOriginShape->GetDim(i);
        }
        OP_CHECK_IF(yOriginTotalRows != inputData.totalRows,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context->GetNodeName(), "y_origin, grad_y",
                        std::to_string(yOriginTotalRows) + ", " + std::to_string(inputData.totalRows),
                        "y_origin outer numel must equal grad_y outer numel"),
                    return ge::GRAPH_FAILED);
    }

    if (groupIndexShape != nullptr) {
        inputData.isGroupIndex = 1;
        OP_CHECK_IF(groupIndexShape->GetDimNum() != 1 || groupIndexShape->GetDim(0) < 1,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "group_index",
                                                           Ops::Base::ToString(*groupIndexShape),
                                                           "group_index must be a non-empty 1D tensor"),
                    return ge::GRAPH_FAILED);
        inputData.groupIndexG = groupIndexShape->GetDim(0);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseAttrs(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    inputData.hasClamp = 0;
    inputData.clampLimit = 0.0f;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* clampLimitAttr = attrs->GetAttrPointer<float>(CLAMPLIMIT_ATTR_INDEX);

    if (clampLimitAttr != nullptr) {
        OP_CHECK_IF(*clampLimitAttr != -1.0f && !(*clampLimitAttr > 0.0f),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "clamp_limit",
                                                          std::to_string(*clampLimitAttr),
                                                          "clamp_limit must be -1.0 (no clamp) or greater than 0.0"),
                    return ge::GRAPH_FAILED);
        if (*clampLimitAttr > 0.0f) {
            inputData.hasClamp = 1;
            inputData.clampLimit = *clampLimitAttr;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetSwigluGroupGradShapeAttrsInfo(gert::TilingContext* context, SwigluGroupGradInputInfo& inputData)
{
    OP_CHECK_IF(CalcDtype(context, inputData) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "CalcDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape(context, inputData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "CheckShape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseOptionalInputs(context, inputData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ParseOptionalInputs failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseAttrs(context, inputData) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "ParseAttrs failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
