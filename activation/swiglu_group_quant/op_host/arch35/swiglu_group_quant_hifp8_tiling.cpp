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
 * \file swiglu_group_quant_hifp8_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include "swiglu_group_quant_hifp8_tiling.h"
#include <string>
#include "graph/utils/type_utils.h"

using namespace ge;
namespace optiling {
namespace {
constexpr int64_t BATCH_MODE = 1;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t UB_RESERVE = 1024;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t SWI_FACTOR = 2;
constexpr int64_t SIZE_OF_FLOAT = 4;
constexpr int64_t UB_FACTOR_DYNAMIC = 7;
constexpr int64_t UB_FACTOR_STATIC = 6;

constexpr float CLAMP_LIMIT_DEFAULT = -1.0f;
constexpr float DST_TYPE_MAX_FINITE_DEFAULT = 15.0f;

constexpr int64_t QUANT_MODE_DYNAMIC = 3;
constexpr int64_t QUANT_MODE_STATIC = 2;

constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
constexpr size_t ATTR_INDEX_CLAMP_LIMIT = 4;
constexpr size_t ATTR_INDEX_DST_TYPE_MAX_FINITE = 5;
constexpr size_t ATTR_INDEX_OUTPUT_ORIGIN = 6;

constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
constexpr size_t INPUT_INDEX_GROUP_INDEX = 2;
constexpr size_t INPUT_INDEX_SCALE = 3;
constexpr size_t OUTPUT_INDEX_Y = 0;
constexpr size_t OUTPUT_INDEX_Y_SCALE = 1;
constexpr size_t OUTPUT_INDEX_Y_ORIGIN = 2;

constexpr int64_t DYNAMIC_HIFP8_TILING_KEY = 4000;
constexpr int64_t STATIC_HIFP8_TILING_KEY = 4100;

int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
std::string ShapeToString(const gert::Shape& shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(shape.GetDim(i));
    }
    result += "]";
    return result;
}
} // namespace

ge::graphStatus SwigluGroupQuantHifp8Tiling::GetPlatformInfo()
{
    return SwigluGroupQuantTiling::GetPlatformInfoCommon(context_, coreNum_, ubSize_);
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckInputDtype() const
{
    auto xDtype = context_->GetInputDesc(INPUT_INDEX_X)->GetDataType();
    OP_CHECK_IF((xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                    "The dtype of input x only supports float16, bfloat16 or float32"),
                return ge::GRAPH_FAILED);

    auto weightDesc = context_->GetOptionalInputDesc(INPUT_INDEX_WEIGHT);
    if (weightDesc != nullptr) {
        auto weightDtype = weightDesc->GetDataType();
        OP_CHECK_IF((weightDtype != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "weight",
                                                          ge::TypeUtils::DataTypeToSerialString(weightDtype).c_str(),
                                                          "The dtype of input weight only supports float32"),
                    return ge::GRAPH_FAILED);
    }

    auto groupDesc = context_->GetOptionalInputDesc(INPUT_INDEX_GROUP_INDEX);
    if (groupDesc != nullptr) {
        auto groupDtype = groupDesc->GetDataType();
        OP_CHECK_IF((groupDtype != ge::DT_INT64),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "group_index",
                                                          ge::TypeUtils::DataTypeToSerialString(groupDtype).c_str(),
                                                          "The dtype of input group_index must be int64"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckOutputDtype() const
{
    auto yDtype = context_->GetOutputDesc(OUTPUT_INDEX_Y)->GetDataType();
    OP_CHECK_IF((yDtype != ge::DT_HIFLOAT8),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y",
                                                      ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                                                      "The dtype of output y must be hifloat8"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto groupListTypeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    groupListType_ = groupListTypeAttr == nullptr ? GROUP_LIST_TYPE_COUNT : *groupListTypeAttr;
    OP_CHECK_IF(groupListType_ != GROUP_LIST_TYPE_COUNT && groupListType_ != GROUP_LIST_TYPE_PAIR,
                OP_LOGE(context_->GetNodeName(), "group_list_type only supports 1 or 2, got %ld.", groupListType_),
                return ge::GRAPH_FAILED);

    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    if (quantModePtr != nullptr) {
        quantMode_ = *quantModePtr;
        OP_CHECK_IF((quantMode_ != QUANT_MODE_DYNAMIC && quantMode_ != QUANT_MODE_STATIC),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "quant_mode", std::to_string(quantMode_),
                        "The value of attr quant_mode must be " + std::to_string(QUANT_MODE_DYNAMIC) +
                            " (Dynamic) or " + std::to_string(QUANT_MODE_STATIC) + " (Static)"),
                    return ge::GRAPH_FAILED);
    }

    auto clampLimitPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_CLAMP_LIMIT);
    clampLimit_ = (clampLimitPtr != nullptr) ? *clampLimitPtr : CLAMP_LIMIT_DEFAULT;
    OP_CHECK_IF(
        (clampLimit_ < 0.0f && std::fabs(clampLimit_ + 1.0f) > 1e-6f),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "clamp_limit", std::to_string(clampLimit_),
                                              "The value of attr clamp_limit must be -1 or greater than 0.0"),
        return ge::GRAPH_FAILED);
    hasClamp_ = (clampLimit_ > 0.0f);

    auto dstTypeMaxPtr = attrs->GetAttrPointer<float>(ATTR_INDEX_DST_TYPE_MAX_FINITE);
    dstTypeMax_ = (dstTypeMaxPtr != nullptr) ? *dstTypeMaxPtr : DST_TYPE_MAX_FINITE_DEFAULT;
    OP_CHECK_IF(
        (dstTypeMax_ <= 0.0f),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "dst_type_max", std::to_string(dstTypeMax_),
                                              "The value of attr dst_type_max must be greater than 0"),
        return ge::GRAPH_FAILED);

    auto outputOriginPtr = attrs->GetAttrPointer<bool>(ATTR_INDEX_OUTPUT_ORIGIN);
    if (outputOriginPtr != nullptr) {
        outputOrigin_ = *outputOriginPtr;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckWeightInfo()
{
    auto weightShape = context_->GetOptionalInputShape(INPUT_INDEX_WEIGHT);
    hasWeight_ = (weightShape != nullptr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckGroupIndexInfo()
{
    auto groupIndexShape = context_->GetOptionalInputShape(INPUT_INDEX_GROUP_INDEX);
    if (groupIndexShape == nullptr) {
        isGroup_ = false;
        groupNum_ = 0;
        OP_CHECK_IF(groupListType_ == GROUP_LIST_TYPE_PAIR,
                    OP_LOGE(context_->GetNodeName(), "group_index is required when group_list_type is 2."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    isGroup_ = true;
    auto groupIndexStorageShape = groupIndexShape->GetStorageShape();
    size_t giDimNum = groupIndexStorageShape.GetDimNum();
    const bool isPair = groupListType_ == GROUP_LIST_TYPE_PAIR;
    OP_CHECK_IF(
        isPair ? (giDimNum != 2 || groupIndexStorageShape.GetDim(1) != GROUP_INDEX_PAIR_COLS) : (giDimNum != 1),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "group_index", std::to_string(giDimNum),
            "The shape dim of input group_index must be 1 for group_list_type 1, or 2 with the second dim equal to " +
                std::to_string(GROUP_INDEX_PAIR_COLS) + " for group_list_type 2"),
        return ge::GRAPH_FAILED);
    groupNum_ = groupIndexStorageShape.GetDim(0);
    OP_CHECK_IF(
        (groupNum_ <= 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "group_index", std::to_string(groupNum_),
                                                  "The shape size of input group_index must be greater than 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckScaleInfo() const
{
    if (quantMode_ != QUANT_MODE_STATIC) {
        return ge::GRAPH_SUCCESS;
    }
    auto scaleDesc = context_->GetOptionalInputDesc(INPUT_INDEX_SCALE);
    OP_CHECK_IF(
        (scaleDesc == nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context_->GetNodeName(), "scale", "The input scale is required when quant_mode is 2 (static hifp8 quant)"),
        return ge::GRAPH_FAILED);
    auto scaleDtype = scaleDesc->GetDataType();
    OP_CHECK_IF((scaleDtype != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "scale",
                                                      ge::TypeUtils::DataTypeToSerialString(scaleDtype).c_str(),
                                                      "The dtype of input scale only supports float32"),
                return ge::GRAPH_FAILED);
    auto scaleShape = context_->GetOptionalInputShape(INPUT_INDEX_SCALE);
    OP_CHECK_IF((scaleShape == nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "scale",
                                                         "The shape of input scale is unavailable"),
                return ge::GRAPH_FAILED);
    auto scaleStorageShape = scaleShape->GetStorageShape();
    size_t scaleDimNum = scaleStorageShape.GetDimNum();
    OP_CHECK_IF((scaleDimNum != 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "scale", std::to_string(scaleDimNum),
                                                         "The shape dim of input scale must be 1"),
                return ge::GRAPH_FAILED);
    int64_t scaleSize = scaleStorageShape.GetDim(0);
    if (isGroup_) {
        OP_CHECK_IF((scaleSize != groupNum_),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        context_->GetNodeName(), "scale and group_index",
                        (std::to_string(scaleSize) + " and " + std::to_string(groupNum_)).c_str(),
                        "The shape size of input scale must be equal to the shape size of input group_index"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((scaleSize != 1),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), "scale", std::to_string(scaleSize),
                        "The shape size of input scale must be 1 when group_index is not present"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckYShape(size_t xDimNum, const gert::Shape& xStorageShape) const
{
    auto yShape = context_->GetOutputShape(OUTPUT_INDEX_Y);
    OP_CHECK_IF(
        (yShape == nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "y", "The shape of output y is unavailable"),
        return ge::GRAPH_FAILED);
    const gert::Shape& yShapeStorage = yShape->GetStorageShape();
    OP_CHECK_IF(
        (yShapeStorage.GetDimNum() != xDimNum),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "y", std::to_string(yShapeStorage.GetDimNum()),
            "The shape dim of output y must be equal to the shape dim of input x (" + std::to_string(xDimNum) + ")"),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xDimNum - 1; i++) {
        OP_CHECK_IF((yShapeStorage.GetDim(i) != xStorageShape.GetDim(i)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "y", ShapeToString(yShapeStorage),
                        "The dim[" + std::to_string(i) + "] of output y must be equal to the dim[" + std::to_string(i) +
                            "] of input x (" + std::to_string(xStorageShape.GetDim(i)) + ")"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((yShapeStorage.GetDim(xDimNum - 1) != dimH_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "y", ShapeToString(yShapeStorage),
                    "The last dim of output y must be equal to dimH (" + std::to_string(dimH_) + ")"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckYScaleShape() const
{
    if (quantMode_ == QUANT_MODE_STATIC) {
        return ge::GRAPH_SUCCESS;
    }
    auto yScaleDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y_SCALE);
    OP_CHECK_IF((yScaleDesc == nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "y_scale",
                                                         "The desc of output y_scale is unavailable"),
                return ge::GRAPH_FAILED);
    auto yScaleDtype = yScaleDesc->GetDataType();
    OP_CHECK_IF((yScaleDtype != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y_scale",
                                                      ge::TypeUtils::DataTypeToSerialString(yScaleDtype).c_str(),
                                                      "The dtype of output y_scale must be float32"),
                return ge::GRAPH_FAILED);
    auto yScaleShape = context_->GetOutputShape(OUTPUT_INDEX_Y_SCALE);
    OP_CHECK_IF((yScaleShape == nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "y_scale",
                                                         "The shape of output y_scale is unavailable"),
                return ge::GRAPH_FAILED);
    const gert::Shape& yScaleShapeStorage = yScaleShape->GetStorageShape();
    OP_CHECK_IF((yScaleShapeStorage.GetDimNum() != 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "y_scale",
                                                         std::to_string(yScaleShapeStorage.GetDimNum()),
                                                         "The shape dim of output y_scale must be 1"),
                return ge::GRAPH_FAILED);
    if (isGroup_) {
        OP_CHECK_IF(
            (yScaleShapeStorage.GetDim(0) != groupNum_),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y_scale", ShapeToString(yScaleShapeStorage),
                                                  "The dim[0] of output y_scale must be equal to groupNum (" +
                                                      std::to_string(groupNum_) + ") in group mode"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            (yScaleShapeStorage.GetDim(0) != 1),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y_scale", ShapeToString(yScaleShapeStorage),
                                                  "The dim[0] of output y_scale must be 1 in non-group mode"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckYOriginShape(size_t xDimNum, const gert::Shape& xStorageShape) const
{
    if (!outputOrigin_) {
        return ge::GRAPH_SUCCESS;
    }
    auto yOriginShape = context_->GetOutputShape(OUTPUT_INDEX_Y_ORIGIN);
    OP_CHECK_IF((yOriginShape == nullptr),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "y_origin",
                    "The shape of output y_origin is unavailable when output_origin is true"),
                return ge::GRAPH_FAILED);
    const gert::Shape& yOriginShapeStorage = yOriginShape->GetStorageShape();
    OP_CHECK_IF((yOriginShapeStorage.GetDimNum() != xDimNum),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), "y_origin", std::to_string(yOriginShapeStorage.GetDimNum()),
                    "The shape dim of output y_origin must be equal to the shape dim of input x (" +
                        std::to_string(xDimNum) + ")"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xDimNum - 1; i++) {
        OP_CHECK_IF((yOriginShapeStorage.GetDim(i) != xStorageShape.GetDim(i)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "y_origin", ShapeToString(yOriginShapeStorage),
                        "The dim[" + std::to_string(i) + "] of output y_origin must be equal to the dim[" +
                            std::to_string(i) + "] of input x (" + std::to_string(xStorageShape.GetDim(i)) + ")"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((yOriginShapeStorage.GetDim(xDimNum - 1) != dimH_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "y_origin", ShapeToString(yOriginShapeStorage),
                    "The last dim of output y_origin must be equal to dimH (" + std::to_string(dimH_) + ")"),
                return ge::GRAPH_FAILED);
    auto yOriginDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y_ORIGIN);
    auto xDtype = context_->GetInputDesc(INPUT_INDEX_X)->GetDataType();
    if (yOriginDesc != nullptr) {
        OP_CHECK_IF((yOriginDesc->GetDataType() != xDtype),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context_->GetNodeName(), "y_origin",
                        ge::TypeUtils::DataTypeToSerialString(yOriginDesc->GetDataType()).c_str(),
                        "The dtype of output y_origin must be equal to the dtype of input x (" +
                            ge::TypeUtils::DataTypeToSerialString(xDtype) + ")"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckOutputInfo() const
{
    auto xShape = context_->GetInputShape(INPUT_INDEX_X);
    auto xStorageShape = xShape->GetStorageShape();
    size_t xDimNum = xStorageShape.GetDimNum();
    if (CheckYShape(xDimNum, xStorageShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckYScaleShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckYOriginShape(xDimNum, xStorageShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::GetShapeAttrsInfoInner()
{
    auto shapeX = context_->GetInputShape(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);

    auto xStorageShape = shapeX->GetStorageShape();
    auto xDesc = context_->GetInputDesc(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);

    size_t xDimNum = xStorageShape.GetDimNum();
    OP_CHECK_IF((xDimNum < 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xDimNum),
                                                         "The shape dim of input x must be greater than or equal to 1"),
                return ge::GRAPH_FAILED);
    totalTokens_ = 1;
    for (size_t i = 0; i < xDimNum - 1; i++) {
        totalTokens_ *= xStorageShape.GetDim(i);
    }
    dim2H_ = xStorageShape.GetDim(xDimNum - 1);
    OP_CHECK_IF((dim2H_ % SWI_FACTOR != 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x", ShapeToString(xStorageShape),
                    "The last dim of input x must be even, but got " + std::to_string(dim2H_)),
                return ge::GRAPH_FAILED);
    dimH_ = dim2H_ / SWI_FACTOR;

    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantHifp8Tiling::CalcTileTokens()
{
    int64_t ubAvailable = static_cast<int64_t>(ubSize_) - UB_RESERVE;
    int64_t ubFactor = (quantMode_ == QUANT_MODE_STATIC) ? UB_FACTOR_STATIC : UB_FACTOR_DYNAMIC;
    int64_t ubPerToken = ubFactor * dimH_ * SIZE_OF_FLOAT;
    if (hasWeight_) {
        ubPerToken += DB_BUFFER * SIZE_OF_FLOAT;
    }

    tileTokens_ = ubAvailable / ubPerToken;
    if (tileTokens_ < 1) {
        tileTokens_ = 1;
    }
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CalcCoreDistribution()
{
    if (!isGroup_) {
        usedCoreNum_ = std::min(static_cast<int64_t>(coreNum_), totalTokens_);
        if (usedCoreNum_ == 0) {
            return ge::GRAPH_FAILED;
        }
        tokensPerCore_ = totalTokens_ / usedCoreNum_;
    } else {
        usedCoreNum_ = std::min({static_cast<int64_t>(coreNum_), groupNum_, MAX_CORE_COUNT});
        tokensPerCore_ = 0;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CalcOpTiling()
{
    CalcTileTokens();
    tileLength_ = tileTokens_ * dimH_;
    return CalcCoreDistribution();
}

void SwigluGroupQuantHifp8Tiling::SetTilingData()
{
    tilingData_.set_totalTokens(totalTokens_);
    tilingData_.set_dim2H(dim2H_);
    tilingData_.set_dimH(dimH_);
    tilingData_.set_isGroup(isGroup_ ? 1 : 0);
    tilingData_.set_hasWeight(hasWeight_ ? 1 : 0);
    tilingData_.set_hasClamp(hasClamp_ ? 1 : 0);
    tilingData_.set_outputOrigin(outputOrigin_ ? 1 : 0);
    tilingData_.set_clampLimit(clampLimit_);
    tilingData_.set_dstTypeMax(dstTypeMax_);
    tilingData_.set_tileTokens(tileTokens_);
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_tokensPerCore(tokensPerCore_);
    tilingData_.set_groupNum(groupNum_);
    tilingData_.set_tileLength(tileLength_);
    tilingData_.set_groupListType(groupListType_);
}

void SwigluGroupQuantHifp8Tiling::SetTilingKey()
{
    tilingKey_ = (quantMode_ == QUANT_MODE_STATIC) ? STATIC_HIFP8_TILING_KEY : DYNAMIC_HIFP8_TILING_KEY;
    context_->SetTilingKey(tilingKey_);
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    // Only dynamic non-group quantization needs workspace for cross-core reduction.
    if (isGroup_ || quantMode_ == QUANT_MODE_STATIC) {
        return ge::GRAPH_SUCCESS;
    }
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    workspaceSize_ += (usedCoreNum_ + 1) * SIZE_OF_FLOAT + BLOCK_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::PostTiling()
{
    context_->SetScheduleMode(BATCH_MODE);
    context_->SetBlockDim(usedCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::CheckAllInputs()
{
    if (CheckInputDtype() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOutputDtype() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetAttr() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckWeightInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckGroupIndexInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckScaleInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckOutputInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::ProcessTiling()
{
    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetTilingData();
    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetTilingKey();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantHifp8Tiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAllInputs() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ProcessTiling();
}

} // namespace optiling
