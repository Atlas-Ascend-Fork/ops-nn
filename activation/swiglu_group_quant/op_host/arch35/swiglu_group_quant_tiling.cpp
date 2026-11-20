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
 * \file swiglu_group_quant_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include <limits>
#include <string>
#include <vector>
#include <graph/utils/type_utils.h>
#include "../../../../common/inc/op_host/op_const_def_nn.h"
#include "../../../swiglu_group_quant_with_dual_axis/op_host/arch35/swiglu_group_quant_mx_tiling.h"
#include "../../../swiglu_group_quant_with_dual_axis/op_host/arch35/swiglu_group_quant_with_dual_axis_policy.h"
#include "swiglu_group_quant_tiling.h"
#include "../../../swiglu_group_quant_with_dual_axis/op_kernel/arch35/swiglu_group_quant_flags.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 0;
ge::graphStatus ReportTilingFailure(const char* opName, const std::string& reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason.c_str());
    return ge::GRAPH_FAILED;
}
void ReportUnsupportedQuantMode(const char* opName, const char* paramName, const std::string& actual,
                                const std::string& reason)
{
    const std::string fullReason = std::string("Parameter ") + paramName + ": " + reason + ", but got quant_mode " +
                                   actual;
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, fullReason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1009", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", fullReason.c_str());
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
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) { return CeilDiv(x, y) * y; }

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t D_LIMIT = 256;
constexpr int64_t MX_QUANT_D_LIMIT = 64;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t FP8_BYTES = 1;
constexpr int64_t B16_BYTES = 2;
constexpr int64_t B32_BYTES = 4;
constexpr int64_t FP8_ALIGN_NUM = BLOCK_SIZE / FP8_BYTES;
constexpr int64_t B16_ALIGN_NUM = BLOCK_SIZE / B16_BYTES;
constexpr int64_t B32_ALIGN_NUM = BLOCK_SIZE / B32_BYTES;
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t PER_MX_FP16 = 32;
constexpr int64_t PER_HIF4 = 64;
constexpr int64_t FP4_PACK_NUM = 2;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2;
constexpr int64_t MX_SCALE_ALIGN_FACTOR = 2;
constexpr size_t MIN_X_DIM_NUM = 2;
constexpr size_t MAX_DIM_NUM = 8;
constexpr int64_t BLOCK_QUANT = 0;
constexpr int64_t MX_QUANT = 1;
constexpr int64_t MX_QUANT_EXTEND = 5;
constexpr int64_t HIF4_QUANT = 4;
constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
constexpr size_t ATTR_INDEX_BLOCK_SIZE = 2;
constexpr size_t ATTR_INDEX_ROUND_SCALE = 3;
constexpr size_t ATTR_INDEX_CLAMP_LIMIT = 4;
constexpr size_t ATTR_INDEX_OUTPUT_ORIGIN = 6;
constexpr size_t ATTR_INDEX_ALPHA = 7;
constexpr size_t ATTR_INDEX_BIAS = 8;
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
constexpr size_t INPUT_INDEX_GROUP_INDEX = 2;
constexpr size_t INPUT_INDEX_SCALE = 3;
constexpr size_t OUTPUT_INDEX_Y = 0;
constexpr size_t OUTPUT_INDEX_Y_SCALE = 1;
constexpr size_t OUTPUT_INDEX_Y_ORIGIN = 2;
constexpr size_t CACHE_LINE_SIZE = 128;
constexpr float DEFAULT_CLAMP_LIMIT = -1.0f;
constexpr float FLOAT_COMPARE_EPSILON = 1e-6f;
constexpr float DEFAULT_ALPHA = 1.0f;
constexpr float DEFAULT_BIAS = 0.0f;
// Weight type encoding consumed by the shared MX kernel.
constexpr uint32_t MX_WEIGHT_TYPE_FP16 = 0U;
constexpr uint32_t MX_WEIGHT_TYPE_BF16 = 1U;
constexpr uint32_t MX_WEIGHT_TYPE_FP32 = 2U;
constexpr int64_t BLOCK_QUANT_TILING_KEY = 1000;
constexpr int64_t BLOCK_QUANT_YORIGIN_TILING_KEY = 1100;
constexpr int64_t MX_QUANT_TILING_KEY = 2000;
constexpr int64_t MX_QUANT_YORIGIN_TILING_KEY = 2100;
constexpr int64_t MXFP4_QUANT_TILING_KEY = 3000;
constexpr int64_t MX_QUANT_EXTEND_TILING_KEY = 5000;
constexpr int64_t MX_QUANT_EXTEND_YORIGIN_TILING_KEY = 5100;
constexpr int64_t MXFP4_QUANT_YORIGIN_TILING_KEY = 3100;
constexpr int64_t HIF4_QUANT_TILING_KEY = 6000;
constexpr int64_t HIF4_QUANT_YORIGIN_TILING_KEY = 6100;

int64_t ShapeElementNum(const gert::Shape& shape)
{
    int64_t elementNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        elementNum *= shape.GetDim(i);
    }
    return elementNum;
}

bool CheckedMul(int64_t lhs, int64_t rhs, int64_t& out)
{
    if (lhs < 0 || rhs < 0 || (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
        return false;
    }
    out = lhs * rhs;
    return true;
}
} // namespace

ge::graphStatus SwigluGroupQuantTiling::GetPlatformInfoCommon(const gert::TilingContext* context, uint64_t& coreNum,
                                                              uint64_t& ubSize)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context->GetCompileInfo<SwigluGroupQuantCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr,
                    ReportTilingFailure(context->GetNodeName(), "Get compile info failed, it is null."),
                    return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize = ubSizePlatForm;
    }
    OP_CHECK_IF(
        (coreNum == 0 || ubSize == 0),
        ReportTilingFailure(context->GetNodeName(), "Get platform info failed, coreNum=" + std::to_string(coreNum) +
                                                        ", ubSize=" + std::to_string(ubSize) + "."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetPlatformInfo()
{
    auto ret = GetPlatformInfoCommon(context_, coreNum_, ubSize_);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    if (context_->GetPlatformInfo() != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetClampLimitAttr(const gert::RuntimeAttrs* attrs)
{
    auto clampLimitAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_CLAMP_LIMIT);
    if (clampLimitAttr != nullptr) {
        OP_CHECK_IF(quantMode_ == MX_QUANT_EXTEND && !std::isfinite(*clampLimitAttr),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "clamp_limit", std::to_string(*clampLimitAttr),
                        "The value of attr clamp_limit must be finite when quant_mode is 5"),
                    return ge::GRAPH_FAILED);
        // DEFAULT_CLAMP_LIMIT means user did not pass clamp_limit.
        if (std::fabs(*clampLimitAttr - DEFAULT_CLAMP_LIMIT) > FLOAT_COMPARE_EPSILON) {
            OP_CHECK_IF(!(*clampLimitAttr > 0.0f),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "clamp_limit",
                                                              std::to_string(*clampLimitAttr),
                                                              "The value of attr clamp_limit must be greater than 0.0"),
                        return ge::GRAPH_FAILED);
            clampLimit_ = *clampLimitAttr;
            hasClampLimit_ = 1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto groupListTypeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUP_LIST_TYPE);
    groupListType_ = groupListTypeAttr == nullptr ? GROUP_LIST_TYPE_COUNT : *groupListTypeAttr;
    OP_CHECK_IF(groupListType_ != GROUP_LIST_TYPE_COUNT && groupListType_ != GROUP_LIST_TYPE_PAIR,
                OP_LOGE(context_->GetNodeName(), "group_list_type only supports 1 or 2, got %ld.", groupListType_),
                return ge::GRAPH_FAILED);

    auto quantModeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    quantMode_ = quantModeAttr == nullptr ? BLOCK_QUANT : *quantModeAttr;
    const bool isHif4Quant = quantMode_ == HIF4_QUANT;
    const bool isMxQuant = quantMode_ == MX_QUANT || quantMode_ == MX_QUANT_EXTEND;

    if (isHif4Quant) {
        // quant_mode=4 always quantizes to HIFLOAT4 and ignores dst_type.
        dstType_ = Ops::NN::DT_HIFLOAT4;
    } else {
        auto dstTypeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DST_TYPE);
        dstType_ = dstTypeAttr == nullptr ? ge::DT_FLOAT8_E4M3FN : static_cast<ge::DataType>(*dstTypeAttr);
        OP_CHECK_IF((dstType_ != ge::DT_FLOAT8_E4M3FN && dstType_ != ge::DT_FLOAT8_E5M2 &&
                     dstType_ != ge::DT_FLOAT4_E2M1 && dstType_ != ge::DT_FLOAT4_E1M2),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "dst_type", std::to_string(static_cast<int64_t>(dstType_)),
                        "The value of attr dst_type must be one of (FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, "
                        "FLOAT4_E1M2)"),
                    return ge::GRAPH_FAILED);
    }
    isMxFp4Quant_ = dstType_ == ge::DT_FLOAT4_E2M1 || dstType_ == ge::DT_FLOAT4_E1M2;

    OP_CHECK_IF((quantMode_ != BLOCK_QUANT && quantMode_ != MX_QUANT && quantMode_ != HIF4_QUANT &&
                 quantMode_ != MX_QUANT_EXTEND),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "quant_mode", std::to_string(quantMode_),
                    "The value of attr quant_mode must be one of 0 (block_quant), 1 (mx_quant), 4 (hif4_quant) or 5 "
                    "(mx_quant_v2)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((isMxFp4Quant_ && quantMode_ != MX_QUANT),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "quant_mode", std::to_string(quantMode_),
                    "The value of attr quant_mode must be 1 (mx_quant) when dst_type is FLOAT4_E2M1 or FLOAT4_E1M2"),
                return ge::GRAPH_FAILED);

    if (isHif4Quant) {
        // quant_mode=4 uses the fixed HIF4 group size and ignores block_size/round_scale.
        splitFactor_ = PER_HIF4;
    } else {
        auto blockSizeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_BLOCK_SIZE);
        int64_t blockSize = blockSizeAttr == nullptr ? 0 : *blockSizeAttr;
        int64_t expectedBlockSize = isMxQuant ? PER_MX_FP16 : PER_BLOCK_FP16;
        OP_CHECK_IF((blockSize != 0 && blockSize != expectedBlockSize),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->GetNodeName(), "block_size", std::to_string(blockSize),
                        "The value of attr block_size must be 0 or " + std::to_string(expectedBlockSize) +
                            " when quant_mode is " + std::to_string(quantMode_)),
                    return ge::GRAPH_FAILED);
        splitFactor_ = expectedBlockSize;
    }

    auto roundScaleAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_ROUND_SCALE);
    if (roundScaleAttr != nullptr) {
        roundScale_ = (*roundScaleAttr) ? 1 : 0;
    }
    OP_CHECK_IF((isMxQuant && roundScale_ != 1),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "round_scale", std::to_string(roundScale_),
                    "The value of attr round_scale must be true when quant_mode is " + std::to_string(quantMode_)),
                return ge::GRAPH_FAILED);

    auto outputOriginAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_OUTPUT_ORIGIN);
    if (outputOriginAttr != nullptr) {
        outputOrigin_ = *outputOriginAttr;
    }

    if (GetClampLimitAttr(attrs) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    auto alphaAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_ALPHA);
    auto biasAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_BIAS);
    alpha_ = alphaAttr == nullptr ? DEFAULT_ALPHA : *alphaAttr;
    bias_ = biasAttr == nullptr ? DEFAULT_BIAS : *biasAttr;
    if (quantMode_ == MX_QUANT_EXTEND) {
        OP_CHECK_IF(
            !std::isfinite(alpha_) || alpha_ <= 0.0f || !std::isfinite(bias_),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "alpha and bias", std::to_string(alpha_) + " and " + std::to_string(bias_),
                "The value of attr alpha must be finite and greater than 0.0, and the value of attr bias must "
                "be finite"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            alpha_ != DEFAULT_ALPHA || bias_ != DEFAULT_BIAS,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context_->GetNodeName(), "alpha and bias", std::to_string(alpha_) + " and " + std::to_string(bias_),
                "The attrs alpha and bias are only supported when quant_mode is 5, but quant_mode is " +
                    std::to_string(quantMode_)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CheckWeightInfo()
{
    auto weightDesc = context_->GetOptionalInputDesc(INPUT_INDEX_WEIGHT);
    if (weightDesc != nullptr) {
        auto weightDtype = weightDesc->GetDataType();
        OP_CHECK_IF((weightDtype != ge::DT_FLOAT16 && weightDtype != ge::DT_BF16 && weightDtype != ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        context_->GetNodeName(), "weight", ge::TypeUtils::DataTypeToSerialString(weightDtype).c_str(),
                        "The dtype of input weight only supports float16, bfloat16 or float32"),
                    return ge::GRAPH_FAILED);
        auto weightShape = context_->GetOptionalInputShape(INPUT_INDEX_WEIGHT);
        if (weightShape != nullptr) {
            auto weightStorageShape = weightShape->GetStorageShape();
            auto weightDimNum = weightStorageShape.GetDimNum();
            OP_CHECK_IF(
                (weightDimNum < 1 || weightDimNum > MAX_DIM_NUM),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), "weight", std::to_string(weightDimNum),
                    "The shape dim of input weight must be in the range [1, " + std::to_string(MAX_DIM_NUM) + "]"),
                return ge::GRAPH_FAILED);
            auto weightElementNum = ShapeElementNum(weightStorageShape);
            OP_CHECK_IF((weightElementNum != bs_),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), "weight", ShapeToString(weightStorageShape),
                            "The number of elements of input weight must be equal to the product of input x dims "
                            "except the last one (" +
                                std::to_string(bs_) + "), but got " + std::to_string(weightElementNum)),
                        return ge::GRAPH_FAILED);
            weightType_ = weightDtype;
            if (weightDtype != ge::DT_FLOAT) {
                OP_CHECK_IF(
                    quantMode_ != MX_QUANT_EXTEND,
                    ReportUnsupportedQuantMode(context_->GetNodeName(), "weight", std::to_string(quantMode_),
                                               "A low-precision input weight is only supported when quant_mode is 5"),
                    return ge::GRAPH_FAILED);
            }
            hasWeight_ = true;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CheckGroupIndexInfo()
{
    auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_INDEX_GROUP_INDEX);
    if (groupIndexDesc != nullptr) {
        OP_CHECK_IF(quantMode_ == MX_QUANT_EXTEND,
                    ReportUnsupportedQuantMode(context_->GetNodeName(), "group_index", std::to_string(quantMode_),
                                               "The input group_index is not supported when quant_mode is 5"),
                    return ge::GRAPH_FAILED);
        auto groupIndexDtype = groupIndexDesc->GetDataType();
        OP_CHECK_IF(
            (groupIndexDtype != ge::DT_INT64),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "group_index",
                                                  ge::TypeUtils::DataTypeToSerialString(groupIndexDtype).c_str(),
                                                  "The dtype of input group_index must be int64"),
            return ge::GRAPH_FAILED);
        auto groupIndexShape = context_->GetOptionalInputShape(INPUT_INDEX_GROUP_INDEX);
        if (groupIndexShape != nullptr) {
            auto groupIndexStorageShape = groupIndexShape->GetStorageShape();
            auto groupIndexDimNum = groupIndexStorageShape.GetDimNum();
            const bool isPair = groupListType_ == GROUP_LIST_TYPE_PAIR;
            OP_CHECK_IF(isPair ? (groupIndexDimNum != 2 || groupIndexStorageShape.GetDim(1) != GROUP_INDEX_PAIR_COLS) :
                                 (groupIndexDimNum != 1),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            context_->GetNodeName(), "group_index", std::to_string(groupIndexDimNum),
                            "The shape dim of input group_index must be 1 for group_list_type 1, or 2 with the "
                            "second dim equal to 2 for group_list_type 2"),
                        return ge::GRAPH_FAILED);
            g_ = ShapeElementNum(groupIndexStorageShape);
            // Empty tensor is not supported: a passed group_index must have a positive element count.
            OP_CHECK_IF((g_ <= 0),
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            context_->GetNodeName(), "group_index", std::to_string(g_),
                            "An empty input group_index is not supported, the shape size must be greater than 0"),
                        return ge::GRAPH_FAILED);
            hasGroupIndex_ = true;
        }
    }
    OP_CHECK_IF(groupListType_ == GROUP_LIST_TYPE_PAIR && !hasGroupIndex_,
                OP_LOGE(context_->GetNodeName(), "group_index is required when group_list_type is 2."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CheckOutputInfo(ge::DataType xDtype, const gert::Shape& xStorageShape)
{
    auto yDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF((yDtype != dstType_),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y",
                                                      ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                                                      "The dtype of output y must be the same as dst_type (" +
                                                          ge::TypeUtils::DataTypeToSerialString(dstType_) + ")"),
                return ge::GRAPH_FAILED);

    auto yScaleDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yScaleDesc);
    auto yScaleDtype = yScaleDesc->GetDataType();
    const bool isMxQuant = quantMode_ == MX_QUANT || quantMode_ == MX_QUANT_EXTEND;
    ge::DataType expectedYScaleDtype;
    if (quantMode_ == HIF4_QUANT) {
        expectedYScaleDtype = Ops::NN::DT_HIFLOAT4_SCALE;
    } else {
        expectedYScaleDtype = isMxQuant ? ge::DT_FLOAT8_E8M0 : ge::DT_FLOAT;
    }
    OP_CHECK_IF(
        (yScaleDtype != expectedYScaleDtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            context_->GetNodeName(), "y_scale", ge::TypeUtils::DataTypeToSerialString(yScaleDtype).c_str(),
            "The dtype of output y_scale must be " + ge::TypeUtils::DataTypeToSerialString(expectedYScaleDtype) +
                " when quant_mode is " + std::to_string(quantMode_)),
        return ge::GRAPH_FAILED);

    auto yOriginDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y_ORIGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yOriginDesc);
    auto yOriginDtype = yOriginDesc->GetDataType();
    OP_CHECK_IF((yOriginDtype != xDtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y_origin",
                                                      ge::TypeUtils::DataTypeToSerialString(yOriginDtype).c_str(),
                                                      "The dtype of output y_origin must be the same as input x (" +
                                                          ge::TypeUtils::DataTypeToSerialString(xDtype) + ")"),
                return ge::GRAPH_FAILED);

    int64_t xDimNum = static_cast<int64_t>(xStorageShape.GetDimNum());

    // Check y shape: [..., D/2].
    auto yShape = context_->GetOutputShape(OUTPUT_INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = yShape->GetStorageShape();
    OP_CHECK_IF((yStorageShape.GetDimNum() != static_cast<size_t>(xDimNum)),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), "y", std::to_string(yStorageShape.GetDimNum()),
                    "The shape dim of output y must be the same as input x (" + std::to_string(xDimNum) + ")"),
                return ge::GRAPH_FAILED);
    for (int64_t i = 0; i < xDimNum - 1; ++i) {
        OP_CHECK_IF((yStorageShape.GetDim(i) != xStorageShape.GetDim(i)),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "y", ShapeToString(yStorageShape),
                        "The dim[" + std::to_string(i) + "] of output y must be equal to the dim[" + std::to_string(i) +
                            "] of input x (" + std::to_string(xStorageShape.GetDim(i)) + ")"),
                    return ge::GRAPH_FAILED);
    }
    int64_t expectedYLastDim = splitD_;
    OP_CHECK_IF((yStorageShape.GetDim(xDimNum - 1) != expectedYLastDim),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y", ShapeToString(yStorageShape),
                                                      "The last dim of output y must be " +
                                                          std::to_string(expectedYLastDim) + " when dst_type is " +
                                                          ge::TypeUtils::DataTypeToSerialString(dstType_)),
                return ge::GRAPH_FAILED);

    // Check y_scale shape.
    auto yScaleShape = context_->GetOutputShape(OUTPUT_INDEX_Y_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yScaleShape);
    auto yScaleStorageShape = yScaleShape->GetStorageShape();
    if (isMxQuant) {
        // [..., ceil(ceil((D/2)/32)/2), 2]
        OP_CHECK_IF((yScaleStorageShape.GetDimNum() != static_cast<size_t>(xDimNum + 1)),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "y_scale", std::to_string(yScaleStorageShape.GetDimNum()),
                        "The shape dim of output y_scale must be " + std::to_string(xDimNum + 1) +
                            " when quant_mode is " + std::to_string(quantMode_)),
                    return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < xDimNum - 1; ++i) {
            OP_CHECK_IF((yScaleStorageShape.GetDim(i) != xStorageShape.GetDim(i)),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), "y_scale", ShapeToString(yScaleStorageShape),
                            "The dim[" + std::to_string(i) + "] of output y_scale must be equal to the dim[" +
                                std::to_string(i) + "] of input x (" + std::to_string(xStorageShape.GetDim(i)) + ")"),
                        return ge::GRAPH_FAILED);
        }
        int64_t expectedMxTailDim = CeilDiv(scaleCol_, MX_SCALE_ALIGN_FACTOR);
        OP_CHECK_IF(
            (yScaleStorageShape.GetDim(xDimNum - 1) != expectedMxTailDim ||
             yScaleStorageShape.GetDim(xDimNum) != MX_SCALE_ALIGN_FACTOR),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "y_scale", ShapeToString(yScaleStorageShape),
                "The last two dims of output y_scale must be [" + std::to_string(expectedMxTailDim) + ", " +
                    std::to_string(MX_SCALE_ALIGN_FACTOR) + "] when quant_mode is " + std::to_string(quantMode_)),
            return ge::GRAPH_FAILED);
    } else if (quantMode_ == HIF4_QUANT) {
        // [..., ceil((D/2)/64), 1]
        OP_CHECK_IF(
            (yScaleStorageShape.GetDimNum() != static_cast<size_t>(xDimNum + 1)),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context_->GetNodeName(), "y_scale", std::to_string(yScaleStorageShape.GetDimNum()),
                "The shape dim of output y_scale must be " + std::to_string(xDimNum + 1) + " when quant_mode is 4"),
            return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < xDimNum - 1; ++i) {
            OP_CHECK_IF((yScaleStorageShape.GetDim(i) != xStorageShape.GetDim(i)),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), "y_scale", ShapeToString(yScaleStorageShape),
                            "The dim[" + std::to_string(i) + "] of output y_scale must be equal to the dim[" +
                                std::to_string(i) + "] of input x (" + std::to_string(xStorageShape.GetDim(i)) + ")"),
                        return ge::GRAPH_FAILED);
        }
        int64_t expectedHif4TailDim = CeilDiv(splitD_, PER_HIF4);
        OP_CHECK_IF(
            (yScaleStorageShape.GetDim(xDimNum - 1) != expectedHif4TailDim || yScaleStorageShape.GetDim(xDimNum) != 1),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y_scale", ShapeToString(yScaleStorageShape),
                                                  "The last two dims of output y_scale must be [" +
                                                      std::to_string(expectedHif4TailDim) +
                                                      ", 1] when quant_mode is 4"),
            return ge::GRAPH_FAILED);
    } else {
        // [..., ceil((D/2)/128)]
        OP_CHECK_IF((yScaleStorageShape.GetDimNum() != static_cast<size_t>(xDimNum)),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "y_scale", std::to_string(yScaleStorageShape.GetDimNum()),
                        "The shape dim of output y_scale must be " + std::to_string(xDimNum) + " when quant_mode is 0"),
                    return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < xDimNum - 1; ++i) {
            OP_CHECK_IF((yScaleStorageShape.GetDim(i) != xStorageShape.GetDim(i)),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), "y_scale", ShapeToString(yScaleStorageShape),
                            "The dim[" + std::to_string(i) + "] of output y_scale must be equal to the dim[" +
                                std::to_string(i) + "] of input x (" + std::to_string(xStorageShape.GetDim(i)) + ")"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(
            (yScaleStorageShape.GetDim(xDimNum - 1) != scaleCol_),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context_->GetNodeName(), "y_scale", ShapeToString(yScaleStorageShape),
                "The last dim of output y_scale must be " + std::to_string(scaleCol_) + " when quant_mode is 0"),
            return ge::GRAPH_FAILED);
    }

    if (outputOrigin_) {
        // Check y_origin shape: [..., D/2].
        auto yOriginShape = context_->GetOutputShape(OUTPUT_INDEX_Y_ORIGIN);
        OP_CHECK_NULL_WITH_CONTEXT(context_, yOriginShape);
        auto yOriginStorageShape = yOriginShape->GetStorageShape();
        OP_CHECK_IF(
            (yOriginStorageShape.GetDimNum() != static_cast<size_t>(xDimNum)),
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context_->GetNodeName(), "y_origin", std::to_string(yOriginStorageShape.GetDimNum()),
                "The shape dim of output y_origin must be the same as input x (" + std::to_string(xDimNum) + ")"),
            return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < xDimNum - 1; ++i) {
            OP_CHECK_IF((yOriginStorageShape.GetDim(i) != xStorageShape.GetDim(i)),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), "y_origin", ShapeToString(yOriginStorageShape),
                            "The dim[" + std::to_string(i) + "] of output y_origin must be equal to the dim[" +
                                std::to_string(i) + "] of input x (" + std::to_string(xStorageShape.GetDim(i)) + ")"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF((yOriginStorageShape.GetDim(xDimNum - 1) != splitD_),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->GetNodeName(), "y_origin", ShapeToString(yOriginStorageShape),
                        "The last dim of output y_origin must be " + std::to_string(splitD_)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetShapeAttrsInfoInner()
{
    // (b, s, hc_mix)
    auto shapeX = context_->GetInputShape(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);

    auto xStorageShape = shapeX->GetStorageShape();
    auto xDesc = context_->GetInputDesc(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF((xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(xDtype).c_str(),
                                                      "The dtype of input x only supports float16 or bfloat16"),
                return ge::GRAPH_FAILED);
    auto xDimNum = xStorageShape.GetDimNum();
    OP_CHECK_IF((xDimNum < MIN_X_DIM_NUM || xDimNum > MAX_DIM_NUM),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xDimNum),
                                                         "The shape dim of input x must be in the range [" +
                                                             std::to_string(MIN_X_DIM_NUM) + ", " +
                                                             std::to_string(MAX_DIM_NUM) + "]"),
                return ge::GRAPH_FAILED);

    // Get Attrs
    // Resolve the mode before applying geometry rules. MX modes 1/5 accept
    // D = 2H in 64-element units; block and HIF4 modes retain the 256-element contract.
    if (GetAttr() == ge::GRAPH_FAILED) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Get attr failed.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((quantMode_ == MX_QUANT_EXTEND && xDimNum != MIN_X_DIM_NUM),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->GetNodeName(), "x", std::to_string(xDimNum),
                    "The shape dim of input x must be " + std::to_string(MIN_X_DIM_NUM) + " when quant_mode is 5"),
                return ge::GRAPH_FAILED);
    bs_ = 1;
    for (size_t i = 0; i < xDimNum - 1; i++) {
        const int64_t extent = xStorageShape.GetDim(i);
        if (quantMode_ == MX_QUANT_EXTEND) {
            int64_t next = 0;
            OP_CHECK_IF(!CheckedMul(bs_, extent, next),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            context_->GetNodeName(), "x", ShapeToString(xStorageShape),
                            "The shape of input x exceeds the int64 range or contains a non-positive dim[" +
                                std::to_string(i) + "] " + std::to_string(extent) +
                                " when quant_mode is 5, accumulated product is " + std::to_string(bs_)),
                        return ge::GRAPH_FAILED);
            bs_ = next;
        } else {
            bs_ = bs_ * extent;
        }
    }
    // Empty tensor is supported: when bs is 0 the input and outputs are all empty and no data
    // needs to be processed, so only negative dims (unknown dim leaked in) are rejected here.
    OP_CHECK_IF(
        (bs_ < 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x", ShapeToString(xStorageShape),
            "The product of input x dims except the last one must be non-negative (" + std::to_string(bs_) + ")"),
        return ge::GRAPH_FAILED);
    d_ = xStorageShape.GetDim(xDimNum - 1);
    // An empty x (last dim 0) is supported by legacy block/MX/HIF4 modes; otherwise the last dim must be
    // greater than or equal to and divisible by the mode-specific D limit. Mode 5 does not support empty x.
    const int64_t dLimit = (quantMode_ == MX_QUANT || quantMode_ == MX_QUANT_EXTEND) ? MX_QUANT_D_LIMIT : D_LIMIT;
    const bool supportsEmpty = quantMode_ == BLOCK_QUANT || quantMode_ == MX_QUANT || quantMode_ == HIF4_QUANT;
    OP_CHECK_IF(((d_ == 0 && !supportsEmpty) || (d_ != 0 && (d_ < dLimit || d_ % dLimit != 0))),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "x", ShapeToString(xStorageShape),
                    "The last dim of input x must be 0 for an empty supported mode or greater than or equal to " +
                        std::to_string(dLimit) + " and divisible by " + std::to_string(dLimit)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(quantMode_ == MX_QUANT_EXTEND && context_->GetOptionalInputDesc(INPUT_INDEX_SCALE) != nullptr,
                ReportUnsupportedQuantMode(context_->GetNodeName(), "scale", std::to_string(quantMode_),
                                           "The input scale is not supported when quant_mode is 5"),
                return ge::GRAPH_FAILED);

    if (CheckWeightInfo() == ge::GRAPH_FAILED || CheckGroupIndexInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    // y_scale has one more dim than x in MX mode and HIF4 mode, and y_scale dim num must be in [1, 8],
    // so x dim num is limited to [2, 7] when quant_mode is 1 or 4.
    if (quantMode_ == MX_QUANT || quantMode_ == HIF4_QUANT) {
        OP_CHECK_IF((xDimNum > MAX_DIM_NUM - 1),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "x", std::to_string(xDimNum),
                        "The shape dim of input x must be in the range [" + std::to_string(MIN_X_DIM_NUM) + ", " +
                            std::to_string(MAX_DIM_NUM - 1) + "] when quant_mode is " + std::to_string(quantMode_)),
                    return ge::GRAPH_FAILED);
    }

    splitD_ = d_ / SWIGLU_SPLIT_FACTOR;
    scaleCol_ = CeilDiv(splitD_, splitFactor_);

    if (CheckOutputInfo(xDtype, xStorageShape) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcGroupIndexTiling()
{
    if (!hasGroupIndex_) {
        return ge::GRAPH_SUCCESS;
    }

    gFactor_ = g_;
    int64_t groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
    int64_t groupIndexSumSize = BLOCK_SIZE;
    if (groupIndexSize + groupIndexSumSize <= static_cast<int64_t>(ubSize_)) {
        gLoop_ = 1;
        tailGFactor_ = gFactor_;
        return ge::GRAPH_SUCCESS;
    }

    int64_t base = 2;
    int64_t maxBase = std::max(g_, base);
    while (base <= maxBase) {
        gFactor_ = CeilDiv(g_, base);
        if (groupListType_ == GROUP_LIST_TYPE_PAIR) {
            gFactor_ = std::max(DownAlign(gFactor_, GROUP_INDEX_PAIR_COLS), GROUP_INDEX_PAIR_COLS);
        }
        groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
        if (groupIndexSize + groupIndexSumSize < static_cast<int64_t>(ubSize_)) {
            break;
        }
        base++;
    }
    gFactor_ = std::max(gFactor_, static_cast<int64_t>(1));
    if (gFactor_ > static_cast<int64_t>(CACHE_LINE_SIZE / sizeof(int64_t))) {
        gFactor_ = DownAlign(gFactor_, CACHE_LINE_SIZE / sizeof(int64_t));
        gFactor_ = std::max(gFactor_, static_cast<int64_t>(1));
    }
    gLoop_ = CeilDiv(g_, gFactor_);
    tailGFactor_ = g_ % gFactor_ == 0 ? gFactor_ : g_ % gFactor_;
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantTiling::InitCoreTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;
}

void SwigluGroupQuantTiling::SetFullDTiling()
{
    dLoop_ = 1;
    dFactor_ = splitD_;
    tailDFactor_ = dFactor_;
}

void SwigluGroupQuantTiling::SetSplitDTiling()
{
    scaleCol_ = CeilDiv(splitD_, splitFactor_);
    dLoop_ = CeilDiv(splitD_, dFactor_);
    tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
}

void SwigluGroupQuantTiling::SetRowLoopTiling()
{
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;
}

int64_t SwigluGroupQuantTiling::AddWeightSize(int64_t totalSize, int64_t rowFactor) const
{
    return hasWeight_ ? totalSize + RoundUp(rowFactor, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER : totalSize;
}

int64_t SwigluGroupQuantTiling::AddYOriginSize(int64_t totalSize, int64_t rowFactor, int64_t dFactor) const
{
    int64_t yOriginSize = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    return outputOrigin_ ? totalSize + yOriginSize : totalSize;
}

int64_t SwigluGroupQuantTiling::CalcMxQuantTotalSize(int64_t rowFactor, int64_t dFactor) const
{
    int64_t tryScaleCol = RoundUp(CeilDiv(dFactor, splitFactor_), MX_SCALE_ALIGN_FACTOR);
    int64_t x0Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t x1Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t ySize = rowFactor * RoundUp(dFactor, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t scaleSize = RoundUp(rowFactor * tryScaleCol, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t yFp32Size = rowFactor * RoundUp(dFactor, B32_ALIGN_NUM) * B32_BYTES;
    int64_t invScaleSize = rowFactor * RoundUp(CeilDiv(dFactor, B32_ALIGN_NUM), B32_ALIGN_NUM) * B32_BYTES;
    int64_t totalSize = x0Size + x1Size + ySize + scaleSize + yFp32Size + invScaleSize;
    totalSize = AddWeightSize(totalSize, rowFactor);
    return AddYOriginSize(totalSize, rowFactor, dFactor);
}

int64_t SwigluGroupQuantTiling::CalcBlockQuantTotalSize(int64_t rowFactor, int64_t dFactor) const
{
    int64_t tryScaleCol = CeilDiv(dFactor, splitFactor_);
    int64_t x0Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t x1Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t ySize = rowFactor * RoundUp(dFactor, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t scaleSize = RoundUp(rowFactor * tryScaleCol, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;
    int64_t totalSize = x0Size + x1Size + ySize + scaleSize;
    totalSize = AddWeightSize(totalSize, rowFactor);
    return AddYOriginSize(totalSize, rowFactor, dFactor);
}

int64_t SwigluGroupQuantTiling::CalcMxFp4QuantTotalSize(int64_t rowFactor, int64_t dFactor) const
{
    int64_t tryScaleCol = CeilDiv(dFactor, splitFactor_);
    int64_t x0Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t x1Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t swigluSize = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES;
    int64_t maxExpSize = rowFactor * RoundUp(tryScaleCol, B16_ALIGN_NUM) * B16_BYTES;
    int64_t invScaleSize = rowFactor * RoundUp(tryScaleCol, B16_ALIGN_NUM) * B16_BYTES;
    int64_t ySize = rowFactor * RoundUp(CeilDiv(dFactor, FP4_PACK_NUM), FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t scaleSize = rowFactor * RoundUp(tryScaleCol, FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t totalSize = x0Size + x1Size + swigluSize + maxExpSize + invScaleSize + ySize + scaleSize;
    totalSize = AddWeightSize(totalSize, rowFactor);
    // When outputOrigin is enabled, the yOrigin queue keeps the pre-weight swiglu result while
    // quantization consumes the swiglu result from the swiglu buffer, so both are reserved in UB.
    return AddYOriginSize(totalSize, rowFactor, dFactor);
}

int64_t SwigluGroupQuantTiling::CalcHif4QuantTotalSize(int64_t rowFactor, int64_t dFactor) const
{
    int64_t b16AlignNum = 4 * REPEAT_SIZE / B16_BYTES;
    int64_t yCnt = RoundUp(rowFactor * RoundUp(dFactor, B16_ALIGN_NUM), b16AlignNum);
    int64_t eACnt = CeilDiv(yCnt, PER_HIF4);
    int64_t eBCnt = CeilDiv(yCnt, PER_HIF4);
    int64_t eCCnt = CeilDiv(yCnt, PER_HIF4 / SWIGLU_SPLIT_FACTOR);

    int64_t x0Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t x1Size = rowFactor * RoundUp(dFactor, B16_ALIGN_NUM) * B16_BYTES * DOUBLE_BUFFER;
    int64_t swigluSize = RoundUp(rowFactor * RoundUp(dFactor, B16_ALIGN_NUM), b16AlignNum) * B16_BYTES;
    int64_t eASize = RoundUp(eACnt, FP8_ALIGN_NUM) * FP8_BYTES;
    int64_t eBSize = RoundUp(eBCnt, FP8_ALIGN_NUM) * FP8_BYTES;
    int64_t eCSize = RoundUp(eCCnt, B16_ALIGN_NUM) * B16_BYTES;
    int64_t ySize = RoundUp(CeilDiv(yCnt, FP4_PACK_NUM), FP8_ALIGN_NUM) * FP8_BYTES * DOUBLE_BUFFER;
    int64_t scaleSize = RoundUp(CeilDiv(yCnt, PER_HIF4), B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER;
    int64_t totalSize = x0Size + x1Size + swigluSize + eASize + eBSize + eCSize + ySize + scaleSize;
    totalSize = AddWeightSize(totalSize, rowFactor);
    // When outputOrigin is enabled, the yOrigin queue keeps the pre-weight swiglu result while
    // quantization consumes the swiglu result from the swiglu buffer, so both are reserved in UB.
    return AddYOriginSize(totalSize, rowFactor, dFactor);
}

void SwigluGroupQuantTiling::CalcDAndRowFactorTiling(int64_t rowOnceLoop, int64_t dStep, TotalSizeFunc calcTotalSize)
{
    rowFactor_ = rowOnceLoop;
    if ((this->*calcTotalSize)(rowOnceLoop, splitD_) <= static_cast<int64_t>(ubSize_)) {
        SetFullDTiling();
    } else {
        int64_t base = 1;
        int64_t maxBase = CeilDiv(splitD_, dStep);
        dFactor_ = dStep;
        while (base <= maxBase) {
            int64_t tryDFactor = base * dStep;
            if (tryDFactor > splitD_) {
                tryDFactor = splitD_;
            }
            if ((this->*calcTotalSize)(rowOnceLoop, tryDFactor) > static_cast<int64_t>(ubSize_)) {
                break;
            }
            dFactor_ = tryDFactor;
            if (tryDFactor == splitD_) {
                break;
            }
            base++;
        }
        SetSplitDTiling();
    }

    if (dFactor_ == splitD_) {
        int64_t lo = 1;
        int64_t hi = rowOfFormerBlock_;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo + 1) / 2;
            if ((this->*calcTotalSize)(mid, dFactor_) <= static_cast<int64_t>(ubSize_)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        rowFactor_ = lo;
    }

    SetRowLoopTiling();
}

ge::graphStatus SwigluGroupQuantTiling::CalcMxQuantOpTiling()
{
    InitCoreTiling();
    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t dChunkAlign = REPEAT_SIZE / B16_BYTES;
    CalcDAndRowFactorTiling(rowOnceLoop, dChunkAlign, &SwigluGroupQuantTiling::CalcMxQuantTotalSize);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcMxQuantExtendOpTiling()
{
    const auto geometry = SwigluGroupQuantMxTiling::MakeGeometry(splitD_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = geometry.usedCoreNum;

    // Conservative budget: the common unweighted 64x384 packed path uses
    // 180224 bytes for single-axis, within the 203776-byte allocation below.
    constexpr int64_t activationInputs = 2;
    constexpr int64_t scalePair = 2;
    constexpr int64_t inputBytes = DOUBLE_BUFFER * activationInputs * SwigluGroupQuantMxTiling::TILE_ROWS *
                                   SwigluGroupQuantMxTiling::TILE_COLS * B16_BYTES;
    constexpr int64_t activationBytes = SwigluGroupQuantMxTiling::TILE_ROWS * SwigluGroupQuantMxTiling::TILE_COLS *
                                        B16_BYTES;
    const bool shareOrigin = SwigluDualAxisPolicy::ShareOrigin(hasWeight_, outputOrigin_ != 0);
    const int64_t outputDepth = (hasWeight_ || outputOrigin_ != 0) && !shareOrigin ? 1 : DOUBLE_BUFFER;
    const int64_t yBytes = outputDepth * SwigluGroupQuantMxTiling::TILE_ROWS * SwigluGroupQuantMxTiling::TILE_COLS *
                           FP8_BYTES;
    const int64_t scaleBytes = outputDepth * SwigluGroupQuantMxTiling::TILE_ROWS * BLOCK_SIZE;
    constexpr int64_t reciprocalBytes = SwigluGroupQuantMxTiling::TILE_ROWS * BLOCK_SIZE;
    constexpr int64_t scratchBytes = SwigluGroupQuantMxTiling::TILE_COLS * scalePair * B16_BYTES;
    const int64_t weightBytes = hasWeight_ ? SwigluGroupQuantMxTiling::TILE_ROWS * B32_BYTES : 0;
    const int64_t originBytes = outputOrigin_ != 0 && !shareOrigin ? activationBytes : 0;
    const int64_t plannedBytes = inputBytes + activationBytes + yBytes + scaleBytes + reciprocalBytes + scratchBytes +
                                 weightBytes + originBytes;
    OP_CHECK_IF(plannedBytes > static_cast<int64_t>(ubSize_),
                ReportTilingFailure(context_->GetNodeName(), "The planned UB bytes " + std::to_string(plannedBytes) +
                                                                 " exceed the available UB size " +
                                                                 std::to_string(ubSize_) + " when quant_mode is 5."),
                return ge::GRAPH_FAILED);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcBlockQuantOpTiling()
{
    InitCoreTiling();
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, static_cast<int64_t>(1));
    CalcDAndRowFactorTiling(rowOnceLoop, splitFactor_, &SwigluGroupQuantTiling::CalcBlockQuantTotalSize);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcMxFp4QuantOpTiling()
{
    InitCoreTiling();
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, static_cast<int64_t>(1));
    CalcDAndRowFactorTiling(rowOnceLoop, splitFactor_, &SwigluGroupQuantTiling::CalcMxFp4QuantTotalSize);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcHif4QuantOpTiling()
{
    InitCoreTiling();
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, static_cast<int64_t>(1));
    CalcDAndRowFactorTiling(rowOnceLoop, splitFactor_, &SwigluGroupQuantTiling::CalcHif4QuantTotalSize);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantTiling::SetEmptyTiling()
{
    // Zero workload: one core is launched whose row loop count is 0, so the kernel returns without
    // touching global memory. rowFactor/dFactor stay positive to keep kernel queue buffers valid.
    // Group-index tiling (gFactor/gLoop) is still computed by CalcGroupIndexTiling so the kernel
    // derives realBs = min(sum(groupIndex), bs) = 0 and exits from the group path as well. The
    // kernel also returns directly when splitD is 0, which covers d == 0 with a non-empty
    // group_index (realBs > 0 recomputes non-zero row loops in that path).
    rowOfFormerBlock_ = 0;
    rowOfTailBlock_ = 0;
    rowLoopOfFormerBlock_ = 0;
    rowLoopOfTailBlock_ = 0;
    tailRowFactorOfFormerBlock_ = 0;
    tailRowFactorOfTailBlock_ = 0;
    rowFactor_ = 1;
    dLoop_ = 1;
    dFactor_ = std::max(splitD_, splitFactor_);
    tailDFactor_ = dFactor_;
    usedCoreNums_ = 1;
}

template <typename TilingDataType>
void SwigluGroupQuantTiling::SetBaseTilingData(TilingDataType& tilingData) const
{
    tilingData.set_bs(bs_);
    tilingData.set_d(d_);
    tilingData.set_splitD(splitD_);
    tilingData.set_scaleCol(scaleCol_);
    tilingData.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData.set_rowFactor(rowFactor_);
    tilingData.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData.set_dLoop(dLoop_);
    tilingData.set_dFactor(dFactor_);
    tilingData.set_tailDFactor(tailDFactor_);
    tilingData.set_roundScale(roundScale_);
    tilingData.set_outputOrigin(outputOrigin_);
    tilingData.set_clampLimit(clampLimit_);
    tilingData.set_hasClampLimit(hasClampLimit_);
    tilingData.set_g(g_);
    tilingData.set_ubSize(ubSize_);
    tilingData.set_gLoop(gLoop_);
    tilingData.set_gFactor(gFactor_);
    tilingData.set_tailGFactor(tailGFactor_);
    tilingData.set_coreNum(coreNum_);
    tilingData.set_groupListType(groupListType_);
}

void SwigluGroupQuantTiling::SetTilingData()
{
    if (quantMode_ == MX_QUANT_EXTEND) {
        const auto geometry = SwigluGroupQuantMxTiling::MakeGeometry(splitD_, static_cast<int64_t>(coreNum_));
        uint32_t flags = 0U;
        flags |= hasWeight_ ? MX_HAS_WEIGHT : 0U;
        flags |= hasClampLimit_ != 0 ? MX_HAS_CLAMP : 0U;
        flags |= outputOrigin_ != 0 ? MX_OUTPUT_ORIGIN : 0U;
        flags |= SwigluDualAxisPolicy::ShareOrigin(hasWeight_, outputOrigin_ != 0) ? MX_SHARE_ORIGIN : 0U;
        mxExtendTilingData_.set_flags(flags);
        mxExtendTilingData_.set_weightType(
            weightType_ == ge::DT_FLOAT16 ? MX_WEIGHT_TYPE_FP16 :
                                            (weightType_ == ge::DT_BF16 ? MX_WEIGHT_TYPE_BF16 : MX_WEIGHT_TYPE_FP32));
        mxExtendTilingData_.set_dimM(bs_);
        mxExtendTilingData_.set_dimN(splitD_);
        mxExtendTilingData_.set_usedCoreNum(geometry.usedCoreNum);
        mxExtendTilingData_.set_dimNBlockNum(geometry.dimNBlockNum);
        mxExtendTilingData_.set_dimNTail(geometry.dimNTail);
        mxExtendTilingData_.set_alpha(alpha_);
        mxExtendTilingData_.set_bias(bias_);
        mxExtendTilingData_.set_clampLimit(static_cast<float>(clampLimit_));
    } else {
        SetBaseTilingData(tilingData_);

        OP_LOGD(context_->GetNodeName(),
                "SwigluGroupQuantTiling shape: bs:%ld, d:%ld, splitD:%ld, scaleCol:%ld, g:%ld, hasGroupIndex:%d, "
                "hasWeight:%d.",
                bs_, d_, splitD_, scaleCol_, g_, static_cast<int32_t>(hasGroupIndex_),
                static_cast<int32_t>(hasWeight_));
        OP_LOGD(context_->GetNodeName(),
                "SwigluGroupQuantTiling rowTiling: rowOfFormerBlock:%ld, rowOfTailBlock:%ld, rowLoopOfFormerBlock:%ld, "
                "rowLoopOfTailBlock:%ld, rowFactor:%ld, tailRowFactorOfFormerBlock:%ld, tailRowFactorOfTailBlock:%ld.",
                rowOfFormerBlock_, rowOfTailBlock_, rowLoopOfFormerBlock_, rowLoopOfTailBlock_, rowFactor_,
                tailRowFactorOfFormerBlock_, tailRowFactorOfTailBlock_);
        OP_LOGD(context_->GetNodeName(),
                "SwigluGroupQuantTiling dTiling: dLoop:%ld, dFactor:%ld, tailDFactor:%ld, clampLimit:%f, "
                "hasClampLimit:%ld, roundScale:%ld, outputOrigin:%d.",
                dLoop_, dFactor_, tailDFactor_, clampLimit_, hasClampLimit_, roundScale_,
                static_cast<int32_t>(outputOrigin_));
        OP_LOGD(context_->GetNodeName(),
                "SwigluGroupQuantTiling groupIndexTiling: gLoop:%ld, gFactor:%ld, tailGFactor:%ld, coreNum:%ld, "
                "ubSize:%luB.",
                gLoop_, gFactor_, tailGFactor_, coreNum_, ubSize_);
    }
}

ge::graphStatus SwigluGroupQuantTiling::CalcOpTiling()
{
    ge::graphStatus status = CalcGroupIndexTiling();
    if (status == ge::GRAPH_FAILED) {
        return status;
    }
    if (bs_ == 0 || d_ == 0) {
        SetEmptyTiling();
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }
    if (quantMode_ == MX_QUANT_EXTEND) {
        status = CalcMxQuantExtendOpTiling();
    } else if (quantMode_ == BLOCK_QUANT) {
        status = CalcBlockQuantOpTiling();
    } else if (quantMode_ == MX_QUANT) {
        if (isMxFp4Quant_) {
            status = CalcMxFp4QuantOpTiling();
        } else {
            status = CalcMxQuantOpTiling();
        }
    } else if (quantMode_ == HIF4_QUANT) {
        status = CalcHif4QuantOpTiling();
    }
    return status;
}

ge::graphStatus SwigluGroupQuantTiling::SetTilingKey()
{
    if (quantMode_ == BLOCK_QUANT) {
        tilingKey_ = outputOrigin_ ? BLOCK_QUANT_YORIGIN_TILING_KEY : BLOCK_QUANT_TILING_KEY;
        return context_->SetTilingKey(tilingKey_);
    }
    if (quantMode_ == MX_QUANT) {
        if (isMxFp4Quant_) {
            tilingKey_ = outputOrigin_ ? MXFP4_QUANT_YORIGIN_TILING_KEY : MXFP4_QUANT_TILING_KEY;
        } else if (outputOrigin_) {
            tilingKey_ = MX_QUANT_YORIGIN_TILING_KEY;
        } else {
            tilingKey_ = MX_QUANT_TILING_KEY;
        }
        return context_->SetTilingKey(tilingKey_);
    }
    if (quantMode_ == MX_QUANT_EXTEND) {
        tilingKey_ = outputOrigin_ ? MX_QUANT_EXTEND_YORIGIN_TILING_KEY : MX_QUANT_EXTEND_TILING_KEY;
        return context_->SetTilingKey(tilingKey_);
    }
    if (quantMode_ == HIF4_QUANT) {
        tilingKey_ = outputOrigin_ ? HIF4_QUANT_YORIGIN_TILING_KEY : HIF4_QUANT_TILING_KEY;
        return context_->SetTilingKey(tilingKey_);
    }
    return context_->SetTilingKey(tilingKey_);
}

ge::graphStatus SwigluGroupQuantTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (SetTilingKey() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context_->GetNodeName(),
            "SwigluGroupQuantTiling done: quantMode:%ld, dstType:%s, tilingKey:%lu, blockDim:%lu, "
            "workspaceSize:%luB.",
            quantMode_, ge::TypeUtils::DataTypeToSerialString(dstType_).c_str(), tilingKey_,
            hasGroupIndex_ ? coreNum_ : usedCoreNums_, workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetWorkspaceSize()
{
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::PostTiling()
{
    if (hasGroupIndex_) {
        if (context_->SetBlockDim(coreNum_) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (context_->SetBlockDim(usedCoreNums_) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    if (quantMode_ == MX_QUANT_EXTEND) {
        mxExtendTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                         context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(mxExtendTilingData_.GetDataSize());
    } else {
        tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
