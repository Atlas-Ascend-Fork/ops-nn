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
 * \file quantized_batch_norm_tiling_base.cc
 * \brief
 */

#include <iostream>
#include <string>
#include "error_util.h"
#include "quantized_batch_norm_tiling.h"

namespace {
static constexpr int64_t NCHW_DIM_NUM = 4;
static constexpr int64_t X_INPUT_IDX = 0;
static constexpr int64_t MEAN_INPUT_IDX = 1;
static constexpr int64_t VAR_INPUT_IDX = 2;
static constexpr int64_t INSCALE_INPUT_IDX = 3;
static constexpr int64_t INZP_INPUT_IDX = 4;
static constexpr int64_t OUTSCALE_INPUT_IDX = 5;
static constexpr int64_t OUTZP_INPUT_IDX = 6;
static constexpr int64_t WEIGHT_INPUT_IDX = 7;
static constexpr int64_t BIAS_INPUT_IDX = 8;
static constexpr int64_t EPS_ATTR_IDX = 0;
static constexpr int64_t DIM_0 = 0;
static constexpr int64_t DIM_1 = 1;
static constexpr int64_t DIM_2 = 2;
static constexpr int64_t DIM_3 = 3;
static constexpr int64_t DIM_4 = 4;
static constexpr uint32_t MINIMAL_WORKSPACE = 16777216; // 16 * 1024 * 1024
static constexpr int64_t I8_BLOCK_ALIGN_NUM = 32;
static constexpr int64_t I32_BLOCK_ALIGN_NUM = 8;
}; // namespace

namespace optiling {
static inline bool IsDtypeSupported(const ge::DataType dtype)
{
    return ((dtype == ge::DT_INT8) || (dtype == ge::DT_UINT8) || (dtype == ge::DT_INT32));
}

static inline bool IsOthersDtypeSupported(const ge::DataType dtype) { return (dtype == ge::DT_FLOAT); }

// Tiling阶段框架契约类失败(框架对象为空/平台信息异常)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const std::string& opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName.c_str(), reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

bool QuantizedBatchNormTilingBase::CheckInputDtype()
{
    OP_CHECK_IF(context_->GetInputDesc(X_INPUT_IDX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "x",
                    "the input desc of required input x (input index 0) is missing in the tiling context; "
                    "check that the graph node provides all required inputs"),
                return false);
    auto xDtype = context_->GetInputDesc(X_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(context_->GetInputDesc(WEIGHT_INPUT_IDX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "weight",
                    "the input desc of weight (input index 7) is missing in the tiling context; weight is optional "
                    "in the op definition but required by this implementation, provide it in the graph (the aclnn "
                    "API fills it with 1.0 automatically)"),
                return false);
    auto weightDtype = context_->GetInputDesc(WEIGHT_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(context_->GetInputDesc(BIAS_INPUT_IDX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "bias",
                    "the input desc of bias (input index 8) is missing in the tiling context; bias is optional in "
                    "the op definition but required by this implementation, provide it in the graph (the aclnn API "
                    "fills it with 0 automatically)"),
                return false);
    auto biasDtype = context_->GetInputDesc(BIAS_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(context_->GetInputDesc(MEAN_INPUT_IDX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "mean",
                    "the input desc of required input mean (input index 1) is missing in the tiling context; check "
                    "that the graph node provides all required inputs"),
                return false);
    auto meanDtype = context_->GetInputDesc(MEAN_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(context_->GetInputDesc(VAR_INPUT_IDX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "var",
                    "the input desc of required input var (input index 2) is missing in the tiling context; check "
                    "that the graph node provides all required inputs"),
                return false);
    auto varDtype = context_->GetInputDesc(VAR_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(context_->GetInputDesc(INSCALE_INPUT_IDX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "input_scale",
                    "the input desc of required input input_scale (input index 3) is missing in the tiling context; "
                    "check that the graph node provides all required inputs"),
                return false);
    auto inScaleDtype = context_->GetInputDesc(INSCALE_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(context_->GetInputDesc(INZP_INPUT_IDX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "input_zero_point",
                    "the input desc of required input input_zero_point (input index 4) is missing in the tiling "
                    "context; check that the graph node provides all required inputs"),
                return false);
    auto inZeroPointDtype = context_->GetInputDesc(INZP_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(context_->GetInputDesc(OUTSCALE_INPUT_IDX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "output_scale",
                    "the input desc of required input output_scale (input index 5) is missing in the tiling "
                    "context; check that the graph node provides all required inputs"),
                return false);
    auto outScaleDtype = context_->GetInputDesc(OUTSCALE_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(context_->GetInputDesc(OUTZP_INPUT_IDX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "output_zero_point",
                    "the input desc of required input output_zero_point (input index 6) is missing in the tiling "
                    "context; check that the graph node provides all required inputs"),
                return false);
    auto outZeroPointDtype = context_->GetInputDesc(OUTZP_INPUT_IDX)->GetDataType();

    OP_CHECK_IF(!IsDtypeSupported(xDtype),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(commonParams.nodeName, "x", Ops::Base::ToString(xDtype).c_str(),
                                                      "The dtype of x must be DT_INT8, DT_UINT8 or DT_INT32."),
                return false);

    OP_CHECK_IF(
        (!IsOthersDtypeSupported(weightDtype)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(commonParams.nodeName, "weight", Ops::Base::ToString(weightDtype).c_str(),
                                              "The dtype of weight must be DT_FLOAT."),
        return false);

    OP_CHECK_IF((weightDtype != biasDtype),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    commonParams.nodeName, "weight, bias",
                    (Ops::Base::ToString(weightDtype) + ", " + Ops::Base::ToString(biasDtype)).c_str(),
                    "The dtypes of weight and bias must be the same."),
                return false);

    OP_CHECK_IF(
        (meanDtype != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(commonParams.nodeName, "mean", Ops::Base::ToString(meanDtype).c_str(),
                                              "The dtype of mean must be DT_FLOAT."),
        return false);

    OP_CHECK_IF(
        (varDtype != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(commonParams.nodeName, "var", Ops::Base::ToString(varDtype).c_str(),
                                              "The dtype of var must be DT_FLOAT."),
        return false);

    OP_CHECK_IF((inScaleDtype != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(commonParams.nodeName, "input_scale",
                                                      Ops::Base::ToString(inScaleDtype).c_str(),
                                                      "The dtype of input_scale must be DT_FLOAT."),
                return false);

    OP_CHECK_IF((inZeroPointDtype != ge::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(commonParams.nodeName, "input_zero_point",
                                                      Ops::Base::ToString(inZeroPointDtype).c_str(),
                                                      "The dtype of input_zero_point must be DT_INT32."),
                return false);

    OP_CHECK_IF((outScaleDtype != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(commonParams.nodeName, "output_scale",
                                                      Ops::Base::ToString(outScaleDtype).c_str(),
                                                      "The dtype of output_scale must be DT_FLOAT."),
                return false);

    OP_CHECK_IF((outZeroPointDtype != ge::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(commonParams.nodeName, "output_zero_point",
                                                      Ops::Base::ToString(outZeroPointDtype).c_str(),
                                                      "The dtype of output_zero_point must be DT_INT32."),
                return false);
    commonParams.xDtype = xDtype;
    return true;
}

bool QuantizedBatchNormTilingBase::CheckInputShape()
{
    auto xShape = context_->GetInputShape(X_INPUT_IDX);
    OP_CHECK_IF(xShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "x",
                    "the input shape of required input x (input index 0) is missing in the tiling context; check "
                    "that the graph node provides all required inputs"),
                return false);
    auto xStorageShape = xShape->GetStorageShape();

    auto weightShape = context_->GetInputShape(WEIGHT_INPUT_IDX);
    OP_CHECK_IF(weightShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "weight",
                    "the input shape of weight (input index 7) is missing in the tiling context; weight is "
                    "optional in the op definition but required by this implementation, provide it in the graph"),
                return false);
    auto weightStorageShape = weightShape->GetStorageShape();

    auto biasShape = context_->GetInputShape(BIAS_INPUT_IDX);
    OP_CHECK_IF(biasShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "bias",
                    "the input shape of bias (input index 8) is missing in the tiling context; bias is optional "
                    "in the op definition but required by this implementation, provide it in the graph"),
                return false);
    auto biasStorageShape = biasShape->GetStorageShape();

    auto meanShape = context_->GetInputShape(MEAN_INPUT_IDX);
    OP_CHECK_IF(meanShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "mean",
                    "the input shape of required input mean (input index 1) is missing in the tiling context; "
                    "check that the graph node provides all required inputs"),
                return false);
    auto meanStorageShape = meanShape->GetStorageShape();

    auto xDesc = context_->GetInputDesc(X_INPUT_IDX);
    auto format = xDesc->GetFormat().GetStorageFormat();
    if (format == ge::FORMAT_NCHW) {
        OP_CHECK_IF(xStorageShape.GetDimNum() != NCHW_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(commonParams.nodeName, "x",
                                                 std::to_string(xStorageShape.GetDimNum()).c_str(), "4"),
                    return false);
        commonParams.patternR1 = xStorageShape.GetDim(DIM_0);
        commonParams.patternA = xStorageShape.GetDim(DIM_1);
        commonParams.patternR0 = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(commonParams.nodeName, "x", Ops::Base::ToString(format).c_str(), "NCHW");
        return false;
    }

    OP_CHECK_IF(
        commonParams.patternR1 <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(commonParams.nodeName, "x", Ops::Base::ToString(xStorageShape).c_str(),
                                              "dim 0 (N) of the x shape must be greater than 0"),
        return false);
    OP_CHECK_IF(
        commonParams.patternA <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(commonParams.nodeName, "x", Ops::Base::ToString(xStorageShape).c_str(),
                                              "dim 1 (C) of the x shape must be greater than 0"),
        return false);
    OP_CHECK_IF(
        commonParams.patternR0 <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(commonParams.nodeName, "x", Ops::Base::ToString(xStorageShape).c_str(),
                                              "dim 2 * dim 3 (H * W) of the x shape must be greater than 0"),
        return false);
    OP_CHECK_IF(
        weightStorageShape.GetShapeSize() != commonParams.patternA,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            commonParams.nodeName, "weight, x",
            (std::to_string(weightStorageShape.GetShapeSize()) + ", " + std::to_string(commonParams.patternA)).c_str(),
            "The shape size of weight must equal dim 1 (C) of the x shape"),
        return false);
    OP_CHECK_IF(
        biasStorageShape.GetShapeSize() != commonParams.patternA,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            commonParams.nodeName, "bias, x",
            (std::to_string(biasStorageShape.GetShapeSize()) + ", " + std::to_string(commonParams.patternA)).c_str(),
            "The shape size of bias must equal dim 1 (C) of the x shape"),
        return false);
    OP_CHECK_IF(
        meanStorageShape.GetShapeSize() != commonParams.patternA,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            commonParams.nodeName, "mean, x",
            (std::to_string(meanStorageShape.GetShapeSize()) + ", " + std::to_string(commonParams.patternA)).c_str(),
            "The shape size of mean must equal dim 1 (C) of the x shape"),
        return false);
    return true;
}

bool QuantizedBatchNormTilingBase::CheckOthersInputShape()
{
    auto varShape = context_->GetInputShape(VAR_INPUT_IDX);
    OP_CHECK_IF(varShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "var",
                    "the input shape of required input var (input index 2) is missing in the tiling context; "
                    "check that the graph node provides all required inputs"),
                return false);
    auto varStorageShape = varShape->GetStorageShape();

    auto inScaleShape = context_->GetInputShape(INSCALE_INPUT_IDX);
    OP_CHECK_IF(inScaleShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "input_scale",
                    "the input shape of required input input_scale (input index 3) is missing in the tiling "
                    "context; check that the graph node provides all required inputs"),
                return false);
    auto inScaleStorageShape = inScaleShape->GetStorageShape();

    auto inZeroPointShape = context_->GetInputShape(INZP_INPUT_IDX);
    OP_CHECK_IF(inZeroPointShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "input_zero_point",
                    "the input shape of required input input_zero_point (input index 4) is missing in the tiling "
                    "context; check that the graph node provides all required inputs"),
                return false);
    auto inZeroPointStorageShape = inZeroPointShape->GetStorageShape();

    auto outScaleShape = context_->GetInputShape(OUTSCALE_INPUT_IDX);
    OP_CHECK_IF(outScaleShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "output_scale",
                    "the input shape of required input output_scale (input index 5) is missing in the tiling "
                    "context; check that the graph node provides all required inputs"),
                return false);
    auto outScaleStorageShape = outScaleShape->GetStorageShape();

    auto outZeroPointShape = context_->GetInputShape(OUTZP_INPUT_IDX);
    OP_CHECK_IF(outZeroPointShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    commonParams.nodeName, "output_zero_point",
                    "the input shape of required input output_zero_point (input index 6) is missing in the tiling "
                    "context; check that the graph node provides all required inputs"),
                return false);
    auto outZeroPointStorageShape = outZeroPointShape->GetStorageShape();

    OP_CHECK_IF(
        varStorageShape.GetShapeSize() != commonParams.patternA,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            commonParams.nodeName, "var, x",
            (std::to_string(varStorageShape.GetShapeSize()) + ", " + std::to_string(commonParams.patternA)).c_str(),
            "The shape size of var must equal dim 1 (C) of the x shape"),
        return false);

    OP_CHECK_IF(inScaleStorageShape.GetShapeSize() != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE(commonParams.nodeName, "input_scale",
                                              std::to_string(inScaleStorageShape.GetShapeSize()).c_str(), "1"),
                return false);
    OP_CHECK_IF(inZeroPointStorageShape.GetShapeSize() != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE(commonParams.nodeName, "input_zero_point",
                                              std::to_string(inZeroPointStorageShape.GetShapeSize()).c_str(), "1"),
                return false);
    OP_CHECK_IF(outScaleStorageShape.GetShapeSize() != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE(commonParams.nodeName, "output_scale",
                                              std::to_string(outScaleStorageShape.GetShapeSize()).c_str(), "1"),
                return false);
    OP_CHECK_IF(outZeroPointStorageShape.GetShapeSize() != 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE(commonParams.nodeName, "output_zero_point",
                                              std::to_string(outZeroPointStorageShape.GetShapeSize()).c_str(), "1"),
                return false);
    return true;
}

ge::graphStatus QuantizedBatchNormTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        commonParams.coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        commonParams.ubSizePlatForm = ubSizePlatForm;
    } else {
        auto compileInfoPtr = context_->GetCompileInfo<QuantizedBatchNormCompileInfo>();
        if (compileInfoPtr == nullptr) {
            ReportTilingStageError(commonParams.nodeName,
                                   "GetPlatformInfo: compile info is null while platform info is unavailable; "
                                   "check the SoC-specific OPP package and its loading logs");
            return ge::GRAPH_FAILED;
        }
        commonParams.coreNum = compileInfoPtr->coreNum;
        commonParams.ubSizePlatForm = compileInfoPtr->ubSize;
    }
    if (commonParams.coreNum == 0) {
        ReportTilingStageError(commonParams.nodeName,
                               "GetPlatformInfo: core num is 0 after the platform query; check the platform "
                               "info, SoC and OPP installation");
        return ge::GRAPH_FAILED;
    }
    if (commonParams.ubSizePlatForm == 0) {
        ReportTilingStageError(commonParams.nodeName,
                               "GetPlatformInfo: UB size is 0 after the platform query; check the platform "
                               "info, SoC and OPP installation");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantizedBatchNormTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        ReportTilingStageError("QuantizedBatchNorm",
                               "GetShapeAttrsInfo: TilingContext is null; the tiling framework contract is broken");
        return ge::GRAPH_FAILED;
    }
    commonParams.nodeName = context_->GetNodeName();
    // 获取attr
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        ReportTilingStageError(commonParams.nodeName,
                               "GetShapeAttrsInfo: GetAttrs() returned null; check the TilingContext "
                               "construction and the attrs of the node");
        return ge::GRAPH_FAILED;
    }
    const float* epsilon = attrs->GetFloat(EPS_ATTR_IDX);
    if (epsilon == nullptr) {
        ReportTilingStageError(commonParams.nodeName,
                               "GetShapeAttrsInfo: required attr epsilon (attr index 0) is missing in the "
                               "tiling context; set the epsilon attribute on the node");
        return ge::GRAPH_FAILED;
    }
    commonParams.epsilon = *epsilon;
    // check输入dtype(发现点已精确首报, 本层只记录日志并透传状态, 不再追加EZ9999)
    OP_CHECK_IF(!QuantizedBatchNormTilingBase::CheckInputDtype(),
                OP_LOGE_WITHOUT_REPORT(commonParams.nodeName, "CheckInputDtype failed."), return ge::GRAPH_FAILED);
    // check输入shape
    OP_CHECK_IF(!QuantizedBatchNormTilingBase::CheckInputShape(),
                OP_LOGE_WITHOUT_REPORT(commonParams.nodeName, "CheckInputShape failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!QuantizedBatchNormTilingBase::CheckOthersInputShape(),
                OP_LOGE_WITHOUT_REPORT(commonParams.nodeName, "CheckOthersInputShape failed."),
                return ge::GRAPH_FAILED);
    commonParams.patternR0Align = (commonParams.xDtype == ge::DT_INT32) ?
                                      Ops::Base::CeilAlign(commonParams.patternR0, I32_BLOCK_ALIGN_NUM) :
                                      Ops::Base::CeilAlign(commonParams.patternR0, I8_BLOCK_ALIGN_NUM);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantizedBatchNormTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        ReportTilingStageError(commonParams.nodeName,
                               "GetWorkspaceSize: GetWorkspaceSizes(1) returned null; required output-slot "
                               "count is 1. Check the TilingContext construction, not the user device allocation");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = MINIMAL_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
