/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_tiling_infer_base.cpp
 * \brief
 */

#include "batch_norm_grad_tiling.h"
#include "batch_norm_grad_tiling_infer_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;
namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "BatchNormGrad", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

constexpr int64_t VL_DOUBLE = 2;

static std::vector<std::array<ge::DataType, INPUT_NUM>> validInferInputDtype = {
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}};

void BatchNormGradInferBase::Reset()
{
    usedCoreNums_ = 0;
    blockSize_ = 0;
    vlFp32_ = 0;
    vlFp16_ = 0;

    bytesPerDy_ = 0;
    bytesPerWeight_ = 0;
    bytesPerRunningVar_ = 0;

    aTileBase_ = 0;
    r1Dim = 0;
    r0Dim = 0;
    aDim = 0;

    dyDimNum_ = 0;
    epsilon_ = DEFAULT_EPSILON;
    enableDx = false;
    enableDgamma = false;
    enableDbeta = false;
}

void BatchNormGradInferBase::CalcBasicInfo()
{
    if (dyDtype_ == ge::DT_FLOAT) {
        bytesPerDy_ = FLOAT32_BYTES;
        aTileBase_ = vlFp32_;
    } else {
        bytesPerDy_ = FLOAT16_BYTES;
        aTileBase_ = vlFp16_;
    }

    if (weightDtype_ == ge::DT_FLOAT) {
        bytesPerWeight_ = FLOAT32_BYTES;
    } else {
        bytesPerWeight_ = FLOAT16_BYTES;
    }

    if (runningVarDtype_ == ge::DT_FLOAT) {
        bytesPerRunningVar_ = FLOAT32_BYTES;
    } else {
        bytesPerRunningVar_ = FLOAT16_BYTES;
    }

    OP_LOGD(context_, "aTileBase_: %ld, bytesPerDy_: %ld, bytesPerWeight_: %ld,bytesPerRunningVar_: %ld.", aTileBase_,
            bytesPerDy_, bytesPerWeight_, bytesPerRunningVar_);
}

ge::graphStatus BatchNormGradInferBase::GetPlatformInfo()
{
    auto compileInfo = static_cast<const BatchNormGradCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfo == nullptr,
                ReportTilingStageError(
                    context_->GetNodeName(),
                    "GetPlatformInfo: compile info is null; check the SoC-specific OPP package and its loading logs"),
                return ge::GRAPH_FAILED);
    blockSize_ = compileInfo->blockSize;
    vlFp32_ = compileInfo->vlFp32;
    vlFp16_ = vlFp32_ * VL_DOUBLE;

    opName_ = context_->GetNodeName();
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.numBlocks = compileInfo->coreNum;
        aicoreParams_.ubSize = compileInfo->ubSize;
    }

    return ge::GRAPH_SUCCESS;
}

const gert::Shape& BatchNormGradInferBase::EnsureNotScalar(const gert::Shape& in_shape)
{
    static const gert::Shape g_vec_1_shape = {1};
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

ge::graphStatus BatchNormGradInferBase::CheckBigShapesFormatValid()
{
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    OP_CHECK_IF(
        dyDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "y_backprop",
                                                 "the input desc of required input y_backprop is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto dyFormat = dyDesc->GetFormat().GetStorageFormat();
    auto dyShape = context_->GetInputShape(INDEX_0);
    OP_CHECK_IF(dyShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "y_backprop",
                    "the input shape of required input y_backprop is missing in the tiling context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    auto dyStorageShape = EnsureNotScalar(dyShape->GetStorageShape());
    int64_t dyDimNum = dyStorageShape.GetDimNum();

    auto xDesc = context_->GetInputDesc(INDEX_1);
    OP_CHECK_IF(
        xDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "x",
                                                 "the input desc of required input x is missing in the tiling context; "
                                                 "check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto xFormat = xDesc->GetFormat().GetStorageFormat();
    auto xShape = context_->GetInputShape(INDEX_1);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "x",
                                                 "the input shape of required input x is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto xStorageShape = EnsureNotScalar(xShape->GetStorageShape());
    int64_t xDimNum = xStorageShape.GetDimNum();

    auto dxDesc = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_IF(dxDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "x_backprop",
                    "the output desc of required output x_backprop is missing in the tiling context; check that the "
                    "graph node provides all required outputs"),
                return ge::GRAPH_FAILED);
    auto dxFormat = dxDesc->GetFormat().GetStorageFormat();
    auto dxShape = context_->GetOutputShape(INDEX_0);
    OP_CHECK_IF(dxShape == nullptr,
                ReportTilingStageError(context_->GetNodeName(),
                                       "the output shape of x_backprop is missing in the tiling context; check that "
                                       "InferShape executed and the framework propagated the output shapes"),
                return ge::GRAPH_FAILED);
    auto dxStorageShape = EnsureNotScalar(dxShape->GetStorageShape());
    int64_t dxDimNum = dxStorageShape.GetDimNum();

    // 校验dim相等
    if (dyDimNum != xDimNum || dyDimNum != dxDimNum) {
        std::string dimsStr = std::to_string(dyDimNum) + ", " + std::to_string(xDimNum) + " and " +
                              std::to_string(dxDimNum);
        std::string reasonMsg = "The shape dims of y_backprop, x and x_backprop must be the same";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "y_backprop, x and x_backprop",
                                                  dimsStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // 校验format
    if (dyFormat != xFormat || dyFormat != dxFormat) {
        std::string formatsStr = ge::TypeUtils::FormatToSerialString(dyFormat) + ", " +
                                 ge::TypeUtils::FormatToSerialString(xFormat) + " and " +
                                 ge::TypeUtils::FormatToSerialString(dxFormat);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "y_backprop, x and x_backprop",
                                                formatsStr.c_str(),
                                                "The formats of y_backprop, x and x_backprop must be the same");
        return ge::GRAPH_FAILED;
    }

    if (dyFormat == FORMAT_NCHW || dyFormat == FORMAT_NHWC) {
        if (dyDimNum != DIM_NUM_4) {
            std::string reason = "the dim size of y_backprop must be 4D with " +
                                 std::string(ge::TypeUtils::FormatToSerialString(dyFormat)) + " format";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "y_backprop",
                                                     std::to_string(dyDimNum).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (dyFormat == FORMAT_NCDHW || dyFormat == FORMAT_NDHWC) {
        if (dyDimNum != DIM_NUM_5) {
            std::string reason = "the dim size of y_backprop must be 5D with " +
                                 std::string(ge::TypeUtils::FormatToSerialString(dyFormat)) + " format";
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "y_backprop",
                                                     std::to_string(dyDimNum).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "y_backprop",
                                   ge::TypeUtils::FormatToSerialString(dyFormat).c_str(),
                                   "NCHW, NHWC, NCDHW and NDHWC");
        return ge::GRAPH_FAILED;
    }

    // 校验shape的dims相同
    for (int64_t i = 0; i < dyDimNum; i++) {
        if (xStorageShape.GetDim(i) != dyStorageShape.GetDim(i) ||
            dxStorageShape.GetDim(i) != dyStorageShape.GetDim(i)) {
            std::string reasonMsg = "dim[" + std::to_string(i) + "] of y_backprop, x and x_backprop must be the same";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y_backprop",
                                                  Ops::Base::ToString(dyStorageShape).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        if (dyStorageShape.GetDim(i) <= 0) {
            std::string reasonMsg = "dim" + std::to_string(i) + " of y_backprop should be greater than 0";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "y_backprop",
                                                  Ops::Base::ToString(dyStorageShape).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::CheckSmallShapesValid()
{
    auto weightDesc = context_->GetInputDesc(INDEX_2);
    OP_CHECK_IF(
        weightDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "scale",
                                                 "the input desc of required input scale is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    weightDtype_ = weightDesc->GetDataType();
    auto dyDesc = context_->GetInputDesc(INDEX_0);
    OP_CHECK_IF(
        dyDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "y_backprop",
                                                 "the input desc of required input y_backprop is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    dyDtype_ = dyDesc->GetDataType();

    for (int i = INDEX_2; i <= INDEX_4; i++) {
        auto shape = context_->GetInputShape(i);
        OP_CHECK_IF(
            shape == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), inputParamNames[i],
                                                     "the input shape of a required input is missing in the tiling "
                                                     "context; check that the graph node provides all required inputs"),
            return ge::GRAPH_FAILED);
        auto storageShape = shape->GetStorageShape();
        if (storageShape.GetDimNum() != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), inputParamNames[i],
                                         std::to_string(storageShape.GetDimNum()).c_str(), "1D");
            return ge::GRAPH_FAILED;
        }

        int64_t a = storageShape.GetDim(INDEX_0);
        if (a != aDim) {
            std::string reasonMsg = "the first dim of " + std::string(inputParamNames[i]) + " should be " +
                                    std::to_string(aDim);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputParamNames[i],
                                                  Ops::Base::ToString(storageShape).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    for (int i = INDEX_1; i <= INDEX_2; i++) {
        auto shape = context_->GetOutputShape(i);
        OP_CHECK_IF(
            shape == nullptr,
            ReportTilingStageError(context_->GetNodeName(),
                                   "the output shape of a required output is missing in the tiling context; check that "
                                   "InferShape executed and the framework propagated the output shapes"),
            return ge::GRAPH_FAILED);
        auto storageShape = shape->GetStorageShape();

        if (storageShape.GetDimNum() != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), outputParamNames[i],
                                         std::to_string(storageShape.GetDimNum()).c_str(), "1D");
            return ge::GRAPH_FAILED;
        }
        int64_t a = storageShape.GetDim(INDEX_0);
        if (a != aDim) {
            std::string reasonMsg = "The first dim of " + std::string(outputParamNames[i]) + " must be equal to" +
                                    std::to_string(aDim);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), outputParamNames[i],
                                                  Ops::Base::ToString(storageShape).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::CheckDtypeValid()
{
    bool invalid = true;
    for (auto& dtypeList : validInferInputDtype) {
        invalid = false;
        for (int i = 0; i < INPUT_NUM; i++) {
            auto inputDesc = (i == INPUT_NUM - 1) ? context_->GetOptionalInputDesc(i) : context_->GetInputDesc(i);
            if (i == INPUT_NUM - 1 && inputDesc == nullptr) {
                continue;
            }
            OP_CHECK_IF(inputDesc == nullptr,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            context_->GetNodeName(), inputParamNames[i],
                            "the input desc of a required input is missing in the tiling context; check that the graph "
                            "node provides all required inputs"),
                        return ge::GRAPH_FAILED);
            auto dtype = inputDesc->GetDataType();
            if (dtype != dtypeList[i]) {
                invalid = true;
                break;
            }
        }
        if (!invalid) {
            break;
        }
    }
    std::string incorrectDtypeStr;
    std::string expectedDtypesStr;
    if (invalid) {
        for (int i = 0; i < INPUT_NUM; i++) {
            auto inputDesc = (i == INPUT_NUM - 1) ? context_->GetOptionalInputDesc(i) : context_->GetInputDesc(i);
            if (i == INPUT_NUM - 1 && inputDesc == nullptr) {
                continue;
            }
            if (!incorrectDtypeStr.empty())
                incorrectDtypeStr += ", ";
            incorrectDtypeStr += ge::TypeUtils::DataTypeToSerialString(inputDesc->GetDataType());
        }
        for (const auto& dtypeList : validInferInputDtype) {
            if (!expectedDtypesStr.empty())
                expectedDtypesStr += " or ";
            expectedDtypesStr += "[";
            for (int i = 0; i < INPUT_NUM; i++) {
                if (i > 0)
                    expectedDtypesStr += ", ";
                expectedDtypesStr += ge::TypeUtils::DataTypeToSerialString(dtypeList[i]);
            }
            expectedDtypesStr += "]";
        }
    }
    OP_CHECK_IF(invalid == true,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(),
                                          "y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3",
                                          incorrectDtypeStr.c_str(), expectedDtypesStr.c_str()),
                return ge::GRAPH_FAILED);

    auto weightDesc = context_->GetInputDesc(PARAM_INPUT_WEIGHT_INDEX);
    OP_CHECK_IF(
        weightDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "scale",
                                                 "the input desc of required input scale is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    weightDtype_ = weightDesc->GetDataType();

    auto runningVarDesc = context_->GetInputDesc(PARAM_INPUT_RUNNINGVAR_INDEX);
    OP_CHECK_IF(runningVarDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "reserve_space_2",
                    "the input desc of required input reserve_space_2 is missing in the tiling context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    runningVarDtype_ = runningVarDesc->GetDataType();

    auto dxDesc = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_IF(dxDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "x_backprop",
                    "the output desc of required output x_backprop is missing in the tiling context; check that the "
                    "graph node provides all required outputs"),
                return ge::GRAPH_FAILED);
    ge::DataType dxDtype = dxDesc->GetDataType();
    auto dweightDesc = context_->GetOutputDesc(INDEX_1);
    OP_CHECK_IF(dweightDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "scale_backprop",
                    "the output desc of required output scale_backprop is missing in the tiling context; check that "
                    "the graph node provides all required outputs"),
                return ge::GRAPH_FAILED);
    ge::DataType dweightDtype = dweightDesc->GetDataType();
    auto dbiasDesc = context_->GetOutputDesc(INDEX_2);
    OP_CHECK_IF(dbiasDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "offset_backprop",
                    "the output desc of required output offset_backprop is missing in the tiling context; check that "
                    "the graph node provides all required outputs"),
                return ge::GRAPH_FAILED);
    ge::DataType dbiasDtype = dbiasDesc->GetDataType();
    if (dxDtype != dyDtype_) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(dxDtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(dyDtype_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x_backprop and y_backprop", dtypesStr.c_str(),
                                               "The dtypes of x_backprop and y_backprop must be the same");
        return ge::GRAPH_FAILED;
    }
    if (dweightDtype != weightDtype_) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(dweightDtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(weightDtype_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "scale_backprop and scale", dtypesStr.c_str(),
                                               "The dtypes of scale_backprop and scale must be the same");
        return ge::GRAPH_FAILED;
    }
    if (dbiasDtype != weightDtype_) {
        std::string dtypesStr = ge::TypeUtils::DataTypeToSerialString(dbiasDtype) + " and " +
                                ge::TypeUtils::DataTypeToSerialString(weightDtype_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "offset_backprop and scale", dtypesStr.c_str(),
                                               "The dtypes of offset_backprop and scale must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        ReportTilingStageError("BatchNormGrad",
                               "GetShapeAttrsInfo: TilingContext is null; the tiling framework contract is broken");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportTilingStageError(context_->GetNodeName(),
                               "GetShapeAttrsInfo: GetAttrs() returned null; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    const bool* isTrainingPtr = attrs->GetBool(PARAM_ATTRS_ISTRAING_INDEX);
    bool isTraining = (isTrainingPtr == nullptr) ? true : *isTrainingPtr;
    OP_CHECK_IF(isTraining == true, OP_LOGD(context_, "This node only support inference."),
                return ge::GRAPH_PARAM_INVALID);

    const float* epsilonPtr = attrs->GetFloat(PARAM_ATTRS_EPSILON_INDEX);
    epsilon_ = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;

    auto outputMask = attrs->GetAttrPointer<gert::ContinuousVector>(PARAM_ATTRS_OUTPUT_MASK);
    if (outputMask == nullptr) {
        enableDx = true;
        enableDgamma = false;
        enableDbeta = false;
    } else {
        OP_CHECK_IF(outputMask->GetSize() != 3,
                    OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "output_mask",
                                                 std::to_string(outputMask->GetSize()).c_str(), "3"),
                    return ge::GRAPH_PARAM_INVALID);
        auto outputMaskData = static_cast<const bool*>(outputMask->GetData());
        enableDx = static_cast<bool>(outputMaskData[0]);
        enableDgamma = static_cast<bool>(outputMaskData[1]);
        enableDbeta = static_cast<bool>(outputMaskData[2]);
    }

    // 大shape shape和format检查
    OP_CHECK_IF(CheckBigShapesFormatValid() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_, "Check shape or format error."), return ge::GRAPH_FAILED);

    auto ret = GetDyInfo();
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(context_, "GetDyInfo failed."), return ret);

    // 小shape 校验
    OP_CHECK_IF(CheckSmallShapesValid() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(context_, "Input small shapes are invalid."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(CheckDtypeValid() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(context_, "Input dtypes are invalid."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::GetDyInfo()
{
    auto dyDesc = context_->GetInputDesc(PARAM_INPUT_DY);
    OP_CHECK_IF(
        dyDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "y_backprop",
                                                 "the input desc of required input y_backprop is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    dyDtype_ = dyDesc->GetDataType();
    auto dyShape = context_->GetInputShape(PARAM_INPUT_DY);
    OP_CHECK_IF(dyShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "y_backprop",
                    "the input shape of required input y_backprop is missing in the tiling context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    auto dyStorageShape = dyShape->GetStorageShape();
    dyDimNum_ = dyStorageShape.GetDimNum();

    dyFormat_ = dyDesc->GetFormat().GetStorageFormat();
    if (dyFormat_ == FORMAT_NHWC) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_4,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y_backprop",
                                                 std::to_string(dyDimNum_).c_str(), "4D with NHWC format"),
                    return ge::GRAPH_FAILED);
        aDim = dyStorageShape.GetDim(DIM_3);
        r1Dim = dyStorageShape.GetDim(DIM_0) * dyStorageShape.GetDim(DIM_1) * dyStorageShape.GetDim(DIM_2);
        r0Dim = 1;
    } else if (dyFormat_ == FORMAT_NDHWC) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_5,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y_backprop",
                                                 std::to_string(dyDimNum_).c_str(), "5D with NDHWC format"),
                    return ge::GRAPH_FAILED);
        aDim = dyStorageShape.GetDim(DIM_4);
        r1Dim = dyStorageShape.GetDim(DIM_0) * dyStorageShape.GetDim(DIM_1) * dyStorageShape.GetDim(DIM_2) *
                dyStorageShape.GetDim(DIM_3);
        r0Dim = 1;
    } else if (dyFormat_ == FORMAT_NCHW) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_4,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y_backprop",
                                                 std::to_string(dyDimNum_).c_str(), "4D with NCHW format"),
                    return ge::GRAPH_FAILED);
        r1Dim = dyStorageShape.GetDim(DIM_0);
        aDim = dyStorageShape.GetDim(DIM_1);
        r0Dim = dyStorageShape.GetDim(DIM_2) * dyStorageShape.GetDim(DIM_3);
    } else if (dyFormat_ == FORMAT_NCDHW) {
        OP_CHECK_IF(dyDimNum_ != DIM_NUM_5,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y_backprop",
                                                 std::to_string(dyDimNum_).c_str(), "5D with NCDHW format"),
                    return ge::GRAPH_FAILED);
        r1Dim = dyStorageShape.GetDim(DIM_0);
        aDim = dyStorageShape.GetDim(DIM_1);
        r0Dim = dyStorageShape.GetDim(DIM_2) * dyStorageShape.GetDim(DIM_3) * dyStorageShape.GetDim(DIM_4);
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "y_backprop",
                                   ge::TypeUtils::FormatToSerialString(dyFormat_).c_str(),
                                   "NCHW, NCDHW, NHWC and NDHWC");
        return ge::GRAPH_PARAM_INVALID;
    }
    // 特殊场景处理, R1AR0, A==1时转（1,1,R1*R0)
    if (aDim == 1) {
        r0Dim = r0Dim * r1Dim;
        r1Dim = 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradInferBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus BatchNormGradInferBase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = MINIMAL_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
