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
 * \file max_pool_v3_grad_tiling_base.cpp
 * \brief Common tiling base for MaxPoolV3Grad templates.
 */

#include <algorithm>
#include <limits>
#include <string>

#include "error_util.h"
#include "op_host/tiling_util.h"
#include "util/math_util.h"
#include "utils/extern_math_util.h"
#include "max_pool_v3_grad_tiling_base.h"

namespace optiling {
namespace {

bool SubOverflow(int64_t left, int64_t right, int64_t& result)
{
    if ((right > 0 && left < std::numeric_limits<int64_t>::min() + right) ||
        (right < 0 && left > std::numeric_limits<int64_t>::max() + right)) {
        return true;
    }

    result = left - right;
    return false;
}

int64_t FloorDiv(int64_t dividend, int64_t divisor)
{
    int64_t quotient = dividend / divisor;
    const int64_t remainder = dividend % divisor;

    if (remainder < 0) {
        --quotient;
    }

    return quotient;
}

template <typename ShapeType>
ge::graphStatus ValidateConcreteShape(gert::TilingContext* context, const ShapeType& shape, const char* inputName)
{
    if (shape.GetDimNum() != 4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), inputName, std::to_string(shape.GetDimNum()).c_str(), "4");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), inputName,
                                                   std::to_string(shape.GetDim(i)).c_str(),
                                                   "dimensions must be nonnegative");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

template <typename ShapeTypeA, typename ShapeTypeB>
bool IsSameShape(const ShapeTypeA& left, const ShapeTypeB& right)
{
    if (left.GetDimNum() != right.GetDimNum()) {
        return false;
    }

    for (size_t i = 0; i < left.GetDimNum(); ++i) {
        if (left.GetDim(i) != right.GetDim(i)) {
            return false;
        }
    }

    return true;
}

ge::graphStatus ValidateAttrs(gert::TilingContext* context, const std::string& paddingMode,
                              const std::string& dataFormat, bool globalPooling, const int64_t* ksize,
                              const int64_t* strides, const int64_t* pads)
{
    if (dataFormat != "NCHW" && dataFormat != "NHWC") {
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "data_format", dataFormat.c_str(), "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }

    if (paddingMode != "SAME" && paddingMode != "VALID" && paddingMode != "CALCULATED") {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "padding_mode", paddingMode.c_str(),
                                               "SAME, VALID or CALCULATED");
        return ge::GRAPH_FAILED;
    }

    const int32_t channelIndex = dataFormat == "NCHW" ? 1 : 3;
    const int32_t heightIndex = dataFormat == "NCHW" ? 2 : 1;
    const int32_t widthIndex = dataFormat == "NCHW" ? 3 : 2;

    if (strides[0] != 1 || strides[channelIndex] != 1) {
        const std::string actual = "N=" + std::to_string(strides[0]) + ", C=" + std::to_string(strides[channelIndex]);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "strides", actual.c_str(),
                                               "N/C dimensions must be 1");
        return ge::GRAPH_FAILED;
    }

    if (strides[heightIndex] < 1 || strides[heightIndex] > 63 || strides[widthIndex] < 1 || strides[widthIndex] > 63) {
        const std::string actual = "H=" + std::to_string(strides[heightIndex]) +
                                   ", W=" + std::to_string(strides[widthIndex]);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "strides", actual.c_str(),
                                               "H/W dimensions must be in [1, 63]");
        return ge::GRAPH_FAILED;
    }

    if (!globalPooling) {
        if (ksize[0] != 1 || ksize[channelIndex] != 1) {
            const std::string actual = "N=" + std::to_string(ksize[0]) + ", C=" + std::to_string(ksize[channelIndex]);
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "ksize", actual.c_str(),
                                                   "N/C dimensions must be 1");
            return ge::GRAPH_FAILED;
        }

        if (ksize[heightIndex] < 1 || ksize[heightIndex] > 255 || ksize[widthIndex] < 1 || ksize[widthIndex] > 255) {
            const std::string actual = "H=" + std::to_string(ksize[heightIndex]) +
                                       ", W=" + std::to_string(ksize[widthIndex]);
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "ksize", actual.c_str(),
                                                   "H/W dimensions must be in [1, 255]");
            return ge::GRAPH_FAILED;
        }

        if (paddingMode == "CALCULATED" && (pads[0] < 0 || pads[1] < 0 || pads[2] < 0 || pads[3] < 0)) {
            const std::string actual = "[" + std::to_string(pads[0]) + ", " + std::to_string(pads[1]) + ", " +
                                       std::to_string(pads[2]) + ", " + std::to_string(pads[3]) + "]";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "pads", actual.c_str(),
                                                   "pads must be nonnegative under CALCULATED mode");
            return ge::GRAPH_FAILED;
        }

        // 该约束与正向 MaxPoolV3 保持一致。
        if (paddingMode == "CALCULATED" && (pads[0] >= ksize[heightIndex] || pads[1] >= ksize[heightIndex] ||
                                            pads[2] >= ksize[widthIndex] || pads[3] >= ksize[widthIndex])) {
            const std::string actual = "pads=[" + std::to_string(pads[0]) + ", " + std::to_string(pads[1]) + ", " +
                                       std::to_string(pads[2]) + ", " + std::to_string(pads[3]) +
                                       "], ksize H=" + std::to_string(ksize[heightIndex]) +
                                       ", W=" + std::to_string(ksize[widthIndex]);
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "pads, ksize", actual.c_str(),
                "pads must be less than the corresponding ksize under CALCULATED mode");
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ComputeOneDim(gert::TilingContext* context, int64_t inputSize, int64_t kernelSize, int64_t stride,
                              const std::string& paddingMode, int64_t explicitPadBefore, int64_t explicitPadAfter,
                              bool ceilMode, int64_t& effectivePadBefore, int64_t& effectivePadAfter,
                              int64_t& outputSize)
{
    const std::string dimActual = "inputSize=" + std::to_string(inputSize) +
                                  ", kernelSize=" + std::to_string(kernelSize) + ", stride=" + std::to_string(stride) +
                                  ", padBefore=" + std::to_string(explicitPadBefore) +
                                  ", padAfter=" + std::to_string(explicitPadAfter);
    if (paddingMode == "SAME") {
        outputSize = Ops::Base::CeilDiv(inputSize, stride);

        int64_t outputMinusOne = outputSize > 0 ? outputSize - 1 : 0;
        int64_t coveredSize = 0;
        int64_t requiredSize = 0;
        int64_t totalPadding = 0;
        if (ge::MulOverflow(outputMinusOne, stride, coveredSize) ||
            ge::AddOverflow(coveredSize, kernelSize, requiredSize) ||
            SubOverflow(requiredSize, inputSize, totalPadding)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "padding parameters", dimActual.c_str(),
                                                   "overflow occurred while computing SAME padding");
            return ge::GRAPH_FAILED;
        }

        totalPadding = std::max(totalPadding, static_cast<int64_t>(0));
        effectivePadBefore = totalPadding / 2;
        effectivePadAfter = totalPadding - effectivePadBefore;
        return ge::GRAPH_SUCCESS;
    }

    int64_t padBefore = 0;
    int64_t padAfter = 0;

    if (paddingMode == "CALCULATED") {
        padBefore = explicitPadBefore;
        padAfter = explicitPadAfter;
    }

    int64_t numerator = 0;
    int64_t inputMinusKernel = 0;

    if (SubOverflow(inputSize, kernelSize, inputMinusKernel) ||
        ge::AddOverflow(inputMinusKernel, padBefore, numerator) || ge::AddOverflow(numerator, padAfter, numerator)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "pooling parameters", dimActual.c_str(),
                                               "overflow occurred while computing the output size");
        return ge::GRAPH_FAILED;
    }

    // ceil_mode 仅对 CALCULATED 模式生效，与正向 MaxPoolV3 的语义保持一致
    const bool useCeilMode = (paddingMode == "CALCULATED") && ceilMode;
    const int64_t quotient = useCeilMode ? Ops::Base::CeilDiv(numerator, stride) : FloorDiv(numerator, stride);

    if (ge::AddOverflow(quotient, 1, outputSize)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "pooling parameters", dimActual.c_str(),
                                               "overflow occurred while computing the output size");
        return ge::GRAPH_FAILED;
    }

    // 与正向 MaxPoolV3 一致的 ceil 末窗口回退：若 (outputSize - 1) * stride >= inputSize + padBefore，
    // 末窗口起始位置落在下侧 pad 填充位，该窗口被舍弃，输出尺寸减 1
    if (useCeilMode) {
        int64_t lastWindowStart = 0;
        int64_t inputWithPadBefore = 0;
        if (ge::MulOverflow(outputSize - 1, stride, lastWindowStart) ||
            ge::AddOverflow(inputSize, padBefore, inputWithPadBefore)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "pooling parameters", dimActual.c_str(),
                                                   "overflow occurred while computing the output size");
            return ge::GRAPH_FAILED;
        }
        if (lastWindowStart >= inputWithPadBefore) {
            --outputSize;
        }
    }

    effectivePadBefore = padBefore;
    effectivePadAfter = padAfter;
    // ceil 模式即使没有显式 pads 也可能保留部分末窗口，需要计入其隐式尾部 padding，
    // 使 NHWC 预留补齐后的输入缓冲，并在 GM 加载时裁剪到有效输入范围。
    if (useCeilMode && outputSize > 0) {
        int64_t lastWindowStart = 0;
        int64_t requiredSize = 0;
        int64_t inputWithPadBefore = 0;
        int64_t requiredPadAfter = 0;
        if (ge::MulOverflow(outputSize - 1, stride, lastWindowStart) ||
            ge::AddOverflow(lastWindowStart, kernelSize, requiredSize) ||
            ge::AddOverflow(inputSize, padBefore, inputWithPadBefore) ||
            SubOverflow(requiredSize, inputWithPadBefore, requiredPadAfter)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "pooling parameters", dimActual.c_str(),
                                                   "overflow occurred while computing ceil-mode padding");
            return ge::GRAPH_FAILED;
        }
        effectivePadAfter = std::max(effectivePadAfter, requiredPadAfter);
    }

    outputSize = std::max(outputSize, static_cast<int64_t>(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ComputePoolParams(gert::TilingContext* context, int64_t inputHeight, int64_t inputWidth,
                                  int64_t kernelHeight, int64_t kernelWidth, int64_t strideHeight, int64_t strideWidth,
                                  const std::string& paddingMode, const int64_t* pads, bool ceilMode, int64_t& padTop,
                                  int64_t& padBottom, int64_t& padLeft, int64_t& padRight, int64_t& outputHeight,
                                  int64_t& outputWidth)
{
    if (ComputeOneDim(context, inputHeight, kernelHeight, strideHeight, paddingMode, pads[0], pads[1], ceilMode, padTop,
                      padBottom, outputHeight) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ComputeOneDim(context, inputWidth, kernelWidth, strideWidth, paddingMode, pads[2], pads[3], ceilMode, padLeft,
                      padRight, outputWidth) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus MaxPoolV3GradTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_PARAM_INVALID;
    }

    auto origInputShapePtr = context_->GetInputShape(ORIG_INPUT_INDEX);
    auto origOutputShapePtr = context_->GetInputShape(ORIG_OUTPUT_INDEX);
    auto gradShapePtr = context_->GetInputShape(GRAD_INDEX);
    auto outGradShapePtr = context_->GetOutputShape(OUT_GRAD_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context_, origInputShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, origOutputShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outGradShapePtr);

    const auto origInputShape = origInputShapePtr->GetStorageShape();
    const auto origOutputShape = origOutputShapePtr->GetStorageShape();
    const auto gradShape = gradShapePtr->GetStorageShape();
    const auto outGradShape = outGradShapePtr->GetStorageShape();
    const std::string origInputShapeText = "orig_input=" + Ops::Base::ToString(origInputShape);
    const std::string origOutputShapeText = "orig_output=" + Ops::Base::ToString(origOutputShape);
    const std::string gradShapeText = "grad=" + Ops::Base::ToString(gradShape);
    const std::string outGradShapeText = "out_grad=" + Ops::Base::ToString(outGradShape);

    if (ValidateConcreteShape(context_, origInputShape, "orig_input") != ge::GRAPH_SUCCESS ||
        ValidateConcreteShape(context_, origOutputShape, "orig_output") != ge::GRAPH_SUCCESS ||
        ValidateConcreteShape(context_, gradShape, "grad") != ge::GRAPH_SUCCESS ||
        ValidateConcreteShape(context_, outGradShape, "out_grad") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (!IsSameShape(origOutputShape, gradShape)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "grad, orig_output",
                                               (gradShapeText + ", " + origOutputShapeText).c_str(),
                                               "grad shape must equal orig_output shape");
        return ge::GRAPH_FAILED;
    }

    if (!IsSameShape(origInputShape, outGradShape)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "out_grad, orig_input",
                                               (outGradShapeText + ", " + origInputShapeText).c_str(),
                                               "out_grad shape must equal orig_input shape");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto ksizeAttr = attrs->GetListInt(ATTR_INDEX_KSIZE);
    auto stridesAttr = attrs->GetListInt(ATTR_INDEX_STRIDES);
    auto padsAttr = attrs->GetListInt(ATTR_INDEX_PADS);

    OP_CHECK_NULL_WITH_CONTEXT(context_, ksizeAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stridesAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padsAttr);

    if (ksizeAttr->GetSize() != ATTR_LIST_SIZE || stridesAttr->GetSize() != ATTR_LIST_SIZE ||
        padsAttr->GetSize() != ATTR_LIST_SIZE) {
        const std::string actual = "ksize=" + std::to_string(ksizeAttr->GetSize()) +
                                   ", strides=" + std::to_string(stridesAttr->GetSize()) +
                                   ", pads=" + std::to_string(padsAttr->GetSize());
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "ksize, strides, pads", actual.c_str(),
                                               "must contain four elements");
        return ge::GRAPH_FAILED;
    }

    const int64_t* ksize = ksizeAttr->GetData();
    const int64_t* strides = stridesAttr->GetData();
    const int64_t* pads = padsAttr->GetData();

    OP_CHECK_NULL_WITH_CONTEXT(context_, ksize);
    OP_CHECK_NULL_WITH_CONTEXT(context_, strides);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pads);

    const char* paddingModeAttr = attrs->GetAttrPointer<char>(ATTR_INDEX_PADDING_MODE);
    const char* dataFormatAttr = attrs->GetAttrPointer<char>(ATTR_INDEX_DATA_FORMAT);
    const bool* globalPoolingAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_GLOBAL_POOLING);
    const bool* ceilModeAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_CEIL_MODE);

    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingModeAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataFormatAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, globalPoolingAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ceilModeAttr);

    const std::string paddingMode(paddingModeAttr);
    const std::string dataFormat(dataFormatAttr);
    const bool globalPooling = *globalPoolingAttr;
    const bool ceilMode = *ceilModeAttr;

    if (ValidateAttrs(context_, paddingMode, dataFormat, globalPooling, ksize, strides, pads) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int32_t channelIndex = dataFormat == "NCHW" ? 1 : 3;
    const int32_t heightIndex = dataFormat == "NCHW" ? 2 : 1;
    const int32_t widthIndex = dataFormat == "NCHW" ? 3 : 2;

    const int64_t n = origInputShape.GetDim(0);
    const int64_t c = origInputShape.GetDim(channelIndex);
    const int64_t inputHeight = origInputShape.GetDim(heightIndex);
    const int64_t inputWidth = origInputShape.GetDim(widthIndex);

    const int64_t outputN = origOutputShape.GetDim(0);
    const int64_t outputC = origOutputShape.GetDim(channelIndex);
    const int64_t outputHeight = origOutputShape.GetDim(heightIndex);
    const int64_t outputWidth = origOutputShape.GetDim(widthIndex);

    if (n != outputN || c != outputC) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "orig_input, orig_output",
                                               (origInputShapeText + ", " + origOutputShapeText).c_str(),
                                               "orig_output N/C dimensions must match orig_input");
        return ge::GRAPH_FAILED;
    }

    int64_t kernelHeight = ksize[heightIndex];
    int64_t kernelWidth = ksize[widthIndex];
    const int64_t strideHeight = strides[heightIndex];
    const int64_t strideWidth = strides[widthIndex];

    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;

    int64_t inputHW = 0;
    int64_t outputHW = 0;
    int64_t nc = 0;
    int64_t totalInputElements = 0;
    int64_t totalOutputElements = 0;

    if (ge::MulOverflow(inputHeight, inputWidth, inputHW) || ge::MulOverflow(outputHeight, outputWidth, outputHW) ||
        ge::MulOverflow(n, c, nc) || ge::MulOverflow(nc, inputHW, totalInputElements) ||
        ge::MulOverflow(nc, outputHW, totalOutputElements)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "orig_input, orig_output",
                                               (origInputShapeText + ", " + origOutputShapeText).c_str(),
                                               "element-count multiplication overflowed");
        return ge::GRAPH_FAILED;
    }

    if (inputHeight == 0 || inputWidth == 0) {
        if (totalOutputElements != 0) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "orig_input, orig_output",
                                                   (origInputShapeText + ", " + origOutputShapeText).c_str(),
                                                   "orig_output must be empty when the input spatial area is empty");
            return ge::GRAPH_FAILED;
        }

        if (globalPooling) {
            kernelHeight = inputHeight;
            kernelWidth = inputWidth;
        }
    } else {
        int64_t computedOutputHeight = 0;
        int64_t computedOutputWidth = 0;

        if (globalPooling) {
            kernelHeight = inputHeight;
            kernelWidth = inputWidth;
            computedOutputHeight = 1;
            computedOutputWidth = 1;
            padTop = 0;
            padBottom = 0;
            padLeft = 0;
            padRight = 0;
        } else if (ComputePoolParams(context_, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight,
                                     strideWidth, paddingMode, pads, ceilMode, padTop, padBottom, padLeft, padRight,
                                     computedOutputHeight, computedOutputWidth) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (outputHeight != computedOutputHeight || outputWidth != computedOutputWidth) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "orig_input, orig_output",
                (origInputShapeText + ", " + origOutputShapeText).c_str(),
                "orig_output spatial dimensions do not match the pooling attributes");
            return ge::GRAPH_FAILED;
        }
    }

    const auto origInputDesc = context_->GetInputDesc(ORIG_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, origInputDesc);

    inputData.hPad = padTop;
    inputData.wPad = padLeft;
    padBottom_ = padBottom;
    padRight_ = padRight;
    inputData.hStride = strideHeight;
    inputData.wStride = strideWidth;
    inputData.hKernel = kernelHeight;
    inputData.wKernel = kernelWidth;
    inputData.hDilation = 1;
    inputData.wDilation = 1;
    inputData.nX = n;
    inputData.cX = c;
    inputData.hX = inputHeight;
    inputData.wX = inputWidth;
    inputData.nGrad = outputN;
    inputData.cGrad = outputC;
    inputData.hGrad = outputHeight;
    inputData.wGrad = outputWidth;
    inputData.ceilMode = ceilMode;
    inputData.gradShapeSize = totalOutputElements;
    inputData.inputDtype = origInputDesc->GetDataType();
    inputData.indexDtype = ge::DT_INT32;
    inputData.inputFormat = dataFormat == "NCHW" ? ge::Format::FORMAT_NCHW : ge::Format::FORMAT_NHWC;
    inputData.isInt32Meet = (totalInputElements > MAX_INT32) ? 1 : 0;

    inputHW_ = inputHW;
    totalInputElements_ = totalInputElements;
    totalOutputElements_ = totalOutputElements;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3GradTilingBase::GetPlatformInfo()
{
    fe::PlatFormInfos* platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    platform_ascendc::PlatformAscendC platform(platformInfo);

    coreNum_ = platform.GetCoreNumAiv();
    if (coreNum_ <= 0 || coreNum_ > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE(context_->GetNodeName(), "The AIV core number is invalid.");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ubSize_ = static_cast<int64_t>(ubSize);

    if (ubSize_ <= static_cast<int64_t>(DCACHE_SIZE)) {
        OP_LOGE(context_->GetNodeName(), "UB size must be greater than the reserved DCache size.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3GradTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

} // namespace optiling
