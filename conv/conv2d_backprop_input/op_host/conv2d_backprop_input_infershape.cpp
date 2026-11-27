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
 * \file conv2d_backprop_input_infershape.cpp
 * \brief InferShape registration for legacy Conv2DBackpropInput, which is converted to
 *        Conv3DBackpropInputV2 via a fusion pass. The shared infer helpers are reused from
 *        conv_backprop_infershape.h in conv/common.
 */

#include "common/op_host/conv_backprop_infershape.h"
#include "common/op_host/conv_common_cube_util.h"

namespace Ops {
namespace NN {
namespace Conv {

static constexpr size_t kConv2dDimSizeLimit = 4;
constexpr size_t kInputSizeIndex = 0;
constexpr size_t kFilterIndex = 1;
constexpr size_t kOutBackpropIndex = 2;
constexpr size_t kStridesIdx = 0;
constexpr size_t kPadsIdx = 1;
constexpr size_t kDilationsIdx = 2;
constexpr size_t kConv2dAttrListLen = 4;
constexpr size_t kPadTopIdx = 0;
constexpr size_t kPadBottomIdx = 1;
constexpr size_t kPadLeftIdx = 2;
constexpr size_t kPadRightIdx = 3;

// 按下标读const input_size的数据值(input_size按data_format序存放输出shape，dtype支持int32/int64)
static int64_t GetInputSizeDimValue(const gert::Tensor* inputSizeTensor, size_t index)
{
    if (inputSizeTensor->GetDataType() == ge::DT_INT32) {
        return static_cast<int64_t>(inputSizeTensor->GetData<int32_t>()[index]);
    }
    return inputSizeTensor->GetData<int64_t>()[index];
}

// input_size为const且out_backprop为静态shape时，filter的H/W维度不支持-1/-2
static ge::graphStatus CheckFilterShapeForConv2DBackpropInput(gert::InferShapeContext* context)
{
    const auto inputSizeTensor = context->GetInputTensor(kInputSizeIndex);
    OP_CHECK_IF(inputSizeTensor == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get input_size tensor."),
                return ge::GRAPH_FAILED);
    const auto outBackpropShape = context->GetInputShape(kOutBackpropIndex);
    OP_CHECK_IF(outBackpropShape == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get out_backprop shape."),
                return ge::GRAPH_FAILED);
    const auto& inputSizeShape = inputSizeTensor->GetOriginShape();
    if (!IsConstSizeTensor(inputSizeTensor) || Ops::Base::IsUnknownRank(inputSizeShape) ||
        Ops::Base::IsUnknownShape(inputSizeShape) || !IsStaticShape(*outBackpropShape)) {
        return ge::GRAPH_SUCCESS;
    }

    const auto filterShape = context->GetInputShape(kFilterIndex);
    OP_CHECK_IF(filterShape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter shape."),
                return ge::GRAPH_FAILED);
    const auto filterDesc = context->GetInputDesc(kFilterIndex);
    OP_CHECK_IF(filterDesc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get filter tensor desc."),
                return ge::GRAPH_FAILED);
    if (Ops::Base::IsUnknownRank(*filterShape)) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "filter", std::to_string(filterShape->GetDimNum()),
            "The filter shape [-2] must have 4 dimensions and its h and w must be positive when input_size is const "
            "and out_backprop shape is static.");
        return ge::GRAPH_FAILED;
    }
    size_t hIndex = 0;
    size_t wIndex = 0;
    if (filterShape->GetDimNum() != kConv2dDimSizeLimit ||
        !GetHwDimIndex(filterDesc->GetOriginFormat(), hIndex, wIndex)) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t filterH = filterShape->GetDim(hIndex);
    const int64_t filterW = filterShape->GetDim(wIndex);

    const auto yShape = context->GetOutputShape(0);
    OP_CHECK_IF(yShape == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get y shape."),
                return ge::GRAPH_FAILED);
    const auto yDesc = context->GetOutputDesc(0);
    OP_CHECK_IF(yDesc == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get y tensor desc."),
                return ge::GRAPH_FAILED);
    size_t yHIndex = 0;
    size_t yWIndex = 0;
    if (yShape->GetDimNum() != kConv2dDimSizeLimit || outBackpropShape->GetDimNum() != kConv2dDimSizeLimit ||
        !GetHwDimIndex(yDesc->GetOriginFormat(), yHIndex, yWIndex)) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t dxH = GetInputSizeDimValue(inputSizeTensor, yHIndex);
    const int64_t dxW = GetInputSizeDimValue(inputSizeTensor, yWIndex);
    const int64_t dyH = outBackpropShape->GetDim(yHIndex);
    const int64_t dyW = outBackpropShape->GetDim(yWIndex);

    const auto runtimeAttrs = context->GetAttrs();
    OP_CHECK_IF(runtimeAttrs == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get runtime attrs."),
                return ge::GRAPH_FAILED);
    const auto stridesList = runtimeAttrs->GetAttrPointer<gert::ContinuousVector>(kStridesIdx);
    const auto padsList = runtimeAttrs->GetAttrPointer<gert::ContinuousVector>(kPadsIdx);
    const auto dilationsList = runtimeAttrs->GetAttrPointer<gert::ContinuousVector>(kDilationsIdx);
    OP_CHECK_IF(stridesList == nullptr || padsList == nullptr || dilationsList == nullptr ||
                    stridesList->GetSize() < kConv2dAttrListLen || padsList->GetSize() < kConv2dAttrListLen ||
                    dilationsList->GetSize() < kConv2dAttrListLen,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get strides/pads/dilations attrs."),
                return ge::GRAPH_FAILED);
    const auto strides = static_cast<const int64_t*>(stridesList->GetData());
    const auto pads = static_cast<const int64_t*>(padsList->GetData());
    const auto dilations = static_cast<const int64_t*>(dilationsList->GetData());
    const int64_t strideH = strides[yHIndex];
    const int64_t strideW = strides[yWIndex];
    const int64_t kernelH = (filterH - 1) * dilations[yHIndex] + 1;
    const int64_t kernelW = (filterW - 1) * dilations[yWIndex] + 1;
    OP_CHECK_IF(strideH <= 0 || strideW <= 0 || dilations[yHIndex] <= 0 || dilations[yWIndex] <= 0,
                OP_LOGE(context->GetNodeName(), "strides and dilations should be positive."), return ge::GRAPH_FAILED);
    const int64_t dyHNew = (dxH + pads[kPadTopIdx] + pads[kPadBottomIdx] - kernelH) / strideH + 1;
    const int64_t dyWNew = (dxW + pads[kPadLeftIdx] + pads[kPadRightIdx] - kernelW) / strideW + 1;
    if (dyHNew != dyH || dyWNew != dyW) {
        if (filterH < 0 || filterW < 0) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                context->GetNodeName(), "filter", "h=" + std::to_string(filterH) + ", w=" + std::to_string(filterW),
                "The filter shape h and w must be positive when input_size is const and out_backprop shape is static.");
        } else {
            OP_LOGE(context->GetNodeName(),
                    "the output Hout=%ld or Wout=%ld is not equal with the Hout'=%ld or Wout'=%ld which is calculated "
                    "by input and attr. \n"
                    "Hout' = (Hin + 2 * (PaddingUp + PaddingDown) - DilationH * (FilterH - 1) - 1) / strideH + 1 \n"
                    "Wout' = (Win + 2 * (PaddingLeft + PaddingRight) - DilationW * (FilterW - 1) - 1) / strideW + 1",
                    dxH, dxW, dyHNew, dyWNew);
        }
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Conv2DBackpropInput(gert::InferShapeContext* context)
{
    auto ret = InferShapeForConvBackprop(context, 0, "input_size", kConv2dDimSizeLimit);
    if (ret == ge::GRAPH_SUCCESS) {
        ret = PartialInferInputShapeWhenConstInvisible(context);
    }
    if (ret == ge::GRAPH_SUCCESS) {
        ret = CheckFilterShapeForConv2DBackpropInput(context);
    }
    if (ret == ge::GRAPH_SUCCESS) {
        auto yShape = context->GetOutputShape(0);
        if (yShape != nullptr) {
            OP_LOGD(context->GetNodeName(), "[InferShape] Conv2DBackpropInput y_shape: %s",
                    Ops::Base::ToString(*yShape).c_str());
        }
    }
    return ret;
}

static ge::graphStatus InferShapeRange4Conv2DBackpropInput(gert::InferShapeRangeContext* context)
{
    return InferShapeRangeForConvBackprop(context, 1, 1);
}

} // namespace Conv

IMPL_OP_INFERSHAPE(Conv2DBackpropInput)
    .InferShape(Ops::NN::Conv::InferShape4Conv2DBackpropInput)
    .InferShapeRange(Ops::NN::Conv::InferShapeRange4Conv2DBackpropInput)
    .InferDataType(Ops::NN::Conv::InferDataTypeForConvBackpropInputV2)
    .InputsDataDependency({0})
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", -1L);

} // namespace NN
} // namespace Ops
