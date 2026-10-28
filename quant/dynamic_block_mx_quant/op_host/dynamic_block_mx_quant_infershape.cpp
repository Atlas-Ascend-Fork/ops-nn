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
 * \file dynamic_block_mx_quant_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "graph/utils/type_utils.h"
#include "../../../foreach/foreach_utils/op_host/common_dtype.h"

using namespace ge;
namespace ops {
// InferShape阶段框架契约类失败(输出shape容器/属性指针为空)统一上报EZ1014, 不再生成EZ9999内部码
static void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "DynamicBlockMxQuant", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

constexpr size_t INDEX_ATTR_DST_TYPE = 1;
constexpr size_t MIN_DIM_NUM = 2;
constexpr size_t MAX_DIM_NUM = 3;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t Y1_OUTTYPE_INDEX = 0;
constexpr int64_t SCALE1_OUTTYPE_INDEX = 1;
constexpr int64_t SCALE2_OUTTYPE_INDEX = 2;

static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2,
                                                                        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2};

graphStatus InferShapeForDynamicBlockMxQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForDynamicBlockMxQuant");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                 "the input shape of required input x is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_IF(yShape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of y (output index 0) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    gert::Shape* scaleShape1 = context->GetOutputShape(1);
    OP_CHECK_IF(scaleShape1 == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of scale1 (output index 1) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    gert::Shape* scaleShape2 = context->GetOutputShape(2);
    OP_CHECK_IF(scaleShape2 == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of scale2 (output index 2) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, scale shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*scaleShape1);
        Ops::Base::SetUnknownRank(*scaleShape2);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        xShape->GetDimNum() < MIN_DIM_NUM || xShape->GetDimNum() > MAX_DIM_NUM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()),
                                                 "The shape dim of x must be within the range [2, 3]"),
        return ge::GRAPH_FAILED);
    *yShape = *xShape;

    auto attrsPtr = context->GetAttrs();
    OP_CHECK_IF(
        attrsPtr == nullptr,
        ReportInferShapeStageError(context->GetNodeName(),
                                   "InferShape: GetAttrs() returned null; check the InferShapeContext construction"),
        return ge::GRAPH_FAILED);
    size_t dim1 = static_cast<size_t>(xShape->GetDimNum() - 1);

    int64_t dimSize1 = 0;
    if (xShape->GetDim(dim1) == UNKNOWN_DIM_VALUE_) {
        dimSize1 = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize1 = Ops::Base::CeilDiv(xShape->GetDim(dim1), BLOCK_SIZE);
        dimSize1 = (dimSize1 + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *scaleShape1 = *xShape;
    scaleShape1->SetDim(dim1, dimSize1);
    scaleShape1->AppendDim(ALIGN_NUM);

    OP_LOGD(context->GetNodeName(), "x shape is : %s, scale1 shape is %s.", Shape2String(*xShape).c_str(),
            Shape2String(*scaleShape1).c_str());

    size_t dim2 = static_cast<size_t>(xShape->GetDimNum() - 2);
    int64_t dimSize2 = 0;
    if (xShape->GetDim(dim2) == UNKNOWN_DIM_VALUE_) {
        dimSize2 = UNKNOWN_DIM_VALUE_;
    } else {
        dimSize2 = Ops::Base::CeilDiv(xShape->GetDim(dim2), BLOCK_SIZE);
        dimSize2 = (dimSize2 + ALIGN_NUM - 1) / ALIGN_NUM;
    }

    *scaleShape2 = *xShape;
    scaleShape2->SetDim(dim2, dimSize2);
    scaleShape2->AppendDim(ALIGN_NUM);

    OP_LOGD(context->GetNodeName(), "x shape is : %s, scale2 shape is %s.", Shape2String(*xShape).c_str(),
            Shape2String(*scaleShape2).c_str());

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForDynamicBlockMxQuant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForDynamicBlockMxQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForDynamicBlockMxQuant");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_IF(attrsPtr == nullptr,
                ReportInferShapeStageError(
                    context->GetNodeName(),
                    "InferDataType: GetAttrs() returned null; check the InferDataTypeContext construction"),
                return ge::GRAPH_FAILED);
    const int32_t* dstDtype = attrsPtr->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_IF(dstDtype == nullptr,
                ReportInferShapeStageError(
                    context->GetNodeName(),
                    "the optional attr dst_type is missing in the infer-shape context; set the dst_type attribute"),
                return ge::GRAPH_FAILED);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);

    OP_CHECK_IF(
        std::find(Y_SUPPORT_DTYPE_SET.begin(), Y_SUPPORT_DTYPE_SET.end(), outDtype) == Y_SUPPORT_DTYPE_SET.end(),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "dst_type", ge::TypeUtils::DataTypeToSerialString(outDtype),
                                  "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2"),
        return ge::GRAPH_FAILED);
    context->SetOutputDataType(Y1_OUTTYPE_INDEX, outDtype);
    context->SetOutputDataType(SCALE1_OUTTYPE_INDEX, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(SCALE2_OUTTYPE_INDEX, ge::DT_FLOAT8_E8M0);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForDynamicBlockMxQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicBlockMxQuant)
    .InferShape(InferShapeForDynamicBlockMxQuant)
    .InferDataType(InferDataTypeForDynamicBlockMxQuant);
} // namespace ops
