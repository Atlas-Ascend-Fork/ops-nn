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
 * \file group_quant.cc
 * \brief
 */
#include <numeric>
#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
// InferShape阶段框架契约类失败(上下文/输出shape容器为空)统一上报EZ1014, 不再生成EZ9999内部码
static void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "GroupQuant", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

static const size_t ATTR_INDEX_OF_DST_TYPE = 0;
static const int32_t DTYPE_INT8 = 2;
static const int32_t DTYPE_INT4 = 29;

static ge::graphStatus InferDataType4GroupQuant(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        ReportInferShapeStageError(
            "GroupQuant", "InferDataType: InferDataTypeContext is null; the infer-shape framework contract is broken"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferDataType4GroupQuant begin");

    ge::DataType yDtype = ge::DT_INT8;
    const int32_t* pDstDtype = context->GetAttrs()->GetAttrPointer<int32_t>(ATTR_INDEX_OF_DST_TYPE);
    if (pDstDtype != nullptr) {
        int32_t dstDtype = *pDstDtype;
        OP_CHECK_IF(dstDtype != DTYPE_INT8 && dstDtype != DTYPE_INT4,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context->GetNodeName(), "dst_type", std::to_string(dstDtype).c_str(),
                        "the value of attr dst_type only supports 2 (DT_INT8) and 29 (DT_INT4)"),
                    return ge::GRAPH_FAILED);
        yDtype = static_cast<ge::DataType>(dstDtype);
    }
    context->SetOutputDataType(0, yDtype);

    OP_LOGD(context->GetNodeName(), "InferDataType4GroupQuant end");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GroupQuant(gert::InferShapeContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        ReportInferShapeStageError(
            "GroupQuant", "InferShape: InferShapeContext is null; the infer-shape framework contract is broken"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "InferShape4GroupQuant begin");

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

    *yShape = *xShape;

    OP_LOGI(context->GetNodeName(), "[InferShape] output0 shape=%s", Ops::Base::ToString(*yShape).c_str());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupQuant).InferDataType(InferDataType4GroupQuant).InferShape(InferShape4GroupQuant);
} // namespace ops
