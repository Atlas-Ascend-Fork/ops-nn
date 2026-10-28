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
 * \file ascend_quant_v2.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
// InferShape阶段框架契约类失败统一上报EZ1014, 不再生成EZ9999内部码
static void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "AscendQuantV2", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

constexpr size_t g_AttrDstType = 2;

static graphStatus InferShapeForAscendQuantV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForAscendQuantV2");
    const gert::Shape* inputXShape = context->GetInputShape(0);
    OP_CHECK_IF(
        inputXShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                 "the input shape of required input x is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);

    gert::Shape* outputShape = context->GetOutputShape(0);

    *outputShape = *inputXShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForAscendQuantV2");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForAscendQuantV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForAscendQuantV2");
    const int32_t* dstDtype = context->GetAttrs()->GetAttrPointer<int32_t>(g_AttrDstType);
    OP_CHECK_IF(
        dstDtype == nullptr,
        ReportInferShapeStageError(context->GetNodeName(), "the required attr dst_dtype is missing in the infer-shape "
                                                           "context; set the dst_dtype attribute on the node"),
        return ge::GRAPH_FAILED);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstDtype);
    context->SetOutputDataType(0, outDtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForAscendQuantV2");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AscendQuantV2).InferShape(InferShapeForAscendQuantV2).InferDataType(InferDataTypeForAscendQuantV2);
} // namespace ops
