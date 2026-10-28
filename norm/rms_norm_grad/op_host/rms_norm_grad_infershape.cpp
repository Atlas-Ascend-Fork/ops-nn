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
 * \file rms_norm_grad_infershape.cpp
 * \brief
 */

#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
// InferShape阶段框架契约类失败(输出shape容器为空)统一上报EZ1014, 不再生成EZ9999内部码
static void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "RmsNormGrad", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

static ge::graphStatus InferShape4RmsNormGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4RmsNormGrad.");
    const gert::Shape* x_shape = context->GetInputShape(1);
    const gert::Shape* gamma_shape = context->GetInputShape(3);
    OP_CHECK_IF(
        x_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                 "the input shape of required input x is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamma_shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "gamma",
                    "the input shape of required input gamma is missing in the infer-shape context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);

    // get output shapes
    gert::Shape* dx_shape = context->GetOutputShape(0);
    gert::Shape* dgamma_shape = context->GetOutputShape(1);
    OP_CHECK_IF(dx_shape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of dx (output index 0) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dgamma_shape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of dgamma (output index 1) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);
    *dx_shape = *x_shape;
    *dgamma_shape = *gamma_shape;

    OP_LOGD(context, "End to do InferShape4RmsNormGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNormGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4RmsNormGrad");
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, DT_FLOAT);
    OP_LOGD(context, "End to do InferDataType4RmsNormGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormGrad).InferShape(InferShape4RmsNormGrad).InferDataType(InferDataType4RmsNormGrad);
} // namespace ops
