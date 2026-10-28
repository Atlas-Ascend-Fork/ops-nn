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
 * \file layer_norm_grad_v3_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
// InferShape阶段框架契约类失败(输出shape容器为空)统一上报EZ1014, 不再生成EZ9999内部码
static void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "LayerNormGradV3", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

static const int64_t INPUT_DY_INDEX = 0;
static const int64_t INPUT_X_INDEX = 1;
static const int64_t INPUT_RSTD_INDEX = 2;
static const int64_t INPUT_GAMMA_INDEX = 4;
static constexpr int OUTPUT_PD_X_INDEX = 0;
static constexpr int OUTPUT_PD_GAMMA_INDEX = 1;
static constexpr int OUTPUT_PD_BETA_INDEX = 2;

static ge::graphStatus InferShapeLayerNormGradV3(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        ReportInferShapeStageError(
            "LayerNormGradV3", "InferShape: InferShapeContext is null; the infer-shape framework contract is broken");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "Begin to do InferShapeLayerNormGradV3.");

    const gert::Shape* dy_shape = context->GetInputShape(INPUT_DY_INDEX);
    OP_CHECK_IF(
        dy_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "dy",
                                                 "the input shape of required input dy is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);

    const gert::Shape* x_shape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_IF(
        x_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                 "the input shape of required input x is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);

    const gert::Shape* rstd_shape = context->GetInputShape(INPUT_RSTD_INDEX);
    OP_CHECK_IF(
        rstd_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "rstd",
                                                 "the input shape of required input rstd is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);

    const gert::Shape* gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    OP_CHECK_IF(gamma_shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "gamma",
                    "the input shape of required input gamma is missing in the infer-shape context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);

    gert::Shape* output_pd_x_shape = context->GetOutputShape(OUTPUT_PD_X_INDEX);
    OP_CHECK_IF(output_pd_x_shape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of pd_x (output index 0) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    gert::Shape* output_pd_gamma_shape = context->GetOutputShape(OUTPUT_PD_GAMMA_INDEX);
    OP_CHECK_IF(output_pd_gamma_shape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of pd_gamma (output index 1) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    gert::Shape* output_pd_beta_shape = context->GetOutputShape(OUTPUT_PD_BETA_INDEX);
    OP_CHECK_IF(output_pd_beta_shape == nullptr,
                ReportInferShapeStageError(context->GetNodeName(),
                                           "the output shape container of pd_beta (output index 2) is missing in the "
                                           "infer-shape context; check the InferShapeContext construction"),
                return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*dy_shape)) {
        Ops::Base::SetUnknownRank(*output_pd_x_shape);
    } else {
        *output_pd_x_shape = *dy_shape;
    }

    if (Ops::Base::IsUnknownRank(*gamma_shape)) {
        Ops::Base::SetUnknownRank(*output_pd_gamma_shape);
        Ops::Base::SetUnknownRank(*output_pd_beta_shape);
    } else {
        *output_pd_gamma_shape = *gamma_shape;
        *output_pd_beta_shape = *gamma_shape;
    }
    OP_LOGD(context, "End to do InferShapeLayerNormGradV3.");
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeLayerNormGradV3(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataTypeLayerNormGradV3");
    context->SetOutputDataType(OUTPUT_PD_X_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    context->SetOutputDataType(OUTPUT_PD_GAMMA_INDEX, context->GetInputDataType(INPUT_GAMMA_INDEX));
    context->SetOutputDataType(OUTPUT_PD_BETA_INDEX, context->GetInputDataType(INPUT_GAMMA_INDEX));
    OP_LOGD(context, "End to do InferDataTypeLayerNormGradV3");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LayerNormGradV3).InferShape(InferShapeLayerNormGradV3).InferDataType(InferDataTypeLayerNormGradV3);
} // namespace ops
