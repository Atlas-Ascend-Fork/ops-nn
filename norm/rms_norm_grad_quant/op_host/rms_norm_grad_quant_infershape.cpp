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
 * \file rms_norm_grad_quant_infershape.cpp
 * \brief
 */

#include <algorithm>
#include <vector>
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {
// InferShape阶段框架契约类失败(输出shape容器为空)统一上报EZ1014, 不再生成EZ9999内部码
static void ReportInferShapeStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "RmsNormGradQuant", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", msgKey, msgValue);
}

static constexpr int RMS_NORM_GRAD_QUANT_IDX_DY = 0;
static constexpr int RMS_NORM_GRAD_QUANT_IDX_GAMMA = 3;
static constexpr int RMS_NORM_GRAD_QUANT_IDX_SCALE_X = 4;
static constexpr int RMS_NORM_GRAD_QUANT_ATTR_IDX_DST_TYPE = 2;

static constexpr int RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DX = 0;
static constexpr int RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DGAMMA = 1;

static const std::vector<ge::DataType> RMS_NORM_GRAD_QUANT_OUT_TYPE_LIST = {ge::DT_HIFLOAT8, ge::DT_INT8};

static ge::graphStatus InferShape4RmsNormGradQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4RmsNormGradQuant.");
    const gert::Shape* dy_shape = context->GetInputShape(RMS_NORM_GRAD_QUANT_IDX_DY);
    const gert::Shape* gamma_shape = context->GetInputShape(RMS_NORM_GRAD_QUANT_IDX_GAMMA);
    OP_CHECK_IF(
        dy_shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "dy",
                                                 "the input shape of required input dy is missing in the infer-shape "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamma_shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "gamma",
                    "the input shape of required input gamma is missing in the infer-shape context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);

    // get output shapes
    gert::Shape* dx_shape = context->GetOutputShape(RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DX);
    gert::Shape* dgamma_shape = context->GetOutputShape(RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DGAMMA);
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
    *dx_shape = *dy_shape;
    *dgamma_shape = *gamma_shape;

    OP_LOGD(context, "End to do InferShape4RmsNormGradQuant.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNormGradQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4RmsNormGradQuant");

    // 1. 设置第一个输出dx的数据类型 (scales_x为必选输入，始终走量化场景)
    // 量化场景，根据dst_type这个attr的值进行设置，默认是INT8
    ge::DataType dxDtype = ge::DT_INT8;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* dstTypePtr = attrs->GetAttrPointer<int64_t>(RMS_NORM_GRAD_QUANT_ATTR_IDX_DST_TYPE);
        if (dstTypePtr != nullptr) {
            dxDtype = static_cast<ge::DataType>(*dstTypePtr);
            OP_CHECK_IF(std::find(RMS_NORM_GRAD_QUANT_OUT_TYPE_LIST.begin(), RMS_NORM_GRAD_QUANT_OUT_TYPE_LIST.end(),
                                  dxDtype) == RMS_NORM_GRAD_QUANT_OUT_TYPE_LIST.end(),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "dst_type", std::to_string(*dstTypePtr).c_str(),
                            "the value of attr dst_type only supports DT_INT8 (2) and DT_HIFLOAT8 (34)"),
                        return ge::GRAPH_FAILED);
        }
    }
    context->SetOutputDataType(RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DX, dxDtype);

    // 2. 设置第二个输出dx的数据类型
    context->SetOutputDataType(RMS_NORM_GRAD_QUANT_OUTPUT_IDX_DGAMMA, DT_FLOAT);
    OP_LOGD(context, "End to do InferDataType4RmsNormGradQuant");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormGradQuant)
    .InferShape(InferShape4RmsNormGradQuant)
    .InferDataType(InferDataType4RmsNormGradQuant);
} // namespace ops
