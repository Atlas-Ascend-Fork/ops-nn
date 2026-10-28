/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_situ_quant_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "util/shape_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;
namespace ops {
constexpr size_t INPUT_IDX_X = 0;
constexpr size_t OUTPUT_IDX_Y = 0;
constexpr size_t OUTPUT_IDX_Y_SCALE = 1;
constexpr int64_t CONST_UNKNOW_SHAPE = -1;
constexpr int64_t NUM_TWO = 2;

graphStatus InferShape4DequantSituQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4DequantSituQuant.");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_IF(xShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context) == nullptr || (context)->GetNodeName() == nullptr ? "dequant_situ_quant" :
                                                                                   (context)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    if (yShape == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "dequant_situ_quant" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object yShape is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object yShape is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    gert::Shape* yScaleShape = context->GetOutputShape(OUTPUT_IDX_Y_SCALE);
    if (yScaleShape == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "dequant_situ_quant" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object yScaleShape is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object yScaleShape is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    *yShape = *xShape;
    OP_CHECK_IF(Ops::Base::IsUnknownRank(*xShape),
                OP_LOGD(context, "End to do InferShape4DequantSituQuant, inputx is [-2]."), return GRAPH_SUCCESS);

    int64_t xShapeRank = static_cast<int64_t>(xShape->GetDimNum());

    auto inputDesc = context->GetInputDesc(INPUT_IDX_X);
    OP_CHECK_IF(inputDesc == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context) == nullptr || (context)->GetNodeName() == nullptr ? "dequant_situ_quant" :
                                                                                   (context)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);
    ge::DataType xDataType = inputDesc->GetDataType();

    bool isInt8 = (xDataType == ge::DT_INT8);
    if (isInt8) {
        OP_CHECK_IF(xShapeRank <= 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(xShapeRank),
                        ("x shape rank must > 1 for int8, but is " + std::to_string(xShapeRank))),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(xShapeRank != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(xShapeRank),
                        ("x shape rank must be 2 for int32/bfloat16/float16, but is " + std::to_string(xShapeRank))),
                    return ge::GRAPH_FAILED);
    }

    int64_t lastDim = xShape->GetDim(xShapeRank - 1);
    int64_t outLastDim = lastDim == CONST_UNKNOW_SHAPE ? CONST_UNKNOW_SHAPE : lastDim / NUM_TWO;
    OP_CHECK_IF((lastDim != CONST_UNKNOW_SHAPE) && (lastDim % NUM_TWO != 0),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context->GetNodeName(), "x", std::to_string(lastDim),
                    ("The last dim of x must be even number, but is " + std::to_string(lastDim))),
                return ge::GRAPH_FAILED);

    yShape->SetDim(xShapeRank - 1, outLastDim);

    *yScaleShape = *yShape;
    if (isInt8) {
        yScaleShape->SetDimNum(xShapeRank - 1);
    } else {
        yScaleShape->SetDimNum(1);
        yScaleShape->SetDim(0, xShape->GetDim(0));
    }

    OP_LOGD(context, "End to do InferShape4DequantSituQuant");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4DequantSituQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDtype4DequantSituQuant enter");
    context->SetOutputDataType(OUTPUT_IDX_Y, DT_INT8);
    context->SetOutputDataType(OUTPUT_IDX_Y_SCALE, DT_FLOAT);
    OP_LOGD(context, "InferDtype4DequantSituQuant end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DequantSituQuant).InferShape(InferShape4DequantSituQuant).InferDataType(InferDtype4DequantSituQuant);
} // namespace ops
