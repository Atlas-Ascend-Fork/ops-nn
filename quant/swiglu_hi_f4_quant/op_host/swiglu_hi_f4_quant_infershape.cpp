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
 * \file swiglu_hi_f4_quant_infershape.cpp
 * \brief
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

using namespace ge;

namespace ops {
constexpr int64_t UNKNOWN_DIM_VALUE = -1;
constexpr int64_t UNKNOWN_RANK_DIM = -2;
constexpr size_t INDEX_INPUT_X = 0;
constexpr size_t INDEX_INPUT_GROUP_INDEX = 1;
constexpr size_t INDEX_OUTPUT_Y = 0;
constexpr size_t INDEX_OUTPUT_SCALE = 1;

constexpr size_t INDEX_ATTR_ACTIVATE_DIM = 0;
constexpr size_t INDEX_ATTR_AXIS = 6;

constexpr int64_t SPLIT_NUM = 2;
constexpr int64_t HIF4_BLOCK_SIZE = 64;
constexpr size_t MAX_DIM_NUM = 7;

static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {Ops::NN::DT_HIFLOAT4};

graphStatus ComputeShapeForSwigluHiF4Quant(gert::InferShapeContext* context, const gert::Shape* xShape,
                                           gert::Shape* yShape, gert::Shape* scaleShape)
{
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);

    const int64_t* activateDim = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_ACTIVATE_DIM);
    OP_CHECK_NULL_WITH_CONTEXT(context, activateDim);
    const int64_t* axis = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, axis);

    int64_t xRank = xShape->GetDimNum();
    OP_CHECK_IF(xRank <= 1, OP_LOGE(context->GetNodeName(), "the rank of x must be greater than 1, but is %ld", xRank),
                return ge::GRAPH_FAILED);
    int64_t activateDimNorm = (*activateDim >= 0) ? static_cast<int64_t>(*activateDim) :
                                                    static_cast<int64_t>(*activateDim + xRank);
    int64_t axisNorm = (*axis >= 0) ? static_cast<int64_t>(*axis) : static_cast<int64_t>(*axis + xRank);
    OP_CHECK_IF(activateDimNorm >= xRank || axisNorm >= xRank,
                OP_LOGE(context->GetNodeName(),
                        "activateDim and axis must < xRank, but xRank is %ld, activateDim is %ld, axis is %ld", xRank,
                        activateDimNorm, axisNorm),
                return ge::GRAPH_FAILED);

    if (xShape->GetDim(activateDimNorm) != UNKNOWN_DIM_VALUE && xShape->GetDim(activateDimNorm) % SPLIT_NUM != 0) {
        OP_LOGE(context->GetNodeName(), "The dimension [%zu] at activate_dim must be divisible by 2, but got [%ld].",
                activateDimNorm, xShape->GetDim(activateDimNorm));
        return ge::GRAPH_FAILED;
    }

    *yShape = *xShape;
    if (xShape->GetDim(activateDimNorm) != UNKNOWN_DIM_VALUE) {
        yShape->SetDim(activateDimNorm, xShape->GetDim(activateDimNorm) / SPLIT_NUM);
    }

    const gert::Shape* groupIndexShape = context->GetOptionalInputShape(INDEX_INPUT_GROUP_INDEX);
    bool hasGroup = (groupIndexShape != nullptr);
    OP_CHECK_IF(
        hasGroup && groupIndexShape->GetDimNum() != 1,
        OP_LOGE(context->GetNodeName(), "the rank of group_index must be 1, but is %d", groupIndexShape->GetDimNum()),
        return ge::GRAPH_FAILED);

    int64_t yAxisSize = 0;
    if (yShape->GetDim(axisNorm) == UNKNOWN_DIM_VALUE) {
        yAxisSize = UNKNOWN_DIM_VALUE;
    } else {
        int64_t yDim = yShape->GetDim(axisNorm);
        yAxisSize = Ops::Base::CeilDiv(yDim, HIF4_BLOCK_SIZE);
    }

    *scaleShape = *yShape;
    scaleShape->SetDim(axisNorm, yAxisSize);
    scaleShape->AppendDim(1);
    OP_LOGI(context->GetNodeName(), "End to do InferShapeForSwigluHiF4Quant");
    return ge::GRAPH_SUCCESS;
}

graphStatus InferShapeForSwigluHiF4Quant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForSwigluHiF4Quant");
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* scaleShape = context->GetOutputShape(INDEX_OUTPUT_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    OP_CHECK_IF(xShape->GetDimNum() < 1 || xShape->GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "Input x rank[%lu] should be in [1, 7].", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, scale shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*scaleShape);
        return ge::GRAPH_SUCCESS;
    }
    return ComputeShapeForSwigluHiF4Quant(context, xShape, yShape, scaleShape);
}

ge::graphStatus InferDataTypeForSwigluHiF4Quant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForSwigluHiF4Quant");
    context->SetOutputDataType(INDEX_OUTPUT_Y, Ops::NN::DT_HIFLOAT4);
    context->SetOutputDataType(INDEX_OUTPUT_SCALE, Ops::NN::DT_HIFLOAT4_SCALE);
    OP_LOGI(context->GetNodeName(), "End to do InferDataTypeForSwigluHiF4Quant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SwigluHiF4Quant)
    .InferShape(InferShapeForSwigluHiF4Quant)
    .InferDataType(InferDataTypeForSwigluHiF4Quant);
} // namespace ops
