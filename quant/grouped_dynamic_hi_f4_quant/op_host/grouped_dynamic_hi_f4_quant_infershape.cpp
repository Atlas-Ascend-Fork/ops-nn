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
 * \file grouped_dynamic_hi_f4_quant_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/shape_util.h"
#include "../../../common/inc/op_host/op_const_def_nn.h"

using namespace ge;
namespace ops {
constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_INPUT_GROUP_INDEX = 1;
constexpr int64_t INDEX_OUTPUT_Y = 0;
constexpr int64_t INDEX_OUTPUT_SCALE = 1;
constexpr int64_t SCALE_DIM_0 = 0;
constexpr int64_t SCALE_DIM_1 = 1;
constexpr int64_t SCALE_DIM_2 = 2;
constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t NUM_ONE = 1;
constexpr size_t SCALE_DIM_NUM = 3;

static const int32_t DTYPE_FLOAT8_E5M2 = 35;
static const int32_t DTYPE_FLOAT8_E4M3FN = 36;

template <typename T>
static std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

graphStatus InferShapeForGroupedDynamicHiF4Quant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForGroupedDynamicHiF4Quant");
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* groupIdxShape = context->GetInputShape(INDEX_INPUT_GROUP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupIdxShape);

    gert::Shape* yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape;

    gert::Shape* scaleShape = context->GetOutputShape(INDEX_OUTPUT_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);

    size_t xShapeSize = xShape->GetDimNum();
    size_t groupIdxShapeSize = groupIdxShape->GetDimNum();
    OP_CHECK_IF(groupIdxShapeSize != 1,
                OP_LOGE(context->GetNodeName(), "group_index's shape must be 1D, but is %lu", groupIdxShapeSize),
                return ge::GRAPH_FAILED);
    int64_t groupIdxDim0 = groupIdxShape->GetDim(0);
    OP_CHECK_IF(groupIdxDim0 == 0, OP_LOGE(context->GetNodeName(), "group_index does not support empty tensor"),
                return ge::GRAPH_FAILED);

    // dynamic -2 (input x)
    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "input x is UnknownRank, set outputs' shape to -2");
        *scaleShape = *xShape;
        return ge::GRAPH_SUCCESS;
    } else {
        OP_CHECK_IF(
            xShapeSize != 2,
            OP_LOGE(context->GetNodeName(), "input x is not UnknownRank, shape must be 2D, but is %lu", xShapeSize),
            return ge::GRAPH_FAILED);
    }

    int64_t dim0Size = (xShape->GetDim(0) / static_cast<int64_t>(BLOCK_SIZE) + groupIdxDim0); // 不带起始0
    // dynamic -2 or -1 (groupIdxShape), dynamic -1 (input x)
    if (Ops::Base::IsUnknownRank(*groupIdxShape) || groupIdxDim0 == -1 || xShape->GetDim(0) == -1) {
        dim0Size = -1;
    }

    // 推导scale的shape
    scaleShape->SetDimNum(SCALE_DIM_NUM);
    scaleShape->SetDim(SCALE_DIM_0, dim0Size);
    scaleShape->SetDim(SCALE_DIM_1, (xShape->GetDim(1) == -1) ? -1 : xShape->GetDim(1));
    scaleShape->SetDim(SCALE_DIM_2, NUM_ONE);
    OP_LOGD(context->GetNodeName(), "mxscale shape is :%s after infershape.", Shape2String(*scaleShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForGroupedDynamicHiF4Quant");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForGroupedDynamicHiF4Quant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForGroupedDynamicHiF4Quant");
    context->SetOutputDataType(0, ge::DT_HIFLOAT4);
    context->SetOutputDataType(1, Ops::NN::DT_HIFLOAT4_SCALE);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForGroupedDynamicHiF4Quant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedDynamicHiF4Quant)
    .InferShape(InferShapeForGroupedDynamicHiF4Quant)
    .InferDataType(InferDataTypeForGroupedDynamicHiF4Quant);
} // namespace ops
