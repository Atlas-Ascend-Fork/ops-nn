/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\\file anti_hi_f4_quant_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/math_util.h"
#include "util/shape_util.h"
#include "graph/utils/type_utils.h"

using namespace ge;
namespace ops {
constexpr size_t MAX_DIM_NUM = 7;
constexpr size_t MIN_DIM_NUM = 2;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t INDEX_INPUT_SCALE = 1;
constexpr int64_t INDEX_OUTPUT_Y = 0;
constexpr size_t INDEX_ATTR_DST_TYPE = 0;
constexpr int64_t BLOCK_SIZE = 64;
constexpr int64_t NUM_TWO = 2;

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

graphStatus InferShapeForAntiHiF4Quant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForAntiHiF4Quant");
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(xShape->GetDimNum() < MIN_DIM_NUM || xShape->GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "Input x rank[%zu] should be in [2, 7].", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "x shape is : %s, y shape is %s.", Shape2String(*xShape).c_str(),
            Shape2String(*yShape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForAntiHiF4Quant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForAntiHiF4Quant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForAntiHiF4Quant");
    auto attrsPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrsPtr);
    const int64_t* dstType = attrsPtr->GetAttrPointer<int64_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstType);

    int64_t outputDtype = *dstType;
    context->SetOutputDataType(INDEX_OUTPUT_Y, static_cast<ge::DataType>(outputDtype));
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForAntiHiF4Quant, output dtype = %ld", outputDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AntiHiF4Quant).InferShape(InferShapeForAntiHiF4Quant).InferDataType(InferDataTypeForAntiHiF4Quant);
} // namespace ops
