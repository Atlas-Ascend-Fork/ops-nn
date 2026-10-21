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
 * \file dynamic_block_hi_f4_quant_infershape.cpp
 * \brief InferShape & InferDataType for DynamicBlockHiF4Quant
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

template <typename T>
std::string Shape2String(const T& shape)
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

constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t ALIGN_NUM_FOR_HIFLOAT4 = 1;
constexpr size_t MAX_DIM_NUM = 7;
constexpr size_t MIN_DIM_NUM = 2;
constexpr size_t DIM_NUM_TWO = 2;
constexpr int64_t BLOCKSIZE4HIFLOAT4 = 64;
constexpr int64_t Y_OUTTYPE_INDEX = 0;
constexpr int64_t SCALE1_OUTTYPE_INDEX = 1;
constexpr int64_t SCALE2_OUTTYPE_INDEX = 2;

graphStatus InferShapeForDynamicBlockHiF4Quant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForDynamicBlockHiF4Quant");
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* scale1Shape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale1Shape);

    gert::Shape* scale2Shape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale2Shape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(context->GetNodeName(), "x shape is UnknownRank, set y, scale1, scale2 shape to (-2, )");
        Ops::Base::SetUnknownRank(*yShape);
        Ops::Base::SetUnknownRank(*scale1Shape);
        Ops::Base::SetUnknownRank(*scale2Shape);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(xShape->GetDimNum() < MIN_DIM_NUM || xShape->GetDimNum() > MAX_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "Input x rank[%lu] should be in [2, 7].", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    // y.shape = x.shape
    *yShape = *xShape;

    // scale1: rank(scale1) = rank(x) + 1
    //         scale1.shape[:-2] = x.shape[:-1]
    //         scale1.shape[-2] = x.shape[-1] // 64
    //         scale1.shape[-1] = 1
    *scale1Shape = *xShape;
    int64_t dimSizeNeg1 = 0;
    if (xShape->GetDim(xShape->GetDimNum() - 1) == UNKNOWN_DIM_VALUE_) {
        dimSizeNeg1 = UNKNOWN_DIM_VALUE_;
    } else {
        dimSizeNeg1 = Ops::Base::CeilDiv(xShape->GetDim(xShape->GetDimNum() - 1), BLOCKSIZE4HIFLOAT4);
    }
    scale1Shape->SetDim(xShape->GetDimNum() - 1, dimSizeNeg1);
    scale1Shape->AppendDim(ALIGN_NUM_FOR_HIFLOAT4);

    // scale2: rank(scale2) = rank(x) + 1
    //         scale2.shape[:-3] = x.shape[:-2]
    //         scale2.shape[-3] = x.shape[-2] // 64
    //         scale2.shape[-2] = x.shape[-1]
    //         scale2.shape[-1] = 1
    *scale2Shape = *xShape;
    int64_t dimSizeNeg2 = 0;
    if (xShape->GetDim(xShape->GetDimNum() - DIM_NUM_TWO) == UNKNOWN_DIM_VALUE_) {
        dimSizeNeg2 = UNKNOWN_DIM_VALUE_;
    } else {
        dimSizeNeg2 = Ops::Base::CeilDiv(xShape->GetDim(xShape->GetDimNum() - DIM_NUM_TWO), BLOCKSIZE4HIFLOAT4);
    }
    scale2Shape->SetDim(xShape->GetDimNum() - DIM_NUM_TWO, dimSizeNeg2);
    scale2Shape->AppendDim(ALIGN_NUM_FOR_HIFLOAT4);

    OP_LOGD(context->GetNodeName(), "x shape is : %s, y shape is %s, scale1 shape is %s, scale2 shape is %s.",
            Shape2String(*xShape).c_str(), Shape2String(*yShape).c_str(), Shape2String(*scale1Shape).c_str(),
            Shape2String(*scale2Shape).c_str());
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForDynamicBlockHiF4Quant");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForDynamicBlockHiF4Quant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForDynamicBlockHiF4Quant");
    context->SetOutputDataType(Y_OUTTYPE_INDEX, ge::DT_HIFLOAT4);
    context->SetOutputDataType(SCALE1_OUTTYPE_INDEX, Ops::NN::DT_HIFLOAT4_SCALE);
    context->SetOutputDataType(SCALE2_OUTTYPE_INDEX, Ops::NN::DT_HIFLOAT4_SCALE);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForDynamicBlockHiF4Quant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DynamicBlockHiF4Quant)
    .InferShape(InferShapeForDynamicBlockHiF4Quant)
    .InferDataType(InferDataTypeForDynamicBlockHiF4Quant);
} // namespace ops
