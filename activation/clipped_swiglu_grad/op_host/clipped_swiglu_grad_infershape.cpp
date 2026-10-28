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
 * \file clipped_swiglu_grad_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/shape_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace {
constexpr size_t GRAD_IN_X = 1;       // x is the second input (index 1)
constexpr size_t GRAD_OUT_GRAD_X = 0; // grad_x is the first output (index 0)
constexpr size_t GRAD_ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForClippedSwigluGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForClippedSwigluGrad");
    auto xShape = context->GetInputShape(GRAD_IN_X);
    OP_CHECK_IF(xShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context) == nullptr || (context)->GetNodeName() == nullptr ? "clipped_swiglu_grad" :
                                                                                   (context)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);
    auto gradXOutShape = context->GetOutputShape(GRAD_OUT_GRAD_X);
    if (gradXOutShape == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "clipped_swiglu_grad" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object gradXOutShape is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object gradXOutShape is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "clipped_swiglu_grad" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    auto splitDimPtr = attrs->GetAttrPointer<int64_t>(GRAD_ATTR_DIM);
    if (splitDimPtr == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "clipped_swiglu_grad" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object splitDimPtr is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object splitDimPtr is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*gradXOutShape);
        return ge::GRAPH_SUCCESS;
    }
    auto splitDim = *splitDimPtr;
    if (splitDim < 0) {
        splitDim += xShape->GetDimNum();
    }
    if (splitDim < 0 || splitDim >= static_cast<int64_t>(xShape->GetDimNum())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "ClippedSwigluGrad", "dim", std::to_string(*splitDimPtr),
            ("The value of attr [dim] must be in the range [-" + std::to_string(xShape->GetDimNum()) + ", " +
             std::to_string(xShape->GetDimNum() - 1) + "], but got [" + std::to_string(splitDim) + "]."));
        return GRAPH_FAILED;
    }
    int64_t splitDimSize = xShape->GetDim(splitDim);
    if (splitDimSize >= 0 && splitDimSize % static_cast<int64_t>(SPLIT_NUM) != 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            "ClippedSwigluGrad", "x", std::to_string(splitDimSize),
            ("The size of x[dim] must be divisible by " + std::to_string(SPLIT_NUM) + ", but got [" +
             std::to_string(splitDimSize) + "]."));
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to generate gradXOutShape");
    // grad_x has the same shape as x
    *gradXOutShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForClippedSwigluGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForClippedSwigluGrad");
    const ge::DataType dtype = context->GetInputDataType(1); // x is index 1
    ge::graphStatus ret = context->SetOutputDataType(0, dtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForClippedSwigluGrad");
    return ret;
}

IMPL_OP_INFERSHAPE(ClippedSwigluGrad)
    .InferShape(InferShapeForClippedSwigluGrad)
    .InferDataType(InferDataTypeForClippedSwigluGrad);
} // namespace ops
