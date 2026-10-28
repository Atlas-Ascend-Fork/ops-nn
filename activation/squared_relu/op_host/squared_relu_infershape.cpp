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
 * \file squared_relu.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;

static ge::graphStatus InferShape4SquaredRelu(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4SquaredRelu");

    // get input shapes
    auto xShape = context->GetInputShape(IDX_0);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "squared_relu" : (context)->GetNodeName()),
            "input"),
        return ge::GRAPH_FAILED);

    // get output shapes
    auto yShape = context->GetOutputShape(IDX_0);
    if (yShape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "squared_relu" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object yShape is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object yShape is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    size_t xDimNum = xShape->GetDimNum();
    yShape->SetDimNum(xDimNum);

    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4SquaredRelu");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4SquaredRelu(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4SquaredRelu");

    auto input_dtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, input_dtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataType4SquaredRelu");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SquaredRelu).InferShape(InferShape4SquaredRelu).InferDataType(InferDataType4SquaredRelu);
} // namespace ops
