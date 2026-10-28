/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_host/infershape_broadcast_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;
namespace ops {

static std::string ShapeCannotBroadcastMsg(const gert::Shape& shape1, const gert::Shape& shape2)
{
    std::string res = "shape ";
    res += Ops::Base::ToString(shape1);
    res += " and ";
    res += Ops::Base::ToString(shape2);
    res += " cannot broadcast!";
    return res;
}

static ge::graphStatus InferShape4FastGeluGrad(gert::InferShapeContext* context)
{
    auto fast_gelu_grad_in_shape1 = context->GetInputShape(0);
    OP_CHECK_IF(
        fast_gelu_grad_in_shape1 == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "fast_gelu_grad" : (context)->GetNodeName()),
            "dy"),
        return ge::GRAPH_FAILED);
    auto fast_gelu_grad_in_shape2 = context->GetInputShape(1);
    OP_CHECK_IF(
        fast_gelu_grad_in_shape2 == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "fast_gelu_grad" : (context)->GetNodeName()),
            "x"),
        return ge::GRAPH_FAILED);
    auto fast_gelu_grad_out_shape = context->GetOutputShape(0);
    if (fast_gelu_grad_out_shape == nullptr) {
        const std::string msgOpName = std::string((
            (context) == nullptr || (context)->GetNodeName() == nullptr ? "fast_gelu_grad" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object fast_gelu_grad_out_shape is unavailable; check the "
                               "operator registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(),
                                      "InferShape framework object fast_gelu_grad_out_shape is unavailable; check the "
                                      "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    OP_CHECK_IF(
        !Ops::Base::BroadcastShape(fast_gelu_grad_in_shape1, fast_gelu_grad_in_shape2, fast_gelu_grad_out_shape),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x, y",
            (Ops::Base::ToString(*fast_gelu_grad_in_shape1) + ", " + Ops::Base::ToString(*fast_gelu_grad_in_shape2)),
            "The shapes of x and y must be broadcastable"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FastGeluGrad).InferShape(InferShape4FastGeluGrad);
} // namespace ops
