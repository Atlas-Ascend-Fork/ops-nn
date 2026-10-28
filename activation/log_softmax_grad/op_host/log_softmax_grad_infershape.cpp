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
 * \file log_softmax_grad_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;
namespace ops {
static ge::graphStatus CopyShapeInput2OutputWithIdx(gert::InferShapeContext* context, int64_t input_idx,
                                                    int64_t output_idx)
{
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_IF(
        in_shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "log_softmax_grad" :
                                                                           (context)->GetNodeName()),
            (input_idx == 0 ? std::string("grad") :
                              (input_idx == 1 ? std::string("x") : "input[" + std::to_string(input_idx) + "]"))),
        return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(output_idx);
    if (out_shape == nullptr) {
        const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                       "log_softmax_grad" :
                                                       (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object out_shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object out_shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForLogSoftmaxGrad(gert::InferShapeContext* context)
{
    return CopyShapeInput2OutputWithIdx(context, 1, 0);
}
IMPL_OP_INFERSHAPE(LogSoftmaxGrad).InferShape(InferShapeForLogSoftmaxGrad);
} // namespace ops
