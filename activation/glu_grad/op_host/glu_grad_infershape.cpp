/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace {
constexpr size_t GLU_GRAD_IN_GRAD_OUT = 0;
constexpr size_t GLU_GRAD_IN_SELF = 1;
constexpr size_t GLU_GRAD_OUT = 0;
constexpr size_t GLU_GRAD_ATTR_DIM = 0;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForGluGrad(gert::InferShapeContext* context)
{
    auto self_shape = context->GetInputShape(GLU_GRAD_IN_SELF);
    OP_CHECK_IF(
        self_shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "glu_grad" : (context)->GetNodeName()),
            "self"),
        return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(GLU_GRAD_OUT);
    if (out_shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "glu_grad" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object out_shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object out_shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "glu_grad" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(GLU_GRAD_ATTR_DIM);
    if (split_dim_ptr == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "glu_grad" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object split_dim_ptr is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object split_dim_ptr is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    *out_shape = *self_shape;
    size_t self_dim_num = self_shape->GetDimNum();

    OP_CHECK_IF(self_dim_num == 0,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    "GLUGrad", "self", std::to_string(self_dim_num),
                    ("input self dim num is " + std::to_string(self_dim_num) + ", not support scalar")),
                return GRAPH_FAILED);

    OP_CHECK_IF(Ops::Base::IsUnknownRank(*self_shape), OP_LOGD("GLUGrad", "input self is unknown rank, no need check."),
                return GRAPH_SUCCESS);

    auto split_dim = *split_dim_ptr;
    if (split_dim < 0) {
        split_dim += static_cast<int64_t>(self_dim_num);
    }

    OP_CHECK_IF((split_dim < 0 || split_dim >= static_cast<int64_t>(self_dim_num)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    "GLUGrad", "dim", std::to_string(*split_dim_ptr),
                    ("The value of attr [dim] must be in the range [-" + std::to_string(self_dim_num) + ", " +
                     std::to_string(self_dim_num - 1) + "], but got [" + std::to_string(split_dim) + "].")),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GLUGrad).InferShape(InferShapeForGluGrad);
} // namespace ops
