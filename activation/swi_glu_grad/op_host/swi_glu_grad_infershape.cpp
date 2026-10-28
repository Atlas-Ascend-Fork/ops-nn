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
 * \file swi_glu_grad_infer.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "error_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeForSwiGluGrad(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(1);
    OP_CHECK_IF(
        x1_shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "swi_glu_grad" : (context)->GetNodeName()),
            "x"),
        return ge::GRAPH_FAILED);

    gert::Shape* output_shape_1 = context->GetOutputShape(0);
    if (output_shape_1 == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "swi_glu_grad" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object output_shape_1 is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object output_shape_1 is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    *output_shape_1 = *x1_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSwiGluGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, dtype);
    return ret;
}

IMPL_OP_INFERSHAPE(SwiGluGrad).InferShape(InferShapeForSwiGluGrad).InferDataType(InferDataTypeForSwiGluGrad);
} // namespace ops
