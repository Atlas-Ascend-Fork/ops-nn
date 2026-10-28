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
 * \file situ_glu_grad_infershape.cpp
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
constexpr size_t GRADY_INDEX = 0;
constexpr size_t X_INDEX = 1;
constexpr size_t GRADX_INDEX = 0;
constexpr size_t ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForSituGluGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForSituGluGrad");
    auto xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu_grad" : (context)->GetNodeName()),
            "x"),
        return ge::GRAPH_FAILED);
    auto gradXShape = context->GetOutputShape(GRADX_INDEX);
    if (gradXShape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu_grad" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object gradXShape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object gradXShape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu_grad" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    // grad_x shape is the same as x shape
    *gradXShape = *xShape;
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForSituGluGrad");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSituGluGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForSituGluGrad");
    const ge::DataType dtype = context->GetInputDataType(X_INDEX);
    ge::graphStatus ret = context->SetOutputDataType(0, dtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForSituGluGrad");
    return ret;
}

IMPL_OP_INFERSHAPE(SituGluGrad).InferShape(InferShapeForSituGluGrad).InferDataType(InferDataTypeForSituGluGrad);
} // namespace ops
