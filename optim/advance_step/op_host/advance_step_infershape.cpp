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
 * \file advance_step_infershape.cc
 * \brief
 */
#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t IDX_2 = 2;
static constexpr int64_t IDX_3 = 3;
static constexpr int64_t IDX_4 = 4;
static constexpr int64_t IDX_5 = 5;

static constexpr size_t SIZE_2 = 2;

static ge::graphStatus InferShape4AdvanceStep(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4AdvanceStep");

    // get input shapes
    auto x1Shape = context->GetInputShape(IDX_0);
    OP_CHECK_IF(
        x1Shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()),
            "input_tokens"),
        return ge::GRAPH_FAILED);

    // get output shapes
    auto y1Shape = context->GetOutputShape(IDX_0);
    auto y2Shape = context->GetOutputShape(IDX_1);
    auto y3Shape = context->GetOutputShape(IDX_2);
    auto y4Shape = context->GetOutputShape(IDX_3);
    if (y1Shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y1Shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y1Shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    if (y2Shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y2Shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y2Shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    if (y3Shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y3Shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y3Shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    if (y4Shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "advance_step" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object y4Shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object y4Shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    *y1Shape = *x1Shape;
    *y2Shape = *x1Shape;
    *y3Shape = *x1Shape;
    *y4Shape = *x1Shape;

    OP_LOGD(context, "End to do InferShape4AdvanceStep");
    OP_LOGI(context->GetNodeName(), "[InferShape] output0 shape=%s", Ops::Base::ToString(*y1Shape).c_str());
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4AdvanceStep(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4AdvanceStep");

    auto input_dtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, input_dtype);
    context->SetOutputDataType(IDX_1, input_dtype);
    context->SetOutputDataType(IDX_2, input_dtype);
    context->SetOutputDataType(IDX_3, input_dtype);

    OP_LOGD(context, "End to do InferDataType4AdvanceStep");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AdvanceStep).InferShape(InferShape4AdvanceStep).InferDataType(InferDataType4AdvanceStep);
} // namespace ops
