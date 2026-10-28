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
 * \file situ_glu_infershape.cpp
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
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
constexpr size_t GLU_ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForSituGlu(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForSituGlu");
    auto xShape = context->GetInputShape(GLU_IN_X);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu" : (context)->GetNodeName()), "x"),
        return ge::GRAPH_FAILED);
    auto yShape = context->GetOutputShape(GLU_OUT_Y);
    if (yShape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object yShape is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object yShape is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    auto splitDimPtr = attrs->GetAttrPointer<int64_t>(GLU_ATTR_DIM);
    if (splitDimPtr == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu" : (context)->GetNodeName()));
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
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }
    auto splitDim = *splitDimPtr;
    if (splitDim < 0) {
        splitDim += xShape->GetDimNum();
    }
    if (splitDim < 0 || splitDim >= static_cast<int64_t>(xShape->GetDimNum())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "SituGlu", "dim", std::to_string(*splitDimPtr),
            ("The value of attr [dim] must be in the range [-" + std::to_string(xShape->GetDimNum()) + ", " +
             std::to_string(xShape->GetDimNum() - 1) + "], but got [" + std::to_string(splitDim) + "]."));
        return GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to generate yShape");
    *yShape = *xShape;
    // dynamic shape
    if (xShape->GetDim(splitDim) == -1) {
        return ge::GRAPH_SUCCESS;
    }
    if (xShape->GetDim(splitDim) < 0 || xShape->GetDim(splitDim) % SPLIT_NUM != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "SituGlu", "x", Ops::Base::ToString(*xShape),
            ("The shape [" + std::string(Ops::Base::ToString(*xShape).c_str()) + "] is not divisible by 2."));
        return GRAPH_FAILED;
    }

    yShape->SetDim(splitDim, xShape->GetDim(splitDim) / SPLIT_NUM);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeForSituGlu");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForSituGlu(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForSituGlu");
    const ge::DataType dtype = context->GetInputDataType(0);
    ge::graphStatus ret = context->SetOutputDataType(0, dtype);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForSituGlu");
    return ret;
}

IMPL_OP_INFERSHAPE(SituGlu).InferShape(InferShapeForSituGlu).InferDataType(InferDataTypeForSituGlu);
} // namespace ops
