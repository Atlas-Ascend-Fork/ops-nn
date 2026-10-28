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
 * \file softmax_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_elewise_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace Ops::Base;

namespace {
const size_t IR_ATTR_NUM_TWO = 2;
} // namespace

using namespace ge;
namespace ops {
static graphStatus InferDtype4SoftmaxV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferDtype4SoftmaxV2 enter");

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "softmax_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto input_dtype = context->GetInputDataType(0);
    bool need_half_to_float = false;
    if (attrs->GetAttrNum() == IR_ATTR_NUM_TWO) {
        const bool* half_to_float = attrs->GetAttrPointer<bool>(1);
        need_half_to_float = half_to_float != nullptr && *half_to_float && input_dtype == DT_FLOAT16;
    }

    DataType output_dtype = need_half_to_float ? DT_FLOAT : input_dtype;
    context->SetOutputDataType(0, output_dtype);

    OP_LOGD(context->GetNodeName(), "set output dtype: %s", ToString(output_dtype).c_str());
    OP_LOGD(context->GetNodeName(), "InferDtype4SoftmaxV2 end");

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4SoftmaxV2(gert::InferShapeContext* context) { return InferShape4Elewise(context); }

IMPL_OP_INFERSHAPE(SoftmaxV2).InferShape(InferShape4SoftmaxV2).InferDataType(InferDtype4SoftmaxV2);
} // namespace ops
