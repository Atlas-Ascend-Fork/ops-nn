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
 * \file gelu_grad_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "infershape_broadcast_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {
static ge::graphStatus CopyShapeInput2OutputWithIdx(gert::InferShapeContext* context, int64_t input_idx,
                                                    int64_t output_idx)
{
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_IF(
        in_shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "gelu_grad_v2" : (context)->GetNodeName()),
            (input_idx == 0 ? std::string("dy") :
                              (input_idx == 1 ? std::string("x") : "input[" + std::to_string(input_idx) + "]"))),
        return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(output_idx);
    if (out_shape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "gelu_grad_v2" : (context)->GetNodeName()));
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

static ge::graphStatus InferShapeForGeluGradV2(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do GeluGradV2InferShape");
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    if ((fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
         ge::GRAPH_SUCCESS)) {
        const std::string msgOpName = std::string(
            (context->GetNodeName() == nullptr ? "gelu_grad_v2" : context->GetNodeName()));
        const std::string msgReason = (std::string(__func__) + ": " + "Cannot get platform info!");
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
        return ge::GRAPH_FAILED;
    };
    OP_LOGD(context->GetNodeName(), "soc version is %s", platform_info.str_info.short_soc_version.c_str());
    if (platform_info.str_info.short_soc_version == "Ascend950") {
        const size_t inputCount = 2;
        std::vector<const gert::Shape*> to_broadcast_shapes(inputCount);
        for (size_t i = 0; i < inputCount; i++) {
            auto in_shape = context->GetInputShape(i);
            OP_CHECK_IF(
                in_shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context) == nullptr || (context)->GetNodeName() == nullptr ? "gelu_grad_v2" :
                                                                                   (context)->GetNodeName()),
                    (i == 0 ? std::string("dy") : (i == 1 ? std::string("x") : "input[" + std::to_string(i) + "]"))),
                return ge::GRAPH_FAILED);
            to_broadcast_shapes[i] = in_shape;
        }
        auto out_shape = context->GetOutputShape(0);
        if (out_shape == nullptr) {
            const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                           "gelu_grad_v2" :
                                                           (context)->GetNodeName()));
            OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                                   "InferShape framework object out_shape is unavailable; check the operator "
                                   "registration and context construction.");
            REPORT_PREDEFINED_ERR_MSG(
                "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                (std::vector<const char*>{msgOpName.c_str(),
                                          "InferShape framework object out_shape is unavailable; check the operator "
                                          "registration and context construction."}));
            return ge::GRAPH_FAILED;
        };

        OP_CHECK_IF(!BroadcastShape(to_broadcast_shapes, out_shape),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "dy, x",
                                                           (Ops::Base::ToString(*to_broadcast_shapes[0]) + ", " +
                                                            Ops::Base::ToString(*to_broadcast_shapes[1])),
                                                           "The shapes of dy and x must be broadcastable"),
                    return ge::GRAPH_FAILED);

        return ge::GRAPH_SUCCESS;
    }

    return CopyShapeInput2OutputWithIdx(context, 1, 0);
}

IMPL_OP_INFERSHAPE(GeluGradV2).InferShape(InferShapeForGeluGradV2);
} // namespace ops
