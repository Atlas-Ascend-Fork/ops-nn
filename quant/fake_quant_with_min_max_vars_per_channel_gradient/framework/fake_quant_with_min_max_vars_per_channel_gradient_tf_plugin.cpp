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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"

namespace domi {
// TensorFlow FakeQuantWithMinMaxVarsPerChannelGradient: auto mapping plus NHWC format setup.
// NHWC only applies to rank-4 tensors; rank-1 inputs (min/max) and rank-1 outputs
// (backprops_wrt_min/backprops_wrt_max) must stay ND, otherwise the format is invalid for them.
static bool SetNhwcFormatForRank4Input(ge::Operator& op, size_t idx)
{
    auto tensor = op.GetInputDesc(idx);
    const bool isNhwc = tensor.GetShape().GetDimNum() == 4U;
    tensor.SetOriginFormat(isNhwc ? ge::FORMAT_NHWC : ge::FORMAT_ND);
    tensor.SetFormat(isNhwc ? ge::FORMAT_NHWC : ge::FORMAT_ND);
    if (op.UpdateInputDesc(idx, tensor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update input format failed.");
        return false;
    }
    return true;
}

static bool SetNhwcFormatForRank4Output(ge::Operator& op, size_t idx)
{
    auto tensor = op.GetOutputDesc(idx);
    const bool isNhwc = tensor.GetShape().GetDimNum() == 4U;
    tensor.SetOriginFormat(isNhwc ? ge::FORMAT_NHWC : ge::FORMAT_ND);
    tensor.SetFormat(isNhwc ? ge::FORMAT_NHWC : ge::FORMAT_ND);
    if (op.UpdateOutputDesc(idx, tensor) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update output format failed.");
        return false;
    }
    return true;
}

static Status ParseParamsFakeQuantWithMinMaxVarsPerChannelGradient(const google::protobuf::Message* op_src,
                                                                   ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    for (size_t i = 0; i < op.GetInputsSize(); i++) {
        if (!SetNhwcFormatForRank4Input(op, i)) {
            return FAILED;
        }
    }
    for (size_t i = 0; i < op.GetOutputsSize(); i++) {
        if (!SetNhwcFormatForRank4Output(op, i)) {
            return FAILED;
        }
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("FakeQuantWithMinMaxVarsPerChannelGradient")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FakeQuantWithMinMaxVarsPerChannelGradient")
    .ParseParamsFn(ParseParamsFakeQuantWithMinMaxVarsPerChannelGradient)
    .ImplyType(ImplyType::TVM);
} // namespace domi
