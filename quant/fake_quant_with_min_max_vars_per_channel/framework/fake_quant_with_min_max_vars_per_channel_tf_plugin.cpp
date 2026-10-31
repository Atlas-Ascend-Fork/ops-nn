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
 * \file fake_quant_with_min_max_vars_per_channel_tf_plugin.cpp
 * \brief
 */
#include "register/register.h"
#include "framework/plugin_util.h"

namespace domi {
// TensorFlow FakeQuantWithMinMaxVarsPerChannel: auto mapping, plus NHWC format setup on
// input/output for rank-4 tensors (NHWC is a 4-D layout; other ranks stay ND).
static Status ParseParamsFakeQuantWithMinMaxVarsPerChannel(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (AutoMappingFn(op_src, op) != SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "auto mapping failed.");
        return FAILED;
    }
    // 本算子按 ND/DT_FLOAT 注册且 x rank∈[1,8]；NHWC 固定为 4 维 [N,H,W,C]，
    // 故仅对 rank==4 的张量设置 NHWC，其它 rank 保持 ND（与同族 gradient 插件惯例一致）。
    auto tensor_in = op.GetInputDesc(0);
    const bool inNhwc = tensor_in.GetShape().GetDimNum() == 4U;
    tensor_in.SetOriginFormat(inNhwc ? ge::FORMAT_NHWC : ge::FORMAT_ND);
    tensor_in.SetFormat(inNhwc ? ge::FORMAT_NHWC : ge::FORMAT_ND);
    if (op.UpdateInputDesc(static_cast<uint32_t>(0), tensor_in) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update input format failed.");
        return FAILED;
    }
    auto tensor_out = op.GetOutputDesc(0);
    const bool outNhwc = tensor_out.GetShape().GetDimNum() == 4U;
    tensor_out.SetOriginFormat(outNhwc ? ge::FORMAT_NHWC : ge::FORMAT_ND);
    tensor_out.SetFormat(outNhwc ? ge::FORMAT_NHWC : ge::FORMAT_ND);
    if (op.UpdateOutputDesc(static_cast<uint32_t>(0), tensor_out) != ge::GRAPH_SUCCESS) {
        OP_LOGE(GetOpName(op).c_str(), "update output format failed.");
        return FAILED;
    }
    return SUCCESS;
}

REGISTER_CUSTOM_OP("FakeQuantWithMinMaxVarsPerChannel")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("FakeQuantWithMinMaxVarsPerChannel")
    .ParseParamsFn(ParseParamsFakeQuantWithMinMaxVarsPerChannel)
    .ImplyType(ImplyType::TVM);
} // namespace domi
