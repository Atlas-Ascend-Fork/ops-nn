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
 * \file soft_margin_loss_grad_infershape.cpp
 * \brief SoftMarginLossGrad 形状/类型推导：out = numpy broadcast(self, target, grad_output)
 */
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static bool BroadcastDim(int64_t a, int64_t b, int64_t& out)
{
    if (a == 1) {
        out = b;
        return true;
    }
    if (b == 1) {
        out = a;
        return true;
    }
    if (a < 0 || b < 0) {
        out = a; // 动态/未知维：按原语义传递，由动态 shape 阶段解析
        return true;
    }
    if (a == b) {
        out = a;
        return true;
    }
    return false; // 广播不兼容（a!=b 且均非 1）
}

static ge::graphStatus InferShape4SoftMarginLossGrad(gert::InferShapeContext* context)
{
    const gert::Shape* s0 = context->GetInputShape(0);
    const gert::Shape* s1 = context->GetInputShape(1);
    const gert::Shape* s2 = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, s0);
    OP_CHECK_NULL_WITH_CONTEXT(context, s1);
    OP_CHECK_NULL_WITH_CONTEXT(context, s2);
    gert::Shape* out = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out);

    size_t r = s0->GetDimNum();
    r = std::max(r, s1->GetDimNum());
    r = std::max(r, s2->GetDimNum());
    out->SetDimNum(r);
    auto dimAt = [](const gert::Shape* s, size_t i, size_t r) -> int64_t {
        size_t sr = s->GetDimNum();
        return (i + sr < r) ? 1 : s->GetDim(i + sr - r);
    };
    for (size_t i = 0; i < r; i++) {
        int64_t d = 0;
        if (!BroadcastDim(dimAt(s0, i, r), dimAt(s1, i, r), d) || !BroadcastDim(d, dimAt(s2, i, r), d)) {
            OP_LOGE(context->GetNodeName(), "broadcast incompatible for predict, label, dout");
            return ge::GRAPH_FAILED;
        }
        out->SetDim(i, d);
    }
    OP_LOGI(context->GetNodeName(), "[InferShape] output shape=%s", Ops::Base::ToString(*out).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftMarginLossGrad).InferShape(InferShape4SoftMarginLossGrad);

} // namespace ops
