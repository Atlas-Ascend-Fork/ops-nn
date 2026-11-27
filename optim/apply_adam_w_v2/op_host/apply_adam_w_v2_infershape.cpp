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
 * \file apply_adam_w_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4ApplyAdamWV2(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4ApplyAdamWV2");
    const gert::Shape* varShape = context->GetInputShape(0);
    const gert::Shape* mShape = context->GetInputShape(1);
    const gert::Shape* vShape = context->GetInputShape(2);
    const gert::Shape* gradShape = context->GetInputShape(3);
    const gert::Shape* stepShape = context->GetInputShape(4);
    const gert::Shape* mgnShape = context->GetOptionalInputShape(5);
    OP_CHECK_IF(varShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("ApplyAdamWV2", "input var"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(mShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("ApplyAdamWV2", "input m"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("ApplyAdamWV2", "input v"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gradShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("ApplyAdamWV2", "input grad"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stepShape == nullptr, OP_LOGE_WITH_INVALID_INPUT("ApplyAdamWV2", "input step"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*mShape != *varShape || *vShape != *varShape || *gradShape != *varShape,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("ApplyAdamWV2", "var, m, v, grad",
                                                       Ops::Base::ToString({varShape, mShape, vShape, gradShape}),
                                                       "The shapes of m, v, grad must be the same as the shape of var"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(stepShape->GetShapeSize() != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("ApplyAdamWV2", "step", Ops::Base::ToString(*stepShape),
                                                      "Step should be a scalar(0D) or have shape size 1"),
                return ge::GRAPH_FAILED);
    if (mgnShape != nullptr && *mgnShape != *varShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("ApplyAdamWV2", "var, max_grad_norm",
                                               Ops::Base::ToString({varShape, mgnShape}),
                                               "The shape of max_grad_norm must be the same as the shape of var");
        return ge::GRAPH_FAILED;
    }
    // 本算子为原地写回（无独立输出声明），InferShape 仅做输入合法性校验，不推导输出
    OP_LOGD(context, "End to do InferShape4ApplyAdamWV2");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4ApplyAdamWV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4ApplyAdamWV2");
    // 原地算子无输出 dtype 推导，仅校验核心输入 dtype 一致性（m/v 与 var 同 dtype）
    auto varDtype = context->GetInputDataType(0);
    auto mDtype = context->GetInputDataType(1);
    auto vDtype = context->GetInputDataType(2);
    if (mDtype != varDtype || vDtype != varDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "ApplyAdamWV2", "var, m, v",
            Ops::Base::ToString(varDtype) + ", " + Ops::Base::ToString(mDtype) + ", " + Ops::Base::ToString(vDtype),
            "The dtypes of m and v must be the same as the dtype of var");
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdamWV2).InferShape(InferShape4ApplyAdamWV2).InferDataType(InferDataType4ApplyAdamWV2);
} // namespace ops
