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
 * \file sigmoid_cross_entropy_with_logits_v2_infershape.cpp
 * \brief sigmoid_cross_entropy_with_logits_v2 infershape
 */
#include <cstring>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t ATTR_INDEX_REDUCTION = 0;
constexpr char DEFAULT_REDUCTION[] = "mean";
} // namespace

static ge::graphStatus InferShape4SigmoidCrossEntropyWithLogitsV2(gert::InferShapeContext* context)
{
    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* reduction = attrs->GetAttrPointer<char>(ATTR_INDEX_REDUCTION);
    if (reduction == nullptr) {
        reduction = DEFAULT_REDUCTION;
    }

    if (std::strcmp(reduction, "none") == 0) {
        const gert::Shape* predictShape = context->GetInputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
        *outputShape = *predictShape;
    } else if (std::strcmp(reduction, "mean") == 0 || std::strcmp(reduction, "sum") == 0) {
        outputShape->SetDimNum(0);
    } else {
        OP_LOGE(context->GetNodeName(),
                "The value of attr reduction[%s] must be none, mean or sum for SigmoidCrossEntropyWithLogitsV2.",
                reduction);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SigmoidCrossEntropyWithLogitsV2).InferShape(InferShape4SigmoidCrossEntropyWithLogitsV2);
} // namespace ops
