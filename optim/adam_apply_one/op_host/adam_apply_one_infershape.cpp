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
 * \file adam_apply_one_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "infershape_broadcast_util.h"

using namespace Ops::Base;
using namespace ge;
namespace ops {
constexpr size_t INPUT0_IDX = 0;
constexpr size_t OUTPUT0_IDX = 0;
constexpr size_t OUTPUT1_IDX = 1;
constexpr size_t OUTPUT2_IDX = 2;
constexpr size_t INPUT_NUM = 10;
constexpr const char* kInputNames[INPUT_NUM] = {"input0", "input1", "input2", "input3", "input4",
                                                "mul0_x", "mul1_x", "mul2_x", "mul3_x", "add2_y"};

static ge::graphStatus InferShape4AdamApplyOne(gert::InferShapeContext* context)
{
    auto output0_shape = context->GetOutputShape(OUTPUT0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output0_shape);
    auto output1_shape = context->GetOutputShape(OUTPUT1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output1_shape);
    auto output2_shape = context->GetOutputShape(OUTPUT2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output2_shape);

    auto input0_shape = context->GetInputShape(INPUT0_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0_shape);
    gert::Shape broadcastShape = *input0_shape;
    for (size_t idx = 1; idx < INPUT_NUM; idx++) {
        auto curShape = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, curShape);
        gert::Shape tmpShape;
        OP_CHECK_IF(!BroadcastShape(&broadcastShape, curShape, &tmpShape),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), kInputNames[idx],
                        (Ops::Base::ToString(*curShape) + " and " + Ops::Base::ToString(broadcastShape)).c_str(),
                        "The shape of this input should be broadcastable with previous inputs"),
                    return ge::GRAPH_FAILED);
        broadcastShape = tmpShape;
    }

    *output0_shape = broadcastShape;
    *output1_shape = broadcastShape;
    *output2_shape = broadcastShape;

    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AdamApplyOne).InferShape(InferShape4AdamApplyOne);
} // namespace ops
