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
 * \file apply_adam_w_infershape.cpp
 * \brief ApplyAdamW 算子输出 shape 推导实现
 *
 * 输出 shape 推导规则：
 *   - var / m / v 为 ref 输出，分别跟随同名输入
 *
 * 参考逻辑：canndev rt1.0 ApplyAdamWInferShape（var->var、m->m、v->v 逐对拷贝）。
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace gert;

namespace ops {
static ge::graphStatus CopyShapeInput2OutputInApplyAdamW(gert::InferShapeContext* context, size_t inputIdx,
                                                         size_t outputIdx)
{
    auto inShape = context->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(outputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ApplyAdamW(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4ApplyAdamW");
    constexpr size_t inputVarIdx = 0;
    constexpr size_t inputMIdx = 1;
    constexpr size_t inputVIdx = 2;
    constexpr size_t outputVarIdx = 0;
    constexpr size_t outputMIdx = 1;
    constexpr size_t outputVIdx = 2;
    OP_CHECK_IF(CopyShapeInput2OutputInApplyAdamW(context, inputVarIdx, outputVarIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "copy input var shape to output var failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CopyShapeInput2OutputInApplyAdamW(context, inputMIdx, outputMIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "copy input m shape to output m failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CopyShapeInput2OutputInApplyAdamW(context, inputVIdx, outputVIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "copy input v shape to output v failed"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "End to do InferShape4ApplyAdamW");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyAdamW).InferShape(InferShape4ApplyAdamW);
} // namespace ops
