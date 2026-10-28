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
 * \file deformable_offsets_grad_infershape.cpp
 * \brief DeformableOffsetsGrad 算子输出 shape 推导实现
 *
 * 输出 shape 推导规则：
 *   - grad_x 跟随 x 输入
 *   - grad_offsets 跟随 offsets 输入
 *
 * 参考逻辑：canndev rt1.0 DeformableOffsetsGradInfer（grad_x 取 x 的 shape/shape range，
 * grad_offsets 取 offsets 的 shape/shape range）。
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace gert;

namespace ops {
static ge::graphStatus CopyShapeInput2OutputInDeformableOffsetsGrad(gert::InferShapeContext* context, size_t inputIdx,
                                                                    size_t outputIdx)
{
    auto inShape = context->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShape);
    auto outShape = context->GetOutputShape(outputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    *outShape = *inShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4DeformableOffsetsGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4DeformableOffsetsGrad");
    constexpr size_t inputXIdx = 1;
    constexpr size_t inputOffsetsIdx = 2;
    constexpr size_t outputGradXIdx = 0;
    constexpr size_t outputGradOffsetsIdx = 1;
    OP_CHECK_IF(CopyShapeInput2OutputInDeformableOffsetsGrad(context, inputXIdx, outputGradXIdx) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "copy input x shape to output grad_x failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CopyShapeInput2OutputInDeformableOffsetsGrad(context, inputOffsetsIdx, outputGradOffsetsIdx) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "copy input offsets shape to output grad_offsets failed"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "End to do InferShape4DeformableOffsetsGrad");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DeformableOffsetsGrad).InferShape(InferShape4DeformableOffsetsGrad);
} // namespace ops
