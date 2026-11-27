/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Runtime shape inference for NormalizeBatch
//   (arch35: ascend950 / ascend960dt / ascend960pr / ascend350).
//   output_y.shape = input_x.shape (n, c, d)；seq_len 形状 (n,) 一致性校验；
//   dtype 推导位于 op_graph，属性值域校验由 tiling 承载。

#include "register/op_impl_registry.h"             // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h" // InferShapeContext, gert::Shape
#include "op_common/log/log.h"                     // OP_CHECK_NULL_WITH_CONTEXT macro
#include "util/shape_util.h"                       // IsUnknownRank, SetUnknownRank

using namespace ge;

namespace ops {

static constexpr size_t INPUT_X_IDX = 0;
static constexpr size_t INPUT_SEQ_LEN_IDX = 1;
static constexpr size_t OUTPUT_Y_IDX = 0;

static constexpr size_t EXPECTED_RANK = 3; // (n, c, d)

// InferShape4NormalizeBatch(context) — shape inference function
//
// input_x rank-3 gate + identity copy: output shape = input shape；
// seq_len rank-1 且第 0 维与 input_x 的 n 一致。
static ge::graphStatus InferShape4NormalizeBatch(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        // 判空分支无 context 可传，算子名字符串豁免（What/Where/Why 齐备，不留静默失败出口）
        OP_LOGE("NormalizeBatch", "InferShape4NormalizeBatch context is nullptr, check the graph builder caller.");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* inputShape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // 动态 shape（-1/-2）交给框架展开后再校验
    if (!Ops::Base::IsUnknownRank(*inputShape)) {
        OP_CHECK_IF(inputShape->GetDimNum() != EXPECTED_RANK,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input_x",
                                                             std::to_string(inputShape->GetDimNum()).c_str(),
                                                             "rank must be 3 (n, c, d)"),
                    return ge::GRAPH_FAILED);
        for (size_t i = 0; i < inputShape->GetDimNum(); ++i) {
            OP_CHECK_IF(inputShape->GetDim(i) < 0,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input_x",
                                                                 std::to_string(inputShape->GetDim(i)).c_str(),
                                                                 "dim must be static (non-negative)"),
                        return ge::GRAPH_FAILED);
        }
    }

    // seq_len 形状 (n,) 一致性
    const gert::Shape* seqShape = context->GetInputShape(INPUT_SEQ_LEN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqShape);
    if (!Ops::Base::IsUnknownRank(*seqShape)) {
        OP_CHECK_IF(
            seqShape->GetDimNum() != 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "seq_len",
                                                     std::to_string(seqShape->GetDimNum()).c_str(), "rank must be 1"),
            return ge::GRAPH_FAILED);
        if (!Ops::Base::IsUnknownRank(*inputShape) && seqShape->GetDim(0) >= 0) {
            OP_CHECK_IF(seqShape->GetDim(0) != inputShape->GetDim(0),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "input_x, seq_len",
                                                               ("input_x n=" + std::to_string(inputShape->GetDim(0)) +
                                                                ", seq_len dim0=" + std::to_string(seqShape->GetDim(0)))
                                                                   .c_str(),
                                                               "seq_len shape (n,) must match input_x dim0"),
                        return ge::GRAPH_FAILED);
        }
    }

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NormalizeBatch).InferShape(InferShape4NormalizeBatch);

} // namespace ops
