/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

#include <string>

using namespace ge;

namespace ops {

static std::string ShapeToString(const gert::Shape& shape)
{
    std::string result = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            result += ",";
        }
        result += std::to_string(shape.GetDim(i));
    }
    result += "]";
    return result;
}

static ge::graphStatus InferShape4AscendRequant(gert::InferShapeContext* context)
{
    const gert::Shape* input_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);

    const gert::Shape* req_scale_shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, req_scale_shape);

    gert::Shape* output_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    const auto is_unknown_rank = [](const gert::Shape& s) -> bool {
        return s.GetDimNum() == 1U && s.GetDim(0U) == ge::UNKNOWN_DIM_NUM;
    };
    const bool xUnknownRank = is_unknown_rank(*input_shape);
    const bool scaleUnknownRank = is_unknown_rank(*req_scale_shape);

    if (!xUnknownRank && (input_shape->GetDimNum() < 1U || input_shape->GetDimNum() > 8U)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                 (std::to_string(input_shape->GetDimNum()) + "D").c_str(),
                                                 "The rank of x must be in [1, 8]");
        return ge::GRAPH_FAILED;
    }
    if (!scaleUnknownRank && req_scale_shape->GetDimNum() > 8U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "req_scale",
                                                 (std::to_string(req_scale_shape->GetDimNum()) + "D").c_str(),
                                                 "The rank of req_scale must be in [0, 8]");
        return ge::GRAPH_FAILED;
    }

    if (!xUnknownRank && !scaleUnknownRank && req_scale_shape->GetDimNum() > input_shape->GetDimNum()) {
        const std::string ranks = std::to_string(req_scale_shape->GetDimNum()) + "D, " +
                                  std::to_string(input_shape->GetDimNum()) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "req_scale, x", ranks.c_str(),
            "The rank of req_scale must not exceed the rank of x, req_scale is not broadcastable to x");
        return ge::GRAPH_FAILED;
    }

    if (!xUnknownRank && !scaleUnknownRank) {
        const size_t xRank = input_shape->GetDimNum();
        const size_t scaleRank = req_scale_shape->GetDimNum();
        if (xRank < 1U || scaleRank > xRank) {
            const std::string ranks = std::to_string(scaleRank) + "D, " + std::to_string(xRank) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), "req_scale, x", ranks.c_str(),
                "The rank of req_scale must be in [0, rank(x)], req_scale is not broadcastable to x");
            return ge::GRAPH_FAILED;
        }
        const auto is_unknown_dim = [](int64_t dim) -> bool { return dim == ge::UNKNOWN_DIM; };
        for (size_t i = 0; i < scaleRank; i++) {
            const int64_t xDim = input_shape->GetDim(xRank - scaleRank + i);
            const int64_t scaleDim = req_scale_shape->GetDim(i);
            if (is_unknown_dim(xDim) || is_unknown_dim(scaleDim)) {
                continue;
            }
            if (scaleDim != 1 && scaleDim != xDim) {
                const std::string shapes = ShapeToString(*req_scale_shape) + ", " + ShapeToString(*input_shape);
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "req_scale, x", shapes.c_str(),
                                                       "Each dim of req_scale must be 1 or equal to the "
                                                       "corresponding x dim, req_scale must broadcast to x");
                return ge::GRAPH_FAILED;
            }
        }
    }

    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AscendRequant).InferShape(InferShape4AscendRequant);

} // namespace ops
