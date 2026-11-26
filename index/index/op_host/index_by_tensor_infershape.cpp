/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include "infershape_broadcast_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

namespace ops {
namespace {
constexpr size_t X_INDEX = 0;
constexpr size_t INDICES_INDEX = 1;
constexpr size_t MASK_INDEX = 0;
constexpr size_t Y_INDEX = 0;

ge::graphStatus ParseIndicesMask(gert::InferShapeContext* context, const gert::TypedContinuousVector<int64_t>& mask,
                                 size_t xRank, size_t indicesNum, size_t& insertAxis)
{
    size_t firstAxis = 0;
    size_t lastAxis = 0;
    size_t indexedAxes = 0;
    for (size_t axis = 0; axis < mask.GetSize(); ++axis) {
        const auto value = mask.GetData()[axis];
        OP_CHECK_IF(
            value != 0 && value != 1,
            OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "indices_mask", std::to_string(value).c_str(), "0 or 1"),
            return ge::GRAPH_FAILED);
        if (value == 1) {
            OP_CHECK_IF(axis >= xRank,
                        OP_LOGE(context->GetNodeName(), "Indexed axis %zu exceeds input rank %zu", axis, xRank),
                        return ge::GRAPH_FAILED);
            if (indexedAxes == 0) {
                firstAxis = axis;
            }
            lastAxis = axis;
            ++indexedAxes;
        }
    }
    OP_CHECK_IF(indexedAxes != indicesNum,
                OP_LOGE(context->GetNodeName(), "indices_mask selects %zu axes, but there are %zu indices", indexedAxes,
                        indicesNum),
                return ge::GRAPH_FAILED);
    // Consecutive indexed axes are replaced in place; otherwise the broadcast dimensions go first.
    insertAxis = lastAxis - firstAxis + 1 == indexedAxes ? firstAxis : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BroadcastIndices(gert::InferShapeContext* context, size_t indicesNum, gert::Shape& replacement)
{
    std::vector<const gert::Shape*> indicesShapes;
    for (size_t i = 0; i < indicesNum; ++i) {
        const auto* shape = context->GetDynamicInputShape(INDICES_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        indicesShapes.push_back(shape);
    }
    OP_CHECK_IF(!Ops::Base::BroadcastShape(indicesShapes, &replacement),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "IndexByTensor indices cannot broadcast"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void AssembleOutputShape(const gert::Shape& xShape, const gert::TypedContinuousVector<int64_t>& mask,
                         const gert::Shape& replacement, size_t insertAxis, gert::Shape& yShape)
{
    const size_t xRank = xShape.GetDimNum();
    yShape.SetDimNum(0);
    for (size_t axis = 0; axis < xRank; ++axis) {
        if (axis == insertAxis) {
            for (size_t i = 0; i < replacement.GetDimNum(); ++i) {
                yShape.AppendDim(replacement.GetDim(i));
            }
        }
        if (axis >= mask.GetSize() || mask.GetData()[axis] == 0) {
            yShape.AppendDim(xShape.GetDim(axis));
        }
    }
}

ge::graphStatus InferShapeIndexByTensor(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Native IndexByTensor V2 shape inference begins");
    const auto* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto* yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* mask = attrs->GetListInt(MASK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, mask);
    const size_t indicesNum = context->GetComputeNodeInputNum() - 1;
    const size_t xRank = xShape->GetDimNum();
    size_t insertAxis = 0;
    if (ParseIndicesMask(context, *mask, xRank, indicesNum, insertAxis) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (indicesNum == 0) {
        *yShape = *xShape;
        return ge::GRAPH_SUCCESS;
    }
    gert::Shape replacement;
    if (BroadcastIndices(context, indicesNum, replacement) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (Ops::Base::IsUnknownRank(replacement)) {
        Ops::Base::SetUnknownRank(*yShape);
        return ge::GRAPH_SUCCESS;
    }

    const size_t outputRank = xRank - indicesNum + replacement.GetDimNum();
    OP_CHECK_IF(outputRank > gert::Shape::kMaxDimNum,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "output rank", std::to_string(outputRank).c_str(),
                                          ("at most " + std::to_string(gert::Shape::kMaxDimNum)).c_str()),
                return ge::GRAPH_FAILED);

    AssembleOutputShape(*xShape, *mask, replacement, insertAxis, *yShape);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(IndexByTensor).InferShape(InferShapeIndexByTensor);
} // namespace ops
