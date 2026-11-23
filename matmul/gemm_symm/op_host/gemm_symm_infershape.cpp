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

namespace {
constexpr size_t kRank2 = 2;
constexpr size_t kRank3 = 3;

int64_t LogicalM(const gert::Shape* shape, bool transposed)
{
    const size_t rank = shape->GetDimNum();
    return transposed ? shape->GetDim(rank - 1) : shape->GetDim(rank - 2);
}

int64_t LogicalK(const gert::Shape* shape, bool transposed)
{
    const size_t rank = shape->GetDimNum();
    return transposed ? shape->GetDim(rank - 2) : shape->GetDim(rank - 1);
}
} // namespace

namespace ops {
static ge::graphStatus InferShapeGemmSymm(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* aShape = context->GetInputShape(0);
    const auto* bShape = context->GetInputShape(1);
    auto* yShape = context->GetOutputShape(0);
    const auto* attrs = context->GetAttrs();
    if (aShape == nullptr || bShape == nullptr || yShape == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* transA = attrs->GetAttrPointer<bool>(2);
    const auto* transB = attrs->GetAttrPointer<bool>(3);
    const bool aT = transA != nullptr && *transA;
    const bool bT = transB != nullptr && *transB;
    const size_t aRank = aShape->GetDimNum();
    if ((aRank != kRank2 && aRank != kRank3) || bShape->GetDimNum() != aRank) {
        return ge::GRAPH_FAILED;
    }
    const int64_t m = LogicalM(aShape, aT);
    const int64_t k = LogicalK(aShape, aT);
    if (LogicalK(bShape, !bT) != k || LogicalM(bShape, !bT) != m) {
        return ge::GRAPH_FAILED;
    }
    yShape->SetDimNum(aRank);
    size_t axis = 0;
    if (aRank == kRank3) {
        yShape->SetDim(0, aShape->GetDim(0));
        axis = 1;
    }
    yShape->SetDim(axis, m);
    yShape->SetDim(axis + 1, m);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGemmSymm(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
} // namespace ops

IMPL_OP_INFERSHAPE(GemmSymm).InferShape(ops::InferShapeGemmSymm).InferDataType(ops::InferDataTypeGemmSymm);
