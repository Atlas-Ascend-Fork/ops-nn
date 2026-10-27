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
 * \file embedding_dense_grad_v2_infershape.cpp
 * \brief
 */
#include <vector>
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace {
constexpr size_t EMBEDDING_DENSE_GRAD_IN_GRAD = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_IN_INDICES = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_IN_POSIDX = 2;
constexpr size_t EMBEDDING_DENSE_GRAD_OUT_Y = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ = 2;
} // namespace

namespace ops {
static ge::graphStatus ReportInferShapeFailure(const char* opName, const char* reason)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {opName, reason};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", reason);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus InferShapeForEmbeddingDenseGradV2(gert::InferShapeContext* context)
{
    auto grad_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_GRAD);
    OP_CHECK_IF(grad_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "grad"), return GRAPH_FAILED);
    auto indices_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_INDICES);
    OP_CHECK_IF(indices_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "sort_indices"),
                return GRAPH_FAILED);
    auto pos_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_POSIDX);
    OP_CHECK_IF(pos_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "pos_idx"),
                return GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(0);
    if (out_shape == nullptr) {
        return ReportInferShapeFailure(context->GetNodeName(), "output y shape is null");
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ReportInferShapeFailure(context->GetNodeName(), "attributes are null");
    }

    auto num_weights = attrs->GetAttrPointer<int64_t>(EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS);
    if (num_weights == nullptr) {
        return ReportInferShapeFailure(context->GetNodeName(), "num_weights attribute is null");
    }
    auto padding_idx = attrs->GetAttrPointer<int64_t>(EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX);
    if (padding_idx == nullptr) {
        return ReportInferShapeFailure(context->GetNodeName(), "padding_idx attribute is null");
    }
    auto scale_grad_by_freq = attrs->GetAttrPointer<bool>(EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ);
    if (scale_grad_by_freq == nullptr) {
        return ReportInferShapeFailure(context->GetNodeName(), "scale_grad_by_freq attribute is null");
    }

    int64_t output_shape_len = 2;
    int64_t input_shape_len = grad_shape->GetDimNum();
    out_shape->SetDimNum(output_shape_len);
    out_shape->SetDim(0, *num_weights);
    out_shape->SetDim(1, grad_shape->GetDim(input_shape_len - 1));

    return GRAPH_SUCCESS;
}

static graphStatus InferDtypeForEmbeddingDenseGradV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDtypeForEmbeddingDenseGrad rt2.0");

    auto grad_dtype = context->GetInputDataType(EMBEDDING_DENSE_GRAD_IN_GRAD);

    context->SetOutputDataType(EMBEDDING_DENSE_GRAD_OUT_Y, grad_dtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDtypeForEmbeddingDenseGrad");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingDenseGradV2)
    .InferShape(InferShapeForEmbeddingDenseGradV2)
    .InferDataType(InferDtypeForEmbeddingDenseGradV2);
} // namespace ops
