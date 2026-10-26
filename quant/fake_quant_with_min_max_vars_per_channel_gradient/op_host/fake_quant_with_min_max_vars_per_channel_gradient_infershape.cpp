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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_IDX_GRADIENTS = 0;
static constexpr size_t INPUT_IDX_X = 1;
static constexpr size_t INPUT_IDX_MIN = 2;
static constexpr size_t INPUT_IDX_MAX = 3;
static constexpr size_t OUTPUT_IDX_BPX = 0;
static constexpr size_t OUTPUT_IDX_BPMIN = 1;
static constexpr size_t OUTPUT_IDX_BPMAX = 2;

static ge::graphStatus FakeQuantWithMinMaxVarsPerChannelGradientInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do FakeQuantWithMinMaxVarsPerChannelGradientInferShape");

    const gert::Shape* gradShape = context->GetInputShape(INPUT_IDX_GRADIENTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradShape);
    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* minShape = context->GetInputShape(INPUT_IDX_MIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, minShape);
    const gert::Shape* maxShape = context->GetInputShape(INPUT_IDX_MAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxShape);

    gert::Shape* bpxShape = context->GetOutputShape(OUTPUT_IDX_BPX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bpxShape);
    gert::Shape* bpMinShape = context->GetOutputShape(OUTPUT_IDX_BPMIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, bpMinShape);
    gert::Shape* bpMaxShape = context->GetOutputShape(OUTPUT_IDX_BPMAX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bpMaxShape);

    // Cross-input constraints (README): x rank in [1, 8]; gradients shape == x shape;
    // min/max are 1-D with length == x last dim C and equal lengths. The kernel indexes
    // gradients/x/min/max linearly by channelNum_ = C, so a mismatch would cause out-of-range
    // device reads/writes. Unknown rank (dim_num == 0 on a dynamic-rank graph) is skipped here
    // and re-checked at tiling with the concrete shape; unknown dims are skipped as well.
    const size_t xRank = xShape->GetDimNum();
    if (Ops::Base::IsUnknownRank(*xShape)) {
        // True unknown rank (dynamic-rank graph): relationships are checked at tiling.
        *bpxShape = *xShape;
        *bpMinShape = *minShape;
        *bpMaxShape = *maxShape;
        return ge::GRAPH_SUCCESS;
    }
    // Known rank, including the illegal rank-0 scalar: x rank must be in [1, 8].
    if (xRank < 1U || xRank > 8U) {
        OP_LOGE(context, "x rank %zu out of range [1, 8]", xRank);
        return ge::GRAPH_FAILED;
    }
    const size_t gradRank = gradShape->GetDimNum();
    if (!Ops::Base::IsUnknownRank(*gradShape)) {
        if (gradRank != xRank) {
            OP_LOGE(context, "gradients rank %zu != x rank %zu", gradRank, xRank);
            return ge::GRAPH_FAILED;
        }
        for (size_t i = 0U; i < xRank; ++i) {
            const int64_t xDim = xShape->GetDim(i);
            const int64_t gDim = gradShape->GetDim(i);
            if (xDim >= 0 && gDim >= 0 && xDim != gDim) {
                OP_LOGE(context, "gradients dim %zu (%ld) != x dim (%ld)", i, gDim, xDim);
                return ge::GRAPH_FAILED;
            }
        }
    }
    const int64_t channelNum = xShape->GetDim(static_cast<int64_t>(xRank) - 1);
    const size_t minRank = minShape->GetDimNum();
    const size_t maxRank = maxShape->GetDimNum();
    const bool minUnknown = Ops::Base::IsUnknownRank(*minShape);
    const bool maxUnknown = Ops::Base::IsUnknownRank(*maxShape);
    // A known rank must be 1 (rank-0 scalar min/max is illegal); unknown rank is deferred to
    // tiling, which only sees concrete shapes.
    if ((!minUnknown && minRank != 1U) || (!maxUnknown && maxRank != 1U)) {
        OP_LOGE(context, "min/max must be 1-D, got min rank %zu max rank %zu", minRank, maxRank);
        return ge::GRAPH_FAILED;
    }
    if (!minUnknown && !maxUnknown) {
        const int64_t minLen = minShape->GetDim(0);
        const int64_t maxLen = maxShape->GetDim(0);
        if (channelNum >= 0 && minLen >= 0 && maxLen >= 0 &&
            (minLen != channelNum || maxLen != channelNum || minLen != maxLen)) {
            OP_LOGE(context, "min/max length (%ld/%ld) must equal channel C (%ld)", minLen, maxLen, channelNum);
            return ge::GRAPH_FAILED;
        }
    }

    *bpxShape = *xShape;
    *bpMinShape = *minShape;
    *bpMaxShape = *maxShape;

    OP_LOGI(context->GetNodeName(),
            "FakeQuantWithMinMaxVarsPerChannelGradient output shapes: bpx=%s, bpMin=%s, bpMax=%s.",
            Ops::Base::ToString(*bpxShape).c_str(), Ops::Base::ToString(*bpMinShape).c_str(),
            Ops::Base::ToString(*bpMaxShape).c_str());
    OP_LOGD(context, "End to do FakeQuantWithMinMaxVarsPerChannelGradientInferShape");
    return ge::GRAPH_SUCCESS;
}

static graphStatus FakeQuantWithMinMaxVarsPerChannelGradientInferDtype(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "FakeQuantWithMinMaxVarsPerChannelGradientInferDtype enter");
    context->SetOutputDataType(OUTPUT_IDX_BPX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_IDX_BPMIN, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_IDX_BPMAX, ge::DT_FLOAT);
    OP_LOGD(context, "FakeQuantWithMinMaxVarsPerChannelGradientInferDtype end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FakeQuantWithMinMaxVarsPerChannelGradient)
    .InferShape(FakeQuantWithMinMaxVarsPerChannelGradientInferShape)
    .InferDataType(FakeQuantWithMinMaxVarsPerChannelGradientInferDtype);
} // namespace ops
