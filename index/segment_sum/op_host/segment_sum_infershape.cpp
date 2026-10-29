/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file segment_sum_infershape.cpp
 * \brief segment_sum infershape
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/const_util.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_X_IDX = 0;
static constexpr size_t INPUT_SEGMENT_IDS_IDX = 1;
static constexpr size_t OUTPUT_Y_IDX = 0;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

static ge::graphStatus InferShape4SegmentSum(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "***runtime2.0 inferShape***");
    OP_LOGD(context->GetNodeName(), "Begin to do SegmentSum InferShape.");
    const gert::Shape* x_shape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    const gert::Shape* segment_ids_shape = context->GetInputShape(INPUT_SEGMENT_IDS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, segment_ids_shape);
    gert::Shape* output_shape = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

    if (Ops::Base::IsUnknownRank(*x_shape)) {
        OP_LOGD(context->GetNodeName(), "x_shape is UnknownRank, set output_shape to (-2, )");
        Ops::Base::SetUnknownRank(*output_shape);
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownRank(*segment_ids_shape)) {
        OP_LOGD(context->GetNodeName(), "segment_ids_shape is UnknownRank, set output_shape to (-2, )");
        Ops::Base::SetUnknownRank(*output_shape);
        return ge::GRAPH_SUCCESS;
    }

    int64_t num_segments = UNKNOWN_DIM_VALUE_;
    auto segment_ids_tensor = context->GetInputTensor(INPUT_SEGMENT_IDS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, segment_ids_tensor);
    int64_t segment_ids_size = segment_ids_shape->GetShapeSize();
    if (segment_ids_size > 0) {
        if (segment_ids_tensor->GetDataType() == ge::DT_INT32) {
            auto segmentIdsData = segment_ids_tensor->GetData<int32_t>();
            if (segmentIdsData != nullptr) {
                num_segments = static_cast<int64_t>(segmentIdsData[segment_ids_size - 1]) + 1;
            } else {
                OP_LOGW(context->GetNodeName(), "segment_ids data is not ready, keep num_segments as unknown dim.");
            }
        } else if (segment_ids_tensor->GetDataType() == ge::DT_INT64) {
            auto segmentIdsData = segment_ids_tensor->GetData<int64_t>();
            if (segmentIdsData != nullptr) {
                num_segments = static_cast<int64_t>(segmentIdsData[segment_ids_size - 1]) + 1;
            } else {
                OP_LOGW(context->GetNodeName(), "segment_ids data is not ready, keep num_segments as unknown dim.");
            }
        }
    } else if (segment_ids_size == 0) {
        num_segments = 0;
    } else {
        // segment_ids含未知维（-1）时GetShapeSize()为负，输出首维保持未知
        num_segments = UNKNOWN_DIM_VALUE_;
    }

    const size_t x_rank = x_shape->GetDimNum();
    output_shape->SetDimNum(x_rank);
    output_shape->SetDim(0, num_segments);
    for (size_t i = 1; i < x_rank; i++) {
        output_shape->SetDim(i, x_shape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End to do SegmentSum InferShape.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SegmentSum).InferShape(InferShape4SegmentSum).InputsDataDependency({INPUT_SEGMENT_IDS_IDX});

} // namespace ops
