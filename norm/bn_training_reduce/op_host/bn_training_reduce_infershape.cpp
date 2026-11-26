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

#include "util/shape_util.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4BNTrainingReduce(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(0);
    gert::Shape* sumShape = context->GetOutputShape(0);
    gert::Shape* squareSumShape = context->GetOutputShape(1);
    if (xShape == nullptr || xDesc == nullptr || sumShape == nullptr || squareSumShape == nullptr) {
        OP_LOGE(context, "BNTrainingReduce requires input x and two outputs.");
        return GRAPH_FAILED;
    }

    const bool isUnknownRank = xShape->GetDimNum() == 1 && xShape->GetDim(0) == UNKNOWN_DIM_NUM;
    if (isUnknownRank) {
        *sumShape = *xShape;
        *squareSumShape = *xShape;
        OP_LOGI(context->GetNodeName(), "[InferShape] output0 shape=%s, output1 shape=%s",
                Ops::Base::ToString(*sumShape).c_str(), Ops::Base::ToString(*squareSumShape).c_str());
        return GRAPH_SUCCESS;
    }

    const size_t rank = xShape->GetDimNum();
    const ge::Format format = xDesc->GetOriginFormat();
    size_t channelIndex = 0;
    if (format == FORMAT_NCHW) {
        if (rank < 2U || rank > 4U) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                     "NCHW input rank must be in [2, 4].");
            return GRAPH_FAILED;
        }
        channelIndex = 1U;
    } else if (format == FORMAT_NHWC) {
        if (rank != 4U) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                     "NHWC input rank must be 4.");
            return GRAPH_FAILED;
        }
        channelIndex = 3U;
    } else if (format == FORMAT_NCDHW) {
        if (rank != 5U) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                     "NCDHW input rank must be 5.");
            return GRAPH_FAILED;
        }
        channelIndex = 1U;
    } else {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "x",
                                               std::to_string(static_cast<int32_t>(format)).c_str(),
                                               "Supported formats are NCHW, NHWC, and NCDHW.");
        return GRAPH_FAILED;
    }

    const int64_t channel = xShape->GetDim(channelIndex);
    sumShape->SetDimNum(1);
    sumShape->SetDim(0, channel);
    squareSumShape->SetDimNum(1);
    squareSumShape->SetDim(0, channel);
    OP_LOGI(context->GetNodeName(), "[InferShape] output0 shape=%s, output1 shape=%s",
            Ops::Base::ToString(*sumShape).c_str(), Ops::Base::ToString(*squareSumShape).c_str());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BNTrainingReduce).InferShape(InferShape4BNTrainingReduce);
} // namespace ops
