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
 * \file fatrelu_mul_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t SIZE_2 = 2;
static constexpr int64_t UNKNOWN_RANK_DIM = -2;
static constexpr int64_t UNKNOWN_DIM_VALUE = -1;
static constexpr int64_t MIN_RANK = 2; // rank(x) >= 2 (no upper bound, row-model rank transparent)

static ge::graphStatus InferShape4FatreluMul(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4FatreluMul");

    // get input shapes
    auto xShape = context->GetInputShape(IDX_0);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "fatrelu_mul" : (context)->GetNodeName()),
            "x"),
        return ge::GRAPH_FAILED);

    // get output shapes
    auto yShape = context->GetOutputShape(IDX_0);
    if (yShape == nullptr) {
        const std::string msgOpName = std::string(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "fatrelu_mul" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object yShape is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object yShape is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    // unknown rank (-2): propagate as-is, output rank depends on input rank
    if (xShape->GetDimNum() == 1 && xShape->GetDim(0) == UNKNOWN_RANK_DIM) {
        yShape->SetDimNum(1);
        yShape->SetDim(IDX_0, UNKNOWN_RANK_DIM);
        OP_LOGD(context, "End to do InferShape4FatreluMul with unknown rank");
        return GRAPH_SUCCESS;
    }

    size_t xDimNum = xShape->GetDimNum();
    // rank contract: rank(x) >= 2, no upper bound (row model
    // is rank transparent); a lower rank has no valid last dim to halve (rank=0
    // would underflow GetDim(xDimNum-1)). -1/-2 dynamic dims keep their dedicated
    // propagation branches above and are unaffected by this guard.
    const int64_t xRank = static_cast<int64_t>(xDimNum);
    OP_CHECK_IF(xRank < MIN_RANK, OP_LOGE(context->GetNodeName(), "The rank of x should be >= 2, but is %ld.", xRank),
                return ge::GRAPH_FAILED);
    yShape->SetDimNum(xDimNum);

    *yShape = *xShape;
    const int64_t lastDim = xShape->GetDim(xDimNum - 1);
    // unknown dim (-1) propagates; known dim is halved
    yShape->SetDim(xDimNum - 1, lastDim < 0 ? lastDim : lastDim / SIZE_2);

    OP_LOGD(context, "End to do InferShape4FatreluMul");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4FatreluMul(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4FatreluMul");

    auto input_dtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, input_dtype);

    OP_LOGD(context, "End to do InferDataType4FatreluMul");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FatreluMul).InferShape(InferShape4FatreluMul).InferDataType(InferDataType4FatreluMul);
} // namespace ops
