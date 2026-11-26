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
 * \file sigmoid_cross_entropy_with_logits_grad_arch35_tiling.cpp
 * \brief
 */
#include "sigmoid_cross_entropy_with_logits_grad_arch35_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "../../op_kernel/arch35/sigmoid_cross_entropy_with_logits_grad_tiling_data.h"
#include <algorithm>
#include <limits>
#include <set>
#include <string>

using namespace ge;

namespace optiling {

static constexpr size_t IDX_PREDICT = 0;
static constexpr size_t IDX_TARGET = 1;
static constexpr size_t IDX_DOUT = 2;
static constexpr size_t WORKSPACE_NUM = 1;
static constexpr size_t MAX_DIM_NUM = 8;

static ge::graphStatus TilingForSigmoidCrossEntropyWithLogitsGrad(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    auto nodeName = context->GetNodeName();
    if (nodeName != nullptr) {
        OP_LOGI(nodeName, "Enter SigmoidCrossEntropyWithLogitsGradTilingFunc");
    }
    auto predictShapePtr = context->GetInputShape(IDX_PREDICT);
    if (predictShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto predictShape = predictShapePtr->GetStorageShape();

    OP_CHECK_IF(predictShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    nodeName, "predict", std::to_string(predictShape.GetDimNum()).c_str(), "rank must be <= 8"),
                return ge::GRAPH_FAILED);

    int64_t totalElements = 1;
    for (size_t i = 0; i < predictShape.GetDimNum(); ++i) {
        int64_t dim = predictShape.GetDim(i);
        OP_CHECK_IF(dim < 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "predict", std::to_string(dim).c_str(),
                                                             "dimensions must be non-negative at tiling time"),
                    return ge::GRAPH_FAILED);
        if (dim != 0 && totalElements > std::numeric_limits<int64_t>::max() / dim) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "predict", Ops::Base::ToString(predictShape).c_str(),
                                                  "the total element count exceeds the supported int64 range");
            return ge::GRAPH_FAILED;
        }
        totalElements *= dim;
    }

    auto predictDesc = context->GetInputDesc(IDX_PREDICT);
    if (predictDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType inputDtype = predictDesc->GetDataType();

    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedDtypes.count(inputDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "predict", Ops::Base::ToString(inputDtype).c_str(),
                                                      "supported dtypes are FLOAT16, FLOAT, and BFLOAT16"),
                return ge::GRAPH_FAILED);

    auto targetDesc = context->GetInputDesc(IDX_TARGET);
    if (targetDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType targetDtype = targetDesc->GetDataType();
    OP_CHECK_IF(supportedDtypes.count(targetDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "target", Ops::Base::ToString(targetDtype).c_str(),
                                                      "supported dtypes are FLOAT16, FLOAT, and BFLOAT16"),
                return ge::GRAPH_FAILED);

    auto doutDesc = context->GetInputDesc(IDX_DOUT);
    if (doutDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType doutDtype = doutDesc->GetDataType();
    OP_CHECK_IF(supportedDtypes.count(doutDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "dout", Ops::Base::ToString(doutDtype).c_str(),
                                                      "supported dtypes are FLOAT16, FLOAT, and BFLOAT16"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(inputDtype != targetDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    nodeName, "predict and target",
                    (Ops::Base::ToString(inputDtype) + ", " + Ops::Base::ToString(targetDtype)).c_str(),
                    "their dtypes must be identical"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputDtype != doutDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    nodeName, "predict and dout",
                    (Ops::Base::ToString(inputDtype) + ", " + Ops::Base::ToString(doutDtype)).c_str(),
                    "their dtypes must be identical"),
                return ge::GRAPH_FAILED);

    auto targetShapePtr = context->GetInputShape(IDX_TARGET);
    OP_CHECK_NULL_WITH_CONTEXT(context, targetShapePtr);
    auto targetShape = targetShapePtr->GetStorageShape();
    OP_CHECK_IF(targetShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    nodeName, "target", std::to_string(targetShape.GetDimNum()).c_str(), "rank must be <= 8"),
                return ge::GRAPH_FAILED);

    auto doutShapePtr = context->GetInputShape(IDX_DOUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, doutShapePtr);
    auto doutShape = doutShapePtr->GetStorageShape();
    OP_CHECK_IF(doutShape.GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    nodeName, "dout", std::to_string(doutShape.GetDimNum()).c_str(), "rank must be <= 8"),
                return ge::GRAPH_FAILED);

    bool targetShapeEqual = (predictShape.GetDimNum() == targetShape.GetDimNum());
    for (size_t i = 0; i < predictShape.GetDimNum() && targetShapeEqual; ++i) {
        if (predictShape.GetDim(i) != targetShape.GetDim(i)) {
            targetShapeEqual = false;
        }
    }
    const std::string targetShapeReason = "must exactly match predict shape " + Ops::Base::ToString(predictShape);
    OP_CHECK_IF(!targetShapeEqual,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "target", Ops::Base::ToString(targetShape).c_str(),
                                                      targetShapeReason.c_str()),
                return ge::GRAPH_FAILED);
    const auto predictElements = predictShape.GetShapeSize();
    const auto doutElements = doutShape.GetShapeSize();
    const std::string doutSizeReason = "element count must equal predict element count " +
                                       std::to_string(predictElements);
    OP_CHECK_IF(predictElements != doutElements,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(nodeName, "dout", std::to_string(doutElements).c_str(),
                                                          doutSizeReason.c_str()),
                return ge::GRAPH_FAILED);

    SigmoidCrossEntropyWithLogitsGradTilingData*
        tilingData = context->GetTilingData<SigmoidCrossEntropyWithLogitsGradTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData->totalLength = static_cast<uint64_t>(totalElements);
    tilingData->tileNum = static_cast<uint32_t>(
        std::min(static_cast<int64_t>(256), std::max(static_cast<int64_t>(1), totalElements)));

    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    if (ws != nullptr) {
        ws[0] = 0;
    }
    context->SetBlockDim(1);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseForSigmoidCrossEntropyWithLogitsGrad(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SigmoidCrossEntropyWithLogitsGrad)
    .Tiling(TilingForSigmoidCrossEntropyWithLogitsGrad)
    .TilingParse<SigmoidCrossEntropyWithLogitsGradCompileInfo>(ParseForSigmoidCrossEntropyWithLogitsGrad);

} // namespace optiling
