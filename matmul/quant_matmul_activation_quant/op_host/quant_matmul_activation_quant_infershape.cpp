/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_activation_quant_infershape.cpp
 * \brief InferShape and InferDataType for QuantMatmulActivationQuant
 */
#include <algorithm>
#include <string_view>

#include "common/op_host/matmul_common_infershape.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "runtime/infer_datatype_context.h"
#include "quant_matmul_activation_quant_host_utils.h"
#include "util/math_util.h"

using Ops::NN::FormatString;

namespace {
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;

constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t Y_SCALE_INDEX = 1;

constexpr uint32_t TRANSPOSE_X1_INDEX = 0;
constexpr uint32_t TRANSPOSE_X2_INDEX = 1;
constexpr uint32_t Y_DTYPE_INDEX = 4;
constexpr uint32_t ACTIVATION_TYPE_INDEX = 3;

constexpr int64_t MX_SCALE_LAST_DIM_SIZE = 2;
constexpr int64_t SWIGLU_BRANCH_COUNT = 2;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_DIM_NUM = -2;
constexpr size_t QUANT_MATMUL_MIN_SHAPE_SIZE = 2;
constexpr size_t QUANT_MATMUL_MAX_SHAPE_SIZE = 6;
constexpr const char* OP_NAME = "QuantMatmulActivationQuant";

const char* GetValidOpNameForInferShape(gert::InferShapeContext* context)
{
    const char* nodeName = context->GetNodeName();
    if (nodeName != nullptr && nodeName[0] != '\0') {
        return nodeName;
    }
    return OP_NAME;
}

static ge::graphStatus InferShape4QuantMatmulActivationQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin InferShape4QuantMatmulActivationQuant.");

    OP_CHECK_IF(context == nullptr, OP_LOGE_WITH_INVALID_INPUT("nil", "context"), return ge::GRAPH_FAILED);
    const char* opName = GetValidOpNameForInferShape(context);

    auto x1Shape = context->GetInputShape(X1_INDEX);
    auto x2Shape = context->GetInputShape(X2_INDEX);

    OP_CHECK_IF(x1Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "x1"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(x2Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "x2"), return ge::GRAPH_FAILED);

    auto x1Rank = x1Shape->GetDimNum();
    auto x2Rank = x2Shape->GetDimNum();
    bool hasUnknownRank = Ops::NN::CheckIsUnknownDimNum(*x1Shape) || Ops::NN::CheckIsUnknownDimNum(*x2Shape);

    auto yShape = context->GetOutputShape(Y_INDEX);
    auto yScaleShape = context->GetOutputShape(Y_SCALE_INDEX);
    OP_CHECK_IF(yShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "y"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(yScaleShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "y_scale"), return ge::GRAPH_FAILED);

    if (hasUnknownRank) {
        yShape->SetDimNum(1);
        yShape->SetDim(0, UNKNOWN_DIM_NUM);
        yScaleShape->SetDimNum(1);
        yScaleShape->SetDim(0, UNKNOWN_DIM_NUM);
        OP_LOGD(opName, "Input rank is unknown; setting y and yScale to unknown rank [-2].");
        return ge::GRAPH_SUCCESS;
    }

    if (x1Rank < QUANT_MATMUL_MIN_SHAPE_SIZE || x1Rank > QUANT_MATMUL_MAX_SHAPE_SIZE ||
        x2Rank < QUANT_MATMUL_MIN_SHAPE_SIZE || x2Rank > QUANT_MATMUL_MAX_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, "x1, x2", FormatString("%zuD, %zuD", x1Rank, x2Rank).c_str(),
                                                  "the input ranks must be in the range [2, 6]");
        return ge::GRAPH_FAILED;
    }
    const size_t yRank = std::max(x1Rank, x2Rank);

    OP_LOGD(opName, "Input shapes: x1=%s, x2=%s", Ops::Base::ToString(*x1Shape).c_str(),
            Ops::Base::ToString(*x2Shape).c_str());

    const auto* attrs = context->GetAttrs();
    const auto* activationType = attrs == nullptr ? nullptr : attrs->GetAttrPointer<char>(ACTIVATION_TYPE_INDEX);
    const bool isSwiglu = activationType != nullptr && std::string_view(activationType) == "swiglu";
    bool transposeX1 = false;
    bool transposeX2 = false;
    if (attrs != nullptr) {
        const bool* transposeX1Attr = attrs->GetAttrPointer<bool>(TRANSPOSE_X1_INDEX);
        const bool* transposeX2Attr = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
        if (transposeX1Attr != nullptr) {
            transposeX1 = *transposeX1Attr;
        }
        if (transposeX2Attr != nullptr) {
            transposeX2 = *transposeX2Attr;
        }
    }

    const int64_t outputM = transposeX1 ? x1Shape->GetDim(x1Rank - 1) : x1Shape->GetDim(x1Rank - 2);
    const int64_t matmulN = transposeX2 ? x2Shape->GetDim(x2Rank - 2) : x2Shape->GetDim(x2Rank - 1);
    const int64_t outputN = isSwiglu && matmulN != UNKNOWN_DIM ? matmulN / SWIGLU_BRANCH_COUNT : matmulN;

    yShape->SetDimNum(yRank);

    // batch 维从右对齐，x1/x2 的 batch 维度数各自不同，不足的补 1
    const size_t x1Offset = yRank - x1Rank;
    const size_t x2Offset = yRank - x2Rank;
    for (size_t i = 0; i < yRank - 2; ++i) {
        const int64_t x1BatchDim = i < x1Offset ? 1 : x1Shape->GetDim(i - x1Offset);
        const int64_t x2BatchDim = i < x2Offset ? 1 : x2Shape->GetDim(i - x2Offset);
        int64_t batchDim;
        if (x1BatchDim == x2BatchDim || x2BatchDim == 1) {
            batchDim = x1BatchDim;
        } else if (x1BatchDim == 1 || x1BatchDim == UNKNOWN_DIM) {
            batchDim = x2BatchDim;
        } else if (x2BatchDim == UNKNOWN_DIM) {
            batchDim = x1BatchDim;
        } else {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName, "x1/x2 batch dim",
                FormatString("dim %zu: %lld, %lld", i, static_cast<long long>(x1BatchDim),
                             static_cast<long long>(x2BatchDim))
                    .c_str(),
                "the batch dimensions of x1 and x2 must be broadcastable");
            return ge::GRAPH_FAILED;
        }

        yShape->SetDim(i, batchDim);
    }

    yShape->SetDim(yRank - 2, outputM);
    yShape->SetDim(yRank - 1, outputN);

    OP_LOGD(opName, "Output y shape: %s", Ops::Base::ToString(*yShape).c_str());

    int64_t scaleN = UNKNOWN_DIM;
    if (outputN != UNKNOWN_DIM) {
        scaleN = Ops::Base::CeilDiv(outputN, QuantMatmulActivationQuantTilingConstant::MX_SCALE_STORAGE_GROUP);
    }

    *yScaleShape = *yShape;
    yScaleShape->SetDimNum(yRank + 1);
    yScaleShape->SetDim(yRank - 1, scaleN);
    yScaleShape->SetDim(yRank, MX_SCALE_LAST_DIM_SIZE);

    OP_LOGD(opName, "Output yScale shape: %s", Ops::Base::ToString(*yScaleShape).c_str());

    OP_LOGI(context, "End InferShape4QuantMatmulActivationQuant.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4QuantMatmulActivationQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin InferDataType4QuantMatmulActivationQuant.");

    OP_CHECK_IF(context == nullptr, OP_LOGE_WITH_INVALID_INPUT("nil", "context"), return ge::GRAPH_FAILED);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* yDtype = attrs->GetAttrPointer<int64_t>(Y_DTYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDtype);

    context->SetOutputDataType(Y_INDEX, static_cast<ge::DataType>(*yDtype));
    context->SetOutputDataType(Y_SCALE_INDEX, ge::DT_FLOAT8_E8M0);

    OP_LOGD(context, "End InferDataType4QuantMatmulActivationQuant.");
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(QuantMatmulActivationQuant)
    .InferShape(InferShape4QuantMatmulActivationQuant)
    .InferDataType(InferDataType4QuantMatmulActivationQuant);
}
