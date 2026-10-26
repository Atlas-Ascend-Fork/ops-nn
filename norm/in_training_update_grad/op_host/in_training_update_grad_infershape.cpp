/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file in_training_update_grad_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "graph/utils/type_utils.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
constexpr size_t INPUT_DY_IDX = 0;
constexpr size_t INPUT_X_IDX = 1;
constexpr size_t INPUT_VARIANCE_IDX = 2;
constexpr size_t INPUT_MEAN_IDX = 3;
constexpr size_t OUTPUT_RES_GAMMA_IDX = 0;
constexpr size_t OUTPUT_RES_BETA_IDX = 1;
constexpr int64_t NDC1HWC0_DIM_NUM = 6;
constexpr int64_t DIM_N = 0;
constexpr int64_t DIM_D = 1;
constexpr int64_t DIM_C1 = 2;
constexpr int64_t DIM_C0 = 5;
// Mirror of the tiling kernel envelope: at most 64 C0 lanes per row and uint32 tiling fields.
constexpr int64_t MAX_C0_LANES = 64;
constexpr int64_t UINT32_LIMIT = 0xFFFFFFFFLL;

// Outputs are the per-(n,c1) statistics of dy: res_gamma.shape = res_beta.shape =
// (N, 1, C1, 1, 1, C0) with the spatial reduction axes (D, H, W) folded to 1.
// Fully concrete shapes validate the cross-input contract here at the Host graph-build gate;
// rank/dim mismatches used to pass through and only fail later at the tiling gate. Unknown or
// dynamic ranks keep the variance passthrough and are re-validated by tiling.
static ge::graphStatus InferShape4INTrainingUpdateGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4INTrainingUpdateGrad");

    const gert::Shape* dyShape = context->GetInputShape(INPUT_DY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyShape);
    const gert::Shape* xShape = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* varianceShape = context->GetInputShape(INPUT_VARIANCE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, varianceShape);
    const gert::Shape* meanShape = context->GetInputShape(INPUT_MEAN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanShape);
    gert::Shape* resGammaShape = context->GetOutputShape(OUTPUT_RES_GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, resGammaShape);
    gert::Shape* resBetaShape = context->GetOutputShape(OUTPUT_RES_BETA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, resBetaShape);

    // Strict validation only when ALL inputs are fully concrete; any unknown rank (-2) or unknown
    // dim (-1) on any input keeps the variance passthrough (dynamic graphs are finalized and
    // re-validated by the tiling gate). IsUnknownRank must be checked per input: a [-2] rank
    // marker on x/variance/mean alongside a concrete dy is a legal mixed dynamic-rank graph.
    if (IsUnknownRank(*dyShape) || IsUnknownShape(*dyShape) || IsUnknownRank(*xShape) || IsUnknownShape(*xShape) ||
        IsUnknownRank(*varianceShape) || IsUnknownShape(*varianceShape) || IsUnknownRank(*meanShape) ||
        IsUnknownShape(*meanShape)) {
        *resGammaShape = *varianceShape;
        *resBetaShape = *varianceShape;
        OP_LOGD(context->GetNodeName(), "End to do InferShape4INTrainingUpdateGrad (dynamic passthrough)");
        return GRAPH_SUCCESS;
    }

    OP_CHECK_IF((dyShape->GetDimNum() != NDC1HWC0_DIM_NUM),
                OP_LOGE(context, "The shape dim of input dy must be 6 (NDC1HWC0); got %lld",
                        static_cast<long long>(dyShape->GetDimNum())),
                return GRAPH_FAILED);
    // Positivity and kernel-envelope gate (README promises rejection for N/C1/C0<=0; the arch35
    // register path bounds C0 to one 64-lane fp32 vector register and uint32 tiling fields).
    OP_CHECK_IF((dyShape->GetDim(DIM_N) <= 0 || dyShape->GetDim(DIM_C1) <= 0 || dyShape->GetDim(DIM_C0) <= 0),
                OP_LOGE(context, "N, C1 and C0 dims of input dy must be positive"), return GRAPH_FAILED);
    OP_CHECK_IF((dyShape->GetDim(DIM_D) < 0 || dyShape->GetDim(3) < 0 || dyShape->GetDim(4) < 0),
                OP_LOGE(context, "Spatial dims (D, H, W) of input dy must not be negative"), return GRAPH_FAILED);
    OP_CHECK_IF((dyShape->GetDim(DIM_C0) > MAX_C0_LANES),
                OP_LOGE(context, "C0 dim of input dy must not exceed 64 (one fp32 vector register); got %lld",
                        static_cast<long long>(dyShape->GetDim(DIM_C0))),
                return GRAPH_FAILED);
    const int64_t numN = dyShape->GetDim(DIM_N);
    const int64_t numD = dyShape->GetDim(DIM_D);
    const int64_t numC1 = dyShape->GetDim(DIM_C1);
    const int64_t numH = dyShape->GetDim(3);
    const int64_t numW = dyShape->GetDim(4);
    const int64_t numC0 = dyShape->GetDim(DIM_C0);
    // Envelope validated by pairwise quotient BEFORE any product is computed, so extreme dim
    // values cannot overflow the int64 check itself (mirrors the tiling gate).
    const auto pairFits = [](int64_t a, int64_t b) {
        return a <= UINT32_LIMIT && b <= (a > 0 ? UINT32_LIMIT / a : UINT32_LIMIT);
    };
    OP_CHECK_IF(!pairFits(numH, numW),
                OP_LOGE(context, "Dim product H*W of input dy exceeds the uint32 kernel "
                                 "envelope"),
                return GRAPH_FAILED);
    const int64_t numHW = numH * numW;
    OP_CHECK_IF(!pairFits(numD, numHW),
                OP_LOGE(context, "Dim product D*H*W of input dy exceeds the uint32 kernel "
                                 "envelope"),
                return GRAPH_FAILED);
    OP_CHECK_IF(!pairFits(numN, numC1),
                OP_LOGE(context, "Dim product N*C1 of input dy exceeds the uint32 kernel "
                                 "envelope"),
                return GRAPH_FAILED);
    const int64_t groupNum = numN * numC1;
    OP_CHECK_IF(!pairFits(numHW, numC0),
                OP_LOGE(context, "Dim product H*W*C0 of input dy exceeds the uint32 "
                                 "kernel envelope"),
                return GRAPH_FAILED);
    OP_CHECK_IF(!pairFits(groupNum, numC0),
                OP_LOGE(context, "Dim product N*C1*C0 of input dy exceeds the uint32 "
                                 "kernel envelope"),
                return GRAPH_FAILED);

    gert::Shape statShape;
    statShape.SetDimNum(NDC1HWC0_DIM_NUM);
    statShape.SetDim(DIM_N, dyShape->GetDim(DIM_N));
    statShape.SetDim(DIM_D, 1);
    statShape.SetDim(DIM_C1, dyShape->GetDim(DIM_C1));
    statShape.SetDim(3, 1);
    statShape.SetDim(4, 1);
    statShape.SetDim(DIM_C0, dyShape->GetDim(DIM_C0));

    OP_CHECK_IF((*xShape != *dyShape), OP_LOGE(context, "The shape of x must be the same as dy"), return GRAPH_FAILED);
    OP_CHECK_IF((*varianceShape != statShape),
                OP_LOGE(context, "The shape of variance must be (N, 1, C1, 1, 1, C0) derived from dy"),
                return GRAPH_FAILED);
    OP_CHECK_IF((*meanShape != statShape),
                OP_LOGE(context, "The shape of mean must be (N, 1, C1, 1, 1, C0) derived from dy"),
                return GRAPH_FAILED);

    *resGammaShape = statShape;
    *resBetaShape = statShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4INTrainingUpdateGrad");
    return GRAPH_SUCCESS;
}

// Both outputs are always float32. Concrete input dtypes are validated here at the Host
// graph-build gate; they used to pass through unvalidated until the tiling gate.
static graphStatus InferDataType4INTrainingUpdateGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4INTrainingUpdateGrad");
    ge::DataType dyDtype = context->GetInputDataType(INPUT_DY_IDX);
    if (dyDtype != ge::DT_UNDEFINED) {
        OP_CHECK_IF(
            (dyDtype != ge::DT_FLOAT16 && dyDtype != ge::DT_FLOAT),
            OP_LOGE(context, "The dtype of input dy must be FLOAT or FLOAT16; got %s", ToString(dyDtype).c_str()),
            return GRAPH_FAILED);
        OP_CHECK_IF((context->GetInputDataType(INPUT_X_IDX) != ge::DT_UNDEFINED &&
                     context->GetInputDataType(INPUT_X_IDX) != dyDtype),
                    OP_LOGE(context, "The dtypes of input dy and input x must be the same"), return GRAPH_FAILED);
        OP_CHECK_IF((context->GetInputDataType(INPUT_VARIANCE_IDX) != ge::DT_UNDEFINED &&
                     context->GetInputDataType(INPUT_VARIANCE_IDX) != ge::DT_FLOAT),
                    OP_LOGE(context, "The dtype of input variance must be FLOAT"), return GRAPH_FAILED);
        OP_CHECK_IF((context->GetInputDataType(INPUT_MEAN_IDX) != ge::DT_UNDEFINED &&
                     context->GetInputDataType(INPUT_MEAN_IDX) != ge::DT_FLOAT),
                    OP_LOGE(context, "The dtype of input mean must be FLOAT"), return GRAPH_FAILED);
    }
    context->SetOutputDataType(OUTPUT_RES_GAMMA_IDX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_RES_BETA_IDX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4INTrainingUpdateGrad");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(INTrainingUpdateGrad)
    .InferShape(InferShape4INTrainingUpdateGrad)
    .InferDataType(InferDataType4INTrainingUpdateGrad);
} // namespace ops
