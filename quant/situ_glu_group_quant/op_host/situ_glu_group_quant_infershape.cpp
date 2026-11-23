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
 * \file situ_glu_group_quant_infershape.cpp
 * \brief Shape inference for SituGluGroupQuant, ported from cann-recipes-infer
 *        proto.cpp (same caliber, ops-nn EZ error reporting style).
 */

#include "graph/utils/type_utils.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "util/math_util.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace ops {
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1;
constexpr int64_t UNKNOWN_RANK_DIM = -2; // 未知 rank（新框架以 1 维 [-2] 表示）
constexpr size_t UNKNOWN_RANK_DIM_NUM = 1;
constexpr size_t INDEX_INPUT_X = 0;
constexpr size_t INDEX_INPUT_GROUP_INDEX = 1;
constexpr size_t INDEX_OUTPUT_Y = 0;
constexpr size_t INDEX_OUTPUT_Y_SCALE = 1;

constexpr int64_t SPLIT_NUM = 2;
constexpr int64_t BLOCK_SIZE = 32; // MXFP8 group size, fixed
constexpr int64_t ALIGN_NUM = 2;   // y_scale last axis interleave factor

static const std::initializer_list<ge::DataType> Y_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E4M3FN};
static const std::initializer_list<ge::DataType> SCALE_SUPPORT_DTYPE_SET = {ge::DT_FLOAT8_E8M0};
static const std::initializer_list<ge::DataType> GROUP_INDEX_DTYPE_SET = {ge::DT_INT64};

static void ReportFrameworkError(gert::InferShapeContext* context, const std::string& reason)
{
    const std::string msgOpName = std::string(((context) == nullptr || (context)->GetNodeName() == nullptr ?
                                                   "situ_glu_group_quant" :
                                                   (context)->GetNodeName()));
    OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", reason.c_str());
    REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                              (std::vector<const char*>{msgOpName.c_str(), reason.c_str()}));
}

// 未知 rank：新框架以 1 维且该维值为 -2 表示
static inline bool IsUnknownRank(const gert::Shape* shape)
{
    return shape->GetDimNum() == UNKNOWN_RANK_DIM_NUM && shape->GetDim(0) == UNKNOWN_RANK_DIM;
}

graphStatus InferShapeForSituGluGroupQuant(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForSituGluGroupQuant");
    const gert::Shape* xShape = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_IF(xShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    ((context) == nullptr || (context)->GetNodeName() == nullptr ? "situ_glu_group_quant" :
                                                                                   (context)->GetNodeName()),
                    "x"),
                return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    if (yShape == nullptr) {
        ReportFrameworkError(context,
                             "InferShape framework object yShape is unavailable; check the operator registration "
                             "and context construction.");
        return ge::GRAPH_FAILED;
    };
    gert::Shape* y_scaleShape = context->GetOutputShape(INDEX_OUTPUT_Y_SCALE);
    if (y_scaleShape == nullptr) {
        ReportFrameworkError(context, "InferShape framework object y_scaleShape is unavailable; check the operator "
                                      "registration and context construction.");
        return ge::GRAPH_FAILED;
    };

    // 未知 rank（-2）：跳过末维校验，y / y_scale 透传未知 rank；具体形状落地后
    // 由 host tiling 完成全部校验（末维正偶、EPR 区间、weight numel）。
    if (IsUnknownRank(xShape)) {
        yShape->SetDimNum(UNKNOWN_RANK_DIM_NUM);
        yShape->SetDim(0, UNKNOWN_RANK_DIM);
        y_scaleShape->SetDimNum(UNKNOWN_RANK_DIM_NUM);
        y_scaleShape->SetDim(0, UNKNOWN_RANK_DIM);
        OP_LOGI(context->GetNodeName(), "Input x has unknown rank (-2); y and y_scale propagate unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    // recipes caliber: rank >= 1 (no upper cap)
    OP_CHECK_IF(xShape->GetDimNum() < 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()),
                    ("rank of x must be >= 1, but is " + std::to_string(xShape->GetDimNum()) + ".")),
                return ge::GRAPH_FAILED);

    const int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    const int64_t lastDimIdx = rank - 1;
    const int64_t lastDim = xShape->GetDim(lastDimIdx);
    // recipes caliber: last dim must be positive and even; -1 stays dynamic
    OP_CHECK_IF(lastDim != UNKNOWN_DIM_VALUE_ && (lastDim <= 0 || lastDim % SPLIT_NUM != 0),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    context->GetNodeName(), "x", std::to_string(lastDim),
                    ("x last dim must be positive and even, but got [" + std::to_string(lastDim) + "].")),
                return ge::GRAPH_FAILED);

    const int64_t epr = (lastDim == UNKNOWN_DIM_VALUE_) ? UNKNOWN_DIM_VALUE_ : lastDim / SPLIT_NUM;

    // y: x with last dim halved
    *yShape = *xShape;
    yShape->SetDim(lastDimIdx, epr);

    // y_scale: x.shape[:-1] + [ceil(ceil(EPR/32)/2), 2]
    *y_scaleShape = *xShape;
    int64_t scaleDim = UNKNOWN_DIM_VALUE_;
    if (epr != UNKNOWN_DIM_VALUE_) {
        scaleDim = (epr + BLOCK_SIZE - 1) / BLOCK_SIZE;    // group count = ceil(EPR/32)
        scaleDim = (scaleDim + ALIGN_NUM - 1) / ALIGN_NUM; // interleave halve
    }
    y_scaleShape->SetDim(lastDimIdx, scaleDim);
    y_scaleShape->AppendDim(ALIGN_NUM);

    OP_LOGI(context->GetNodeName(), "End to do InferShapeForSituGluGroupQuant");
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDataTypeForSituGluGroupQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForSituGluGroupQuant");
    // recipes caliber: y is always e4m3fn and y_scale is always e8m0 (fixed dtypes)
    context->SetOutputDataType(INDEX_OUTPUT_Y, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(INDEX_OUTPUT_Y_SCALE, ge::DT_FLOAT8_E8M0);
    OP_LOGI(context->GetNodeName(), "End to do InferDataTypeForSituGluGroupQuant");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SituGluGroupQuant)
    .InferShape(InferShapeForSituGluGroupQuant)
    .InferDataType(InferDataTypeForSituGluGroupQuant);
} // namespace ops
