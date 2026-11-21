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
 * \file ge_glu_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;

namespace {
constexpr size_t GLU_IN_X = 0;
constexpr size_t GLU_OUT_Y = 0;
constexpr size_t GLU_OUT_Y_GLU = 1;
constexpr size_t GLU_ATTR_DIM = 0;
const size_t SPLIT_NUM = 2;

const char* GetInferShapeNodeName(const gert::InferShapeContext* context)
{
    return (context == nullptr || context->GetNodeName() == nullptr) ? "ge_glu_v2" : context->GetNodeName();
}

void ReportInferShapeError(gert::InferShapeContext* context, const std::string& msgReason)
{
    const std::string msgOpName = std::string(GetInferShapeNodeName(context));
    OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", msgReason.c_str());
    REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                              (std::vector<const char*>{msgOpName.c_str(), msgReason.c_str()}));
}

ge::graphStatus GetInferShapeInputs(gert::InferShapeContext* context, const gert::Shape*& xShape,
                                    gert::Shape*& outShapeY, gert::Shape*& outShapeYGlu, const int64_t*& splitDimPtr)
{
    xShape = context->GetInputShape(GLU_IN_X);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(GetInferShapeNodeName(context), "x"),
                return ge::GRAPH_FAILED);
    outShapeY = context->GetOutputShape(GLU_OUT_Y);
    if (outShapeY == nullptr) {
        ReportInferShapeError(context, "InferShape framework object out_shape_y is unavailable; check the operator "
                                       "registration and context construction.");
        return ge::GRAPH_FAILED;
    };
    outShapeYGlu = context->GetOutputShape(GLU_OUT_Y_GLU);
    if (outShapeYGlu == nullptr) {
        ReportInferShapeError(context, "InferShape framework object out_shape_y_glu is unavailable; check the operator "
                                       "registration and context construction.");
        return ge::GRAPH_FAILED;
    };
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        ReportInferShapeError(context,
                              "InferShape framework object attrs is unavailable; check the operator registration and "
                              "context construction.");
        return ge::GRAPH_FAILED;
    };

    splitDimPtr = attrs->GetAttrPointer<int64_t>(GLU_ATTR_DIM);
    if (splitDimPtr == nullptr) {
        ReportInferShapeError(context, "InferShape framework object split_dim_ptr is unavailable; check the operator "
                                       "registration and context construction.");
        return ge::GRAPH_FAILED;
    };
    return ge::GRAPH_SUCCESS;
}

bool NormalizeAndCheckSplitDim(int64_t& splitDim, size_t xDimNum, int64_t rawSplitDim)
{
    if (splitDim < 0) {
        splitDim += xDimNum;
    }
    OP_CHECK_IF((splitDim < 0 || splitDim >= static_cast<int64_t>(xDimNum)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    "GEGLUV2", "dim", std::to_string(rawSplitDim),
                    ("The value of attr [dim] must be in the range [-" + std::to_string(xDimNum) + ", " +
                     std::to_string(xDimNum - 1) + "], but got [" + std::to_string(splitDim) + "].")),
                return false);
    return true;
}
} // namespace

namespace ops {
static ge::graphStatus InferShapeForGeGluV2(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = nullptr;
    gert::Shape* outShapeY = nullptr;
    gert::Shape* outShapeYGlu = nullptr;
    const int64_t* splitDimPtr = nullptr;
    if (GetInferShapeInputs(context, xShape, outShapeY, outShapeYGlu, splitDimPtr) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto split_dim = *splitDimPtr;
    *outShapeY = *xShape;
    *outShapeYGlu = *xShape;
    size_t x_dim_num = xShape->GetDimNum();

    OP_CHECK_IF(x_dim_num == 0,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    "GEGLUV2", "x", std::to_string(x_dim_num),
                    ("input x dim num is " + std::to_string(x_dim_num) + ", not support input x is scalar")),
                return GRAPH_FAILED);

    OP_CHECK_IF(Ops::Base::IsUnknownRank(*xShape), OP_LOGD("GEGLUV2", "input x is unknown rank, no need check."),
                return GRAPH_SUCCESS);

    if (!NormalizeAndCheckSplitDim(split_dim, x_dim_num, *splitDimPtr)) {
        return GRAPH_FAILED;
    }

    // 动态shape场景split_dim_value传入-1不做处理
    auto split_dim_value = xShape->GetDim(split_dim);
    if (split_dim_value == -1) {
        return GRAPH_SUCCESS;
    }
    OP_CHECK_IF(split_dim_value < 0 || split_dim_value % SPLIT_NUM != 0,
                OP_LOGE("GEGLUV2", "The length of the split dimension [%ld] must be an even number.", split_dim_value),
                return GRAPH_FAILED);
    outShapeY->SetDim(split_dim, split_dim_value / SPLIT_NUM);
    outShapeYGlu->SetDim(split_dim, split_dim_value / SPLIT_NUM);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GeGluV2).InferShape(InferShapeForGeGluV2);
} // namespace ops
