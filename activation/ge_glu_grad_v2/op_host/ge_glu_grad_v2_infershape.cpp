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
 * \file ge_glu_grad_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;
namespace ops {
namespace geglugradv2 {

constexpr size_t GE_GLU_GRAD_IN_DY = 0;
constexpr size_t GE_GLU_GRAD_IN_X = 1;
constexpr size_t GE_GLU_GRAD_IN_GELU = 2;
constexpr size_t GE_GLU_GRAD_OUT_DX = 0;
constexpr size_t GE_GLU_ATTR_DIM = 0;
constexpr int64_t GE_GLU_NUM_TWO = 2;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

ge::graphStatus CopyShapeInput2OutputWithIdx(gert::InferShapeContext* context, int64_t input_idx, int64_t output_idx)
{
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_IF(
        in_shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_grad_v2" : (context)->GetNodeName()),
            (input_idx == 0 ? std::string("dy") :
                              (input_idx == 1 ? std::string("x") :
                                                (input_idx == 2 ? std::string("gelu") :
                                                                  "input[" + std::to_string(input_idx) + "]")))),
        return ge::GRAPH_FAILED);
    auto out_shape = context->GetOutputShape(output_idx);
    if (out_shape == nullptr) {
        const std::string msgOpName = std::string((
            (context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_grad_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object out_shape is unavailable; check the operator registration "
                               "and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object out_shape is unavailable; check "
                                                         "the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSplitDimValues(int64_t dim_value_x, int64_t dim_value_dy, int64_t dim_value_gelu)
{
    OP_CHECK_IF((dim_value_x % GE_GLU_NUM_TWO != 0),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON("GEGLUGRADV2", "x", std::to_string(dim_value_x),
                                                          "The split-axis dimension must be even."),
                return GRAPH_FAILED);
    OP_CHECK_IF((dim_value_dy != UNKNOWN_DIM_VALUE_ && dim_value_x != GE_GLU_NUM_TWO * dim_value_dy),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    "GEGLUGRADV2", "x,dy", std::to_string(dim_value_x) + "," + std::to_string(dim_value_dy),
                    "The checked x dimension must equal 2 times the corresponding dy dimension."),
                return GRAPH_FAILED);
    OP_CHECK_IF((dim_value_gelu != UNKNOWN_DIM_VALUE_ && dim_value_x != GE_GLU_NUM_TWO * dim_value_gelu),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    "GEGLUGRADV2", "x,gelu", std::to_string(dim_value_x) + "," + std::to_string(dim_value_gelu),
                    "The checked x dimension must equal 2 times the corresponding gelu dimension."),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckNonSplitDimValues(int64_t dim_value_x, int64_t dim_value_dy, int64_t dim_value_gelu)
{
    OP_CHECK_IF((dim_value_dy != UNKNOWN_DIM_VALUE_ && dim_value_x != dim_value_dy),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    "GEGLUGRADV2", "x,dy", std::to_string(dim_value_x) + "," + std::to_string(dim_value_dy),
                    "The checked x dimension must equal 1 times the corresponding dy dimension."),
                return GRAPH_FAILED);
    OP_CHECK_IF((dim_value_gelu != UNKNOWN_DIM_VALUE_ && dim_value_x != dim_value_gelu),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    "GEGLUGRADV2", "x,gelu", std::to_string(dim_value_x) + "," + std::to_string(dim_value_gelu),
                    "The checked x dimension must equal 1 times the corresponding gelu dimension."),
                return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckDimValuesConsistency(const gert::Shape* x_shape, const gert::Shape* dy_shape,
                                                 const gert::Shape* gelu_shape, int64_t split_dim)
{
    size_t x_dim_num = x_shape->GetDimNum();
    for (size_t i = 0; i < x_dim_num; i++) {
        int64_t dim_value_x = x_shape->GetDim(i);
        int64_t dim_value_dy = dy_shape->GetDim(i);
        int64_t dim_value_gelu = gelu_shape->GetDim(i);
        if (dim_value_x == UNKNOWN_DIM_VALUE_) {
            OP_CHECK_IF(
                (dim_value_dy != UNKNOWN_DIM_VALUE_ && dim_value_gelu != UNKNOWN_DIM_VALUE_ &&
                 dim_value_dy != dim_value_gelu),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    "GEGLUGRADV2", "dy,gelu", std::to_string(dim_value_dy) + "," + std::to_string(dim_value_gelu),
                    "The input shape dy and gelu are different, check failed."),
                return GRAPH_FAILED);
            continue;
        }
        if (split_dim == static_cast<int64_t>(i)) {
            auto status = CheckSplitDimValues(dim_value_x, dim_value_dy, dim_value_gelu);
            OP_CHECK_IF(status != GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT("GEGLUGRADV2", "CheckSplitDimValues failed at dim %zu.", i),
                        return GRAPH_FAILED);
        } else {
            auto status = CheckNonSplitDimValues(dim_value_x, dim_value_dy, dim_value_gelu);
            OP_CHECK_IF(status != GRAPH_SUCCESS,
                        OP_LOGE_WITHOUT_REPORT("GEGLUGRADV2", "CheckNonSplitDimValues failed at dim %zu.", i),
                        return GRAPH_FAILED);
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForGeGluGradV2(gert::InferShapeContext* context)
{
    auto dy_shape = context->GetInputShape(GE_GLU_GRAD_IN_DY);
    OP_CHECK_IF(
        dy_shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_grad_v2" : (context)->GetNodeName()),
            "dy"),
        return ge::GRAPH_FAILED);
    auto x_shape = context->GetInputShape(GE_GLU_GRAD_IN_X);
    OP_CHECK_IF(
        x_shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_grad_v2" : (context)->GetNodeName()),
            "x"),
        return ge::GRAPH_FAILED);
    auto gelu_shape = context->GetInputShape(GE_GLU_GRAD_IN_GELU);
    OP_CHECK_IF(
        gelu_shape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            ((context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_grad_v2" : (context)->GetNodeName()),
            "gelu"),
        return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string((
            (context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_grad_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object attrs is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    auto split_dim_ptr = attrs->GetAttrPointer<int64_t>(GE_GLU_ATTR_DIM);
    if (split_dim_ptr == nullptr) {
        const std::string msgOpName = std::string((
            (context) == nullptr || (context)->GetNodeName() == nullptr ? "ge_glu_grad_v2" : (context)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "InferShape framework object split_dim_ptr is unavailable; check the operator "
                               "registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "InferShape framework object split_dim_ptr is unavailable; "
                                                         "check the operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };

    auto split_dim = *split_dim_ptr;
    size_t dy_dim_num = dy_shape->GetDimNum();
    size_t x_dim_num = x_shape->GetDimNum();
    size_t gelu_dim_num = gelu_shape->GetDimNum();
    OP_CHECK_IF(x_dim_num == 0 || dy_dim_num == 0 || gelu_dim_num == 0,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    "GEGLUGRADV2", "x,dy,gelu",
                    std::to_string(x_dim_num) + "," + std::to_string(dy_dim_num) + "," + std::to_string(gelu_dim_num),
                    "scalar inputs are not supported, check failed"),
                return GRAPH_FAILED);

    OP_CHECK_IF(Ops::Base::IsUnknownRank(*x_shape) || Ops::Base::IsUnknownRank(*dy_shape) ||
                    Ops::Base::IsUnknownRank(*gelu_shape),
                OP_LOGD("GEGLUGRADV2", "inputs have unknown rank, no need check."),
                return CopyShapeInput2OutputWithIdx(context, 1, 0));

    OP_CHECK_IF(
        x_dim_num != dy_dim_num || x_dim_num != gelu_dim_num,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            "GEGLUGRADV2", "x, dy, gelu",
            (std::to_string(x_dim_num) + ", " + std::to_string(dy_dim_num) + ", " + std::to_string(gelu_dim_num)),
            "The shape dims of x, dy and gelu must be the same"),
        return GRAPH_FAILED);

    if (split_dim < 0) {
        split_dim += x_dim_num;
    }
    OP_CHECK_IF(
        (split_dim < 0 || split_dim >= static_cast<int64_t>(x_dim_num)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dim", std::to_string(split_dim),
                                              "The value of dim must be in the range [-" + std::to_string(x_dim_num) +
                                                  ", " + std::to_string(x_dim_num - 1) + "]"),
        return GRAPH_FAILED);

    auto status = CheckDimValuesConsistency(x_shape, dy_shape, gelu_shape, split_dim);
    OP_CHECK_IF(status != GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT("GEGLUGRADV2", "CheckDimValuesConsistency failed."),
                return GRAPH_FAILED);
    return CopyShapeInput2OutputWithIdx(context, 1, 0);
}

IMPL_OP_INFERSHAPE(GeGluGradV2).InferShape(InferShapeForGeGluGradV2);
} // namespace geglugradv2
} // namespace ops
