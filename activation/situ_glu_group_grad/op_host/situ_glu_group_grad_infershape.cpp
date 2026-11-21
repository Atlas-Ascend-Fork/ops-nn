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
 * \file situ_glu_group_grad_infershape.cpp
 * \brief InferShape for SituGluGroupGrad: grad_x equals x; grad_weight equals weight (empty when absent).
 */

#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "util/shape_util.h"
using namespace ge;

namespace {
constexpr size_t GRAD_IN_X = 1;
constexpr size_t GRAD_IN_WEIGHT = 2;
constexpr size_t GRAD_OUT_X = 0;
constexpr size_t GRAD_OUT_WEIGHT = 1;
} // namespace

namespace ops {
static ge::graphStatus InferShapeForSituGluGroupGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start InferShape for SituGluGroupGrad");
    auto xShape = context->GetInputShape(GRAD_IN_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto gradXShape = context->GetOutputShape(GRAD_OUT_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gradXShape);
    auto gradWeightShape = context->GetOutputShape(GRAD_OUT_WEIGHT);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gradWeightShape);

    // grad_x has the same shape as x.
    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*gradXShape);
    } else {
        *gradXShape = *xShape;
    }

    // grad_weight is a required output: same shape as weight when weight is provided, empty ([0])
    // otherwise (no weight in forward means no weight gradient chain).
    auto weightShape = context->GetOptionalInputShape(GRAD_IN_WEIGHT);
    if (weightShape != nullptr && !Ops::Base::IsUnknownRank(*weightShape)) {
        *gradWeightShape = *weightShape;
    } else {
        gradWeightShape->SetDimNum(1);
        gradWeightShape->SetDim(0, 0);
    }
    OP_LOGD(context->GetNodeName(), "Finish InferShape for SituGluGroupGrad");
    return ge::GRAPH_SUCCESS;
}

// InferDataType is implemented in op_graph/situ_glu_group_grad_graph_infer.cpp.
IMPL_OP_INFERSHAPE(SituGluGroupGrad).InferShape(InferShapeForSituGluGroupGrad);
} // namespace ops
