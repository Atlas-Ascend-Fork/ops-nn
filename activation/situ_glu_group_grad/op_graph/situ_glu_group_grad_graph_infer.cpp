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
 * \file situ_glu_group_grad_graph_infer.cpp
 * \brief InferDataType for SituGluGroupGrad (must live in op_graph).
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace {
constexpr size_t GRAD_IN_X = 1;
constexpr size_t GRAD_OUT_X = 0;
constexpr size_t GRAD_OUT_WEIGHT = 1;
} // namespace

namespace ops {
ge::graphStatus InferDataTypeForSituGluGroupGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    // grad_x follows the x dtype; grad_weight is always float32 regardless of whether weight is present.
    const ge::DataType dtype = context->GetInputDataType(GRAD_IN_X);
    ge::graphStatus ret = context->SetOutputDataType(GRAD_OUT_X, dtype);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "grad_x", "set dtype failed",
                                                      "set grad_x dtype failed"),
                return ge::GRAPH_FAILED);
    ret = context->SetOutputDataType(GRAD_OUT_WEIGHT, ge::DT_FLOAT);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "grad_weight", "set dtype failed",
                                                      "set grad_weight dtype failed"),
                return ge::GRAPH_FAILED);
    return ret;
}

IMPL_OP(SituGluGroupGrad).InferDataType(InferDataTypeForSituGluGroupGrad);
} // namespace ops
