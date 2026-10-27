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
 * \file in_training_update_v2_graph_infer.cpp
 * \brief Graph data type inference for INTrainingUpdateV2.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_X = 0;
constexpr size_t OUTPUT_Y = 0;
constexpr size_t OUTPUT_BATCH_MEAN = 1;
constexpr size_t OUTPUT_BATCH_VARIANCE = 2;
} // namespace

ge::graphStatus InferDataTypeForINTrainingUpdateV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE("INTrainingUpdateV2", "infer data type context is null");
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(OUTPUT_Y, context->GetRequiredInputDataType(INPUT_X)) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(OUTPUT_BATCH_MEAN, ge::DT_FLOAT) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(OUTPUT_BATCH_VARIANCE, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output dtype", "setter returned failure",
                                              "all inferred output dtypes must be writable");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(INTrainingUpdateV2).InferDataType(InferDataTypeForINTrainingUpdateV2);
} // namespace ops
