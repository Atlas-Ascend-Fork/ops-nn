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
 * \file inplace_cross_entropy_sum_exp_and_index_logit_graph_infer.cpp
 * \brief InplaceCrossEntropySumExpAndIndexLogit dtype inference
 *
 * - predicted_logits / sum_exp_logits: fixed float32;
 * - vocab_parallel_logits (output2): follows input dtype (in-place alias);
 * - target_offset / target_mask: fixed int32.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

namespace {
constexpr size_t OUTPUT_PREDICTED_LOGITS = 0;
constexpr size_t OUTPUT_SUM_EXP_LOGITS = 1;
constexpr size_t OUTPUT_VOCAB_PARALLEL_LOGITS = 2;
constexpr size_t OUTPUT_TARGET_OFFSET = 3;
constexpr size_t OUTPUT_TARGET_MASK = 4;
constexpr size_t INPUT_VOCAB_PARALLEL_LOGITS = 0;
} // namespace

static ge::graphStatus InferDataType4InplaceCrossEntropySumExpAndIndexLogit(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_PREDICTED_LOGITS, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_SUM_EXP_LOGITS, ge::DT_FLOAT);
    // vocab_parallel_logits output follows input dtype (in-place alias)
    context->SetOutputDataType(OUTPUT_VOCAB_PARALLEL_LOGITS, context->GetInputDataType(INPUT_VOCAB_PARALLEL_LOGITS));
    context->SetOutputDataType(OUTPUT_TARGET_OFFSET, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_TARGET_MASK, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(InplaceCrossEntropySumExpAndIndexLogit).InferDataType(InferDataType4InplaceCrossEntropySumExpAndIndexLogit);

} // namespace ops
