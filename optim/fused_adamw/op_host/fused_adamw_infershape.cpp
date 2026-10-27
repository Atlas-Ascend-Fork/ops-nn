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
 * \file fused_adamw_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
static constexpr size_t INPUT_PARAMS_INDEX = 0;
static constexpr size_t INPUT_GRADS_INDEX = 1;
static constexpr size_t INPUT_EXP_AVGS_INDEX = 2;
static constexpr size_t INPUT_EXP_AVG_SQS_INDEX = 3;
static constexpr size_t INPUT_MAX_EXP_AVG_SQS_INDEX = 4;
static constexpr size_t OUTPUT_PARAMS_INDEX = 0;
static constexpr size_t OUTPUT_EXP_AVGS_INDEX = 1;
static constexpr size_t OUTPUT_EXP_AVG_SQS_INDEX = 2;
static constexpr size_t OUTPUT_MAX_EXP_AVG_SQS_INDEX = 3;

namespace ops {
static ge::graphStatus InferShapeForFusedAdamW(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShapeForFusedAdamW.");

    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_IF(computeNodeInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "computeNodeInfo",
                                                         "failed to get compute node info from the context"),
                return ge::GRAPH_FAILED);

    auto paramsInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_PARAMS_INDEX);
    OP_CHECK_IF(
        paramsInstanceInfo == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "paramsInstanceInfo",
                                                 "failed to get the input instance info of params from the context"),
        return ge::GRAPH_FAILED);
    auto inputNum = paramsInstanceInfo->GetInstanceNum();
    if (inputNum == 0) {
        OP_LOGE_FOR_INVALID_TENSORNUM(context->GetNodeName(), "params", static_cast<int64_t>(inputNum),
                                      "greater than 0");
        return ge::GRAPH_FAILED;
    }

    auto paramsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_PARAMS_INDEX);
    OP_CHECK_IF(
        paramsOutInstanceInfo == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "paramsOutInstanceInfo",
                                                 "failed to get the output instance info of params from the context"),
        return ge::GRAPH_FAILED);
    auto expAvgsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_EXP_AVGS_INDEX);
    OP_CHECK_IF(
        expAvgsOutInstanceInfo == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "expAvgsOutInstanceInfo",
                                                 "failed to get the output instance info of exp_avgs from the context"),
        return ge::GRAPH_FAILED);
    auto expAvgSqsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_EXP_AVG_SQS_INDEX);
    OP_CHECK_IF(expAvgSqsOutInstanceInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "expAvgSqsOutInstanceInfo",
                    "failed to get the output instance info of exp_avg_sqs from the context"),
                return ge::GRAPH_FAILED);
    auto maxExpAvgSqsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_MAX_EXP_AVG_SQS_INDEX);
    OP_CHECK_IF(maxExpAvgSqsOutInstanceInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "maxExpAvgSqsOutInstanceInfo",
                    "failed to get the output instance info of max_exp_avg_sqs from the context"),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputNum; i++) {
        const gert::Shape* paramsShape = context->GetDynamicInputShape(INPUT_PARAMS_INDEX, i);
        OP_CHECK_IF(paramsShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "paramsShape",
                                                             "failed to get the input shape from the context"),
                    return ge::GRAPH_FAILED);
        const gert::Shape* gradsShape = context->GetDynamicInputShape(INPUT_GRADS_INDEX, i);
        OP_CHECK_IF(gradsShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "gradsShape",
                                                             "failed to get the input shape from the context"),
                    return ge::GRAPH_FAILED);
        const gert::Shape* expAvgsShape = context->GetDynamicInputShape(INPUT_EXP_AVGS_INDEX, i);
        OP_CHECK_IF(expAvgsShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "expAvgsShape",
                                                             "failed to get the input shape from the context"),
                    return ge::GRAPH_FAILED);
        const gert::Shape* expAvgSqsShape = context->GetDynamicInputShape(INPUT_EXP_AVG_SQS_INDEX, i);
        OP_CHECK_IF(expAvgSqsShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "expAvgSqsShape",
                                                             "failed to get the input shape from the context"),
                    return ge::GRAPH_FAILED);
        const gert::Shape* maxExpAvgSqsShape = context->GetDynamicInputShape(INPUT_MAX_EXP_AVG_SQS_INDEX, i);
        OP_CHECK_IF(maxExpAvgSqsShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "maxExpAvgSqsShape",
                                                             "failed to get the input shape from the context"),
                    return ge::GRAPH_FAILED);

        if (*paramsShape != *gradsShape || *paramsShape != *expAvgsShape || *paramsShape != *expAvgSqsShape ||
            *paramsShape != *maxExpAvgSqsShape) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "params/grads/exp_avgs/exp_avg_sqs/max_exp_avg_sqs",
                Ops::Base::ToString(*paramsShape) + "," + Ops::Base::ToString(*gradsShape) + "," +
                    Ops::Base::ToString(*expAvgsShape) + "," + Ops::Base::ToString(*expAvgSqsShape) + "," +
                    Ops::Base::ToString(*maxExpAvgSqsShape),
                "All tensors in the params, grads, exp_avgs, exp_avg_sqs and max_exp_avg_sqs tensor lists "
                "must have the same shape");
            return ge::GRAPH_FAILED;
        }

        gert::Shape* paramsRefShape = context->GetOutputShape(paramsOutInstanceInfo->GetInstanceStart() + i);
        OP_CHECK_IF(paramsRefShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "paramsRefShape",
                                                             "failed to get the output shape from the context"),
                    return ge::GRAPH_FAILED);
        gert::Shape* expAvgsRefShape = context->GetOutputShape(expAvgsOutInstanceInfo->GetInstanceStart() + i);
        OP_CHECK_IF(expAvgsRefShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "expAvgsRefShape",
                                                             "failed to get the output shape from the context"),
                    return ge::GRAPH_FAILED);
        gert::Shape* expAvgSqsRefShape = context->GetOutputShape(expAvgSqsOutInstanceInfo->GetInstanceStart() + i);
        OP_CHECK_IF(expAvgSqsRefShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "expAvgSqsRefShape",
                                                             "failed to get the output shape from the context"),
                    return ge::GRAPH_FAILED);
        gert::Shape* maxExpAvgSqsRefShape = context->GetOutputShape(maxExpAvgSqsOutInstanceInfo->GetInstanceStart() +
                                                                    i);
        OP_CHECK_IF(maxExpAvgSqsRefShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "maxExpAvgSqsRefShape",
                                                             "failed to get the output shape from the context"),
                    return ge::GRAPH_FAILED);

        *paramsRefShape = *paramsShape;
        *expAvgsRefShape = *expAvgsShape;
        *expAvgSqsRefShape = *expAvgSqsShape;
        *maxExpAvgSqsRefShape = *maxExpAvgSqsShape;
    }

    OP_LOGD(context, "End to do InferShapeForFusedAdamW.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForFusedAdamW(gert::InferDataTypeContext* context)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_IF(computeNodeInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "computeNodeInfo",
                                                         "failed to get compute node info from the context"),
                return ge::GRAPH_FAILED);

    auto paramsInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_PARAMS_INDEX);
    OP_CHECK_IF(
        paramsInstanceInfo == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "paramsInstanceInfo",
                                                 "failed to get the input instance info of params from the context"),
        return ge::GRAPH_FAILED);
    auto inputNum = paramsInstanceInfo->GetInstanceNum();

    auto paramsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_PARAMS_INDEX);
    OP_CHECK_IF(
        paramsOutInstanceInfo == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "paramsOutInstanceInfo",
                                                 "failed to get the output instance info of params from the context"),
        return ge::GRAPH_FAILED);
    auto expAvgsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_EXP_AVGS_INDEX);
    OP_CHECK_IF(
        expAvgsOutInstanceInfo == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "expAvgsOutInstanceInfo",
                                                 "failed to get the output instance info of exp_avgs from the context"),
        return ge::GRAPH_FAILED);
    auto expAvgSqsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_EXP_AVG_SQS_INDEX);
    OP_CHECK_IF(expAvgSqsOutInstanceInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "expAvgSqsOutInstanceInfo",
                    "failed to get the output instance info of exp_avg_sqs from the context"),
                return ge::GRAPH_FAILED);
    auto maxExpAvgSqsOutInstanceInfo = context->GetIrOutputInstanceInfo(OUTPUT_MAX_EXP_AVG_SQS_INDEX);
    OP_CHECK_IF(maxExpAvgSqsOutInstanceInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "maxExpAvgSqsOutInstanceInfo",
                    "failed to get the output instance info of max_exp_avg_sqs from the context"),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputNum; i++) {
        context->SetOutputDataType(paramsOutInstanceInfo->GetInstanceStart() + i,
                                   context->GetDynamicInputDataType(INPUT_PARAMS_INDEX, i));
        context->SetOutputDataType(expAvgsOutInstanceInfo->GetInstanceStart() + i,
                                   context->GetDynamicInputDataType(INPUT_EXP_AVGS_INDEX, i));
        context->SetOutputDataType(expAvgSqsOutInstanceInfo->GetInstanceStart() + i,
                                   context->GetDynamicInputDataType(INPUT_EXP_AVG_SQS_INDEX, i));
        context->SetOutputDataType(maxExpAvgSqsOutInstanceInfo->GetInstanceStart() + i,
                                   context->GetDynamicInputDataType(INPUT_MAX_EXP_AVG_SQS_INDEX, i));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedAdamw).InferShape(InferShapeForFusedAdamW).InferDataType(InferDataTypeForFusedAdamW);
} // namespace ops
