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
 * \file bn3d_training_reduce_graph_infer.cpp
 * \brief BN3DTrainingReduce 图推导：输出恒 fp32，并规范 dense 通道向量的格式。
 *        InferDataType 仅图场景使用，故落在 op_graph；InferShape 图与单算子共用，留在 op_host。
 */

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
// 与 op_host/bn3d_training_reduce_tiling.h 里的同名常量取值一致；两处分属 op_graph 与
// op_host，不跨层 include，故各留一份，靠 REG_OP 的输出序（sum, square_sum）约束。
constexpr size_t OUTPUT_SUM_INDEX = 0;
constexpr size_t OUTPUT_SQUARE_SUM_INDEX = 1;

bool IsDenseChannelPlatform()
{
    fe::PlatFormInfos infos;
    fe::OptionalInfos options;
    std::string arch;
    return fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(infos, options) == ge::GRAPH_SUCCESS &&
           infos.GetPlatformRes("version", "NpuArch", arch) && arch == "3510";
}
} // namespace

static ge::graphStatus InferDataType4BN3DTrainingReduce(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4BN3DTrainingReduce");
    // 输出恒 fp32（不随输入 dtype，fp16 / bf16 在内核内提升为 fp32 累加）。
    context->SetOutputDataType(OUTPUT_SUM_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_SQUARE_SUM_INDEX, ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4BN3DTrainingReduce");
    return ge::GRAPH_SUCCESS;
}

static ge::Status NormalizeChannelVectorFormat(ge::GNode& node)
{
    ge::TensorDesc inputDesc;
    if (node.GetInputDesc(0, inputDesc) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const auto inputFormat = ge::GetPrimaryFormat(inputDesc.GetOriginFormat());
    if (inputFormat != ge::FORMAT_NDHWC && inputFormat != ge::FORMAT_NCDHW) {
        return ge::GRAPH_SUCCESS;
    }
    ge::TensorDesc outputDescs[2];
    bool normalize[2] = {false, false};
    for (int32_t i = 0; i < 2; ++i) {
        if (node.GetOutputDesc(i, outputDescs[i]) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        normalize[i] = ge::GetPrimaryFormat(outputDescs[i].GetOriginFormat()) == ge::FORMAT_NDHWC &&
                       outputDescs[i].GetOriginShape().GetDimNum() == 1;
    }
    // Channel vectors have no spatial axes. Keeping NDHWC as their origin
    // format makes GE insert a five-axis output Transpose for unknown rank.
    // Use the registered dense output storage without changing the [C] shape.
    for (int32_t i = 0; i < 2; ++i) {
        if (!normalize[i]) {
            continue;
        }
        auto& outputDesc = outputDescs[i];
        outputDesc.SetOriginFormat(ge::FORMAT_NCDHW);
        outputDesc.SetFormat(ge::FORMAT_NCDHW);
        if (node.UpdateOutputDesc(i, outputDesc) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        for (const auto& consumer : node.GetOutDataNodesAndPortIndexs(i)) {
            ge::TensorDesc input;
            if (consumer.first == nullptr ||
                consumer.first->GetInputDesc(consumer.second, input) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
            input.SetOriginFormat(ge::FORMAT_NCDHW);
            input.SetFormat(ge::FORMAT_NCDHW);
            if (consumer.first->UpdateInputDesc(consumer.second, input) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    OP_LOGD("BN3DTrainingReduceOutputFormatPass", "Normalized channel-vector output formats");
    return ge::GRAPH_SUCCESS;
}

static ge::Status NormalizeBN3DOutputFormat(ge::GraphPtr& graph, ge::CustomPassContext&)
{
    if (graph == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (!IsDenseChannelPlatform()) {
        return ge::GRAPH_SUCCESS;
    }
    for (auto& node : graph->GetAllNodes()) {
        ge::AscendString type;
        if (node.GetType(type) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (type == ge::AscendString("BN3DTrainingReduce") && NormalizeChannelVectorFormat(node) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("BN3DTrainingReduceOutputFormatPass")
    .CustomPassFn(NormalizeBN3DOutputFormat)
    .Stage(ge::CustomPassStage::kAfterInferShape);

IMPL_OP(BN3DTrainingReduce).InferDataType(InferDataType4BN3DTrainingReduce);
} // namespace ops
