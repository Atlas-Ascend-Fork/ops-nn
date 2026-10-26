/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <set>
#include <string>

#include "common/inc/error_util.h"
#include "softmax_v2_onnx_fusion_pass.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"

using namespace ge;
using namespace fusion;

namespace ops {
namespace {
const std::string kPassName = "ASoftmaxFusionPass";
const int64_t kCaptureSoftmaxIdx = 0;

// 平台判定：与 layer_norm 融合规则一致，用 NpuArch（跨版本稳定）判断是否 regbase 架构。
const std::set<std::string> kSupportedNpuArch = {"3510", "5102"};

bool IsRegbase()
{
    fe::PlatFormInfos platform_infos;
    fe::OptionalInfos optional_infos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) !=
        SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Get platform info failed, not regbase.");
        return false;
    }
    std::string arch_str;
    if (!platform_infos.GetPlatformRes("version", "NpuArch", arch_str)) {
        OPS_LOG_D(kPassName.c_str(), "NpuArch not found, not regbase.");
        return false;
    }
    OPS_LOG_D(kPassName.c_str(), "Platform NpuArch: %s", arch_str.c_str());
    if (kSupportedNpuArch.count(arch_str) == 0U) {
        OPS_LOG_D(kPassName.c_str(), "NpuArch=%s is not regbase, skip.", arch_str.c_str());
        return false;
    }
    return true;
}

struct SoftmaxInfo {
    std::vector<int64_t> dims;
    int64_t axis = 0;
    DataType dtype = DT_FLOAT;
    Format format = FORMAT_ND;
    DataType out_dtype = DT_FLOAT; // 原节点输出 dtype（half_to_float=true 时与输入不同）
    Format out_format = FORMAT_ND;
    bool half_to_float = false;
    bool valid = false;
};

bool GetSoftmaxNode(const std::unique_ptr<MatchResult>& match_result, GNode& softmax_node)
{
    NodeIo node_io;
    OP_LOGE_IF(match_result->GetCapturedTensor(static_cast<size_t>(kCaptureSoftmaxIdx), node_io) != SUCCESS, false,
               kPassName.c_str(), "Failed to get captured SoftmaxV2 node.");
    softmax_node = node_io.node;
    return true;
}

SoftmaxInfo GetSoftmaxInfo(const GNode& softmax_node)
{
    SoftmaxInfo info;
    TensorDesc input_desc;
    if (softmax_node.GetInputDesc(0, input_desc) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "GetInputDesc(0) failed.");
        return info;
    }
    info.dims = input_desc.GetShape().GetDims();
    info.dtype = input_desc.GetDataType();
    info.format = input_desc.GetFormat();

    int64_t need_fusion = 0;
    if (softmax_node.GetAttr(AscendString("need_fusion"), need_fusion) != GRAPH_SUCCESS || need_fusion == 0) {
        OPS_LOG_D(kPassName.c_str(), "need_fusion attr missing or zero, skip.");
        return info;
    }

    std::vector<int64_t> axes;
    if (softmax_node.GetAttr(AscendString("axes"), axes) != GRAPH_SUCCESS || axes.empty()) {
        OPS_LOG_D(kPassName.c_str(), "axes attr missing or empty, skip.");
        return info;
    }

    int64_t raw_axis = axes[0];
    info.axis = raw_axis >= 0 ? raw_axis : raw_axis + static_cast<int64_t>(info.dims.size());
    (void)softmax_node.GetAttr(AscendString("half_to_float"), info.half_to_float);

    // 读取原节点输出 desc（half_to_float=true 时输出 dtype 可能与输入不同）
    TensorDesc output_desc;
    if (softmax_node.GetOutputDesc(0, output_desc) == GRAPH_SUCCESS) {
        info.out_dtype = output_desc.GetDataType();
        info.out_format = output_desc.GetFormat();
    } else {
        info.out_dtype = info.dtype;
        info.out_format = info.format;
    }
    info.valid = true;
    return info;
}

bool CheckGuards(const SoftmaxInfo& info)
{
    if (!info.valid) {
        return false;
    }
    if (info.dims.size() < 3UL) {
        OPS_LOG_D(kPassName.c_str(), "dims.size() < 3, skip.");
        return false;
    }
    if (info.axis == static_cast<int64_t>(info.dims.size()) - 1) {
        OPS_LOG_D(kPassName.c_str(), "axis is last dim, no need reshape.");
        return false;
    }
    return true;
}

void SetNodeOutputDesc(GNode& node, int32_t out_idx, const std::vector<int64_t>& out_dims, DataType dtype,
                       Format format)
{
    TensorDesc out_desc;
    out_desc.SetDataType(dtype);
    out_desc.SetFormat(format);
    out_desc.SetOriginFormat(format);
    out_desc.SetShape(Shape(out_dims));
    out_desc.SetOriginShape(Shape(out_dims));
    node.UpdateOutputDesc(out_idx, out_desc);
}

es::EsTensorHolder BuildReshapeNode(es::EsGraphBuilder& graph_builder, const es::EsTensorHolder& input,
                                    const std::vector<int64_t>& target_dims, DataType dtype, Format format,
                                    const char* base_name)
{
    auto shape_const = graph_builder.CreateConst(target_dims, {static_cast<int64_t>(target_dims.size())});
    auto* c_builder = graph_builder.GetCGraphBuilder();
    auto* graph = c_builder->GetGraph();
    GNode reshape_node = es::CompliantNodeBuilder(graph)
                             .OpType("Reshape")
                             .Name(c_builder->GenerateNodeName(base_name).GetString())
                             .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                           {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), reshape_node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *shape_const.GetProducer(), shape_const.GetProducerOutIndex(), reshape_node,
                                 1);
    SetNodeOutputDesc(reshape_node, 0, target_dims, dtype, format);
    return es::EsTensorHolder(c_builder->GetTensorHolderFromNode(reshape_node, 0));
}

es::EsTensorHolder BuildSoftmaxV2Node(es::EsGraphBuilder& graph_builder, const es::EsTensorHolder& input,
                                      const std::vector<int64_t>& axes, bool half_to_float)
{
    auto* c_builder = graph_builder.GetCGraphBuilder();
    auto* graph = c_builder->GetGraph();
    GNode softmax_node = es::CompliantNodeBuilder(graph)
                             .OpType("SoftmaxV2")
                             .Name(c_builder->GenerateNodeName("SoftmaxV2").GetString())
                             .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .IrDefAttrs(
                                 {{"axes", es::CompliantNodeBuilder::kEsAttrOptional, "ListInt", es::CreateFrom(axes)},
                                  {"half_to_float", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                   es::CreateFrom(half_to_float)}})
                             .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(), softmax_node, 0);
    return es::EsTensorHolder(c_builder->GetTensorHolderFromNode(softmax_node, 0));
}
} // namespace

std::vector<PatternUniqPtr> ASoftmaxFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for ASoftmaxFusionPass.");
    std::vector<PatternUniqPtr> patterns;
    if (!IsRegbase()) {
        OPS_LOG_D(kPassName.c_str(), "Not regbase platform, skip ASoftmaxFusionPass.");
        return patterns;
    }
    auto graph_builder = es::EsGraphBuilder(kPassName.c_str());
    auto x = graph_builder.CreateInput(0);
    auto* c_builder = graph_builder.GetCGraphBuilder();
    auto* graph = c_builder->GetGraph();
    GNode softmax_node = es::CompliantNodeBuilder(graph)
                             .OpType("SoftmaxV2")
                             .Name(c_builder->GenerateNodeName("SoftmaxV2").GetString())
                             .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                             .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), softmax_node, 0);
    auto y = es::EsTensorHolder(c_builder->GetTensorHolderFromNode(softmax_node, 0));
    auto pattern_graph = graph_builder.BuildAndReset({y});
    OP_LOGE_IF(pattern_graph == nullptr, patterns, kPassName.c_str(), "BuildAndReset for pattern failed.");
    auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph));
    pattern->CaptureTensor({softmax_node, 0});
    patterns.emplace_back(std::move(pattern));
    return patterns;
}

bool ASoftmaxFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for ASoftmaxFusionPass.");
    GNode softmax_node;
    if (!GetSoftmaxNode(match_result, softmax_node)) {
        return false;
    }
    auto info = GetSoftmaxInfo(softmax_node);
    return CheckGuards(info);
}

GraphUniqPtr ASoftmaxFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for ASoftmaxFusionPass.");
    GNode softmax_node;
    OP_LOGE_IF(!GetSoftmaxNode(match_result, softmax_node), nullptr, kPassName.c_str(),
               "Get captured SoftmaxV2 node failed.");
    auto info = GetSoftmaxInfo(softmax_node);
    OP_LOGE_IF(!CheckGuards(info), nullptr, kPassName.c_str(), "Guards check failed in Replacement.");

    int64_t dim_1 = 1;
    int64_t dim_2 = 1;
    for (size_t idx = 0; idx < info.dims.size(); idx++) {
        if (idx < static_cast<size_t>(info.axis)) {
            dim_1 *= info.dims[idx];
        } else {
            dim_2 *= info.dims[idx];
        }
    }
    OPS_LOG_D(kPassName.c_str(), "Reshape to [dim_1=%ld, dim_2=%ld], origin dims size=%zu.", dim_1, dim_2,
              info.dims.size());

    auto graph_builder = es::EsGraphBuilder("replacement");
    auto x = graph_builder.CreateInput(0, "x", info.dtype, info.format, info.dims);

    auto reshape1_out = BuildReshapeNode(graph_builder, x, {dim_1, dim_2}, info.dtype, info.format, "ReshapeFlatten");

    auto softmax_out = BuildSoftmaxV2Node(graph_builder, reshape1_out, {1}, info.half_to_float);
    GNode softmax_new_node = *softmax_out.GetProducer();
    SetNodeOutputDesc(softmax_new_node, 0, {dim_1, dim_2}, info.out_dtype, info.out_format);

    auto reshape2_out = BuildReshapeNode(graph_builder, softmax_out, info.dims, info.out_dtype, info.out_format,
                                         "ReshapeRestore");

    GraphUniqPtr replace_graph = graph_builder.BuildAndReset({reshape2_out});
    OP_LOGE_IF(replace_graph == nullptr, nullptr, kPassName.c_str(), "BuildAndReset for replacement failed.");
    if (GeUtils::InferShape(*replace_graph, {Shape(info.dims)}) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }
    return replace_graph;
}

REG_FUSION_PASS(ASoftmaxFusionPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);
} // namespace ops
