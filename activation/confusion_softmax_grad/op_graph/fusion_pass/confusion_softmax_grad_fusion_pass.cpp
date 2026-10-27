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

#include "securec.h"
#include "common/inc/error_util.h"
#include "confusion_softmax_grad_fusion_pass.h"
#include "arch_utils.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "ge/ge_utils.h"
#include "graph/utils/type_utils.h"

using namespace ge;
using namespace fusion;

namespace ops {
namespace {
const std::string kPassName = "ZConfusionSoftmaxGradFusionPass";

const int64_t kCaptureSumIdx = 0;
const int64_t kSubgraphInputGrad = 0;
const int64_t kSubgraphInputX = 1;
const int32_t kReduceSumAxesInputIdx = 1;
const size_t kMinSubgraphInputCount = 2;

const int64_t kSubgraphInputAxes = 2;
const int32_t kBinaryInputX1Idx = 0;
const int32_t kBinaryInputX2Idx = 1;
const int32_t kReduceSumInputXIdx = 0;
const int32_t kNodeOutputIdx = 0;
// 平台判定：用 NpuArch（跨版本稳定）判断平台。
const std::set<std::string> kSupportedNpuArchs = {"3510", "9201", "9202"};
// 输入最后一维超过 30000 时，kernel 需要使用 workspace 且无法开启多核，不融合
constexpr int64_t kMaxLength = 30000;

bool IsUnknownShape(const std::vector<int64_t>& dims)
{
    for (auto dim : dims) {
        if (dim < 0) {
            return true;
        }
    }
    return false;
}

bool CheckMulShapeValid(const std::unique_ptr<MatchResult>& match_result)
{
    // Match the original pass: reject unknown dimensions/rank on Mul inputs and output.
    for (const auto& node : match_result->GetMatchedNodes()) {
        AscendString type;
        if (node.GetType(type) != GRAPH_SUCCESS || type != "Mul") {
            continue;
        }
        for (int32_t index = 0; index < 2; ++index) {
            TensorDesc desc;
            if (node.GetInputDesc(index, desc) != GRAPH_SUCCESS || IsUnknownShape(desc.GetShape().GetDims())) {
                return false;
            }
        }
        TensorDesc desc;
        return node.GetOutputDesc(kNodeOutputIdx, desc) == GRAPH_SUCCESS && !IsUnknownShape(desc.GetShape().GetDims());
    }
    return false;
}

bool GetCapturedSumNode(const std::unique_ptr<MatchResult>& match_result, GNode& sum_node)
{
    NodeIo node_io;
    OP_LOGE_IF(match_result->GetCapturedTensor(kCaptureSumIdx, node_io) != SUCCESS, false, kPassName.c_str(),
               "Failed to get captured sum node.");
    sum_node = node_io.node;
    return true;
}

// 被替换的 Sub 节点出度必须为 1（出度!=1 时不融合）。
bool CheckSubOutDegree(const std::unique_ptr<MatchResult>& match_result)
{
    for (const auto& matched_node : match_result->GetMatchedNodes()) {
        AscendString node_type;
        if (matched_node.GetType(node_type) != GRAPH_SUCCESS || node_type != "Sub") {
            continue;
        }
        const auto out_nodes = matched_node.GetOutDataNodesAndPortIndexs(kNodeOutputIdx);
        if (out_nodes.size() != 1U) {
            OPS_LOG_D(kPassName.c_str(), "Sub node out-degree is %zu (expect 1), skip fusion.", out_nodes.size());
            return false;
        }
        return true;
    }
    OPS_LOG_E(kPassName.c_str(), "Matched Sub node not found for out-degree check.");
    return false;
}

// ReduceSum 在新 IR 中 `axes` 为第 1 个 const 输入，`keep_dims` 为属性。
bool GetAxisFromReduceSum(const GNode& sum_node, int64_t& axis_value, bool& keep_dims)
{
    Tensor axes_tensor;
    if (sum_node.GetInputConstData(kReduceSumAxesInputIdx, axes_tensor) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get axes const input from ReduceSum.");
        return false;
    }
    const DataType dtype = axes_tensor.GetDataType();
    const int32_t elem_size = GetSizeByDataType(dtype);
    if (elem_size <= 0 || axes_tensor.GetSize() != static_cast<size_t>(elem_size)) {
        OPS_LOG_D(kPassName.c_str(), "ReduceSum axes must be a single int, but size is %zu.", axes_tensor.GetSize());
        return false;
    }
    const auto* data = axes_tensor.GetData();
    if (data == nullptr) {
        OPS_LOG_D(kPassName.c_str(), "ReduceSum axes const data is nullptr.");
        return false;
    }
    if (dtype == DT_INT64) {
        int64_t tmp = 0;
        auto memRet = memcpy_s(&tmp, sizeof(int64_t), data, sizeof(int64_t));
        OP_LOGE_IF(memRet != EOK, false, kPassName.c_str(), "memcpy_s for int64 axes failed, ret=%d.", memRet);
        axis_value = tmp;
    } else if (dtype == DT_INT32) {
        int32_t tmp = 0;
        auto memRet = memcpy_s(&tmp, sizeof(int32_t), data, sizeof(int32_t));
        OP_LOGE_IF(memRet != EOK, false, kPassName.c_str(), "memcpy_s for int32 axes failed, ret=%d.", memRet);
        axis_value = tmp;
    } else {
        OPS_LOG_D(kPassName.c_str(), "ReduceSum axes dtype %d is not supported.", static_cast<int32_t>(dtype));
        return false;
    }

    if (sum_node.GetAttr(AscendString("keep_dims"), keep_dims) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get keep_dims attr from ReduceSum.");
        return false;
    }
    OPS_LOG_D(kPassName.c_str(), "ReduceSum axis=%lld, keep_dims=%d.", axis_value, static_cast<int32_t>(keep_dims));
    return true;
}

// Mul 输出（即 ReduceSum 输入 0）的 dtype 守卫：仅支持 fp16/fp32/bf16。
bool IsLegalDtype(const GNode& sum_node)
{
    TensorDesc mul_out_desc;
    if (sum_node.GetInputDesc(kReduceSumInputXIdx, mul_out_desc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get input desc of ReduceSum.");
        return false;
    }
    const DataType mul_dtype = mul_out_desc.GetDataType();
    if (mul_dtype != DT_FLOAT16 && mul_dtype != DT_FLOAT && mul_dtype != DT_BF16) {
        OPS_LOG_D(kPassName.c_str(), "Dtype of mul output only support fp16, fp32, bf16, but got [%s].",
                  TypeUtils::DataTypeToAscendString(mul_dtype).GetString());
        return false;
    }
    return true;
}

// ConfusionSoftmaxGrad 要求 grad 与 x 的 shape 一致。
bool IsSubgraphInputsShapeEqual(const std::vector<SubgraphInput>& subgraph_inputs)
{
    if (subgraph_inputs.size() < kMinSubgraphInputCount) {
        OPS_LOG_D(kPassName.c_str(), "Subgraph inputs size %zu is less than %zu.", subgraph_inputs.size(),
                  kMinSubgraphInputCount);
        return false;
    }
    TensorDesc grad_desc;
    TensorDesc x_desc;
    const auto grad_inputs = subgraph_inputs[kSubgraphInputGrad].GetAllInputs();
    const auto x_inputs = subgraph_inputs[kSubgraphInputX].GetAllInputs();
    if (grad_inputs.empty() || x_inputs.empty()) {
        return false;
    }
    if (grad_inputs.at(0).node.GetInputDesc(grad_inputs.at(0).index, grad_desc) != GRAPH_SUCCESS ||
        x_inputs.at(0).node.GetInputDesc(x_inputs.at(0).index, x_desc) != GRAPH_SUCCESS) {
        return false;
    }
    if (grad_desc.GetShape().GetDims() != x_desc.GetShape().GetDims()) {
        OPS_LOG_D(kPassName.c_str(), "The input0 shape and input1 shape are not equal, can't be fused.");
        return false;
    }
    return true;
}

// 守卫：axis 为最后一维（兼容 -1 与 dimNum-1 等写法）、输入非标量、最后一维不超过 30000。
bool CheckAxisAndLength(const GNode& sum_node, int64_t axis_value)
{
    TensorDesc reduce_sum_input_desc;
    if (sum_node.GetInputDesc(kReduceSumInputXIdx, reduce_sum_input_desc) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Failed to get input desc of ReduceSum.");
        return false;
    }
    const std::vector<int64_t> reduce_sum_dims = reduce_sum_input_desc.GetShape().GetDims();
    const int64_t dim_num = static_cast<int64_t>(reduce_sum_dims.size());
    const int64_t axis = axis_value < 0 ? axis_value + dim_num : axis_value;

    // 输入为标量，不融合
    if (dim_num == 0) {
        OPS_LOG_D(kPassName.c_str(), "ReduceSum input dim num is 0.");
        return false;
    }

    // axis 不是最后一维，不融合
    if ((axis % dim_num) != (dim_num - 1)) {
        OPS_LOG_D(kPassName.c_str(), "ReduceSum axis %lld is not the last dim, dim num is %lld.", axis_value, dim_num);
        return false;
    }

    // 最后一维超过 30000，不融合
    if (reduce_sum_dims[dim_num - 1] > kMaxLength) {
        OPS_LOG_D(kPassName.c_str(), "Last dim %lld is more than %lld.", reduce_sum_dims[dim_num - 1], kMaxLength);
        return false;
    }
    return true;
}

std::vector<es::EsTensorHolder> CreateReplacementInputs(es::EsGraphBuilder& graph_builder,
                                                        const std::vector<SubgraphInput>& subgraph_inputs)
{
    std::vector<es::EsTensorHolder> inputs;
    for (size_t i = 0; i < subgraph_inputs.size(); ++i) {
        const auto all_inputs = subgraph_inputs[i].GetAllInputs();
        if (all_inputs.empty()) {
            OPS_LOG_E(kPassName.c_str(), "Subgraph input %zu is empty.", i);
            return {};
        }
        TensorDesc tensor_desc;
        const auto match_node = all_inputs.at(0);
        if (match_node.node.GetInputDesc(match_node.index, tensor_desc) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Get subgraph input %zu desc failed.", i);
            return {};
        }
        auto data = graph_builder.CreateInput(
            static_cast<int64_t>(i), ("replacement_input_" + std::to_string(i)).c_str(), tensor_desc.GetDataType(),
            tensor_desc.GetFormat(), tensor_desc.GetShape().GetDims());
        inputs.emplace_back(data);
    }
    return inputs;
}

// Build a two-input one-output element-wise node (Mul/Sub) with CompliantNodeBuilder.
es::EsTensorHolder BuildBinaryNode(es::EsGraphBuilder& graph_builder, const es::EsTensorHolder& input0,
                                   const es::EsTensorHolder& input1, const char* op_type)
{
    auto* c_builder = graph_builder.GetCGraphBuilder();
    auto* graph = c_builder->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType(op_type)
                     .Name(c_builder->GenerateNodeName(op_type).GetString())
                     .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    ES_ASSERT_GRAPH_SUCCESS(es::AddEdgeAndUpdatePeerDesc(*graph, *input0.GetProducer(), input0.GetProducerOutIndex(),
                                                         node, kBinaryInputX1Idx));
    ES_ASSERT_GRAPH_SUCCESS(es::AddEdgeAndUpdatePeerDesc(*graph, *input1.GetProducer(), input1.GetProducerOutIndex(),
                                                         node, kBinaryInputX2Idx));
    return es::EsTensorHolder(c_builder->GetTensorHolderFromNode(node, kNodeOutputIdx));
}

// Keep axes on the boundary so its Const may also serve consumers outside the pattern.
es::EsTensorHolder BuildPatternReduceSum(es::EsGraphBuilder& graph_builder, const es::EsTensorHolder& input,
                                         const es::EsTensorHolder& axes)
{
    auto* c_builder = graph_builder.GetCGraphBuilder();
    auto* graph = c_builder->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
                     .OpType("ReduceSum")
                     .Name(c_builder->GenerateNodeName("ReduceSum").GetString())
                     .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs(
                         {{"keep_dims", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(true)},
                          {"noop_with_empty_axes", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                           es::CreateFrom(true)}})
                     .Build();
    ES_ASSERT_GRAPH_SUCCESS(es::AddEdgeAndUpdatePeerDesc(*graph, *input.GetProducer(), input.GetProducerOutIndex(),
                                                         node, kReduceSumInputXIdx));
    ES_ASSERT_GRAPH_SUCCESS(es::AddEdgeAndUpdatePeerDesc(*graph, *axes.GetProducer(), axes.GetProducerOutIndex(), node,
                                                         kReduceSumAxesInputIdx));
    return es::EsTensorHolder(c_builder->GetTensorHolderFromNode(node, kNodeOutputIdx));
}

// Pattern: input0 -> Mul(input0, input1) -> ReduceSum(axes, keep_dims) -> Sub(input0, sum) -> output
// input0 同时作为 Mul 与 Sub 的输入，表达“Sub 与 Mul 共享 input0”的约束。
PatternUniqPtr MakePatternConfusionSoftmaxGrad(bool axes_on_boundary)
{
    auto graph_builder = es::EsGraphBuilder(kPassName.c_str());
    auto input0 = graph_builder.CreateInput(kSubgraphInputGrad, "grad");
    auto input1 = graph_builder.CreateInput(kSubgraphInputX, "x");

    auto mul = BuildBinaryNode(graph_builder, input0, input1, "Mul");
    auto axes = axes_on_boundary ? graph_builder.CreateInput(kSubgraphInputAxes, "axes") :
                                   graph_builder.CreateConst(std::vector<int64_t>{-1}, std::vector<int64_t>{1});
    auto sum = BuildPatternReduceSum(graph_builder, mul, axes);
    auto sub = BuildBinaryNode(graph_builder, input0, sum, "Sub");

    auto graph = graph_builder.BuildAndReset({sub});
    OP_LOGE_IF(graph == nullptr, nullptr, kPassName.c_str(), "BuildAndReset for pattern failed.");
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*sum.GetProducer(), kNodeOutputIdx});
    return pattern;
}

// 融合节点输出 desc 沿用原 Sub 节点（pattern 输出）的输出 desc。
bool CopySubOutputDesc(const std::unique_ptr<MatchResult>& match_result, GNode& fused_node)
{
    for (const auto& matched_node : match_result->GetMatchedNodes()) {
        AscendString node_type;
        if (matched_node.GetType(node_type) != GRAPH_SUCCESS || node_type != "Sub") {
            continue;
        }
        TensorDesc sub_out_desc;
        OP_LOGE_IF(matched_node.GetOutputDesc(kNodeOutputIdx, sub_out_desc) != GRAPH_SUCCESS, false, kPassName.c_str(),
                   "Get output desc of matched Sub node failed.");
        TensorDesc fused_out_desc;
        fused_node.GetOutputDesc(kNodeOutputIdx, fused_out_desc);
        fused_out_desc.SetDataType(sub_out_desc.GetDataType());
        fused_out_desc.SetFormat(sub_out_desc.GetFormat());
        fused_out_desc.SetShape(sub_out_desc.GetShape());
        fused_out_desc.SetOriginFormat(sub_out_desc.GetOriginFormat());
        fused_out_desc.SetOriginShape(sub_out_desc.GetOriginShape());
        OP_LOGE_IF(fused_node.UpdateOutputDesc(kNodeOutputIdx, fused_out_desc) != GRAPH_SUCCESS, false,
                   kPassName.c_str(), "Update output desc of fused node failed.");
        return true;
    }
    OPS_LOG_E(kPassName.c_str(), "Matched Sub node not found.");
    return false;
}
} // namespace

std::vector<PatternUniqPtr> ZConfusionSoftmaxGradFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for ZConfusionSoftmaxGradFusionPass.");
    std::vector<PatternUniqPtr> patterns;
    if (!CheckNpuArch(kPassName, kSupportedNpuArchs)) {
        OPS_LOG_D(kPassName.c_str(), "Unsupported platform, skip ZConfusionSoftmaxGradFusionPass.");
        return patterns;
    }
    // Consume an unshared Const with the original pattern; use a boundary input
    // as a fallback for shared axes so external consumers remain connected.
    for (bool axes_on_boundary : {false, true}) {
        auto pattern = MakePatternConfusionSoftmaxGrad(axes_on_boundary);
        if (pattern == nullptr) {
            OPS_LOG_E(kPassName.c_str(), "Create fusion pattern failed.");
            return {};
        }
        patterns.emplace_back(std::move(pattern));
    }
    return patterns;
}

bool ZConfusionSoftmaxGradFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter MeetRequirements for ZConfusionSoftmaxGradFusionPass.");
    if (!CheckMulShapeValid(match_result)) {
        return false;
    }

    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
    if (!IsSubgraphInputsShapeEqual(subgraph_inputs)) {
        return false;
    }

    GNode sum_node;
    OP_LOGE_IF(!GetCapturedSumNode(match_result, sum_node), false, kPassName.c_str(),
               "Get captured ReduceSum node failed.");

    if (!CheckSubOutDegree(match_result)) {
        return false;
    }

    if (!IsLegalDtype(sum_node)) {
        return false;
    }

    int64_t axis_value = 0;
    bool keep_dims = true;
    OP_LOGE_IF(!GetAxisFromReduceSum(sum_node, axis_value, keep_dims), false, kPassName.c_str(),
               "Failed to get axis/keep_dims from ReduceSum.");
    if (!keep_dims) {
        OPS_LOG_D(kPassName.c_str(), "keep_dims is false, skip fusion.");
        return false;
    }

    return CheckAxisAndLength(sum_node, axis_value);
}

GraphUniqPtr ZConfusionSoftmaxGradFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "Enter Replacement for ZConfusionSoftmaxGradFusionPass.");

    std::vector<SubgraphInput> subgraph_inputs;
    match_result->ToSubgraphBoundary()->GetAllInputs(subgraph_inputs);
    OP_LOGE_IF(subgraph_inputs.size() < kMinSubgraphInputCount, nullptr, kPassName.c_str(),
               "Subgraph inputs size %zu is less than %zu.", subgraph_inputs.size(), kMinSubgraphInputCount);

    auto graph_builder = es::EsGraphBuilder("replacement");
    // Keep the boundary input indexes, including the unused axes placeholder in
    // the shared-Const pattern. Only grad and x feed the fused operator.
    auto replacement_inputs = CreateReplacementInputs(graph_builder, subgraph_inputs);
    OP_LOGE_IF(replacement_inputs.size() < kMinSubgraphInputCount, nullptr, kPassName.c_str(),
               "Create replacement inputs failed.");

    // ConfusionSoftmaxGrad(grad, x): grad=input0, x=input1，输出接管 Sub 的外部消费。
    auto confusion_softmax_grad = es::ConfusionSoftmaxGrad(replacement_inputs[kSubgraphInputGrad],
                                                           replacement_inputs[kSubgraphInputX]);

    GraphUniqPtr replace_graph = graph_builder.BuildAndReset({confusion_softmax_grad});
    OP_LOGE_IF(replace_graph == nullptr, nullptr, kPassName.c_str(), "BuildAndReset for replacement failed.");
    // 按子图边界取输入 shape，做全图 shape/dtype 推导（GeUtils::InferShape，8.5.0 接口）。
    std::vector<Shape> input_shapes;
    for (const auto& subgraph_input : subgraph_inputs) {
        const auto all_inputs = subgraph_input.GetAllInputs();
        if (all_inputs.empty()) {
            OPS_LOG_E(kPassName.c_str(), "Subgraph input is empty when infer shape.");
            return nullptr;
        }
        TensorDesc input_desc;
        if (all_inputs.at(0).node.GetInputDesc(all_inputs.at(0).index, input_desc) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "Get subgraph input desc failed when infer shape.");
            return nullptr;
        }
        input_shapes.emplace_back(input_desc.GetShape());
    }
    if (GeUtils::InferShape(*replace_graph, input_shapes) != SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "InferShape for replacement failed.");
        return nullptr;
    }
    OP_LOGE_IF(!CopySubOutputDesc(match_result, *confusion_softmax_grad.GetProducer()), nullptr, kPassName.c_str(),
               "Copy output desc from Sub node failed.");
    return replace_graph;
}

REG_FUSION_PASS(ZConfusionSoftmaxGradFusionPass).Stage(CustomPassStage::kAfterBuiltinFusionPass);

} // namespace ops
