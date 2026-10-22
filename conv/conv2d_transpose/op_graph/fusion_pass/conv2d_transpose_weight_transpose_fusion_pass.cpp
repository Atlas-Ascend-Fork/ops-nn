/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file conv2d_transpose_weight_transpose_fusion_pass.cpp
 *  \brief Insert the constant weight layout conversion before Conv2DTranspose.
 */
#include "conv2d_transpose_weight_transpose_fusion_pass.h"

#include <cstdint>
#include <string>
#include <vector>

#include "conv/common/op_host/conv_npu_arch_resolver.h"
#include "ge/compliant_node_builder.h"
#include "graph/graph.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"

namespace ops {
using namespace ge;
using namespace ge::es;

namespace {
constexpr char kPassName[] = "Conv2DTransposeWeightTransFusionPass";
constexpr char kProcessedAttr[] = "_nn_conv2d_transpose_weight_transposed";
constexpr int32_t kXIndex = 1;
constexpr int32_t kFilterIndex = 2;
constexpr int32_t kOutputIndex = 0;
constexpr int64_t kInt8C0 = 32;
constexpr int64_t kFp16C0 = 16;
constexpr int64_t kN0 = 16;

int64_t CeilDiv(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }

bool IsFuseSoc()
{
    fe::PlatFormInfos platformInfo;
    fe::OptionalInfos optionalInfo;
    return fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) ==
               SUCCESS &&
           conv_arch::IsCubeVectorFuseSoc(platformInfo);
}

bool GetFourDimShape(const TensorDesc& desc, std::vector<int64_t>& dims)
{
    const auto source = desc.GetShape().GetDims();
    if (source.empty() || source.size() > 4 || (desc.GetFormat() != FORMAT_NCHW && desc.GetFormat() != FORMAT_NHWC)) {
        return false;
    }
    if (source.size() == 1) {
        dims = {1, source[0], 1, 1};
    } else if (source.size() == 2) {
        dims = {1, source[0], source[1], 1};
    } else if (source.size() == 3) {
        dims = {1, source[0], source[1], source[2]};
    } else {
        dims = source;
    }
    return true;
}

std::vector<int64_t> Permute(const std::vector<int64_t>& dims, const std::vector<int64_t>& perm)
{
    std::vector<int64_t> result;
    result.reserve(perm.size());
    for (auto axis : perm) {
        result.push_back(dims[static_cast<size_t>(axis)]);
    }
    return result;
}

TensorDesc WithShapeAndFormat(TensorDesc desc, const std::vector<int64_t>& dims, Format format)
{
    desc.SetShape(Shape(dims));
    desc.SetOriginShape(Shape(dims));
    desc.SetFormat(format);
    desc.SetOriginFormat(format);
    return desc;
}

bool MakeNode(Graph& graph, const std::string& name, const char* type, const TensorDesc& input,
              const TensorDesc& output, GNode& node)
{
    node = CompliantNodeBuilder(&graph)
               .OpType(type)
               .Name(name.c_str())
               .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
               .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
               .Build();
    return node.UpdateInputDesc(kOutputIndex, input) == GRAPH_SUCCESS &&
           node.UpdateOutputDesc(kOutputIndex, output) == GRAPH_SUCCESS;
}

bool MakeReshape(Graph& graph, const std::string& name, const TensorDesc& input, const std::vector<int64_t>& shape,
                 Format format, GNode& node, TensorDesc& output)
{
    output = WithShapeAndFormat(input, shape, format);
    auto shapeAttr = shape;
    return MakeNode(graph, name, "Reshape", input, output, node) && node.SetAttr("shape", shapeAttr) == GRAPH_SUCCESS;
}

bool MakeTranspose(Graph& graph, const std::string& name, const TensorDesc& input, const std::vector<int64_t>& perm,
                   Format format, int64_t groups, bool isA16W8, GNode& node, TensorDesc& output)
{
    TensorDesc transposeInput = input;
    transposeInput.SetFormat(FORMAT_ND);
    transposeInput.SetOriginFormat(FORMAT_ND);
    output = WithShapeAndFormat(input, Permute(input.GetShape().GetDims(), perm), format);
    auto permAttr = perm;
    if (!MakeNode(graph, name, "TransposeD", transposeInput, output, node) ||
        node.SetAttr("perm", permAttr) != GRAPH_SUCCESS) {
        return false;
    }
    return !isA16W8 || node.SetAttr("groups", groups) == GRAPH_SUCCESS;
}

bool MakeReFormat(Graph& graph, const std::string& name, const TensorDesc& input, int64_t groups, bool isA16W8,
                  GNode& node, TensorDesc& output)
{
    TensorDesc reformatInput = input;
    if (isA16W8) {
        const auto dims = input.GetOriginShape().GetDims();
        if (dims.size() != 4 || dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0 || dims[3] <= 0) {
            return false;
        }
        const int64_t c0 = (input.GetDataType() == DT_INT8 || input.GetDataType() == DT_UINT8) ? kInt8C0 : kFp16C0;
        const std::vector<int64_t> fzShape = {dims[2] * dims[3] * CeilDiv(dims[1], c0), CeilDiv(dims[0], kN0), kN0, c0};
        reformatInput.SetFormat(static_cast<Format>(
            GetFormatFromSub(static_cast<int32_t>(FORMAT_FRACTAL_Z), static_cast<int32_t>(groups))));
        reformatInput.SetShape(Shape(fzShape));
        output = reformatInput;
        output.SetFormat(FORMAT_FRACTAL_Z);
        output.SetOriginFormat(FORMAT_NCHW);
    } else {
        output = WithShapeAndFormat(input, input.GetShape().GetDims(), FORMAT_NHWC);
    }
    if (!MakeNode(graph, name, "ReFormat", reformatInput, output, node)) {
        return false;
    }
    return !isA16W8 || node.SetAttr("groups", groups) == GRAPH_SUCCESS;
}

void UpdateFloatDesc(TensorDesc& desc)
{
    if (desc.GetDataType() == DT_FLOAT) {
        desc.SetDataType(DT_FLOAT16);
    }
}

bool UpdateDtypes(GNode& conv, const TensorDesc& weightDesc)
{
    TensorDesc xDesc;
    TensorDesc outputDesc;
    if (conv.GetInputDesc(kXIndex, xDesc) != GRAPH_SUCCESS ||
        conv.GetOutputDesc(kOutputIndex, outputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    TensorDesc adjustedWeight = weightDesc;
    UpdateFloatDesc(xDesc);
    UpdateFloatDesc(adjustedWeight);
    UpdateFloatDesc(outputDesc);
    if (conv.UpdateInputDesc(kXIndex, xDesc) != GRAPH_SUCCESS ||
        conv.UpdateInputDesc(kFilterIndex, adjustedWeight) != GRAPH_SUCCESS ||
        conv.UpdateOutputDesc(kOutputIndex, outputDesc) != GRAPH_SUCCESS) {
        return false;
    }
    if (adjustedWeight.GetDataType() != DT_INT8 || xDesc.GetDataType() != DT_FLOAT16) {
        return true;
    }
    for (const auto& consumer : conv.GetOutDataNodesAndPortIndexs(kOutputIndex)) {
        if (consumer.first == nullptr) {
            continue;
        }
        AscendString type;
        if (consumer.first->GetType(type) != GRAPH_SUCCESS || type != "AscendDequant") {
            continue;
        }
        TensorDesc input;
        TensorDesc output;
        if (consumer.first->GetInputDesc(kOutputIndex, input) != GRAPH_SUCCESS ||
            consumer.first->GetOutputDesc(kOutputIndex, output) != GRAPH_SUCCESS) {
            return false;
        }
        UpdateFloatDesc(input);
        UpdateFloatDesc(output);
        if (consumer.first->UpdateInputDesc(kOutputIndex, input) != GRAPH_SUCCESS ||
            consumer.first->UpdateOutputDesc(kOutputIndex, output) != GRAPH_SUCCESS) {
            return false;
        }
    }
    return true;
}

struct WeightPlan {
    std::vector<int64_t> sourceShape;
    std::vector<int64_t> reshapeIn;
    std::vector<int64_t> perm;
    std::vector<int64_t> reshapeOut;
    int64_t groups = 1;
    bool isA16W8 = false;
};

bool PreparePlan(const GNode& conv, const TensorDesc& filter, WeightPlan& plan)
{
    if (!GetFourDimShape(filter, plan.sourceShape)) {
        return false;
    }
    if (conv.GetAttr("groups", plan.groups) != GRAPH_SUCCESS) {
        plan.groups = 1;
    }
    if (plan.sourceShape.size() != 4 || plan.groups <= 0) {
        return false;
    }
    TensorDesc x;
    if (conv.GetInputDesc(kXIndex, x) != GRAPH_SUCCESS) {
        return false;
    }
    plan.isA16W8 = (x.GetDataType() == DT_FLOAT16 || x.GetDataType() == DT_FLOAT) && filter.GetDataType() == DT_INT8;
    if (plan.isA16W8) {
        for (auto dim : plan.sourceShape) {
            if (dim <= 0) {
                return false;
            }
        }
    }
    if (!plan.isA16W8) {
        plan.perm = filter.GetFormat() == FORMAT_NCHW ? std::vector<int64_t>{0, 2, 3, 1} :
                                                        std::vector<int64_t>{0, 1, 2, 3};
        return true;
    }
    if (plan.groups == 1) {
        plan.perm = filter.GetFormat() == FORMAT_NCHW ? std::vector<int64_t>{1, 0, 2, 3} :
                                                        std::vector<int64_t>{3, 0, 1, 2};
        return true;
    }
    const auto& dims = plan.sourceShape;
    if (dims[0] % plan.groups != 0) {
        return false;
    }
    const int64_t c = dims[filter.GetFormat() == FORMAT_NCHW ? 1 : 3];
    const int64_t h = dims[filter.GetFormat() == FORMAT_NCHW ? 2 : 1];
    const int64_t w = dims[filter.GetFormat() == FORMAT_NCHW ? 3 : 2];
    plan.reshapeIn = filter.GetFormat() == FORMAT_NCHW ?
                         std::vector<int64_t>{plan.groups, dims[0] / plan.groups, c, h, w} :
                         std::vector<int64_t>{plan.groups, dims[0] / plan.groups, h, w, c};
    plan.perm = filter.GetFormat() == FORMAT_NCHW ? std::vector<int64_t>{0, 2, 1, 3, 4} :
                                                    std::vector<int64_t>{0, 4, 1, 2, 3};
    plan.reshapeOut = {plan.groups * c, dims[0] / plan.groups, h, w};
    return true;
}

bool InsertWeightConversion(Graph& graph, GNode& conv, const GNodePtr& filterNode, int32_t filterPort,
                            const TensorDesc& filterDesc, const WeightPlan& plan)
{
    AscendString convName;
    if (conv.GetName(convName) != GRAPH_SUCCESS) {
        return false;
    }
    const std::string prefix = std::string(convName.GetString()) + "_weight_const_fold";
    std::vector<GNode> nodes;
    TensorDesc previousDesc = filterDesc;
    if (!plan.reshapeIn.empty()) {
        GNode reshape;
        TensorDesc output;
        if (!MakeReshape(graph, prefix + "_reshape_group_in", previousDesc, plan.reshapeIn, FORMAT_ND, reshape,
                         output)) {
            return false;
        }
        nodes.push_back(reshape);
        previousDesc = output;
    }

    GNode transpose;
    TensorDesc transposeOutput;
    const Format transposeFormat = plan.isA16W8 && plan.reshapeIn.empty() ? FORMAT_NCHW : FORMAT_ND;
    if (!MakeTranspose(graph, prefix + "_transpose_nc", previousDesc, plan.perm, transposeFormat, plan.groups,
                       plan.isA16W8, transpose, transposeOutput)) {
        return false;
    }
    nodes.push_back(transpose);
    previousDesc = transposeOutput;

    if (!plan.reshapeOut.empty()) {
        GNode reshape;
        TensorDesc output;
        if (!MakeReshape(graph, prefix + "_reshape_group_out", previousDesc, plan.reshapeOut, FORMAT_NCHW, reshape,
                         output)) {
            return false;
        }
        nodes.push_back(reshape);
        previousDesc = output;
    }

    GNode reformat;
    TensorDesc weightDesc;
    if (!MakeReFormat(graph, prefix + "_reformat", previousDesc, plan.groups, plan.isA16W8, reformat, weightDesc)) {
        return false;
    }
    nodes.push_back(reformat);

    if (graph.RemoveEdge(*filterNode, filterPort, conv, kFilterIndex) != GRAPH_SUCCESS) {
        return false;
    }
    if (graph.AddDataEdge(*filterNode, filterPort, nodes.front(), kOutputIndex) != GRAPH_SUCCESS) {
        return false;
    }
    for (size_t i = 1; i < nodes.size(); ++i) {
        if (graph.AddDataEdge(nodes[i - 1], kOutputIndex, nodes[i], kOutputIndex) != GRAPH_SUCCESS) {
            return false;
        }
    }
    bool processed = true;
    if (graph.AddDataEdge(nodes.back(), kOutputIndex, conv, kFilterIndex) != GRAPH_SUCCESS ||
        !UpdateDtypes(conv, weightDesc) || conv.SetAttr(kProcessedAttr, processed) != GRAPH_SUCCESS) {
        return false;
    }
    OP_LOGI(kPassName, "[CONV2D_TRANSPOSE_WEIGHT_TRANSPOSE] applied to %s (A16W8=%d, groups=%ld)", convName.GetString(),
            static_cast<int>(plan.isA16W8), static_cast<long>(plan.groups));
    return true;
}
} // namespace

Status Conv2DTransposeWeightTransFusionPass::Run(GraphPtr& graph, CustomPassContext& passContext)
{
    (void)passContext;
    OP_LOGI(kPassName, "[CONV2D_TRANSPOSE_WEIGHT_TRANSPOSE] pass invoked");
    if (graph == nullptr || !graph->IsValid() || !IsFuseSoc()) {
        return GRAPH_NOT_CHANGED;
    }
    int32_t changed = 0;
    for (auto conv : graph->GetAllNodes()) {
        AscendString type;
        bool processed = false;
        if (conv.GetType(type) != GRAPH_SUCCESS || type != "Conv2DTranspose" ||
            (conv.GetAttr(kProcessedAttr, processed) == GRAPH_SUCCESS && processed)) {
            continue;
        }
        const auto filterPair = conv.GetInDataNodesAndPortIndexs(kFilterIndex);
        if (filterPair.first == nullptr) {
            continue;
        }
        AscendString filterType;
        if (filterPair.first->GetType(filterType) != GRAPH_SUCCESS ||
            (filterType != "Const" && filterType != "Constant")) {
            continue;
        }
        TensorDesc filterDesc;
        if (filterPair.first->GetOutputDesc(filterPair.second, filterDesc) != GRAPH_SUCCESS) {
            return FAILED;
        }
        WeightPlan plan;
        if (!PreparePlan(conv, filterDesc, plan)) {
            continue;
        }
        if (!InsertWeightConversion(*graph, conv, filterPair.first, filterPair.second, filterDesc, plan)) {
            OP_LOGE(kPassName, "Insert weight conversion failed.");
            return FAILED;
        }
        ++changed;
    }
    return changed == 0 ? GRAPH_NOT_CHANGED : SUCCESS;
}

REG_FUSION_PASS(Conv2DTransposeWeightTransFusionPass).Stage(CustomPassStage::kCompatibleInherited);

} // namespace ops
