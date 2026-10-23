/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file biasadd_conv_fusion_pass.cpp
 *  \brief conv + biasadd/add --> conv with bias
 */
#include "biasadd_conv_fusion_pass.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "conv/common/op_graph/fusion_pass/conv_fusion_utils_pass.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"

namespace Ops {
using namespace NN;
using namespace Conv;
using namespace BiasaddConvFusion;
using namespace ConvFusionUtils;
using namespace ge;
using namespace fe;
using namespace fusion;

namespace {

void SyncOriginDataType(TensorDesc& desc)
{
#if GE_COMPILER_VERSION_NUM >= 90300000U
    if (desc.GetOriginDataType() == DT_UNDEFINED) {
        desc.SetOriginDataType(desc.GetDataType());
    }
#endif
}

bool IsConstCandidate(const GNode& consumer, int32_t inputIndex)
{
    const auto source = consumer.GetInDataNodesAndPortIndexs(inputIndex).first;
    if (source == nullptr) {
        return false;
    }
    AscendString sourceType;
    source->GetType(sourceType);
    if (sourceType == CONST) {
        return true;
    }
    if (sourceType != "Data") {
        return false;
    }
    const auto parent = source->GetInDataNodesAndPortIndexs(0).first;
    if (parent != nullptr) {
        AscendString parentType;
        parent->GetType(parentType);
        return parentType == CONST;
    }
    return false;
}

// Conv2D/DepthwiseConv2D/Conv3D require the bias channel dim to match Cout
bool CheckCoutMatch(const TensorDesc& convOutputDesc, const AscendString& convType, int64_t biasDim)
{
    size_t channelIndex = 0U;
    if (convType == CONV2D || convType == DEPTHWISE_CONV2D) {
        if (convOutputDesc.GetOriginFormat() == FORMAT_NCHW) {
            channelIndex = 1U;
        } else if (convOutputDesc.GetOriginFormat() == FORMAT_NHWC) {
            channelIndex = NHWC_CHANNEL_AXIS;
        } else if (convOutputDesc.GetOriginFormat() == FORMAT_HWCN) {
            channelIndex = HWCN_CHANNEL_AXIS;
        } else {
            return false;
        }
    } else if (convType == CONV3D) {
        if (convOutputDesc.GetOriginFormat() == FORMAT_NCDHW) {
            channelIndex = 1U;
        } else if (convOutputDesc.GetOriginFormat() == FORMAT_NDHWC) {
            channelIndex = NDHWC_CHANNEL_AXIS;
        } else {
            return false;
        }
    } else {
        return true;
    }
    const auto outputDims = convOutputDesc.GetOriginShape().GetDims();
    return channelIndex < outputDims.size() && biasDim == outputDims[channelIndex];
}

bool NormalizeConstShape(const TensorDesc& convOutputDesc, const TensorDesc& biasDesc, const AscendString& convType,
                         std::vector<int64_t>& normalizedShape)
{
    const auto dims = biasDesc.GetShape().GetDims();
    if (dims.empty()) {
        return false;
    }
    if (dims.size() == 1U) {
        if (!CheckCoutMatch(convOutputDesc, convType, dims[0])) {
            return false;
        }
        normalizedShape = dims;
        return true;
    }
    int64_t oneCount = 0;
    int64_t lastNonOne = 1;
    for (const auto dim : dims) {
        if (dim == -1 || dim == UNKNOWN_DIM_SENTINEL) {
            return false;
        }
        if (dim == 1) {
            ++oneCount;
        } else {
            lastNonOne = dim;
        }
    }
    if (oneCount < MIN_ONE_DIM_COUNT) {
        return false;
    }
    if (!CheckCoutMatch(convOutputDesc, convType, lastNonOne)) {
        return false;
    }
    normalizedShape = {lastNonOne};
    return true;
}

bool CheckConstBias(const GNode& conv, const Tensor& biasTensor, const AscendString& convType)
{
    TensorDesc convOutputDesc;
    FUSION_PASS_CHECK(conv.GetOutputDesc(0, convOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get conv output desc failed, no fusion."), return false);
    std::vector<int64_t> normalizedShape;
    FUSION_PASS_CHECK(!NormalizeConstShape(convOutputDesc, biasTensor.GetTensorDesc(), convType, normalizedShape),
                      OP_LOGD(FUSION_NAME, "Const bias shape, Cout or format does not meet fusion requirements."),
                      return false);
    return true;
}

bool CheckPlaceholderBias(const GNode& conv, const GNode& addNode, const AscendString& convType)
{
    const auto source = addNode.GetInDataNodesAndPortIndexs(1).first;
    FUSION_PASS_CHECK(convType == CONV3D, OP_LOGD(FUSION_NAME, "Conv3D does not support Placeholder fusion."),
                      return false);
    AscendString sourceType;
    TensorDesc desc;
    source->GetType(sourceType);
    FUSION_PASS_CHECK(sourceType != "Data", OP_LOGD(FUSION_NAME, "Placeholder bias is not direct Data, no fusion."),
                      return false);
    FUSION_PASS_CHECK(source->GetOutputDesc(0, desc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get Placeholder bias desc failed, no fusion."), return false);
    const auto dims = desc.GetShape().GetDims();
    FUSION_PASS_CHECK(dims.size() != 1U, OP_LOGD(FUSION_NAME, "Placeholder bias rank is not one, no fusion."),
                      return false);
    FUSION_PASS_CHECK(dims[0] == -1 || dims[0] == UNKNOWN_DIM_SENTINEL,
                      OP_LOGD(FUSION_NAME, "Placeholder bias has unknown dimension, no fusion."), return false);
    // The placeholder path has no const payload to normalize, so guard the channel length here: a
    // mismatched Data bias (top-level or subgraph parent-Const mapped) would fuse into a Conv whose
    // bias port shape violates the IR and fails downstream shape checks.
    TensorDesc convOutputDesc;
    FUSION_PASS_CHECK(conv.GetOutputDesc(0, convOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get conv output desc failed, no fusion."), return false);
    FUSION_PASS_CHECK(!CheckCoutMatch(convOutputDesc, convType, dims[0]),
                      OP_LOGD(FUSION_NAME, "Placeholder bias length mismatches Cout, no fusion."), return false);
    OP_LOGD(FUSION_NAME, "Platform supports direct Conv and Placeholder bias fusion.");
    return true;
}

bool CheckReshapeBias(const GNode& reshape)
{
    // Bias must be a direct Const node and the shape input must not be a direct Const.
    const auto biasSource = reshape.GetInDataNodesAndPortIndexs(0).first;
    FUSION_PASS_CHECK(biasSource == nullptr, OP_LOGD(FUSION_NAME, "Reshape bias source node is null, no fusion."),
                      return false);
    AscendString biasSourceType;
    biasSource->GetType(biasSourceType);
    FUSION_PASS_CHECK(biasSourceType != CONST, OP_LOGD(FUSION_NAME, "Reshape bias is not a direct Const, no fusion."),
                      return false);
    const auto shapeSource = reshape.GetInDataNodesAndPortIndexs(1).first;
    if (shapeSource != nullptr) {
        AscendString shapeSourceType;
        shapeSource->GetType(shapeSourceType);
        FUSION_PASS_CHECK(shapeSourceType == CONST,
                          OP_LOGD(FUSION_NAME, "Reshape shape is an additional Const candidate, no fusion."),
                          return false);
    }
    Tensor tensor;
    TensorDesc reshapeInputDesc;
    TensorDesc biasOutputDesc;
    FUSION_PASS_CHECK(reshape.GetInputConstData(0, tensor) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get Reshape bias Const data failed, no fusion."), return false);
    FUSION_PASS_CHECK(biasSource->GetOutputDesc(0, biasOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get Reshape bias source desc failed, no fusion."), return false);
    // Fall back to the bias source output desc when the reshape input desc is missing or empty.
    if (reshape.GetInputDesc(0, reshapeInputDesc) != GRAPH_SUCCESS || reshapeInputDesc.GetShape().GetDims().empty()) {
        reshapeInputDesc = biasOutputDesc;
    }
    const auto dims = reshapeInputDesc.GetShape().GetDims();
    const auto dtype = biasOutputDesc.GetDataType();
    FUSION_PASS_CHECK(dims.size() != 1U, OP_LOGD(FUSION_NAME, "Reshape bias rank is not one, no fusion."),
                      return false);
    // The R-mode dtype whitelist is intentionally conservative: BFLOAT16 is excluded because
    // the reshaped bias feeds the Conv3D bias port, whose BFLOAT16 support is not guaranteed.
    FUSION_PASS_CHECK(dtype != DT_FLOAT16 && dtype != DT_FLOAT,
                      OP_LOGD(FUSION_NAME, "Reshape bias dtype is not FP16 or FP32, no fusion."), return false);
    return true;
}

bool QueryPlatform(PlatformInfo& platformInfo)
{
    OptionalInfo optionalInfo;
    FUSION_PASS_CHECK(
        PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS,
        OP_LOGD(FUSION_NAME, "Get platform info failed, no fusion."), return false);
    return true;
}

bool SupportsFixPipe(const PlatformInfo& platformInfo)
{
    return platformInfo.ai_core_intrinsic_dtype_map.find(INTRINSIC_FIX_PIPE_L0C2OUT) !=
           platformInfo.ai_core_intrinsic_dtype_map.end();
}

GNode BuildConvNode(Graph* graph, const char_t* type, const char_t* name)
{
    return es::CompliantNodeBuilder(graph)
        .OpType(type)
        .Name(name)
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"filter", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
}

GNode BuildAddNode(Graph* graph, const std::string& type)
{
    if (type == "BiasAdd") {
        return es::CompliantNodeBuilder(graph)
            .OpType("BiasAdd")
            .Name("BiasAdd_pattern")
            .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"bias", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();
    }
    return es::CompliantNodeBuilder(graph)
        .OpType("Add")
        .Name("Add_pattern")
        .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
}

GNode BuildReshapeNode(Graph* graph, const std::string& name)
{
    return es::CompliantNodeBuilder(graph)
        .OpType("Reshape")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
}

es::EsTensorHolder BuildPatternConv(es::EsGraphBuilder& builder, const es::EsTensorHolder& x,
                                    const es::EsTensorHolder& filter, const std::string& type)
{
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    const std::string convName = type + "_pattern";
    GNode conv = BuildConvNode(graph, type.c_str(), convName.c_str());
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), conv, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), conv, 1);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(conv, 0));
}

PatternUniqPtr MakeOrdinaryPattern(const std::string& convType, const std::string& addType, bool reversed)
{
    const std::string suffix = reversed ? REVERSED_ADD_PATTERN_SUFFIX : "";
    es::EsGraphBuilder builder((std::string(FUSION_NAME) + "_" + convType + "_" + addType + suffix).c_str());
    auto x = builder.CreateInput(static_cast<int64_t>(INPUT_FMAP_INDEX));
    auto filter = builder.CreateInput(static_cast<int64_t>(INPUT_FILTER_INDEX));
    auto bias = builder.CreateInput(static_cast<int64_t>(INPUT_BIAS_INDEX));
    auto convOutput = BuildPatternConv(builder, x, filter, convType);
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode addNode = BuildAddNode(graph, addType);
    if (reversed) {
        es::AddEdgeAndUpdatePeerDesc(*graph, *bias.GetProducer(), bias.GetProducerOutIndex(), addNode, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, *convOutput.GetProducer(), convOutput.GetProducerOutIndex(), addNode, 1);
    } else {
        es::AddEdgeAndUpdatePeerDesc(*graph, *convOutput.GetProducer(), convOutput.GetProducerOutIndex(), addNode, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, *bias.GetProducer(), bias.GetProducerOutIndex(), addNode, 1);
    }
    auto output = es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(addNode, 0));
    auto patternGraph = builder.BuildAndReset({output});
    if (patternGraph == nullptr) {
        return nullptr;
    }
    auto pattern = std::make_unique<Pattern>(std::move(*patternGraph));
    pattern->CaptureTensor({*convOutput.GetProducer(), 0});
    pattern->CaptureTensor({*output.GetProducer(), 0});
    return pattern;
}

PatternUniqPtr MakeReshapePattern(bool hasDualOutputs)
{
    const std::string name = std::string(FUSION_NAME) + "_Conv3D_Add_R_" +
                             (hasDualOutputs ? "dual_output" : "single_output");
    es::EsGraphBuilder builder(name.c_str());
    auto x = builder.CreateInput(static_cast<int64_t>(INPUT_FMAP_INDEX));
    auto filter = builder.CreateInput(static_cast<int64_t>(INPUT_FILTER_INDEX));
    auto bias = builder.CreateInput(static_cast<int64_t>(INPUT_BIAS_INDEX));
    auto shape = builder.CreateInput(static_cast<int64_t>(PATTERN_SHAPE_INPUT_INDEX));
    auto convOutput = BuildPatternConv(builder, x, filter, "Conv3D");
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode reshape = BuildReshapeNode(graph, "Reshape_pattern");
    GNode addNode = BuildAddNode(graph, "Add");
    es::AddEdgeAndUpdatePeerDesc(*graph, *bias.GetProducer(), bias.GetProducerOutIndex(), reshape, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *shape.GetProducer(), shape.GetProducerOutIndex(), reshape, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *convOutput.GetProducer(), convOutput.GetProducerOutIndex(), addNode, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, reshape, 0, addNode, 1);
    auto reshapeOutput = es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(reshape, 0));
    auto addOutput = es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(addNode, 0));
    std::vector<es::EsTensorHolder> outputs = {addOutput};
    if (hasDualOutputs) {
        outputs.emplace_back(reshapeOutput);
    }
    auto patternGraph = builder.BuildAndReset(outputs);
    if (patternGraph == nullptr) {
        return nullptr;
    }
    auto pattern = std::make_unique<Pattern>(std::move(*patternGraph));
    pattern->CaptureTensor({*convOutput.GetProducer(), 0});
    pattern->CaptureTensor({addNode, 0});
    pattern->CaptureTensor({reshape, 0});
    return pattern;
}

bool BuildReshapeReplacement(es::EsGraphBuilder& builder, const GNode& source,
                             const std::vector<es::EsTensorHolder>& boundaryTensors,
                             std::vector<es::EsTensorHolder>& outputs)
{
    TensorDesc biasDesc;
    TensorDesc shapeDesc;
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(source.GetInputDesc(0, biasDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get reshape bias desc failed."), return false);
    FUSION_PASS_CHECK(source.GetInputDesc(1, shapeDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get reshape shape desc failed."), return false);
    FUSION_PASS_CHECK(source.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get reshape output desc failed."), return false);
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode target = BuildReshapeNode(graph, "BiasaddConvFusionReshape");
    es::AddEdgeAndUpdatePeerDesc(*graph, *boundaryTensors[INPUT_BIAS_INDEX].GetProducer(),
                                 boundaryTensors[INPUT_BIAS_INDEX].GetProducerOutIndex(), target, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *boundaryTensors[PATTERN_SHAPE_INPUT_INDEX].GetProducer(),
                                 boundaryTensors[PATTERN_SHAPE_INPUT_INDEX].GetProducerOutIndex(), target, 1);
    SyncOriginDataType(biasDesc);
    SyncOriginDataType(shapeDesc);
    SyncOriginDataType(outputDesc);
    FUSION_PASS_CHECK(target.UpdateInputDesc(0, biasDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Update reshape bias desc failed."), return false);
    FUSION_PASS_CHECK(target.UpdateInputDesc(1, shapeDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Update reshape shape desc failed."), return false);
    int64_t intValue = 0;
    if (source.GetAttr("axis", intValue) == GRAPH_SUCCESS) {
        target.SetAttr("axis", intValue);
    }
    if (source.GetAttr("num_axes", intValue) == GRAPH_SUCCESS) {
        target.SetAttr("num_axes", intValue);
    }
    FUSION_PASS_CHECK(target.UpdateOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Update reshape output desc failed."), return false);
    outputs.emplace_back(builder.GetCGraphBuilder()->GetTensorHolderFromNode(target, 0));
    return true;
}

bool CreateReplacementInputs(const std::unique_ptr<MatchResult>& matchResult, es::EsGraphBuilder& builder,
                             std::vector<es::EsTensorHolder>& inputTensors, std::vector<TensorDesc>& inputDescs)
{
    std::vector<SubgraphInput> boundaryInputs;
    const auto boundary = matchResult->ToSubgraphBoundary();
    FUSION_PASS_CHECK(boundary == nullptr, OP_LOGD(FUSION_NAME, "Get subgraph boundary failed."), return false);
    FUSION_PASS_CHECK(boundary->GetAllInputs(boundaryInputs) != SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get boundary inputs failed."), return false);
    for (size_t index = 0; index < boundaryInputs.size(); ++index) {
        const auto matchedInputs = boundaryInputs[index].GetAllInputs();
        FUSION_PASS_CHECK(matchedInputs.empty(), OP_LOGD(FUSION_NAME, "Boundary input is empty."), return false);
        const auto matchedInput = matchedInputs[0];
        TensorDesc desc;
        FUSION_PASS_CHECK(matchedInput.node.GetInputDesc(matchedInput.index, desc) != GRAPH_SUCCESS,
                          OP_LOGD(FUSION_NAME, "Get boundary input desc failed."), return false);
        inputDescs.emplace_back(desc);
        inputTensors.emplace_back(builder.CreateInput(static_cast<int64_t>(index),
                                                      (std::string("input_") + std::to_string(index)).c_str(),
                                                      desc.GetDataType(), desc.GetFormat(), desc.GetShape().GetDims()));
    }
    return true;
}

bool GetReplacementBiasShape(const GNode& conv, const GNode& biasConsumer, int32_t biasInputIndex,
                             std::vector<int64_t>& normalizedShape)
{
    if (!IsConstCandidate(biasConsumer, biasInputIndex)) {
        return true;
    }
    AscendString convType;
    TensorDesc convOutputDesc;
    Tensor tensor;
    FUSION_PASS_CHECK(conv.GetType(convType) != GRAPH_SUCCESS, OP_LOGD(FUSION_NAME, "Get conv type failed."),
                      return false);
    FUSION_PASS_CHECK(conv.GetOutputDesc(0, convOutputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get conv output desc failed."), return false);
    FUSION_PASS_CHECK(biasConsumer.GetInputConstData(biasInputIndex, tensor) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get Const bias data failed."), return false);
    NormalizeConstShape(convOutputDesc, tensor.GetTensorDesc(), convType, normalizedShape);
    return true;
}

// Note: also rewrites the bias source node's output desc in the original graph.
bool PrepareTargetBiasDesc(const TensorDesc& convSideInputDesc, GNode& biasSource,
                           const std::vector<int64_t>& normalizedShape, const TensorDesc& firstInputDesc,
                           TensorDesc& targetBiasDesc)
{
    TensorDesc sourceDesc;
    FUSION_PASS_CHECK(biasSource.GetOutputDesc(0, sourceDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get bias source output desc failed."), return false);
    const auto biasShape = normalizedShape.empty() ? sourceDesc.GetShape().GetDims() : normalizedShape;
    sourceDesc.SetShape(Shape(biasShape));
    sourceDesc.SetOriginShape(Shape(biasShape));
    sourceDesc.SetFormat(static_cast<Format>(GetPrimaryFormat(convSideInputDesc.GetFormat())));
    sourceDesc.SetOriginFormat(convSideInputDesc.GetOriginFormat());
    SyncOriginDataType(sourceDesc);
    FUSION_PASS_CHECK(biasSource.UpdateOutputDesc(0, sourceDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Update bias source output desc failed."), return false);
    targetBiasDesc = sourceDesc;
    targetBiasDesc.SetOriginFormat(firstInputDesc.GetOriginFormat());
    SyncOriginDataType(targetBiasDesc);
    return true;
}

void CopyConvAttrs(const GNode& source, GNode& target)
{
    std::vector<int64_t> listValue;
    if (source.GetAttr(STRIDES, listValue) == GRAPH_SUCCESS) {
        target.SetAttr(STRIDES, listValue);
    }
    if (source.GetAttr(PADS, listValue) == GRAPH_SUCCESS) {
        target.SetAttr(PADS, listValue);
    }
    if (source.GetAttr(DILATIONS, listValue) == GRAPH_SUCCESS) {
        target.SetAttr(DILATIONS, listValue);
    }
    AscendString stringValue;
    if (source.GetAttr(DATA_FORMAT, stringValue) == GRAPH_SUCCESS) {
        target.SetAttr(DATA_FORMAT, stringValue);
    }
    if (source.GetAttr(PADDING, stringValue) == GRAPH_SUCCESS) {
        target.SetAttr(PADDING, stringValue);
    }
    if (source.GetAttr(AUTO_PAD, stringValue) == GRAPH_SUCCESS) {
        target.SetAttr(AUTO_PAD, stringValue);
    }
    int64_t intValue = 0;
    if (source.GetAttr(OFFSET_X, intValue) == GRAPH_SUCCESS) {
        target.SetAttr(OFFSET_X, intValue);
    }
    if (source.GetAttr(OP_IMPL_MODE_ENUM, intValue) == GRAPH_SUCCESS) {
        target.SetAttr(OP_IMPL_MODE_ENUM, intValue);
    }
    if (source.GetAttr(GROUPS, intValue) == GRAPH_SUCCESS) {
        target.SetAttr(GROUPS, intValue);
    }
}

GNode BuildTargetConv(Graph& graph, const GNode& source, const std::vector<es::EsTensorHolder>& boundaryTensors)
{
    AscendString convType;
    AscendString convName;
    source.GetType(convType);
    source.GetName(convName);
    GNode target = BuildConvNode(&graph, convType.GetString(), convName.GetString());
    es::AddEdgeAndUpdatePeerDesc(graph, *boundaryTensors[INPUT_FMAP_INDEX].GetProducer(),
                                 boundaryTensors[INPUT_FMAP_INDEX].GetProducerOutIndex(), target, 0);
    es::AddEdgeAndUpdatePeerDesc(graph, *boundaryTensors[INPUT_FILTER_INDEX].GetProducer(),
                                 boundaryTensors[INPUT_FILTER_INDEX].GetProducerOutIndex(), target, 1);
    es::AddEdgeAndUpdatePeerDesc(graph, *boundaryTensors[INPUT_BIAS_INDEX].GetProducer(),
                                 boundaryTensors[INPUT_BIAS_INDEX].GetProducerOutIndex(), target, INPUT_BIAS_INDEX);
    CopyConvAttrs(source, target);
    return target;
}

bool UpdateTargetConvDescs(GNode& target, const GNode& source, const std::vector<TensorDesc>& inputDescs,
                           const TensorDesc& targetBiasDesc)
{
    TensorDesc outputDesc;
    FUSION_PASS_CHECK(source.GetOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get conv output desc failed."), return false);
    TensorDesc xDesc = inputDescs[INPUT_FMAP_INDEX];
    TensorDesc filterDesc = inputDescs[INPUT_FILTER_INDEX];
    TensorDesc biasDesc = targetBiasDesc;
    SyncOriginDataType(outputDesc);
    SyncOriginDataType(xDesc);
    SyncOriginDataType(filterDesc);
    SyncOriginDataType(biasDesc);
    FUSION_PASS_CHECK(target.UpdateInputDesc(0, xDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Update target conv input desc failed."), return false);
    FUSION_PASS_CHECK(target.UpdateInputDesc(1, filterDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Update target conv filter desc failed."), return false);
    FUSION_PASS_CHECK(target.UpdateInputDesc(INPUT_BIAS_INDEX, biasDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Update target conv bias desc failed."), return false);
    FUSION_PASS_CHECK(target.UpdateOutputDesc(0, outputDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Update target conv output desc failed."), return false);
    return true;
}

struct MatchInfo {
    bool hasDualOutputs = false;
    bool isReshapeMode = false;
    bool isReversedAdd = false;
    GNode conv;
    GNode addNode;
};

bool ReadMatchInfo(const std::unique_ptr<MatchResult>& matchResult, MatchInfo& info)
{
    AscendString patternName;
    FUSION_PASS_CHECK(matchResult->GetPatternGraph().GetName(patternName) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get pattern graph name failed."), return false);
    const std::string name(patternName.GetString());
    info.hasDualOutputs = name == RESHAPE_DUAL_OUTPUT_PATTERN;
    info.isReshapeMode = info.hasDualOutputs || name == RESHAPE_SINGLE_OUTPUT_PATTERN;
    info.isReversedAdd = !info.isReshapeMode && name.size() > REVERSED_ADD_PATTERN_SUFFIX.size() &&
                         name.compare(name.size() - REVERSED_ADD_PATTERN_SUFFIX.size(),
                                      REVERSED_ADD_PATTERN_SUFFIX.size(), REVERSED_ADD_PATTERN_SUFFIX) == 0;
    NodeIo convOutput;
    NodeIo addOutput;
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(CONV_CAPTURE, convOutput) != SUCCESS,
                      OP_LOGD(FUSION_NAME, "Failed to get captured conv tensor."), return false);
    FUSION_PASS_CHECK(matchResult->GetCapturedTensor(ADD_CAPTURE, addOutput) != SUCCESS,
                      OP_LOGD(FUSION_NAME, "Failed to get captured add tensor."), return false);
    info.conv = convOutput.node;
    info.addNode = addOutput.node;
    return true;
}

} // namespace

std::vector<PatternUniqPtr> AABiasaddConvFusion::Patterns()
{
    std::vector<PatternUniqPtr> patterns;
    patterns.emplace_back(MakeReshapePattern(false));
    patterns.emplace_back(MakeReshapePattern(true));

    for (const auto& convType : convTypes) {
        for (const auto& addType : addTypes) {
            patterns.emplace_back(MakeOrdinaryPattern(convType, addType, false));
            if (addType == "Add") {
                patterns.emplace_back(MakeOrdinaryPattern(convType, addType, true));
            }
        }
    }
    return patterns;
}

bool AABiasaddConvFusion::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    MatchInfo info;
    FUSION_PASS_CHECK(!ReadMatchInfo(matchResult, info), OP_LOGD(FUSION_NAME, "Read match info failed, no fusion."),
                      return false);
    const GNode& conv = info.conv;
    const GNode& addNode = info.addNode;
    const char* candidateType = info.hasDualOutputs ?
                                    "R dual-output" :
                                    (info.isReshapeMode ? "R single-output" :
                                                          (info.isReversedAdd ? "ordinary reversed" : "ordinary"));
    OP_LOGD(FUSION_NAME, "Begin to check BiasaddConv fusion requirements, candidate is %s.", candidateType);
    FUSION_PASS_CHECK(!(conv.GetInDataNodesAndPortIndexs(INPUT_BIAS_INDEX).first == nullptr),
                      OP_LOGD(FUSION_NAME, "Conv already has bias, no fusion."), return false);
    PlatformInfo platformInfo;
    FUSION_PASS_CHECK_NOLOG(!QueryPlatform(platformInfo), return false);
    if (info.isReshapeMode) {
        NodeIo reshapeOutput;
        FUSION_PASS_CHECK(matchResult->GetCapturedTensor(RESHAPE_CAPTURE, reshapeOutput) != SUCCESS,
                          OP_LOGD(FUSION_NAME, "Failed to get captured reshape tensor."), return false);
        return CheckReshapeBias(reshapeOutput.node);
    }
    AscendString convType;
    conv.GetType(convType);
    const int32_t biasInputIndex = info.isReversedAdd ? 0 : 1;
    const auto bias = addNode.GetInDataNodesAndPortIndexs(biasInputIndex).first;
    FUSION_PASS_CHECK(bias == nullptr, OP_LOGD(FUSION_NAME, "Bias source node is null, no fusion."), return false);
    TensorDesc biasDesc;
    FUSION_PASS_CHECK(bias->GetOutputDesc(0, biasDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get bias output desc failed, no fusion."), return false);
    FUSION_PASS_CHECK(supportedBiasDtypes.find(biasDesc.GetDataType()) == supportedBiasDtypes.end(),
                      OP_LOGD(FUSION_NAME, "Bias dtype is not supported, no fusion."), return false);
    AscendString biasType;
    bias->GetType(biasType);
    if (biasType == "Variable") {
        // Only fuses a reversed Add whose bias resolves to a Const weight, so keep
        // Variable fusion to the normal input order.
        FUSION_PASS_CHECK(info.isReversedAdd,
                          OP_LOGD(FUSION_NAME, "Reversed Add does not support Variable bias, no fusion."),
                          return false);
        AscendString addType;
        addNode.GetType(addType);
        FUSION_PASS_CHECK(addType != "BiasAdd", OP_LOGD(FUSION_NAME, "Variable bias only supports BiasAdd, no fusion."),
                          return false);
        OP_LOGD(FUSION_NAME, "Variable bias meets requirements, fuse in training mode.");
        return true;
    }

    if (IsConstCandidate(addNode, biasInputIndex)) {
        Tensor biasTensor;
        if (addNode.GetInputConstData(biasInputIndex, biasTensor) == GRAPH_SUCCESS) {
            return CheckConstBias(conv, biasTensor, convType);
        }
    }

    FUSION_PASS_CHECK(info.isReversedAdd,
                      OP_LOGD(FUSION_NAME, "Reversed Add only supports Const candidate bias, no fusion."),
                      return false);
    const bool supportsFixPipe = SupportsFixPipe(platformInfo);
    if (supportsFixPipe) {
        return CheckPlaceholderBias(conv, addNode, convType);
    }
    return false;
}

GraphUniqPtr AABiasaddConvFusion::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(FUSION_NAME, "Begin to build BiasaddConv replacement.");
    MatchInfo info;
    FUSION_PASS_CHECK(!ReadMatchInfo(matchResult, info), OP_LOGD(FUSION_NAME, "Read match info failed."),
                      return nullptr);
    const GNode& conv = info.conv;
    const GNode& addNode = info.addNode;
    es::EsGraphBuilder builder(FUSION_NAME.c_str());
    std::vector<es::EsTensorHolder> boundaryTensors;
    std::vector<TensorDesc> inputDescs;
    FUSION_PASS_CHECK(!CreateReplacementInputs(matchResult, builder, boundaryTensors, inputDescs),
                      OP_LOGD(FUSION_NAME, "Create replacement inputs failed."), return nullptr);
    GNode sourceReshape;
    if (info.isReshapeMode) {
        NodeIo reshapeOutput;
        FUSION_PASS_CHECK(matchResult->GetCapturedTensor(RESHAPE_CAPTURE, reshapeOutput) != SUCCESS,
                          OP_LOGD(FUSION_NAME, "Failed to get captured reshape tensor."), return nullptr);
        sourceReshape = reshapeOutput.node;
    }
    const GNode& biasConsumer = info.isReshapeMode ? sourceReshape : addNode;
    const int32_t biasInputIndex = info.isReshapeMode ? 0 : (info.isReversedAdd ? 0 : 1);
    std::vector<int64_t> normalizedShape;
    FUSION_PASS_CHECK(!GetReplacementBiasShape(conv, biasConsumer, biasInputIndex, normalizedShape),
                      OP_LOGD(FUSION_NAME, "Get replacement bias shape failed."), return nullptr);
    const auto biasSource = biasConsumer.GetInDataNodesAndPortIndexs(biasInputIndex).first;
    FUSION_PASS_CHECK(biasSource == nullptr, OP_LOGD(FUSION_NAME, "Bias source node is null."), return nullptr);
    TensorDesc convSideDesc;
    const int32_t convInputIndex = info.isReshapeMode ? 0 : (info.isReversedAdd ? 1 : 0);
    FUSION_PASS_CHECK(addNode.GetInputDesc(convInputIndex, convSideDesc) != GRAPH_SUCCESS,
                      OP_LOGD(FUSION_NAME, "Get add conv side desc failed."), return nullptr);
    TensorDesc targetBiasDesc;
    FUSION_PASS_CHECK(!PrepareTargetBiasDesc(convSideDesc, *biasSource, normalizedShape, inputDescs[0], targetBiasDesc),
                      OP_LOGD(FUSION_NAME, "Prepare target bias desc failed."), return nullptr);
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();
    GNode targetConv = BuildTargetConv(*graph, conv, boundaryTensors);
    FUSION_PASS_CHECK(!UpdateTargetConvDescs(targetConv, conv, inputDescs, targetBiasDesc),
                      OP_LOGD(FUSION_NAME, "Update target conv descs failed."), return nullptr);
    std::vector<es::EsTensorHolder> outputs = {builder.GetCGraphBuilder()->GetTensorHolderFromNode(targetConv, 0)};
    if (info.isReshapeMode && info.hasDualOutputs) {
        FUSION_PASS_CHECK(!BuildReshapeReplacement(builder, sourceReshape, boundaryTensors, outputs),
                          OP_LOGD(FUSION_NAME, "Build reshape replacement failed."), return nullptr);
    }
    AscendString convName;
    conv.GetName(convName);
    OP_LOGD(FUSION_NAME, "Build BiasaddConv replacement success for node %s.", convName.GetString());
    return builder.BuildAndReset(outputs);
}

#if GE_COMPILER_VERSION_NUM >= 90300000U
REG_FUSION_PASS(AABiasaddConvFusion).Stage(CustomPassStage::kCompatibleInherited);
#endif

} // namespace Ops
