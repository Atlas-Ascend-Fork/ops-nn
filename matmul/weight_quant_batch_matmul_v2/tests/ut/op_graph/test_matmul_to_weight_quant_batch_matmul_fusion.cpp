/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "common/inc/op_graph/cube_utils/cube_fp16_t.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "../../../op_graph/fusion_pass/matmul_to_weight_quant_batch_matmul_fusion.h"

namespace {
using namespace ge;
using ops::fp16_t;

constexpr int64_t M = 4;
constexpr int64_t K = 64;
constexpr int64_t N = 32;
constexpr int64_t GROUPS = 2;

struct Config {
    std::string matmulType = "MatMulV2";
    Format weightFormat = FORMAT_ND;
    DataType scaleDtype = DT_FLOAT;
    bool constWeight = true;
    bool transposeX = false;
    bool transposeWeight = false;
    bool explicitXTranspose = false;
    bool explicitWeightTranspose = false;
    bool withOffset = true;
    bool withAdd = false;
    bool withExternalBias = false;
    bool sqrtMode = false;
    bool withQuant = false;
    bool quantV2 = true;
    bool quantPerChannel = false;
    bool quantScaleConst = true;
    bool withQuantOffset = false;
    bool quantSqrtMode = false;
    float quantOffset = 0.0F;
    std::string quantRoundMode = "round";
    int64_t quantAxis = -1;
    bool quantInvalidParamShape = false;
    int64_t mSize = M;
    int64_t groups = GROUPS;
};

TensorDesc MakeDesc(const std::vector<int64_t>& storageDims, const std::vector<int64_t>& originDims, DataType dtype,
                    Format format, Format originFormat = FORMAT_ND)
{
    TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetShape(Shape(storageDims));
    desc.SetFormat(format);
    desc.SetOriginShape(Shape(originDims));
    desc.SetOriginFormat(originFormat);
    return desc;
}

TensorDesc MakeNd(const std::vector<int64_t>& dims, DataType dtype) { return MakeDesc(dims, dims, dtype, FORMAT_ND); }

int64_t CeilDiv16(int64_t value) { return value / 16 + (value % 16 == 0 ? 0 : 1); }

void SetNibble(std::vector<uint8_t>& data, size_t index, uint8_t value)
{
    if (index % 2 == 0) {
        data[index / 2] = (data[index / 2] & 0xF0U) | (value & 0x0FU);
    } else {
        data[index / 2] = (data[index / 2] & 0x0FU) | ((value & 0x0FU) << 4);
    }
}

uint8_t GetNibble(const uint8_t* data, size_t index)
{
    return index % 2 == 0 ? data[index / 2] & 0x0FU : (data[index / 2] >> 4) & 0x0FU;
}

std::vector<uint8_t> MakeNdWeight(bool transposed)
{
    const int64_t rows = transposed ? N : K;
    const int64_t cols = transposed ? K : N;
    std::vector<uint8_t> data(static_cast<size_t>(rows * cols / 2), 0U);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
            SetNibble(data, static_cast<size_t>(row * cols + col), static_cast<uint8_t>((row * cols + col) % 16));
        }
    }
    return data;
}

std::vector<uint8_t> NdToNz(const std::vector<uint8_t>& source, bool transposed)
{
    const int64_t n1 = CeilDiv16(N);
    const int64_t k1 = CeilDiv16(K);
    std::vector<uint8_t> target(static_cast<size_t>(n1 * k1 * 16 * 16 / 2), 0U);
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t n = 0; n < N; ++n) {
            const size_t sourceIndex = transposed ? static_cast<size_t>(n * K + k) : static_cast<size_t>(k * N + n);
            const size_t targetIndex = static_cast<size_t>(((n / 16 * k1 + k / 16) * 16 + k % 16) * 16 + n % 16);
            SetNibble(target, targetIndex, GetNibble(source.data(), sourceIndex));
        }
    }
    return target;
}

std::vector<uint8_t> NdToTransposedNz(const std::vector<uint8_t>& source)
{
    const int64_t n1 = CeilDiv16(N);
    const int64_t k1 = CeilDiv16(K);
    std::vector<uint8_t> target(static_cast<size_t>(n1 * k1 * 16 * 16 / 2), 0U);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            const size_t sourceIndex = static_cast<size_t>(n * K + k);
            const size_t targetIndex = static_cast<size_t>(((k / 16 * n1 + n / 16) * 16 + n % 16) * 16 + k % 16);
            SetNibble(target, targetIndex, GetNibble(source.data(), sourceIndex));
        }
    }
    return target;
}

es::EsTensorHolder MakeInt4Const(es::EsGraphBuilder& builder, const std::string& name,
                                 const std::vector<int64_t>& storageDims, const std::vector<int64_t>& originDims,
                                 Format format, const std::vector<uint8_t>& data)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto desc = MakeDesc(storageDims, originDims, DT_INT4, format);
    Tensor tensor(desc, data);
    AttrValue value;
    EXPECT_EQ(value.SetAttrValue(tensor), GRAPH_SUCCESS);
    auto node = es::CompliantNodeBuilder(graph)
                    .OpType("Const")
                    .Name(name.c_str())
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .IrDefAttrs({{"value", es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", value}})
                    .InstanceOutputShape("y", storageDims)
                    .InstanceOutputDataType("y", DT_INT4)
                    .InstanceOutputFormat("y", format)
                    .Build();
    EXPECT_EQ(node.UpdateOutputDesc(0, desc), GRAPH_SUCCESS);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

GNode BuildAntiQuant(Graph* graph, bool withOffset, bool sqrtMode)
{
    auto builder = es::CompliantNodeBuilder(graph);
    builder.OpType("AscendAntiQuantV2").Name("anti_quant_v2");
    if (withOffset) {
        builder.IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"scale", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"offset", es::CompliantNodeBuilder::kEsIrInputOptional, ""}});
    } else {
        builder.IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"scale", es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    }
    return builder.IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({{"dst_type", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                      es::CreateFrom(static_cast<int64_t>(DT_FLOAT16))},
                     {"sqrt_mode", es::CompliantNodeBuilder::kEsAttrOptional, "Bool", es::CreateFrom(sqrtMode)}})
        .Build();
}

GNode BuildMatMul(Graph* graph, const std::string& opType, bool transposeX, bool transposeWeight, bool withBias)
{
    auto builder = es::CompliantNodeBuilder(graph);
    builder.OpType(opType.c_str()).Name("matmul");
    if (withBias) {
        builder.IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""}});
    } else {
        builder.IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    }
    const bool isBatch = opType.find("BatchMatMul") == 0;
    return builder.IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({{isBatch ? "adj_x1" : "transpose_x1", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                      es::CreateFrom(transposeX)},
                     {isBatch ? "adj_x2" : "transpose_x2", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                      es::CreateFrom(transposeWeight)}})
        .Build();
}

GNode BuildAdd(Graph* graph)
{
    return es::CompliantNodeBuilder(graph)
        .OpType("Add")
        .Name("bias_add")
        .IrDefInputs({{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
}

GNode BuildQuant(Graph* graph, const Config& config)
{
    auto builder = es::CompliantNodeBuilder(graph);
    builder.OpType(config.quantV2 ? "AscendQuantV2" : "AscendQuant").Name("output_quant");
    if (config.quantV2) {
        builder.IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"scale", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                             {"offset", es::CompliantNodeBuilder::kEsIrInputOptional, ""}});
    } else {
        builder.IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}});
    }
    auto quant = builder.IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs({{"sqrt_mode", es::CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                   es::CreateFrom(config.quantSqrtMode)},
                                  {"round_mode", es::CompliantNodeBuilder::kEsAttrOptional, "String",
                                   es::CreateFrom(AscendString(config.quantRoundMode.c_str()))},
                                  {"dst_type", es::CompliantNodeBuilder::kEsAttrOptional, "Int",
                                   es::CreateFrom(static_cast<int64_t>(DT_INT8))}})
                     .Build();
    if (config.quantV2) {
        auto axis = config.quantAxis;
        quant.SetAttr("axis", axis);
    }
    return quant;
}

GNode BuildTransposeD(Graph* graph, const std::string& name)
{
    return es::CompliantNodeBuilder(graph)
        .OpType("TransposeD")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .IrDefAttrs({{"perm", es::CompliantNodeBuilder::kEsAttrRequired, "ListInt",
                      es::CreateFrom(std::vector<int64_t>{1, 0})}})
        .Build();
}

GNode BuildTranspose(Graph* graph, const std::string& name)
{
    return es::CompliantNodeBuilder(graph)
        .OpType("Transpose")
        .Name(name.c_str())
        .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                      {"perm", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
}

std::shared_ptr<Graph> BuildTestGraph(const Config& config)
{
    es::EsGraphBuilder builder("anti_quant_v2_test");
    auto* rawGraph = builder.GetCGraphBuilder()->GetGraph();
    const std::vector<int64_t> xDims = config.transposeX ? std::vector<int64_t>{K, config.mSize} :
                                                           std::vector<int64_t>{config.mSize, K};
    const std::vector<int64_t> weightOrigin = config.transposeWeight ? std::vector<int64_t>{N, K} :
                                                                       std::vector<int64_t>{K, N};
    const std::vector<int64_t> weightStorage = config.weightFormat == FORMAT_ND ?
                                                   weightOrigin :
                                                   (config.transposeWeight ?
                                                        std::vector<int64_t>{CeilDiv16(K), CeilDiv16(N), 16, 16} :
                                                        std::vector<int64_t>{CeilDiv16(N), CeilDiv16(K), 16, 16});
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, xDims);
    x.GetProducer()->UpdateOutputDesc(0, MakeNd(xDims, DT_FLOAT16));

    es::EsTensorHolder weight(nullptr);
    if (config.constWeight) {
        auto ndData = MakeNdWeight(config.transposeWeight);
        auto data = config.weightFormat == FORMAT_ND ?
                        ndData :
                        (config.transposeWeight ? NdToTransposedNz(ndData) : NdToNz(ndData, false));
        weight = MakeInt4Const(builder, "weight", weightStorage, weightOrigin, config.weightFormat, data);
    } else {
        weight = builder.CreateInput(1, "weight", DT_INT4, config.weightFormat, weightStorage);
        weight.GetProducer()->UpdateOutputDesc(0, MakeDesc(weightStorage, weightOrigin, DT_INT4, config.weightFormat));
    }

    std::vector<float> scaleValues(static_cast<size_t>(config.groups * N), 0.5F);
    std::vector<uint16_t> halfValues(static_cast<size_t>(config.groups * N), fp16_t(0.5F).val);
    auto scale = config.scaleDtype == DT_FLOAT ? builder.CreateConst(scaleValues, {config.groups, N}) :
                                                 builder.CreateConst(halfValues, {config.groups, N}, DT_FLOAT16);
    scale.GetProducer()->UpdateOutputDesc(0, MakeNd({config.groups, N}, config.scaleDtype));
    es::EsTensorHolder offset(nullptr);
    if (config.withOffset) {
        offset = builder.CreateConst(halfValues, {config.groups, N}, DT_FLOAT16);
        offset.GetProducer()->UpdateOutputDesc(0, MakeNd({config.groups, N}, DT_FLOAT16));
    }

    auto anti = BuildAntiQuant(rawGraph, config.withOffset, config.sqrtMode);
    auto mm = BuildMatMul(rawGraph, config.matmulType, config.explicitXTranspose ? false : config.transposeX,
                          config.explicitWeightTranspose ? false : config.transposeWeight, config.withExternalBias);
    es::AddEdgeAndUpdatePeerDesc(*rawGraph, *weight.GetProducer(), 0, anti, 0);
    es::AddEdgeAndUpdatePeerDesc(*rawGraph, *scale.GetProducer(), 0, anti, 1);
    if (config.withOffset) {
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, *offset.GetProducer(), 0, anti, 2);
    }
    anti.UpdateInputDesc(0, MakeDesc(weightStorage, weightOrigin, DT_INT4, config.weightFormat));
    anti.UpdateInputDesc(1, MakeNd({config.groups, N}, config.scaleDtype));
    if (config.withOffset) {
        anti.UpdateInputDesc(2, MakeNd({config.groups, N}, DT_FLOAT16));
    }
    anti.UpdateOutputDesc(0, MakeNd(weightOrigin, DT_FLOAT16));

    if (config.explicitXTranspose) {
        auto transpose = BuildTranspose(rawGraph, "x_transpose");
        auto perm = builder.CreateConst(std::vector<int32_t>{1, 0}, {2});
        perm.GetProducer()->UpdateOutputDesc(0, MakeNd({2}, DT_INT32));
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, *x.GetProducer(), 0, transpose, 0);
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, *perm.GetProducer(), 0, transpose, 1);
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, transpose, 0, mm, 0);
        transpose.UpdateInputDesc(0, MakeNd(xDims, DT_FLOAT16));
        transpose.UpdateInputDesc(1, MakeNd({2}, DT_INT32));
        transpose.UpdateOutputDesc(0, MakeNd({config.mSize, K}, DT_FLOAT16));
        mm.UpdateInputDesc(0, MakeNd({config.mSize, K}, DT_FLOAT16));
    } else {
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, *x.GetProducer(), 0, mm, 0);
        mm.UpdateInputDesc(0, MakeNd(xDims, DT_FLOAT16));
    }

    const std::vector<int64_t> mmWeightDims = config.explicitWeightTranspose ? std::vector<int64_t>{K, N} :
                                                                               weightOrigin;
    if (config.explicitWeightTranspose) {
        auto transpose = BuildTransposeD(rawGraph, "weight_transpose");
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, anti, 0, transpose, 0);
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, transpose, 0, mm, 1);
        transpose.UpdateInputDesc(0, MakeNd(weightOrigin, DT_FLOAT16));
        transpose.UpdateOutputDesc(0, MakeNd(mmWeightDims, DT_FLOAT16));
    } else {
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, anti, 0, mm, 1);
    }
    mm.UpdateInputDesc(1, MakeNd(mmWeightDims, DT_FLOAT16));
    mm.UpdateOutputDesc(0, MakeNd({config.mSize, N}, DT_FLOAT16));

    if (config.withExternalBias) {
        auto bias = builder.CreateInput(config.constWeight ? 1 : 2, "bias", DT_FLOAT16, FORMAT_ND, {N});
        bias.GetProducer()->UpdateOutputDesc(0, MakeNd({N}, DT_FLOAT16));
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, *bias.GetProducer(), 0, mm, 2);
        mm.UpdateInputDesc(2, MakeNd({N}, DT_FLOAT16));
    }

    GNode output = mm;
    if (config.withAdd) {
        auto add = BuildAdd(rawGraph);
        std::vector<float> biasValues(static_cast<size_t>(N), 2.0F);
        auto bias = builder.CreateConst(biasValues, {N});
        bias.GetProducer()->UpdateOutputDesc(0, MakeNd({N}, DT_FLOAT));
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, mm, 0, add, 0);
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, *bias.GetProducer(), 0, add, 1);
        add.UpdateInputDesc(0, MakeNd({config.mSize, N}, DT_FLOAT16));
        add.UpdateInputDesc(1, MakeNd({N}, DT_FLOAT));
        add.UpdateOutputDesc(0, MakeNd({config.mSize, N}, DT_FLOAT16));
        output = add;
    }

    if (config.withQuant) {
        auto quant = BuildQuant(rawGraph, config);
        es::AddEdgeAndUpdatePeerDesc(*rawGraph, output, 0, quant, 0);
        quant.UpdateInputDesc(0, MakeNd({config.mSize, N}, DT_FLOAT16));
        if (config.quantV2) {
            const int64_t paramSize = config.quantPerChannel ? N : 1;
            const std::vector<int64_t> paramDims = config.quantInvalidParamShape ? std::vector<int64_t>{paramSize, 1} :
                                                                                   std::vector<int64_t>{paramSize};
            std::vector<float> quantScaleValues(static_cast<size_t>(paramSize), 0.25F);
            auto quantScale = config.quantScaleConst ?
                                  builder.CreateConst(quantScaleValues, paramDims) :
                                  builder.CreateInput(1, "quant_scale", DT_FLOAT, FORMAT_ND, {paramSize});
            quantScale.GetProducer()->UpdateOutputDesc(0, MakeNd(paramDims, DT_FLOAT));
            es::AddEdgeAndUpdatePeerDesc(*rawGraph, *quantScale.GetProducer(), 0, quant, 1);
            quant.UpdateInputDesc(1, MakeNd(paramDims, DT_FLOAT));
            if (config.withQuantOffset) {
                std::vector<float> quantOffsetValues(static_cast<size_t>(paramSize), config.quantOffset);
                auto quantOffset = builder.CreateConst(quantOffsetValues, paramDims);
                quantOffset.GetProducer()->UpdateOutputDesc(0, MakeNd(paramDims, DT_FLOAT));
                es::AddEdgeAndUpdatePeerDesc(*rawGraph, *quantOffset.GetProducer(), 0, quant, 2);
                quant.UpdateInputDesc(2, MakeNd(paramDims, DT_FLOAT));
            }
        } else {
            float quantScaleValue = 0.25F;
            float quantOffsetValue = config.quantOffset;
            quant.SetAttr("scale", quantScaleValue);
            quant.SetAttr("offset", quantOffsetValue);
        }
        quant.UpdateOutputDesc(0, MakeNd({config.mSize, N}, DT_INT8));
        output = quant;
    }

    auto graph = builder.BuildAndReset();
    EXPECT_NE(graph, nullptr);
    graph->SetOutputs({{output, 0}});
    return graph;
}

GNode FindNode(const std::shared_ptr<Graph>& graph, const std::string& type)
{
    for (auto& node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type.c_str()) {
            return node;
        }
    }
    return GNode();
}

bool HasNode(const std::shared_ptr<Graph>& graph, const std::string& type)
{
    auto node = FindNode(graph, type);
    AscendString nodeType;
    return node.GetType(nodeType) == GRAPH_SUCCESS;
}

Tensor GetInputConst(const GNode& node, int64_t port)
{
    auto input = node.GetInDataNodesAndPortIndexs(port).first;
    Tensor tensor;
    EXPECT_NE(input, nullptr);
    EXPECT_EQ(input->GetAttr(AscendString("value"), tensor), GRAPH_SUCCESS);
    return tensor;
}

uint64_t ExpectedQuantScale(float scale, float offset)
{
    uint32_t bits = 0U;
    std::memcpy(&bits, &scale, sizeof(bits));
    uint64_t result = static_cast<uint64_t>(bits & 0xFFFFE000U);
    if (offset == 0.0F) {
        return result;
    }
    const auto offsetInt = static_cast<int64_t>(offset);
    result |= (static_cast<uint64_t>(offsetInt) << 37UL) & 0x00001FE000000000UL;
    if (offsetInt < 0) {
        result |= 0x0000200000000000UL;
    }
    return result | 0x0000400000000000UL;
}

class MatmulToWeightQuantBatchMatmulFusionTest : public testing::Test {
protected:
    void SetUp() override
    {
        fe::PlatformInfo info;
        fe::OptionalInfo optional;
        info.soc_info.arch_type = 5102;
        info.str_info.short_soc_version = "MC62CM12A";
        info.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"s8s4"};
        optional.soc_version = "MC62CM12A";
        fe::PlatformInfoManager::Instance().platform_info_map_["MC62CM12A"] = info;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional);
    }

    void TearDown() override { fe::PlatformInfoManager::Instance().platform_info_map_.clear(); }

    ops::MatmulToWeightQuantBatchMatmulFusion pass;
    CustomPassContext context;
};

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, NdConstScaleOffsetAndAddFusion)
{
    Config config;
    config.withAdd = true;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_FALSE(HasNode(graph, "AscendAntiQuantV2"));
    EXPECT_FALSE(HasNode(graph, "MatMulV2"));
    EXPECT_FALSE(HasNode(graph, "Add"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    int64_t groupSize = 0;
    bool transposeWeight = false;
    EXPECT_EQ(fused.GetAttr(AscendString("antiquant_group_size"), groupSize), GRAPH_SUCCESS);
    (void)fused.GetAttr(AscendString("transpose_weight"), transposeWeight);
    EXPECT_EQ(groupSize, 32);
    EXPECT_FALSE(transposeWeight);

    auto weight = GetInputConst(fused, 1);
    const auto& weightDesc = weight.GetTensorDesc();
    EXPECT_EQ(weightDesc.GetDataType(), DT_INT4);
    EXPECT_EQ(weightDesc.GetFormat(), FORMAT_FRACTAL_NZ);
    EXPECT_EQ(weightDesc.GetShape().GetDims(), (std::vector<int64_t>{2, 4, 16, 16}));
    EXPECT_EQ(weightDesc.GetOriginShape().GetDims(), (std::vector<int64_t>{K, N}));
    auto expected = NdToNz(MakeNdWeight(false), false);
    ASSERT_EQ(weight.GetSize(), expected.size());
    EXPECT_EQ(std::vector<uint8_t>(weight.GetData(), weight.GetData() + weight.GetSize()), expected);

    auto scale = GetInputConst(fused, 2);
    EXPECT_EQ(scale.GetTensorDesc().GetDataType(), DT_FLOAT16);
    EXPECT_EQ(scale.GetTensorDesc().GetShape().GetDims(), (std::vector<int64_t>{GROUPS, N}));
    auto offset = GetInputConst(fused, 3);
    EXPECT_EQ(offset.GetTensorDesc().GetDataType(), DT_FLOAT16);
    auto bias = GetInputConst(fused, 6);
    EXPECT_EQ(bias.GetTensorDesc().GetDataType(), DT_FLOAT16);
    EXPECT_EQ(reinterpret_cast<const uint16_t*>(bias.GetData())[0], fp16_t(2.0F).val);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, DynamicNzAndTransposeXFusion)
{
    Config config;
    config.weightFormat = FORMAT_FRACTAL_NZ;
    config.constWeight = false;
    config.transposeX = true;
    config.withOffset = false;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    bool transposeX = false;
    EXPECT_EQ(fused.GetAttr(AscendString("transpose_x"), transposeX), GRAPH_SUCCESS);
    EXPECT_TRUE(transposeX);
    auto weightInput = fused.GetInDataNodesAndPortIndexs(1).first;
    ASSERT_NE(weightInput, nullptr);
    AscendString type;
    EXPECT_EQ(weightInput->GetType(type), GRAPH_SUCCESS);
    EXPECT_EQ(type, AscendString("Data"));
    EXPECT_EQ(fused.GetInDataNodesAndPortIndexs(3).first, nullptr);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, TransposedNdConstantIsAbsorbed)
{
    Config config;
    config.transposeWeight = true;
    config.withOffset = false;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    bool transposeWeight = false;
    (void)fused.GetAttr(AscendString("transpose_weight"), transposeWeight);
    EXPECT_FALSE(transposeWeight);
    auto weight = GetInputConst(fused, 1);
    auto expected = NdToNz(MakeNdWeight(true), true);
    ASSERT_EQ(weight.GetSize(), expected.size());
    EXPECT_EQ(std::vector<uint8_t>(weight.GetData(), weight.GetData() + weight.GetSize()), expected);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, ExplicitTransposesAreAbsorbed)
{
    Config config;
    config.transposeX = true;
    config.transposeWeight = true;
    config.explicitXTranspose = true;
    config.explicitWeightTranspose = true;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_FALSE(HasNode(graph, "Transpose"));
    EXPECT_FALSE(HasNode(graph, "TransposeD"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    bool transposeX = false;
    bool transposeWeight = false;
    EXPECT_EQ(fused.GetAttr(AscendString("transpose_x"), transposeX), GRAPH_SUCCESS);
    (void)fused.GetAttr(AscendString("transpose_weight"), transposeWeight);
    EXPECT_TRUE(transposeX);
    EXPECT_FALSE(transposeWeight);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, TransposedNzConstantIsAbsorbed)
{
    Config config;
    config.weightFormat = FORMAT_FRACTAL_NZ;
    config.transposeWeight = true;
    config.scaleDtype = DT_FLOAT16;
    config.withOffset = false;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    auto weight = GetInputConst(fused, 1);
    auto expected = NdToNz(MakeNdWeight(true), true);
    ASSERT_EQ(weight.GetSize(), expected.size());
    EXPECT_EQ(std::vector<uint8_t>(weight.GetData(), weight.GetData() + weight.GetSize()), expected);
    auto scale = GetInputConst(fused, 2);
    EXPECT_EQ(scale.GetTensorDesc().GetDataType(), DT_FLOAT16);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, BatchMatMulV3Fusion)
{
    Config config;
    config.matmulType = "BatchMatMulV3";
    config.transposeWeight = true;
    config.withOffset = false;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_FALSE(HasNode(graph, "BatchMatMulV3"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    int64_t groupSize = 0;
    EXPECT_EQ(fused.GetAttr(AscendString("antiquant_group_size"), groupSize), GRAPH_SUCCESS);
    EXPECT_EQ(groupSize, 32);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, TransposedDynamicNzDoesNotFuse)
{
    Config config;
    config.weightFormat = FORMAT_FRACTAL_NZ;
    config.constWeight = false;
    config.transposeWeight = true;
    auto graph = BuildTestGraph(config);
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
    EXPECT_TRUE(HasNode(graph, "AscendAntiQuantV2"));
    EXPECT_TRUE(HasNode(graph, "MatMulV2"));
    EXPECT_FALSE(HasNode(graph, "WeightQuantBatchMatmulV2"));
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, InvalidGroupSizeDoesNotFuse)
{
    Config config;
    config.groups = 1;
    auto graph = BuildTestGraph(config);
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNode(graph, "WeightQuantBatchMatmulV2"));
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, ExistingBiasKeepsFollowingAdd)
{
    Config config;
    config.withExternalBias = true;
    config.withAdd = true;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_TRUE(HasNode(graph, "Add"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    auto bias = fused.GetInDataNodesAndPortIndexs(6).first;
    ASSERT_NE(bias, nullptr);
    AscendString type;
    EXPECT_EQ(bias->GetType(type), GRAPH_SUCCESS);
    EXPECT_EQ(type, AscendString("Data"));
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, SqrtModeDoesNotFuse)
{
    Config config;
    config.sqrtMode = true;
    auto graph = BuildTestGraph(config);
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNode(graph, "WeightQuantBatchMatmulV2"));
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, PlatformWithoutS8S4DoesNotFuse)
{
    auto& info = fe::PlatformInfoManager::Instance().platform_info_map_["MC62CM12A"];
    info.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"s8s8"};
    auto graph = BuildTestGraph(Config());
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNode(graph, "WeightQuantBatchMatmulV2"));
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, QuantV2PerTensorOffsetAndAddFusion)
{
    Config config;
    config.withAdd = true;
    config.withQuant = true;
    config.withQuantOffset = true;
    config.quantOffset = -3.0F;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_FALSE(HasNode(graph, "AscendQuantV2"));
    EXPECT_FALSE(HasNode(graph, "Add"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    TensorDesc outDesc;
    ASSERT_EQ(fused.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_INT8);
    int64_t dtype = -1;
    EXPECT_EQ(fused.GetAttr(AscendString("dtype"), dtype), GRAPH_SUCCESS);
    EXPECT_EQ(dtype, static_cast<int64_t>(DT_INT8));
    auto quantScale = GetInputConst(fused, 4);
    EXPECT_EQ(quantScale.GetTensorDesc().GetDataType(), DT_UINT64);
    EXPECT_EQ(quantScale.GetTensorDesc().GetShape().GetDims(), (std::vector<int64_t>{1}));
    ASSERT_EQ(quantScale.GetSize(), sizeof(uint64_t));
    EXPECT_EQ(*reinterpret_cast<const uint64_t*>(quantScale.GetData()), ExpectedQuantScale(0.25F, -3.0F));
    EXPECT_EQ(GetInputConst(fused, 6).GetTensorDesc().GetDataType(), DT_FLOAT16);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, QuantV2PerChannelWithDynamicNzWeight)
{
    Config config;
    config.weightFormat = FORMAT_FRACTAL_NZ;
    config.constWeight = false;
    config.withOffset = false;
    config.withQuant = true;
    config.quantPerChannel = true;
    config.withQuantOffset = true;
    config.quantOffset = 4.0F;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_FALSE(HasNode(graph, "AscendQuantV2"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    auto quantScale = GetInputConst(fused, 4);
    EXPECT_EQ(quantScale.GetTensorDesc().GetShape().GetDims(), (std::vector<int64_t>{N}));
    ASSERT_EQ(quantScale.GetSize(), static_cast<size_t>(N) * sizeof(uint64_t));
    const auto* data = reinterpret_cast<const uint64_t*>(quantScale.GetData());
    EXPECT_EQ(data[0], ExpectedQuantScale(0.25F, 4.0F));
    EXPECT_EQ(data[N - 1], ExpectedQuantScale(0.25F, 4.0F));
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, QuantV1AttrFusion)
{
    Config config;
    config.withQuant = true;
    config.quantV2 = false;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_FALSE(HasNode(graph, "AscendQuant"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    auto quantScale = GetInputConst(fused, 4);
    ASSERT_EQ(quantScale.GetSize(), sizeof(uint64_t));
    EXPECT_EQ(*reinterpret_cast<const uint64_t*>(quantScale.GetData()), ExpectedQuantScale(0.25F, 0.0F));
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, QuantSqrtModeFallsBackToFp16)
{
    Config config;
    config.withQuant = true;
    config.quantSqrtMode = true;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_TRUE(HasNode(graph, "AscendQuantV2"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    TensorDesc outDesc;
    ASSERT_EQ(fused.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(fused.GetInDataNodesAndPortIndexs(4).first, nullptr);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, NonConstQuantScaleFallsBackToFp16)
{
    Config config;
    config.withQuant = true;
    config.quantScaleConst = false;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    auto quant = FindNode(graph, "AscendQuantV2");
    EXPECT_NE(quant.GetInDataNodesAndPortIndexs(1).first, nullptr);
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    TensorDesc outDesc;
    ASSERT_EQ(fused.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, NonIntegerQuantOffsetFallsBackToFp16)
{
    Config config;
    config.withQuant = true;
    config.withQuantOffset = true;
    config.quantOffset = 1.5F;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_TRUE(HasNode(graph, "AscendQuantV2"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    TensorDesc outDesc;
    ASSERT_EQ(fused.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, NonLastAxisQuantFallsBackToFp16)
{
    Config config;
    config.withQuant = true;
    config.quantPerChannel = true;
    config.quantAxis = 0;
    config.mSize = N;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_TRUE(HasNode(graph, "AscendQuantV2"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    TensorDesc outDesc;
    ASSERT_EQ(fused.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(fused.GetInDataNodesAndPortIndexs(4).first, nullptr);
}

TEST_F(MatmulToWeightQuantBatchMatmulFusionTest, NonLastAxisParamShapeFallsBackToFp16)
{
    Config config;
    config.withQuant = true;
    config.quantPerChannel = true;
    config.quantInvalidParamShape = true;
    config.quantAxis = -2;
    config.mSize = N;
    auto graph = BuildTestGraph(config);
    ASSERT_EQ(pass.Run(graph, context), SUCCESS);
    EXPECT_TRUE(HasNode(graph, "AscendQuantV2"));
    auto fused = FindNode(graph, "WeightQuantBatchMatmulV2");
    TensorDesc outDesc;
    ASSERT_EQ(fused.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(fused.GetInDataNodesAndPortIndexs(4).first, nullptr);
}
} // namespace
