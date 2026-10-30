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
#include <string>
#include <vector>

#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "ge/ge_utils.h"
#include "es_nn_ops.h"
#include "../../../op_graph/fusion_pass/fused_weight_quant_mat_mul_fusion_pass.h"
#include "register/register_custom_pass.h"
#include "securec.h"
#include "ut_op_util.h"

using namespace ut_util;
using namespace ge;
using namespace fe;
using namespace ops;

namespace {
constexpr int64_t DIM_M = 64;
constexpr int64_t DIM_K = 128;
constexpr int64_t DIM_N = 256;
constexpr int64_t SCALAR_DIM = 1;
constexpr int64_t MISMATCHED_OFFSET_DIM = 2;

constexpr int64_t X_INPUT_INDEX = 0;
constexpr int64_t WEIGHT_INPUT_INDEX = 1;
constexpr int64_t ANTIQUANT_SCALE_INPUT_INDEX = 2;
constexpr int64_t ANTIQUANT_OFFSET_INPUT_INDEX = 3;
constexpr int64_t BIAS_INPUT_INDEX = 3;
constexpr int64_t BIAS_WITH_CONST_SCALE_INPUT_INDEX = 2;
constexpr int64_t QUANT_SCALE_INPUT_INDEX = 2;
constexpr int64_t QUANT_OFFSET_INPUT_INDEX = 2;
constexpr int64_t PERTOKEN_SCALE_INPUT_INDEX = 4;

constexpr int64_t OUTPUT_PORT = 0;
constexpr int64_t X_PORT = 0;
constexpr int64_t WEIGHT_PORT = 1;
constexpr int64_t ANTIQUANT_SCALE_PORT = 2;
constexpr int64_t ANTIQUANT_OFFSET_PORT = 3;
constexpr int64_t FE_BIAS_PORT = 4;
constexpr int64_t FE_PERTOKEN_SCALE_PORT = 5;
constexpr int64_t BIAS_PORT = 6;
constexpr int64_t QUANT_NODE_SCALE_PORT = 1;
constexpr int64_t QUANT_NODE_OFFSET_PORT = 2;

constexpr int64_t FUSED_INPUT_NUM_WITHOUT_BIAS = 3;
constexpr int64_t FUSED_INPUT_NUM_WITH_BIAS = 4;
constexpr int64_t DEFAULT_GROUP_SIZE = 0;
constexpr int64_t DEFAULT_DTYPE = -1;
constexpr int64_t DEFAULT_INNER_PRECISE = 0;
constexpr int64_t QUANT_DST_TYPE_INT8 = 2;
constexpr int64_t DEFAULT_QUANT_AXIS = -1;
constexpr int64_t INVALID_GROUP_SIZE = -1;

constexpr float ANTIQUANT_SCALE_BASE = 0.01F;
constexpr float QUANT_SCALE = 0.005F;
constexpr float V1_QUANT_SCALE = 0.125F;
constexpr float UNIT_SCALE = 1.0F;
constexpr float GOLDEN_QUANT_SCALE = 2.0F;
constexpr float GOLDEN_MM_RESULT = 4.0F;
constexpr float GOLDEN_QUANT_RESULT = 8.0F;
constexpr float PER_CHANNEL_QUANT_SCALE_BASE = 0.002F;
constexpr float PER_CHANNEL_QUANT_SCALE_STEP = 0.001F;
constexpr int64_t PER_CHANNEL_QUANT_SCALE_PERIOD = 8;
constexpr float ZERO_OFFSET = 0.0F;
constexpr float NEGATIVE_OFFSET = -8.0F;
constexpr float POSITIVE_OFFSET = 3.0F;
constexpr float NON_INTEGER_OFFSET = 1.5F;
constexpr float OUT_OF_RANGE_OFFSET = 200.0F;
constexpr float INT8_OFFSET_MIN = -128.0F;
constexpr float INT8_OFFSET_MAX = 127.0F;
constexpr float NEGATIVE_ONE_OFFSET = -1.0F;
constexpr float TWO_OFFSET = 2.0F;
constexpr int64_t OFFSET_PATTERN_PERIOD = 3;

constexpr uint64_t FLOAT_BITS_MASK = 0xFFFFFFFFULL;
constexpr uint64_t DEQ_SCALE_MASK = 0xFFFFE000ULL;
constexpr int64_t OFFSET_VALUE_SHIFT = 37;
constexpr uint64_t OFFSET_VALUE_MASK = 0x00001FE000000000ULL;
constexpr uint64_t OFFSET_SIGN_MASK = 0x0000200000000000ULL;
constexpr uint64_t OFFSET_SIGNED_MASK = 0x0000400000000000ULL;
constexpr uint64_t ENCODED_NEGATIVE_OFFSET_VALUE = 0x00001F0000000000ULL;
constexpr uint64_t ENCODED_POSITIVE_OFFSET_VALUE = 0x0000006000000000ULL;

struct FusedResult {
    bool hasFused = false;
    bool hasWqbmm = false;
    bool hasRelu = false;
    bool hasQuant = false;
    bool transX = false;
    bool transWeight = false;
    int64_t groupSize = INVALID_GROUP_SIZE;
    int64_t fusedDtype = DEFAULT_DTYPE;
    std::string fusedOpType;
    int64_t fusedInputNum = 0;
    DataType fusedYDtype = DT_UNDEFINED;
    DataType fusedScaleDtype = DT_UNDEFINED;
};

static bool GetNodeType(const GNode& node, std::string& type)
{
    AscendString nodeType;
    if (node.GetType(nodeType) != GRAPH_SUCCESS) {
        return false;
    }
    type = nodeType.GetString();
    return true;
}

static bool GetStringAttr(const GNode& node, const std::string& name, std::string& value)
{
    AscendString attrValue;
    if (node.GetAttr(name.c_str(), attrValue) != GRAPH_SUCCESS) {
        return false;
    }
    value = attrValue.GetString();
    return true;
}

FusedResult CheckFusionResult(Graph& graph)
{
    FusedResult result;
    for (auto node : graph.GetAllNodes()) {
        std::string type;
        if (!GetNodeType(node, type)) {
            continue;
        }
        if (type == "FusedWeightQuantMatMul") {
            result.hasFused = true;
            (void)node.GetAttr("transpose_x", result.transX);
            (void)node.GetAttr("transpose_weight", result.transWeight);
            (void)node.GetAttr("antiquant_group_size", result.groupSize);
            (void)node.GetAttr("dtype", result.fusedDtype);
            (void)GetStringAttr(node, "fused_op_type", result.fusedOpType);
            result.fusedInputNum = static_cast<int64_t>(node.GetInputsSize());
            TensorDesc yDesc;
            if (node.GetOutputDesc(OUTPUT_PORT, yDesc) == GRAPH_SUCCESS) {
                result.fusedYDtype = yDesc.GetDataType();
            }
            TensorDesc scaleDesc;
            if (node.GetInputDesc(ANTIQUANT_SCALE_PORT, scaleDesc) == GRAPH_SUCCESS) {
                result.fusedScaleDtype = scaleDesc.GetDataType();
            }
        } else if (type == "WeightQuantBatchMatmulV2") {
            result.hasWqbmm = true;
        } else if (type == "Relu") {
            result.hasRelu = true;
        } else if (type == "AscendQuant" || type == "AscendQuantV2") {
            result.hasQuant = true;
        }
    }
    return result;
}

// ===== 子模式 b（y INT8 输出量化）折叠参考实现（独立于 pass 重算，交叉验证） =====

static float RefDecodeDeqScale(uint64_t encoded)
{
    uint32_t bits = static_cast<uint32_t>(encoded & FLOAT_BITS_MASK);
    float value = 0.0F;
    if (memcpy_s(&value, sizeof(value), &bits, sizeof(bits)) != EOK) {
        ADD_FAILURE() << "Failed to decode reference deq scale";
        return 0.0F;
    }
    return value;
}

static uint64_t RefEncodeDeqScale(float value)
{
    uint32_t bits = 0U;
    if (memcpy_s(&bits, sizeof(bits), &value, sizeof(value)) != EOK) {
        ADD_FAILURE() << "Failed to encode reference deq scale";
        return 0UL;
    }
    return static_cast<uint64_t>(bits & DEQ_SCALE_MASK);
}

// 非对称量化 offset 高位编码参考实现（独立于 pass 重算，交叉验证）：
// 值（整型补码低 8 bit）左移 37 + 负数置 bit45 符号 + bit46 signed 恒置；offset 为 0 时高位全零
static uint64_t RefEncodeDeqScaleWithOffset(float scale, float offset)
{
    uint64_t encoded = RefEncodeDeqScale(scale);
    if (offset == ZERO_OFFSET) {
        return encoded;
    }
    auto offsetInt = static_cast<int64_t>(offset);
    encoded |= (static_cast<uint64_t>(offsetInt) << OFFSET_VALUE_SHIFT) & OFFSET_VALUE_MASK;
    if (offsetInt < 0) {
        encoded |= OFFSET_SIGN_MASK;
    }
    encoded |= OFFSET_SIGNED_MASK;
    return encoded;
}

// 读取融合节点 antiquant_scale 输入的 Const 数据（子模式 b 为 pass 折叠生成的新 Const）
static bool ReadFusedScaleConst(Graph& graph, std::vector<uint64_t>& data)
{
    for (auto node : graph.GetAllNodes()) {
        std::string type;
        if (!GetNodeType(node, type) || type != "FusedWeightQuantMatMul") {
            continue;
        }
        auto peer = node.GetInDataNodesAndPortIndexs(ANTIQUANT_SCALE_PORT);
        if (peer.first == nullptr) {
            return false;
        }
        Tensor tensor;
        if (peer.first->GetAttr(AscendString("value"), tensor) != GRAPH_SUCCESS || tensor.GetData() == nullptr) {
            return false;
        }
        if (tensor.GetSize() % sizeof(uint64_t) != 0UL) {
            return false;
        }
        const uint64_t* ptr = reinterpret_cast<const uint64_t*>(tensor.GetData());
        data.assign(ptr, ptr + tensor.GetSize() / sizeof(uint64_t));
        return true;
    }
    return false;
}

static TensorDesc MakeTD(const std::vector<int64_t>& shape, DataType dtype)
{
    TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetShape(Shape(shape));
    desc.SetOriginShape(Shape(shape));
    desc.SetFormat(FORMAT_ND);
    desc.SetOriginFormat(FORMAT_ND);
    return desc;
}

static es::EsTensorHolder BuildRelu(es::EsGraphBuilder& gb, const es::EsTensorHolder& x)
{
    auto* graph = gb.GetCGraphBuilder()->GetGraph();
    const std::string name = "relu_" + std::to_string(graph->GetAllNodes().size());
    auto node = es::CompliantNodeBuilder(graph)
                    .OpType("Relu")
                    .Name(name.c_str())
                    .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();
    (void)es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, X_PORT);
    return es::EsTensorHolder(gb.GetCGraphBuilder()->GetTensorHolderFromNode(node, OUTPUT_PORT));
}

static es::EsTensorHolder BuildAscendQuantV2(es::EsGraphBuilder& gb, const es::EsTensorHolder& x,
                                             const es::EsTensorHolder& scale, const es::EsTensorHolder* offset,
                                             bool sqrtMode, const char* roundMode, int64_t dstType, int64_t axis)
{
    auto* graph = gb.GetCGraphBuilder()->GetGraph();
    const std::string name = "quant_v2_" + std::to_string(graph->GetAllNodes().size());
    auto node = es::CompliantNodeBuilder(graph)
                    .OpType("AscendQuantV2")
                    .Name(name.c_str())
                    .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"scale", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"offset", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();
    (void)es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, X_PORT);
    (void)es::AddEdgeAndUpdatePeerDesc(*graph, *scale.GetProducer(), scale.GetProducerOutIndex(), node,
                                       QUANT_NODE_SCALE_PORT);
    if (offset != nullptr) {
        (void)es::AddEdgeAndUpdatePeerDesc(*graph, *offset->GetProducer(), offset->GetProducerOutIndex(), node,
                                           QUANT_NODE_OFFSET_PORT);
    }
    ge::AscendString roundModeAttr(roundMode);
    (void)node.SetAttr("sqrt_mode", sqrtMode);
    (void)node.SetAttr("round_mode", roundModeAttr);
    (void)node.SetAttr("dst_type", dstType);
    (void)node.SetAttr("axis", axis);
    return es::EsTensorHolder(gb.GetCGraphBuilder()->GetTensorHolderFromNode(node, OUTPUT_PORT));
}

static es::EsTensorHolder BuildAscendQuant(es::EsGraphBuilder& gb, const es::EsTensorHolder& x, float scale,
                                           float offset, bool sqrtMode, const char* roundMode, int64_t dstType)
{
    auto* graph = gb.GetCGraphBuilder()->GetGraph();
    const std::string name = "quant_" + std::to_string(graph->GetAllNodes().size());
    auto node = es::CompliantNodeBuilder(graph)
                    .OpType("AscendQuant")
                    .Name(name.c_str())
                    .IrDefInputs({{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();
    (void)es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, X_PORT);
    ge::AscendString roundModeAttr(roundMode);
    (void)node.SetAttr("scale", scale);
    (void)node.SetAttr("offset", offset);
    (void)node.SetAttr("sqrt_mode", sqrtMode);
    (void)node.SetAttr("round_mode", roundModeAttr);
    (void)node.SetAttr("dst_type", dstType);
    return es::EsTensorHolder(gb.GetCGraphBuilder()->GetTensorHolderFromNode(node, OUTPUT_PORT));
}

// es 建点仅连边不落输入 desc/attr，需显式 UpdateInputDesc + SetAttr
// （对齐 anti_quant/transpose UT 做法；生产图中 pass 运行于 inferShape 后，desc/attr 必已物化）
static void SetWqbmmInputDescs(const es::EsTensorHolder& wq, bool hasBias, DataType biasDtype,
                               const std::vector<int64_t>& xShape, const std::vector<int64_t>& wShape,
                               const std::vector<int64_t>& sShape, int64_t nDim, bool transX = false,
                               bool transWeight = false)
{
    auto* node = wq.GetProducer();
    node->UpdateInputDesc(X_PORT, MakeTD(xShape, DT_FLOAT16));
    node->UpdateInputDesc(WEIGHT_PORT, MakeTD(wShape, DT_INT8));
    node->UpdateInputDesc(ANTIQUANT_SCALE_PORT, MakeTD(sShape, DT_UINT64));
    if (hasBias) {
        node->UpdateInputDesc(BIAS_PORT, MakeTD({nDim}, biasDtype));
    }
    node->UpdateOutputDesc(OUTPUT_PORT, MakeTD({xShape[X_PORT], nDim}, DT_FLOAT16));
    bool tx = transX;
    node->SetAttr("transpose_x", tx);
    bool tw = transWeight;
    node->SetAttr("transpose_weight", tw);
    int64_t groupSize = DEFAULT_GROUP_SIZE;
    node->SetAttr("antiquant_group_size", groupSize);
}
} // namespace

class FusedWeightQuantMatMulFusionPassTest : public testing::Test {
protected:
    class TestablePass : public ops::FusedWeightQuantMatMulFusionPass {};

    void SetUp() override { MockS8S4Platform(); }

    static void MockS8S4Platform()
    {
        fe::PlatformInfoManager::Instance().platform_info_map_.clear();
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"s8s4"};
        optiCompilationInfo.soc_version = "soc_version";
        fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    static void SetUpTestCase() { MockS8S4Platform(); }

    static void TearDownTestCase() { fe::PlatformInfoManager::Instance().platform_info_map_.clear(); }
};

// 基本融合：WQBMMV2(A16W8 PerChannel) + Relu → FusedWeightQuantMatMul
TEST_F(FusedWeightQuantMatMulFusionPassTest, BasicFusion)
{
    es::EsGraphBuilder gb("fwqm_basic_fusion");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasWqbmm);
    EXPECT_FALSE(r.hasRelu);
    EXPECT_EQ(r.fusedOpType, "relu");
    EXPECT_EQ(r.groupSize, DEFAULT_GROUP_SIZE);
    EXPECT_EQ(r.fusedInputNum, FUSED_INPUT_NUM_WITHOUT_BIAS); // x/weight/antiquant_scale，无 bias
}

// 带 bias（fp16）融合：fused 节点 4 输入
TEST_F(FusedWeightQuantMatMulFusionPassTest, FusionWithBias)
{
    es::EsGraphBuilder gb("fwqm_fusion_with_bias");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto b = gb.CreateInput(BIAS_INPUT_INDEX, "bias", DT_FLOAT16, FORMAT_ND, {DIM_N});
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, b, false, false, DEFAULT_GROUP_SIZE,
                                           DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);

    SetWqbmmInputDescs(wq, true, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasWqbmm);
    EXPECT_EQ(r.fusedInputNum, FUSED_INPUT_NUM_WITH_BIAS);
}

// WQBMMV2 输出非单一 consumer（直连 Relu 与另一节点）：不融合
// （Relu 自身输出扇出不限，融合边界以 Relu 输出为准）
TEST_F(FusedWeightQuantMatMulFusionPassTest, NoFusionWhenReluNotSoleConsumer)
{
    es::EsGraphBuilder gb("fwqm_wqbmm_multi_consumer");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    // WQBMMV2 输出另接一个 Relu（两个 consumer）
    auto relu0 = BuildRelu(gb, wq);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    relu0.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu, relu0});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(*graph);
    EXPECT_FALSE(r.hasFused);
    EXPECT_TRUE(r.hasWqbmm);
}

// WQBMMV2 带 antiquant_offset：不融合（底座 checker 要求为空，pass 预判双保险）
TEST_F(FusedWeightQuantMatMulFusionPassTest, NoFusionWhenOffsetPresent)
{
    es::EsGraphBuilder gb("fwqm_offset_present");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto o = gb.CreateInput(ANTIQUANT_OFFSET_INPUT_INDEX, "antiquant_offset", DT_FLOAT, FORMAT_ND, {DIM_N});
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, o, nullptr, nullptr, nullptr, false, false, DEFAULT_GROUP_SIZE,
                                           DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(*graph);
    EXPECT_FALSE(r.hasFused);
    EXPECT_TRUE(r.hasWqbmm);
}

// per-tensor scale（shape (1,)）：不融合（一期仅 PerChannel）
TEST_F(FusedWeightQuantMatMulFusionPassTest, NoFusionWhenPerTensorScale)
{
    es::EsGraphBuilder gb("fwqm_per_tensor_scale");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {SCALAR_DIM});
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {SCALAR_DIM}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(*graph);
    EXPECT_FALSE(r.hasFused);
    EXPECT_TRUE(r.hasWqbmm);
}

TEST_F(FusedWeightQuantMatMulFusionPassTest, NoFusionWhenScaleHasInvalid2DLayout)
{
    es::EsGraphBuilder gb("fwqm_invalid_scale_layout");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {2, DIM_N / 2});
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {2, DIM_N / 2}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(*graph);
    EXPECT_FALSE(r.hasFused);
    EXPECT_TRUE(r.hasWqbmm);
}

// y 为 INT8（输出量化场景）：不融合
TEST_F(FusedWeightQuantMatMulFusionPassTest, NoFusionWhenYInt8)
{
    es::EsGraphBuilder gb("fwqm_y_int8");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    wq.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(*graph);
    EXPECT_FALSE(r.hasFused);
    EXPECT_TRUE(r.hasWqbmm);
}

TEST_F(FusedWeightQuantMatMulFusionPassTest, NoFusionWithoutS8S4)
{
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    optiCompilationInfo.soc_version = "soc_version";
    fe::PlatformInfoManager::Instance().platform_info_map_["soc_version"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    es::EsGraphBuilder gb("fwqm_without_s8s4");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(*graph);
    EXPECT_FALSE(r.hasFused);
    EXPECT_TRUE(r.hasWqbmm);
    EXPECT_TRUE(r.hasRelu);
}

TEST_F(FusedWeightQuantMatMulFusionPassTest, NoFusionWithoutPlatformInfo)
{
    fe::PlatformInfoManager::Instance().platform_info_map_.clear();
    es::EsGraphBuilder gb("fwqm_without_platform_info");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(*graph);
    EXPECT_FALSE(r.hasFused);
    EXPECT_TRUE(r.hasWqbmm);
    EXPECT_TRUE(r.hasRelu);
}

// ===== 子模式 b（y INT8 输出量化） =====

// 生成 n 个编码合法的反量化 scale：s_w[i] = 0.01 * (i + 1)
static std::vector<uint64_t> GenAntiScaleConst(int64_t n)
{
    std::vector<uint64_t> data(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) {
        data[static_cast<size_t>(i)] = RefEncodeDeqScale(ANTIQUANT_SCALE_BASE * static_cast<float>(i + 1));
    }
    return data;
}

// V2 + per-tensor Const scale（fp32 标量）：三算子融合，y INT8，折叠值逐元素校验
TEST_F(FusedWeightQuantMatMulFusionPassTest, QuantFusionV2PerTensorScale)
{
    es::EsGraphBuilder gb("fwqm_quant_v2_per_tensor");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto antiScaleData = GenAntiScaleConst(DIM_N);
    auto s = gb.CreateConst(antiScaleData, {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    auto quant = BuildAscendQuantV2(gb, relu, qScale, nullptr, false, "round", QUANT_DST_TYPE_INT8, DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasWqbmm);
    EXPECT_FALSE(r.hasRelu);
    EXPECT_FALSE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_INT8);
    EXPECT_EQ(r.fusedDtype, static_cast<int64_t>(DT_INT8));
    EXPECT_EQ(r.fusedScaleDtype, DT_UINT64);
    EXPECT_EQ(r.fusedOpType, "relu");
    EXPECT_EQ(r.fusedInputNum, FUSED_INPUT_NUM_WITHOUT_BIAS); // x/weight/antiquant_scale(Const)，无 bias

    // 折叠值逐元素断言：combined[i] = encode(decode(anti[i]) * s_q)，per-tensor s_q 广播
    std::vector<uint64_t> folded;
    ASSERT_TRUE(ReadFusedScaleConst(*graph, folded));
    ASSERT_EQ(folded.size(), static_cast<size_t>(DIM_N));
    for (int64_t i = 0; i < DIM_N; i++) {
        float sW = RefDecodeDeqScale(antiScaleData[static_cast<size_t>(i)]);
        uint64_t expected = RefEncodeDeqScale(sW * QUANT_SCALE);
        EXPECT_EQ(folded[static_cast<size_t>(i)], expected) << "idx " << i;
    }
}

TEST_F(FusedWeightQuantMatMulFusionPassTest, QuantFusionMatchesOriginalMultiplicativeScale)
{
    es::EsGraphBuilder gb("fwqm_quant_multiplicative_scale");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, SCALAR_DIM});
    auto s = gb.CreateConst(std::vector<uint64_t>{RefEncodeDeqScale(UNIT_SCALE)}, {SCALAR_DIM}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{GOLDEN_QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    auto quant = BuildAscendQuantV2(gb, relu, qScale, nullptr, false, "round", QUANT_DST_TYPE_INT8, DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, SCALAR_DIM}, {SCALAR_DIM}, SCALAR_DIM);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, SCALAR_DIM}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(OUTPUT_PORT, MakeTD({DIM_M, SCALAR_DIM}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, SCALAR_DIM}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);
    CustomPassContext passContext;
    TestablePass pass;
    ASSERT_EQ(pass.Run(graph, passContext), SUCCESS);

    std::vector<uint64_t> folded;
    ASSERT_TRUE(ReadFusedScaleConst(*graph, folded));
    ASSERT_EQ(folded.size(), 1UL);
    const float fusedResult = GOLDEN_MM_RESULT * RefDecodeDeqScale(folded[0]);
    const float originalResult = GOLDEN_MM_RESULT * UNIT_SCALE * GOLDEN_QUANT_SCALE;
    EXPECT_EQ(fusedResult, GOLDEN_QUANT_RESULT);
    EXPECT_EQ(fusedResult, originalResult);
}

// V2 + per-channel Const scale（fp32 (n,)）：per-channel 折叠 + bias 透传
TEST_F(FusedWeightQuantMatMulFusionPassTest, QuantFusionV2PerChannelScaleWithBias)
{
    es::EsGraphBuilder gb("fwqm_quant_v2_per_channel");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto antiScaleData = GenAntiScaleConst(DIM_N);
    auto s = gb.CreateConst(antiScaleData, {DIM_N}, DT_UINT64, FORMAT_ND);
    auto b = gb.CreateInput(BIAS_WITH_CONST_SCALE_INPUT_INDEX, "bias", DT_FLOAT16, FORMAT_ND,
                            {DIM_N}); // 索引连续（x=0/weight=1/bias=2）
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, b, false, false, DEFAULT_GROUP_SIZE,
                                           DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    std::vector<float> qScaleData(static_cast<size_t>(DIM_N));
    for (int64_t i = 0; i < DIM_N; i++) {
        qScaleData[static_cast<size_t>(i)] = PER_CHANNEL_QUANT_SCALE_BASE +
                                             PER_CHANNEL_QUANT_SCALE_STEP *
                                                 static_cast<float>(i % PER_CHANNEL_QUANT_SCALE_PERIOD);
    }
    auto qScale = gb.CreateConst(qScaleData, {DIM_N}, DT_FLOAT, FORMAT_ND);
    auto quant = BuildAscendQuantV2(gb, relu, qScale, nullptr, false, "round", QUANT_DST_TYPE_INT8, DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, true, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasWqbmm);
    EXPECT_FALSE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_INT8);
    EXPECT_EQ(r.fusedInputNum, FUSED_INPUT_NUM_WITH_BIAS); // x/weight/antiquant_scale(Const)/bias

    std::vector<uint64_t> folded;
    ASSERT_TRUE(ReadFusedScaleConst(*graph, folded));
    ASSERT_EQ(folded.size(), static_cast<size_t>(DIM_N));
    for (int64_t i = 0; i < DIM_N; i++) {
        float sW = RefDecodeDeqScale(antiScaleData[static_cast<size_t>(i)]);
        uint64_t expected = RefEncodeDeqScale(sW * qScaleData[static_cast<size_t>(i)]);
        EXPECT_EQ(folded[static_cast<size_t>(i)], expected) << "idx " << i;
    }
}

// V1（AscendQuant，scale/offset 为标量属性）+ uint64 编码 Const scale：属性折叠 + 解码通路
TEST_F(FusedWeightQuantMatMulFusionPassTest, QuantFusionV1AttrScale)
{
    es::EsGraphBuilder gb("fwqm_quant_v1_attr");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto antiScaleData = GenAntiScaleConst(DIM_N);
    auto s = gb.CreateConst(antiScaleData, {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto quant = BuildAscendQuant(gb, relu, V1_QUANT_SCALE, ZERO_OFFSET, false, "Round", QUANT_DST_TYPE_INT8);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_INT8);

    std::vector<uint64_t> folded;
    ASSERT_TRUE(ReadFusedScaleConst(*graph, folded));
    ASSERT_EQ(folded.size(), static_cast<size_t>(DIM_N));
    for (int64_t i = 0; i < DIM_N; i++) {
        float sW = RefDecodeDeqScale(antiScaleData[static_cast<size_t>(i)]);
        uint64_t expected = RefEncodeDeqScale(sW * V1_QUANT_SCALE);
        EXPECT_EQ(folded[static_cast<size_t>(i)], expected) << "idx " << i;
    }
}

TEST_F(FusedWeightQuantMatMulFusionPassTest, FallbackToReluWhenNDimIsNotPositive)
{
    constexpr int64_t ZERO_N_DIM = 0;
    constexpr int64_t UNKNOWN_N_DIM = -1;
    constexpr int64_t INVALID_N_DIM = -2;
    for (int64_t nDim : {ZERO_N_DIM, UNKNOWN_N_DIM, INVALID_N_DIM}) {
        SCOPED_TRACE(nDim);
        es::EsGraphBuilder gb("fwqm_quant_invalid_n");
        auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
        auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
        auto s = gb.CreateConst(GenAntiScaleConst(DIM_N), {DIM_N}, DT_UINT64, FORMAT_ND);
        auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                               DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
        auto relu = BuildRelu(gb, wq);
        auto quant = BuildAscendQuant(gb, relu, V1_QUANT_SCALE, ZERO_OFFSET, false, "Round", QUANT_DST_TYPE_INT8);

        SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, nDim}, {DIM_N}, DIM_N);
        relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, nDim}, DT_FLOAT16));
        quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, nDim}, DT_INT8));

        std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
        ASSERT_NE(graph, nullptr);
        CustomPassContext passContext;
        TestablePass pass;
        const auto status = pass.Run(graph, passContext);
        auto r = CheckFusionResult(*graph);
        EXPECT_TRUE(r.hasQuant);
        if (nDim == INVALID_N_DIM) {
            EXPECT_EQ(status, GRAPH_NOT_CHANGED);
            EXPECT_FALSE(r.hasFused);
            continue;
        }
        EXPECT_EQ(status, SUCCESS);
        EXPECT_TRUE(r.hasFused);
        EXPECT_EQ(r.fusedYDtype, DT_FLOAT16);
    }
}

// ===== 非对称量化（offset 高位编码） =====

// V2 + per-tensor 整数 offset（负零点）：offset 编入高位，折叠值逐元素校验（低 32 位 scale +
// bits 37-44 值 + bit45 符号 + bit46 signed）
TEST_F(FusedWeightQuantMatMulFusionPassTest, QuantFusionV2PerTensorOffset)
{
    es::EsGraphBuilder gb("fwqm_quant_v2_per_tensor_offset");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto antiScaleData = GenAntiScaleConst(DIM_N);
    auto s = gb.CreateConst(antiScaleData, {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    auto qOffset = gb.CreateConst(std::vector<float>{NEGATIVE_OFFSET}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND); // 负零点
    auto quant = BuildAscendQuantV2(gb, relu, qScale, &qOffset, false, "round", QUANT_DST_TYPE_INT8,
                                    DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasWqbmm);
    EXPECT_FALSE(r.hasRelu);
    EXPECT_FALSE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_INT8);

    // 折叠值逐元素断言：combined[i] = encode(s_w[i] * s_q) 高位编入 offset（per-tensor 广播）
    std::vector<uint64_t> folded;
    ASSERT_TRUE(ReadFusedScaleConst(*graph, folded));
    ASSERT_EQ(folded.size(), static_cast<size_t>(DIM_N));
    for (int64_t i = 0; i < DIM_N; i++) {
        float sW = RefDecodeDeqScale(antiScaleData[static_cast<size_t>(i)]);
        uint64_t expected = RefEncodeDeqScaleWithOffset(sW * QUANT_SCALE, NEGATIVE_OFFSET);
        EXPECT_EQ(folded[static_cast<size_t>(i)], expected) << "idx " << i;
    }
    // 高位字段抽查：offset=-8 → 值字段 = 补码 0xF8（bits 37-44），bit45 符号置位，bit46 置位
    EXPECT_EQ(folded.front() & OFFSET_VALUE_MASK, ENCODED_NEGATIVE_OFFSET_VALUE);
    EXPECT_NE(folded.front() & OFFSET_SIGN_MASK, 0ULL);   // bit45
    EXPECT_NE(folded.front() & OFFSET_SIGNED_MASK, 0ULL); // bit46
}

// V2 + per-channel offset（(n,)，含正/负/值域边界）：逐通道编码
TEST_F(FusedWeightQuantMatMulFusionPassTest, QuantFusionV2PerChannelOffset)
{
    es::EsGraphBuilder gb("fwqm_quant_v2_per_channel_offset");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto antiScaleData = GenAntiScaleConst(DIM_N);
    auto s = gb.CreateConst(antiScaleData, {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    // per-channel offset：值域边界（-128/127）与小值负零点混合
    std::vector<float> qOffsetData(static_cast<size_t>(DIM_N));
    for (int64_t i = 0; i < DIM_N; i++) {
        qOffsetData[static_cast<size_t>(i)] = (i % OFFSET_PATTERN_PERIOD == 0) ?
                                                  INT8_OFFSET_MIN :
                                                  ((i % OFFSET_PATTERN_PERIOD == 1) ? INT8_OFFSET_MAX :
                                                                                      NEGATIVE_ONE_OFFSET);
    }
    auto qOffset = gb.CreateConst(qOffsetData, {DIM_N}, DT_FLOAT, FORMAT_ND);
    auto quant = BuildAscendQuantV2(gb, relu, qScale, &qOffset, false, "round", QUANT_DST_TYPE_INT8,
                                    DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_INT8);

    std::vector<uint64_t> folded;
    ASSERT_TRUE(ReadFusedScaleConst(*graph, folded));
    ASSERT_EQ(folded.size(), static_cast<size_t>(DIM_N));
    for (int64_t i = 0; i < DIM_N; i++) {
        float sW = RefDecodeDeqScale(antiScaleData[static_cast<size_t>(i)]);
        uint64_t expected = RefEncodeDeqScaleWithOffset(sW * QUANT_SCALE, qOffsetData[static_cast<size_t>(i)]);
        EXPECT_EQ(folded[static_cast<size_t>(i)], expected) << "idx " << i;
    }
}

// V1（AscendQuant）+ 非零 offset 属性（正零点）：属性式 offset 折叠
TEST_F(FusedWeightQuantMatMulFusionPassTest, QuantFusionV1AttrOffset)
{
    es::EsGraphBuilder gb("fwqm_quant_v1_attr_offset");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto antiScaleData = GenAntiScaleConst(DIM_N);
    auto s = gb.CreateConst(antiScaleData, {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto quant = BuildAscendQuant(gb, relu, V1_QUANT_SCALE, POSITIVE_OFFSET, false, "Round",
                                  QUANT_DST_TYPE_INT8); // offset 属性 = 3

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_INT8);

    std::vector<uint64_t> folded;
    ASSERT_TRUE(ReadFusedScaleConst(*graph, folded));
    ASSERT_EQ(folded.size(), static_cast<size_t>(DIM_N));
    for (int64_t i = 0; i < DIM_N; i++) {
        float sW = RefDecodeDeqScale(antiScaleData[static_cast<size_t>(i)]);
        uint64_t expected = RefEncodeDeqScaleWithOffset(sW * V1_QUANT_SCALE, POSITIVE_OFFSET);
        EXPECT_EQ(folded[static_cast<size_t>(i)], expected) << "idx " << i;
    }
    // 高位字段抽查：offset=3 → 值字段 = 3（bits 38,37），无符号位，bit46 置位
    EXPECT_EQ(folded.front() & OFFSET_VALUE_MASK, ENCODED_POSITIVE_OFFSET_VALUE);
    EXPECT_EQ(folded.front() & OFFSET_SIGN_MASK, 0ULL);   // bit45 不置位
    EXPECT_NE(folded.front() & OFFSET_SIGNED_MASK, 0ULL); // bit46 置位
}

// V2 全零 offset Const：对称语义，编码与不接 offset 完全一致（高位全零）
TEST_F(FusedWeightQuantMatMulFusionPassTest, QuantFusionV2ZeroOffsetConst)
{
    es::EsGraphBuilder gb("fwqm_quant_v2_zero_offset");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto antiScaleData = GenAntiScaleConst(DIM_N);
    auto s = gb.CreateConst(antiScaleData, {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    auto qOffset = gb.CreateConst(std::vector<float>{ZERO_OFFSET}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND); // 全零 offset
    auto quant = BuildAscendQuantV2(gb, relu, qScale, &qOffset, false, "round", QUANT_DST_TYPE_INT8,
                                    DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_INT8);

    std::vector<uint64_t> folded;
    ASSERT_TRUE(ReadFusedScaleConst(*graph, folded));
    ASSERT_EQ(folded.size(), static_cast<size_t>(DIM_N));
    for (int64_t i = 0; i < DIM_N; i++) {
        float sW = RefDecodeDeqScale(antiScaleData[static_cast<size_t>(i)]);
        uint64_t expected = RefEncodeDeqScale(sW * QUANT_SCALE); // 高位全零，与对称一致
        EXPECT_EQ(folded[static_cast<size_t>(i)], expected) << "idx " << i;
    }
}

// V2 scale 非 Const（子图输入）：子模式 b 不满足 → fallback 子模式 a（量化节点保留）
TEST_F(FusedWeightQuantMatMulFusionPassTest, FallbackToReluWhenScaleNotConst)
{
    es::EsGraphBuilder gb("fwqm_quant_scale_not_const");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateConst(GenAntiScaleConst(DIM_N), {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateInput(QUANT_SCALE_INPUT_INDEX, "quant_scale", DT_FLOAT, FORMAT_ND,
                                 {SCALAR_DIM}); // 非 Const；索引连续
    auto quant = BuildAscendQuantV2(gb, relu, qScale, nullptr, false, "round", QUANT_DST_TYPE_INT8, DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    // fallback a：WQBMMV2+Relu 融合（y FP16），AscendQuantV2 保留为独立节点
    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasWqbmm);
    EXPECT_FALSE(r.hasRelu);
    EXPECT_TRUE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_FLOAT16);
}

// V2 offset 非整数（1.5）：硬件 offset 字段为整型码偏移，round 引入 ≤0.5 code 误差，
// 保守准入拒绝 → fallback 子模式 a
TEST_F(FusedWeightQuantMatMulFusionPassTest, FallbackToReluWhenOffsetNotInteger)
{
    es::EsGraphBuilder gb("fwqm_quant_offset_not_integer");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateConst(GenAntiScaleConst(DIM_N), {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    auto qOffset = gb.CreateConst(std::vector<float>{NON_INTEGER_OFFSET}, {SCALAR_DIM}, DT_FLOAT,
                                  FORMAT_ND); // 非整数 offset
    auto quant = BuildAscendQuantV2(gb, relu, qScale, &qOffset, false, "round", QUANT_DST_TYPE_INT8,
                                    DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasRelu);
    EXPECT_TRUE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_FLOAT16);
}

// V2 sqrt_mode=true：数值语义与 fixpipe requant 不一致，fallback 子模式 a
TEST_F(FusedWeightQuantMatMulFusionPassTest, FallbackToReluWhenSqrtMode)
{
    es::EsGraphBuilder gb("fwqm_quant_sqrt_mode");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateConst(GenAntiScaleConst(DIM_N), {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    auto quant = BuildAscendQuantV2(gb, relu, qScale, nullptr, true, "round", QUANT_DST_TYPE_INT8,
                                    DEFAULT_QUANT_AXIS); // sqrt_mode

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_TRUE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_FLOAT16);
}

// V2 offset 超出 int8 零点值域（200）：8 bit 值字段无法承载，fallback 子模式 a
TEST_F(FusedWeightQuantMatMulFusionPassTest, FallbackToReluWhenOffsetOutOfRange)
{
    es::EsGraphBuilder gb("fwqm_quant_offset_out_of_range");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateConst(GenAntiScaleConst(DIM_N), {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    auto qOffset = gb.CreateConst(std::vector<float>{OUT_OF_RANGE_OFFSET}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND); // 超值域
    auto quant = BuildAscendQuantV2(gb, relu, qScale, &qOffset, false, "round", QUANT_DST_TYPE_INT8,
                                    DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_TRUE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_FLOAT16);
}

// V2 offset 非 Const（子图输入）：无法在编译期读取编码，fallback 子模式 a
TEST_F(FusedWeightQuantMatMulFusionPassTest, FallbackToReluWhenOffsetNotConst)
{
    es::EsGraphBuilder gb("fwqm_quant_offset_not_const");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateConst(GenAntiScaleConst(DIM_N), {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    auto qOffset = gb.CreateInput(QUANT_OFFSET_INPUT_INDEX, "quant_offset", DT_FLOAT, FORMAT_ND,
                                  {SCALAR_DIM}); // 非 Const
    auto quant = BuildAscendQuantV2(gb, relu, qScale, &qOffset, false, "round", QUANT_DST_TYPE_INT8,
                                    DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_TRUE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_FLOAT16);
}

// V2 per-channel offset 尺寸与 n 不符（size=2, n=256）：广播规则不满足，fallback 子模式 a
TEST_F(FusedWeightQuantMatMulFusionPassTest, FallbackToReluWhenOffsetSizeMismatch)
{
    es::EsGraphBuilder gb("fwqm_quant_offset_size_mismatch");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateConst(GenAntiScaleConst(DIM_N), {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    // size=2 且非 1：per-channel 尺寸与 n=256 不符
    auto qOffset = gb.CreateConst(std::vector<float>{NEGATIVE_ONE_OFFSET, TWO_OFFSET}, {MISMATCHED_OFFSET_DIM},
                                  DT_FLOAT, FORMAT_ND);
    auto quant = BuildAscendQuantV2(gb, relu, qScale, &qOffset, false, "round", QUANT_DST_TYPE_INT8,
                                    DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_TRUE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_FLOAT16);
}

// 上游 antiquant_scale 自带高位字段（bits 32-63 非零，如老代 pass 复合编码产物）：
// 二次折叠会静默丢弃上游高位，保守拒绝子模式 b → fallback 子模式 a
// （a 模式 scale 直通不改写，原图语义不受影响）
TEST_F(FusedWeightQuantMatMulFusionPassTest, FallbackToReluWhenAntiScaleHighBitsNonZero)
{
    es::EsGraphBuilder gb("fwqm_quant_anti_scale_high_bits");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    // 构造带 bit46 signed 标志的表项（模拟老代 pass 复合编码）
    auto antiScaleData = GenAntiScaleConst(DIM_N);
    for (auto& v : antiScaleData) {
        v |= OFFSET_SIGNED_MASK;
    }
    auto s = gb.CreateConst(antiScaleData, {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);
    auto qScale = gb.CreateConst(std::vector<float>{QUANT_SCALE}, {SCALAR_DIM}, DT_FLOAT, FORMAT_ND);
    auto quant = BuildAscendQuantV2(gb, relu, qScale, nullptr, false, "round", QUANT_DST_TYPE_INT8, DEFAULT_QUANT_AXIS);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quant.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));

    std::shared_ptr<Graph> graph = gb.BuildAndReset({quant});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    // fallback a：WQBMMV2+Relu 融合（y FP16，antiquant_scale 原样直通），量化节点保留
    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_TRUE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_FLOAT16);
}

// V1 AscendQuant 缺 offset 属性（scale/offset 均为 REQUIRED 属性，缺失即异常）：不能按
// 对称量化（offset=0）假定继续（非零 offset 会被静默丢弃），保守 fallback 子模式 a
TEST_F(FusedWeightQuantMatMulFusionPassTest, FallbackToReluWhenV1OffsetAttrMissing)
{
    es::EsGraphBuilder gb("fwqm_v1_offset_missing");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateConst(GenAntiScaleConst(DIM_N), {DIM_N}, DT_UINT64, FORMAT_ND);
    auto wq = es::WeightQuantBatchMatmulV2(x, w, s, nullptr, nullptr, nullptr, nullptr, false, false,
                                           DEFAULT_GROUP_SIZE, DEFAULT_DTYPE, DEFAULT_INNER_PRECISE);
    auto relu = BuildRelu(gb, wq);

    // 手工建 AscendQuant 节点：仅设 scale 属性，不设 offset（模拟属性缺失/未物化）
    auto* graph = gb.GetCGraphBuilder()->GetGraph();
    ge::GNode quantNode = ge::es::CompliantNodeBuilder(graph)
                              .OpType("AscendQuant")
                              .Name("quant")
                              .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                              .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .Build();
    (void)ge::es::AddEdgeAndUpdatePeerDesc(*graph, *relu.GetProducer(), relu.GetProducerOutIndex(), quantNode, X_PORT);
    quantNode.UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));
    quantNode.UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_INT8));
    float scale = V1_QUANT_SCALE;
    (void)quantNode.SetAttr("scale", scale);

    SetWqbmmInputDescs(wq, false, DT_FLOAT16, {DIM_M, DIM_K}, {DIM_K, DIM_N}, {DIM_N}, DIM_N);
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, DIM_N}, DT_FLOAT16));

    auto quantOut = es::EsTensorHolder(gb.GetCGraphBuilder()->GetTensorHolderFromNode(quantNode, OUTPUT_PORT));
    std::shared_ptr<Graph> graphOut = gb.BuildAndReset({quantOut});
    ASSERT_NE(graphOut, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graphOut, passContext), SUCCESS);

    // fallback a：WQBMMV2+Relu 融合（y FP16），AscendQuant 保留为独立节点
    auto r = CheckFusionResult(*graphOut);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasRelu);
    EXPECT_TRUE(r.hasQuant);
    EXPECT_EQ(r.fusedYDtype, DT_FLOAT16);
}

// ===== FE 布局（bias@4 / pertoken_scale@5，无 quant 输入）路径 =====
// FE 生成的 WQBMMV2 节点为另一输入布局（x=0/weight=1/antiquant_scale=2/antiquant_offset=3/
// bias=4/pertoken_scale=5），端口经 _input_name_key/_input_name_value 名字映射或节点 IR 名
// 解析（见 ResolveInputPorts）；底座 checker 要求 pertoken_scale 为空，接入即拒绝融合

namespace {

// 构造 FE 布局 WQBMMV2 节点（CompliantNodeBuilder 自定义 IR：bias@4/pertoken_scale@5），
// 按 FE 物理布线连边（bias→端口 4、pertoken→端口 5），可选写入 _input_name_key 名字映射
static ge::GNode MakeFeLayoutWqbmm(es::EsGraphBuilder& gb, const es::EsTensorHolder& x, const es::EsTensorHolder& w,
                                   const es::EsTensorHolder& s, const es::EsTensorHolder* bias,
                                   const es::EsTensorHolder* pertoken, bool withNameMap, int64_t nDim)
{
    auto* graph = gb.GetCGraphBuilder()->GetGraph();
    ge::GNode node = ge::es::CompliantNodeBuilder(graph)
                         .OpType("WeightQuantBatchMatmulV2")
                         .Name("fe_wqbmm")
                         .IrDefInputs({{"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"weight", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"antiquant_scale", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"antiquant_offset", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                       {"bias", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""},
                                       {"pertoken_scale", ge::es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                         .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
    // 按 FE 物理布线连边：x@0/weight@1/antiquant_scale@2/bias@4/pertoken_scale@5
    (void)ge::es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, X_PORT);
    (void)ge::es::AddEdgeAndUpdatePeerDesc(*graph, *w.GetProducer(), w.GetProducerOutIndex(), node, WEIGHT_PORT);
    (void)ge::es::AddEdgeAndUpdatePeerDesc(*graph, *s.GetProducer(), s.GetProducerOutIndex(), node,
                                           ANTIQUANT_SCALE_PORT);
    if (bias != nullptr) {
        (void)ge::es::AddEdgeAndUpdatePeerDesc(*graph, *bias->GetProducer(), bias->GetProducerOutIndex(), node,
                                               FE_BIAS_PORT);
    }
    if (pertoken != nullptr) {
        (void)ge::es::AddEdgeAndUpdatePeerDesc(*graph, *pertoken->GetProducer(), pertoken->GetProducerOutIndex(), node,
                                               FE_PERTOKEN_SCALE_PORT);
    }
    // es 建点仅连边不落输入 desc/attr，需显式 UpdateInputDesc + SetAttr
    node.UpdateInputDesc(X_PORT, MakeTD({DIM_M, DIM_K}, DT_FLOAT16));
    node.UpdateInputDesc(WEIGHT_PORT, MakeTD({DIM_K, DIM_N}, DT_INT8));
    node.UpdateInputDesc(ANTIQUANT_SCALE_PORT, MakeTD({DIM_N}, DT_UINT64));
    if (bias != nullptr) {
        node.UpdateInputDesc(FE_BIAS_PORT, MakeTD({nDim}, DT_FLOAT16));
    }
    node.UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, nDim}, DT_FLOAT16));
    bool tx = false;
    bool tw = false;
    int64_t groupSize = DEFAULT_GROUP_SIZE;
    (void)node.SetAttr("transpose_x", tx);
    (void)node.SetAttr("transpose_weight", tw);
    (void)node.SetAttr("antiquant_group_size", groupSize);
    if (withNameMap) {
        // 名字映射为权威端口来源（FE 后续阶段写入）：x=0/weight=1/antiquant_scale=2/
        // antiquant_offset=3/bias=4[/pertoken_scale=5]，映射未列出的输入视为不存在
        std::vector<std::string> nameKeys = {"x", "weight", "antiquant_scale", "antiquant_offset", "bias"};
        std::vector<int64_t> namePorts = {X_PORT, WEIGHT_PORT, ANTIQUANT_SCALE_PORT, ANTIQUANT_OFFSET_PORT,
                                          FE_BIAS_PORT};
        if (pertoken != nullptr) {
            nameKeys.emplace_back("pertoken_scale");
            namePorts.push_back(FE_PERTOKEN_SCALE_PORT);
        }
        std::vector<ge::AscendString> nameKeyAttrs;
        nameKeyAttrs.reserve(nameKeys.size());
        for (const auto& nameKey : nameKeys) {
            nameKeyAttrs.emplace_back(nameKey.c_str());
        }
        (void)node.SetAttr(ge::AscendString("_input_name_key"), nameKeyAttrs);
        (void)node.SetAttr(ge::AscendString("_input_name_value"), namePorts);
    }
    return node;
}

// FE 节点输出接 Relu（唯一 consumer），返回 Relu holder 供构图收尾
static es::EsTensorHolder ReluOn(es::EsGraphBuilder& gb, const ge::GNode& node, int64_t nDim)
{
    auto relu = BuildRelu(gb, es::EsTensorHolder(gb.GetCGraphBuilder()->GetTensorHolderFromNode(node, OUTPUT_PORT)));
    relu.GetProducer()->UpdateOutputDesc(OUTPUT_PORT, MakeTD({DIM_M, nDim}, DT_FLOAT16));
    return relu;
}
} // namespace

// FE 布局 + _input_name_key 名字映射：bias 接在端口 4，融合后 bias 边不丢失
TEST_F(FusedWeightQuantMatMulFusionPassTest, FusionWithFeLayoutBiasViaNameMap)
{
    es::EsGraphBuilder gb("fwqm_fe_layout_name_map");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto b = gb.CreateInput(BIAS_INPUT_INDEX, "bias", DT_FLOAT16, FORMAT_ND, {DIM_N});
    auto wqNode = MakeFeLayoutWqbmm(gb, x, w, s, &b, nullptr, true, DIM_N);
    auto relu = ReluOn(gb, wqNode, DIM_N);

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasWqbmm);
    EXPECT_FALSE(r.hasRelu);
    EXPECT_EQ(r.fusedInputNum, FUSED_INPUT_NUM_WITH_BIAS); // x/weight/antiquant_scale/bias，FE 布局 bias@4 不丢失
}

// FE 布局 + 名字映射 + pertoken_scale 接入（端口 5）：不融合（底座 checker 要求为空，
// 端口 5 连接边无论语义为 pertoken_scale 还是 quant_offset 均拒绝）
TEST_F(FusedWeightQuantMatMulFusionPassTest, NoFusionWhenPertokenScalePresent)
{
    es::EsGraphBuilder gb("fwqm_fe_layout_pertoken");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto b = gb.CreateInput(BIAS_INPUT_INDEX, "bias", DT_FLOAT16, FORMAT_ND, {DIM_N});
    auto p = gb.CreateInput(PERTOKEN_SCALE_INPUT_INDEX, "pertoken_scale", DT_FLOAT, FORMAT_ND, {SCALAR_DIM});
    auto wqNode = MakeFeLayoutWqbmm(gb, x, w, s, &b, &p, true, DIM_N);
    auto relu = ReluOn(gb, wqNode, DIM_N);

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);

    auto r = CheckFusionResult(*graph);
    EXPECT_FALSE(r.hasFused);
    EXPECT_TRUE(r.hasWqbmm);
    EXPECT_TRUE(r.hasRelu);
}

// FE 布局无名字映射：端口按节点 IR 名解析（GetInputIndexByName），bias@4 融合不丢失
TEST_F(FusedWeightQuantMatMulFusionPassTest, FusionWithFeLayoutBiasViaIrNames)
{
    es::EsGraphBuilder gb("fwqm_fe_layout_ir_names");
    auto x = gb.CreateInput(X_INPUT_INDEX, "x", DT_FLOAT16, FORMAT_ND, {DIM_M, DIM_K});
    auto w = gb.CreateInput(WEIGHT_INPUT_INDEX, "weight", DT_INT8, FORMAT_ND, {DIM_K, DIM_N});
    auto s = gb.CreateInput(ANTIQUANT_SCALE_INPUT_INDEX, "antiquant_scale", DT_UINT64, FORMAT_ND, {DIM_N});
    auto b = gb.CreateInput(BIAS_INPUT_INDEX, "bias", DT_FLOAT16, FORMAT_ND, {DIM_N});
    auto wqNode = MakeFeLayoutWqbmm(gb, x, w, s, &b, nullptr, false, DIM_N);
    auto relu = ReluOn(gb, wqNode, DIM_N);

    std::shared_ptr<Graph> graph = gb.BuildAndReset({relu});
    ASSERT_NE(graph, nullptr);

    CustomPassContext passContext;
    TestablePass pass;
    EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);

    auto r = CheckFusionResult(*graph);
    EXPECT_TRUE(r.hasFused);
    EXPECT_FALSE(r.hasWqbmm);
    EXPECT_EQ(r.fusedInputNum, FUSED_INPUT_NUM_WITH_BIAS);
}
