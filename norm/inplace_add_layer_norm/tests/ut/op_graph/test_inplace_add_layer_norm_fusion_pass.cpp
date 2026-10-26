/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/inplace_add_layer_norm_proto.h"
#include "../../../op_graph/fusion_pass/inplace_add_layer_norm_fusion_pass.h"

using namespace fe;
using namespace ge;
using namespace ops;

namespace {
constexpr float kEpsilon = 0.00001f;
constexpr int64_t kX1Idx = 0;
constexpr int64_t kX2Idx = 1;
constexpr int64_t kGammaIdx = 2;
constexpr int64_t kBetaIdx = 3;
constexpr int64_t kBiasIdx = 4;

// 守卫 G6 要求 l2/8 <= x1_bytes <= 2*l2。
constexpr int64_t kL2Size = 134217728L;
constexpr int64_t kRowsInBand = 16384L;    // fp16 [16384,1024] = 32MB，落在 [16MB,256MB]
constexpr int64_t kRowsTooSmall = 256L;    // 0.5MB，低于下界
constexpr int64_t kRowsTooLarge = 262144L; // 512MB，高于上界
constexpr int64_t kCols = 1024L;
constexpr int64_t kDynamicReduceAxis = 5120L; // 动态 shape 下的经验归约轴

const char* const kAddLayerNorm = "AddLayerNorm";
const char* const kInplaceAddLayerNorm = "InplaceAddLayerNorm";

class ZInplaceAddLayerNormFusionPassTest : public testing::Test {
protected:
    // 部分 CANN 环境未提供 es::Relu，额外消费者统一用 es::Cast 构造。
    // GNode 是轻量句柄，与图内节点共享底层对象，故可跨改图操作保留并观察接线变化。
    GNode extraConsumer_;

    void SetUp() override
    {
        extraConsumer_ = GNode();
        SetPlatform("Ascend950");
    }

    static void SetPlatform(const std::string& soc, int64_t l2Size = kL2Size)
    {
        fe::PlatFormInfos infos;
        (void)infos.Init();
        const std::map<std::string, std::string> archs = {{"Ascend950", "3510"},    {"Ascend350", "3510"},
                                                          {"MC62", "5102"},         {"Ascend910B", "2201"},
                                                          {"Ascend910_93", "2201"}, {"Ascend310P", "2002"}};
        const auto arch = archs.find(soc);
        std::map<std::string, std::string> version = {{"NpuArch", arch == archs.end() ? "" : arch->second}};
        infos.SetPlatformRes("version", version);
        std::map<std::string, std::string> socInfo = {{"ai_core_cnt", "64"}, {"l2_size", std::to_string(l2Size)}};
        infos.SetPlatformRes("SoCInfo", socInfo);
        fe::OptionalInfos options;
        (void)options.Init();
        options.SetSocVersion(soc);
        PlatformInfoManager::Instance().platform_infos_map_[soc] = infos;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(options);
    }

    // 构造单个 AddLayerNorm 节点的图。额外消费者经成员 extraConsumer_ 暴露给用例。
    std::shared_ptr<Graph> BuildGraph(bool withBias, int64_t rows = kRowsInBand, int64_t cols = kCols,
                                      DataType xDtype = DT_FLOAT16, DataType gammaDtype = DT_FLOAT16,
                                      bool extraConsumerOnX1 = false, bool extraConsumerOnY = false,
                                      int32_t directDataInput = -1)
    {
        const std::vector<int64_t> xShape = {rows, cols};
        const std::vector<int64_t> pShape = {cols};

        auto builder = es::EsGraphBuilder("inplace_add_layer_norm_fusion_test");
        auto x1 = builder.CreateInput(kX1Idx, "x1", xDtype, FORMAT_ND, xShape);
        auto x2 = builder.CreateInput(kX2Idx, "x2", xDtype, FORMAT_ND, xShape);
        if (directDataInput != kX1Idx) {
            x1 = es::Cast(x1, xDtype);
            UpdateInputDesc(x1, 0, xDtype, xShape);
            UpdateOutputDesc(x1, 0, xDtype, xShape);
        }
        if (directDataInput != kX2Idx) {
            x2 = es::Cast(x2, xDtype);
            UpdateInputDesc(x2, 0, xDtype, xShape);
            UpdateOutputDesc(x2, 0, xDtype, xShape);
        }
        auto gamma = builder.CreateInput(kGammaIdx, "gamma", gammaDtype, FORMAT_ND, pShape);
        auto beta = builder.CreateInput(kBetaIdx, "beta", gammaDtype, FORMAT_ND, pShape);

        es::AddLayerNormOutput out;
        if (withBias) {
            auto bias = builder.CreateInput(kBiasIdx, "bias", gammaDtype, FORMAT_ND, pShape);
            out = es::AddLayerNorm(x1, x2, gamma, beta, bias, kEpsilon, false);
        } else {
            out = es::AddLayerNorm(x1, x2, gamma, beta, nullptr, kEpsilon, false);
        }

        UpdateInputDesc(out.y, 0, xDtype, xShape);
        UpdateInputDesc(out.y, 1, xDtype, xShape);
        UpdateInputDesc(out.y, 2, gammaDtype, pShape);
        UpdateInputDesc(out.y, 3, gammaDtype, pShape);
        if (withBias) {
            UpdateInputDesc(out.y, 4, gammaDtype, pShape);
        }
        UpdateOutputDesc(out.y, 0, xDtype, xShape);
        UpdateOutputDesc(out.mean, 1, DT_FLOAT, {rows, 1});
        UpdateOutputDesc(out.rstd, 2, DT_FLOAT, {rows, 1});
        UpdateOutputDesc(out.x, 3, xDtype, xShape);

        std::vector<es::EsTensorHolder> outputs = {out.y, out.mean, out.rstd, out.x};
        es::EsTensorHolder extraConsumer;
        if (extraConsumerOnX1) {
            extraConsumer = es::Cast(x1, DT_FLOAT);
            outputs.emplace_back(extraConsumer);
        }
        if (extraConsumerOnY) {
            extraConsumer = es::Cast(out.y, DT_FLOAT);
            outputs.emplace_back(extraConsumer);
        }
        std::shared_ptr<Graph> graph = builder.BuildAndReset(outputs);
        if (extraConsumer.GetProducer() != nullptr) {
            extraConsumer_ = *extraConsumer.GetProducer();
        }
        return graph;
    }

    static void UpdateInputDesc(const es::EsTensorHolder& tensor, int32_t index, DataType dtype,
                                const std::vector<int64_t>& shape)
    {
        TensorDesc desc;
        tensor.GetProducer()->GetInputDesc(index, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(FORMAT_ND);
        desc.SetShape(Shape(shape));
        tensor.GetProducer()->UpdateInputDesc(index, desc);
    }

    static void UpdateOutputDesc(const es::EsTensorHolder& tensor, int32_t index, DataType dtype,
                                 const std::vector<int64_t>& shape)
    {
        TensorDesc desc;
        tensor.GetProducer()->GetOutputDesc(index, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(FORMAT_ND);
        desc.SetShape(Shape(shape));
        tensor.GetProducer()->UpdateOutputDesc(index, desc);
    }

    static Status RunPass(std::shared_ptr<Graph>& graph)
    {
        CustomPassContext passContext;
        ZInplaceAddLayerNormFusionPass pass;
        return pass.Run(graph, passContext);
    }

    static int CountOpType(const std::shared_ptr<Graph>& graph, const char* opType)
    {
        int count = 0;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == AscendString(opType)) {
                ++count;
            }
        }
        return count;
    }

    static bool FindNodeByType(const std::shared_ptr<Graph>& graph, const char* opType, GNode& found)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == AscendString(opType)) {
                found = node;
                return true;
            }
        }
        return false;
    }
};

TEST_F(ZInplaceAddLayerNormFusionPassTest, control_edges_transferred_to_new_node)
{
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_FLOAT16, DT_FLOAT16, false, true);

    GNode addLn;
    ASSERT_TRUE(FindNodeByType(graph, kAddLayerNorm, addLn));
    ASSERT_NE(extraConsumer_.GetInDataNodesAndPortIndexs(0).first, nullptr)
        << "extra consumer node was not built or not wired";

    // 上游输入节点 --ctrl--> AddLayerNorm --ctrl--> 下游消费者。
    auto inputProducer = addLn.GetInDataNodesAndPortIndexs(kX1Idx).first;
    ASSERT_NE(inputProducer, nullptr);
    ASSERT_EQ(graph->AddControlEdge(*inputProducer, addLn), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddControlEdge(addLn, extraConsumer_), GRAPH_SUCCESS);
    ASSERT_EQ(addLn.GetInControlNodes().size(), 1U);
    ASSERT_EQ(addLn.GetOutControlNodes().size(), 1U);

    ASSERT_EQ(RunPass(graph), SUCCESS);

    GNode inplaceNode;
    ASSERT_TRUE(FindNodeByType(graph, kInplaceAddLayerNorm, inplaceNode));
    EXPECT_EQ(inplaceNode.GetInControlNodes().size(), 1U);
    EXPECT_EQ(inplaceNode.GetOutControlNodes().size(), 1U);
}

// ---------- 正向：两种 bias 形态 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, fusion_success_without_bias)
{
    auto graph = BuildGraph(false);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 0);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, fusion_success_with_bias)
{
    auto graph = BuildGraph(true);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 0);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, fusion_success_bf16)
{
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_BF16, DT_BF16);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 1);
}

// ---------- graph 路线特有：接线与 attr 必须逐位保住 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, inputs_and_outputs_rewired)
{
    auto graph = BuildGraph(true);
    ASSERT_EQ(RunPass(graph), SUCCESS);

    GNode newNode;
    ASSERT_TRUE(FindNodeByType(graph, kInplaceAddLayerNorm, newNode));
    EXPECT_EQ(newNode.GetOutputsSize(), 4U);
    for (int32_t i = 0; i < 5; ++i) {
        EXPECT_NE(newNode.GetInDataNodesAndPortIndexs(i).first, nullptr) << "input " << i << " 未接线";
    }
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, downstream_consumer_rewired_to_new_node)
{
    // y 除了作为图输出，还被一个 Cast 消费；替换后该 Cast 的生产者必须是新节点
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_FLOAT16, DT_FLOAT16, false, true);
    ASSERT_EQ(RunPass(graph), SUCCESS);

    auto producer = extraConsumer_.GetInDataNodesAndPortIndexs(0).first;
    ASSERT_NE(producer, nullptr);
    AscendString producerType;
    producer->GetType(producerType);
    EXPECT_EQ(producerType, AscendString(kInplaceAddLayerNorm));
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, attrs_transferred_to_new_node)
{
    auto graph = BuildGraph(false);
    ASSERT_EQ(RunPass(graph), SUCCESS);

    GNode newNode;
    ASSERT_TRUE(FindNodeByType(graph, kInplaceAddLayerNorm, newNode));
    float32_t epsilon = 0.0F;
    EXPECT_EQ(newNode.GetAttr(AscendString("epsilon"), epsilon), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(epsilon, kEpsilon);
    bool additionalOutput = true;
    EXPECT_EQ(newNode.GetAttr(AscendString("additional_output"), additionalOutput), GRAPH_SUCCESS);
    EXPECT_FALSE(additionalOutput);
}

// ---------- 守卫 G2：平台校验 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_unsupported_platform)
{
    SetPlatform("Ascend310P");
    auto graph = BuildGraph(false);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 0);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, platform_ascend910b_rejected)
{
    SetPlatform("Ascend910B");
    auto graph = BuildGraph(false);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 0);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, platform_ascend910_93_rejected)
{
    SetPlatform("Ascend910_93");
    auto graph = BuildGraph(false);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 0);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_direct_data_input)
{
    for (int32_t inputIdx = 0; inputIdx < 2; ++inputIdx) {
        SCOPED_TRACE(inputIdx);
        auto graph = BuildGraph(false, kRowsInBand, kCols, DT_FLOAT16, DT_FLOAT16, false, false, inputIdx);
        GNode addLn;
        ASSERT_TRUE(FindNodeByType(graph, kAddLayerNorm, addLn));
        auto producer = addLn.GetInDataNodesAndPortIndexs(inputIdx).first;
        ASSERT_NE(producer, nullptr);
        AscendString type;
        ASSERT_EQ(producer->GetType(type), GRAPH_SUCCESS);
        ASSERT_EQ(type, AscendString("Data"));
        EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
        EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
        EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 0);
    }
}

// ---------- 守卫 G4：x1 被其它节点消费时不可原地 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_input_shared_by_other_consumers)
{
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_FLOAT16, DT_FLOAT16, true);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 0);
}

// ---------- 守卫 G6：shape 必须落在 L2 区间 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_shape_below_l2_band)
{
    auto graph = BuildGraph(false, kRowsTooSmall);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_shape_above_l2_band)
{
    auto graph = BuildGraph(false, kRowsTooLarge);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
}

// ---------- 守卫 G6：动态 shape 走末维判定 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, fusion_success_dynamic_shape_on_reduce_axis)
{
    auto graph = BuildGraph(false, -1L, kDynamicReduceAxis);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 1);
}

TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_dynamic_shape_other_axis)
{
    auto graph = BuildGraph(false, -1L, kCols);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
}

// ---------- 守卫 G7：x1/x2/gamma dtype 必须一致 ----------
TEST_F(ZInplaceAddLayerNormFusionPassTest, guard_reject_mixed_input_dtype)
{
    auto graph = BuildGraph(false, kRowsInBand, kCols, DT_FLOAT16, DT_FLOAT);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountOpType(graph, kAddLayerNorm), 1);
    EXPECT_EQ(CountOpType(graph, kInplaceAddLayerNorm), 0);
}
} // namespace
