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

#include <memory>
#include <map>
#include <string>
#include <vector>

#include "../../../op_graph/fusion_pass/conv2dt_quant_process_fusion_pass.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "version/ge-compiler_version.h"

#if GE_COMPILER_VERSION_NUM >= 90300000U
namespace {
using namespace ge;
using namespace ge::es;

void SetPlatform(const std::string& soc, bool fuse, bool hasShortSocInfo = true)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    if (hasShortSocInfo) {
        fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    } else {
        fe::PlatformInfoManager::Instance().platform_info_map_.erase(soc);
    }
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);

    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optionalInfos;
    platformInfos.Init();
    optionalInfos.Init();
    std::map<std::string, std::string> socInfos = {{"cube_vector_combine", fuse ? "fuse" : "split"}};
    platformInfos.SetPlatformRes("SoCInfo", socInfos);
    optionalInfos.SetSocVersion(soc);
    fe::PlatformInfoManager::Instance().platform_infos_map_[soc] = platformInfos;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfos);
}

GNode AddNode(Graph* graph, const char* name, const char* type,
              const std::vector<CompliantNodeBuilder::IrInputDef>& inputs)
{
    return CompliantNodeBuilder(graph)
        .OpType(type)
        .Name(name)
        .IrDefInputs(inputs)
        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
}

GraphPtr MakeGraph(const char* cubeType, GNodePtr& cubeNode)
{
    EsGraphBuilder builder("conv2dt_quant_process_test");
    auto* rawGraph = builder.GetCGraphBuilder()->GetGraph();
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {1, 16, 4, 4});
    auto filter = builder.CreateInput(1, "filter", DT_INT8, FORMAT_NCHW, {16, 16, 1, 1});
    auto scale = builder.CreateInput(2, "scale", DT_UINT64, FORMAT_ND, {1});
    auto inputSize = builder.CreateInput(3, "input_size", DT_INT64, FORMAT_ND, {4});
    auto quant = AddNode(rawGraph, "quant", "AscendQuant", {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    auto cube = AddNode(rawGraph, "cube", cubeType,
                        {{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                         {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                         {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                         {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""}});
    auto dequant = AddNode(rawGraph, "dequant", "AscendDequant",
                           {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                            {"deq_scale", CompliantNodeBuilder::kEsIrInputRequired, ""}});
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*rawGraph, *x.GetProducer(), 0, quant, 0), GRAPH_SUCCESS);
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*rawGraph, *inputSize.GetProducer(), 0, cube, 0), GRAPH_SUCCESS);
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*rawGraph, quant, 0, cube, 1), GRAPH_SUCCESS);
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*rawGraph, *filter.GetProducer(), 0, cube, 2), GRAPH_SUCCESS);
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*rawGraph, *scale.GetProducer(), 0, dequant, 1), GRAPH_SUCCESS);
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*rawGraph, cube, 0, dequant, 0), GRAPH_SUCCESS);
    cubeNode = std::make_shared<GNode>(cube);
    auto output = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(dequant, 0));
    return builder.BuildAndReset({output});
}

class ProbeConv2DTQuantProcessFusionPass : public ops::Conv2DTQuantProcessFusionPass {
public:
    bool visited = false;
    ops::QuantProcessMode reportedMode = ops::QuantProcessMode::QUANT_UNDEFINED;
    void SetReverse(const GNodePtr& node) { SetCinCoutReverse(node); }
    uint32_t BiasIndex() const { return GetBiasInputIndex(); }
    uint32_t WeightIndex() const { return GetWeightInputIndex(); }
    uint32_t BiasMode() const { return GetBiasOptimizationModeIndex(); }

protected:
    Status GetQuantProcessMode(const GNodePtr& quant, const GNodePtr& cube, ops::QuantProcessMode& mode) override
    {
        visited = true;
        const auto status = ops::Conv2DTQuantProcessFusionPass::GetQuantProcessMode(quant, cube, reportedMode);
        mode = ops::QuantProcessMode::QUANT_UNDEFINED;
        return status;
    }
};
} // namespace

TEST(Conv2DTQuantProcessFusionPassTest, MatchesQuantCubeDequantChain)
{
    SetPlatform("Ascend950", false);
    GNodePtr cube;
    auto graph = MakeGraph("Conv2DTranspose", cube);
    ASSERT_NE(graph, nullptr);
    ProbeConv2DTQuantProcessFusionPass pass;
    CustomPassContext context;
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
    EXPECT_TRUE(pass.visited);
    EXPECT_EQ(pass.reportedMode, ops::QuantProcessMode::BIAS_OPTIMIZE);
    EXPECT_EQ(pass.BiasIndex(), 3U);
    EXPECT_EQ(pass.WeightIndex(), 2U);
    EXPECT_EQ(pass.BiasMode(), static_cast<uint32_t>(ops::HostOpMode::BIAS_OPTIMIZATION_MODE_CONVDT));
}

TEST(Conv2DTQuantProcessFusionPassTest, SkipsOtherCubeType)
{
    SetPlatform("Ascend950", false);
    GNodePtr cube;
    auto graph = MakeGraph("Conv2DTransposeD", cube);
    ASSERT_NE(graph, nullptr);
    ProbeConv2DTQuantProcessFusionPass pass;
    CustomPassContext context;
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(pass.visited);
}

TEST(Conv2DTQuantProcessFusionPassTest, SkipsUnsupportedSoc)
{
    SetPlatform("MC62", false);
    GNodePtr cube;
    auto graph = MakeGraph("Conv2DTranspose", cube);
    ASSERT_NE(graph, nullptr);
    ProbeConv2DTQuantProcessFusionPass pass;
    CustomPassContext context;
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(pass.visited);
}

TEST(Conv2DTQuantProcessFusionPassTest, SupportsFuseSocVersion)
{
    SetPlatform("MC62", true, false);
    GNodePtr cube;
    auto graph = MakeGraph("Conv2DTranspose", cube);
    ASSERT_NE(graph, nullptr);
    ProbeConv2DTQuantProcessFusionPass pass;
    CustomPassContext context;
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
    EXPECT_TRUE(pass.visited);
}

TEST(Conv2DTQuantProcessFusionPassTest, SetsCinCoutReverse)
{
    GNodePtr cube;
    auto graph = MakeGraph("Conv2DTranspose", cube);
    ASSERT_NE(graph, nullptr);
    ProbeConv2DTQuantProcessFusionPass pass;
    pass.SetReverse(cube);
    bool reverse = false;
    EXPECT_EQ(cube->GetAttr("quant_cin_cout_reverse", reverse), GRAPH_SUCCESS);
    EXPECT_TRUE(reverse);
}
#endif
