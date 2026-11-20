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
 * \file test_apply_adagrad_to_inplace_fusion_pass.cpp
 * \brief ApplyAdagradToInplaceFusionPass UT: GEIR 图中 ApplyAdagrad 节点改写为 InplaceApplyAdagrad。
 */

#include <gtest/gtest.h>
#include <vector>

#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "register/register_custom_pass.h"
#include "ut_op_util.h"

#include "../../../op_graph/fusion_pass/apply_adagrad_to_inplace_fusion_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace ge::es;
using namespace ge::fusion;
using namespace fe;

namespace {
constexpr int64_t kInputNum = 4;

void SetPlatformInfo950()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    optionalInfo.soc_version = "Ascend950";
    platformInfo.str_info.short_soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

void SetPlatformInfo910B()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 24;
    optionalInfo.soc_version = "Ascend910B1";
    platformInfo.str_info.short_soc_version = "Ascend910B";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

GNode CreateApplyAdagradNode(es::EsGraphBuilder& graphBuilder, const std::string& nodeName)
{
    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    auto applyAdagrad = es::CompliantNodeBuilder(graph)
                            .OpType("ApplyAdagrad")
                            .Name(nodeName.c_str())
                            .IrDefInputs({{"var", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                          {"accum", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                          {"lr", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                          {"grad", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                            .IrDefOutputs({{"var", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .Build();
    return applyAdagrad;
}

std::shared_ptr<Graph> BuildApplyAdagradGraph(es::EsGraphBuilder& graphBuilder, const std::vector<int64_t>& dims,
                                              DataType dtype, bool updateSlots)
{
    const std::vector<int64_t> scalarDims = {};
    auto var = graphBuilder.CreateInput(0, "var", dtype, FORMAT_ND, dims);
    auto accum = graphBuilder.CreateInput(1, "accum", dtype, FORMAT_ND, dims);
    auto lr = graphBuilder.CreateInput(2, "lr", dtype, FORMAT_ND, scalarDims);
    auto grad = graphBuilder.CreateInput(3, "grad", dtype, FORMAT_ND, dims);
    auto applyAdagrad = CreateApplyAdagradNode(graphBuilder, "ApplyAdagradNode");

    bool useLocking = false;
    applyAdagrad.SetAttr("update_slots", updateSlots);
    applyAdagrad.SetAttr("use_locking", useLocking);

    auto* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    const std::vector<std::pair<es::EsTensorHolder, int32_t>> inputs = {{var, 0}, {accum, 1}, {lr, 2}, {grad, 3}};
    for (const auto& input : inputs) {
        auto status = es::AddEdgeAndUpdatePeerDesc(*graph, *input.first.GetProducer(),
                                                   input.first.GetProducerOutIndex(), applyAdagrad, input.second);
        EXPECT_EQ(status, GRAPH_SUCCESS);
    }

    // 显式回填各 Data 输出与 ApplyAdagrad 输入/输出的 TensorDesc，
    // 保证 MeetRequirements 读到真实 dtype/shape（与 bucketize UT 的做法一致）。
    const std::vector<std::vector<int64_t>> inputShapes = {dims, dims, {}, dims};
    for (size_t i = 0; i < inputs.size(); ++i) {
        TensorDesc inputDesc(Shape(inputShapes[i]), FORMAT_ND, dtype);
        inputDesc.SetOriginShape(Shape(inputShapes[i]));
        inputDesc.SetOriginFormat(FORMAT_ND);
        EXPECT_EQ(inputs[i].first.GetProducer()->UpdateOutputDesc(0, inputDesc), GRAPH_SUCCESS);
        EXPECT_EQ(applyAdagrad.UpdateInputDesc(static_cast<int32_t>(i), inputDesc), GRAPH_SUCCESS);
    }

    TensorDesc varOutDesc;
    applyAdagrad.GetOutputDesc(0, varOutDesc);
    varOutDesc.SetDataType(dtype);
    varOutDesc.SetShape(Shape(dims));
    applyAdagrad.UpdateOutputDesc(0, varOutDesc);

    auto* varHolder = graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(applyAdagrad, 0);
    return graphBuilder.BuildAndReset({es::EsTensorHolder(varHolder)});
}

bool HasNodeType(const std::shared_ptr<Graph>& graph, const char* opType)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == opType) {
            return true;
        }
    }
    return false;
}

bool FindNodeByType(const std::shared_ptr<Graph>& graph, const char* opType, GNode& foundNode)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == opType) {
            foundNode = node;
            return true;
        }
    }
    return false;
}
} // namespace

class ApplyAdagradToInplaceFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatformInfo950(); }

    static void TearDownTestCase() {}
};

TEST_F(ApplyAdagradToInplaceFusionPassTest, fusionSuccess950Float)
{
    SetPlatformInfo950();
    std::vector<int64_t> dims{2, 3};
    auto graphBuilder = es::EsGraphBuilder("apply_adagrad_fusion_float_test");
    auto graph = BuildApplyAdagradGraph(graphBuilder, dims, DT_FLOAT, true);
    ASSERT_NE(graph, nullptr);

    ops::ApplyAdagradToInplaceFusionPass pass({AscendString("ApplyAdagrad")});
    CustomPassContext passContext;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, SUCCESS);
    EXPECT_TRUE(HasNodeType(graph, "InplaceApplyAdagrad"));
    EXPECT_FALSE(HasNodeType(graph, "ApplyAdagrad"));
}

TEST_F(ApplyAdagradToInplaceFusionPassTest, fusionSuccess950Bfloat16AttrKept)
{
    SetPlatformInfo950();
    std::vector<int64_t> dims{8, 16};
    auto graphBuilder = es::EsGraphBuilder("apply_adagrad_fusion_bf16_test");
    auto graph = BuildApplyAdagradGraph(graphBuilder, dims, DT_BF16, false);
    ASSERT_NE(graph, nullptr);

    ops::ApplyAdagradToInplaceFusionPass pass({AscendString("ApplyAdagrad")});
    CustomPassContext passContext;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, SUCCESS);
    EXPECT_TRUE(HasNodeType(graph, "InplaceApplyAdagrad"));
    EXPECT_FALSE(HasNodeType(graph, "ApplyAdagrad"));

    GNode inplaceNode;
    ASSERT_TRUE(FindNodeByType(graph, "InplaceApplyAdagrad", inplaceNode));
    bool updateSlots = true;
    EXPECT_EQ(inplaceNode.GetAttr("update_slots", updateSlots), GRAPH_SUCCESS);
    EXPECT_FALSE(updateSlots);

    TensorDesc varOutDesc;
    EXPECT_EQ(inplaceNode.GetOutputDesc(0, varOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(varOutDesc.GetDataType(), DT_BF16);
    EXPECT_EQ(varOutDesc.GetShape().GetDims(), dims);
}

TEST_F(ApplyAdagradToInplaceFusionPassTest, noRewriteOnUnsupportedPlatform)
{
    SetPlatformInfo910B();
    std::vector<int64_t> dims{2, 3};
    auto graphBuilder = es::EsGraphBuilder("apply_adagrad_fusion_910b_test");
    auto graph = BuildApplyAdagradGraph(graphBuilder, dims, DT_FLOAT, true);
    ASSERT_NE(graph, nullptr);

    ops::ApplyAdagradToInplaceFusionPass pass({AscendString("ApplyAdagrad")});
    CustomPassContext passContext;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNodeType(graph, "InplaceApplyAdagrad"));
    EXPECT_TRUE(HasNodeType(graph, "ApplyAdagrad"));
}

TEST_F(ApplyAdagradToInplaceFusionPassTest, noRewriteOnUnsupportedDtype)
{
    SetPlatformInfo950();
    std::vector<int64_t> dims{2, 3};
    auto graphBuilder = es::EsGraphBuilder("apply_adagrad_fusion_int32_test");
    auto graph = BuildApplyAdagradGraph(graphBuilder, dims, DT_INT32, true);
    ASSERT_NE(graph, nullptr);

    ops::ApplyAdagradToInplaceFusionPass pass({AscendString("ApplyAdagrad")});
    CustomPassContext passContext;
    Status runStatus = pass.Run(graph, passContext);
    EXPECT_EQ(runStatus, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasNodeType(graph, "InplaceApplyAdagrad"));
    EXPECT_TRUE(HasNodeType(graph, "ApplyAdagrad"));
}
