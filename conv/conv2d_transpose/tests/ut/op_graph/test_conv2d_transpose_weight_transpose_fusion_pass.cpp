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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/conv2d_transpose_weight_transpose_fusion_pass.h"

using namespace ge;
using namespace ge::es;

namespace {
void SetPlatform(const std::string& cubeVectorCombine)
{
    fe::PlatFormInfos platformInfo;
    fe::OptionalInfos optionalInfo;
    platformInfo.Init();
    optionalInfo.Init();
    std::map<std::string, std::string> socInfo{{"cube_vector_combine", cubeVectorCombine}};
    platformInfo.SetPlatformRes("SoCInfo", socInfo);
    optionalInfo.SetSocVersion("weight_transpose_ut");
    fe::PlatformInfoManager::Instance().platform_infos_map_["weight_transpose_ut"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

GNodePtr FindNode(const GraphPtr& graph, const char* type)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        if (node.GetType(nodeType) == GRAPH_SUCCESS && nodeType == type) {
            return std::make_shared<GNode>(node);
        }
    }
    return nullptr;
}

GraphPtr MakeGraph(int64_t groups, Format weightFormat, DataType xType, DataType weightType)
{
    EsGraphBuilder builder("weight_transpose_test");
    auto size = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {4});
    auto x = builder.CreateInput(1, "x", xType, FORMAT_NCHW, {1, 64, 8, 8});
    const std::vector<int64_t> weightShape = weightFormat == FORMAT_NCHW ? std::vector<int64_t>{64, 32, 3, 3} :
                                                                           std::vector<int64_t>{64, 3, 3, 32};
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto weight = CompliantNodeBuilder(graph)
                      .OpType("Const")
                      .Name("weight")
                      .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                      .InstanceOutputDataType("y", weightType)
                      .InstanceOutputShape("y", weightShape)
                      .InstanceOutputFormat("y", weightFormat)
                      .Build();
    auto conv = CompliantNodeBuilder(graph)
                    .OpType("Conv2DTranspose")
                    .Name("conv")
                    .IrDefInputs({{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", DT_FLOAT16)
                    .InstanceOutputShape("y", {1, 64, 8, 8})
                    .InstanceOutputFormat("y", FORMAT_NCHW)
                    .Build();
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*graph, *size.GetProducer(), size.GetProducerOutIndex(), conv, 0),
              GRAPH_SUCCESS);
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), conv, 1), GRAPH_SUCCESS);
    EXPECT_EQ(AddEdgeAndUpdatePeerDesc(*graph, weight, 0, conv, 2), GRAPH_SUCCESS);
    EXPECT_EQ(conv.SetAttr("groups", groups), GRAPH_SUCCESS);
    auto y = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(conv, 0));
    return builder.BuildAndReset({y});
}
} // namespace

TEST(Conv2DTransposeWeightTransFusionPassTest, A16W8NonGrouped)
{
    SetPlatform("fuse");
    auto graph = MakeGraph(1, FORMAT_NCHW, DT_FLOAT16, DT_INT8);
    CustomPassContext context;
    ops::Conv2DTransposeWeightTransFusionPass pass;
    EXPECT_EQ(pass.Run(graph, context), SUCCESS);
    auto conv = FindNode(graph, "Conv2DTranspose");
    ASSERT_NE(conv, nullptr);
    auto reformat = conv->GetInDataNodesAndPortIndexs(2).first;
    ASSERT_NE(reformat, nullptr);
    TensorDesc desc;
    ASSERT_EQ(reformat->GetOutputDesc(0, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetFormat(), FORMAT_FRACTAL_Z);
    auto transpose = reformat->GetInDataNodesAndPortIndexs(0).first;
    ASSERT_NE(transpose, nullptr);
    std::vector<int64_t> perm;
    EXPECT_EQ(transpose->GetAttr("perm", perm), GRAPH_SUCCESS);
    EXPECT_EQ(perm, (std::vector<int64_t>{1, 0, 2, 3}));
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
}

TEST(Conv2DTransposeWeightTransFusionPassTest, A16W8Grouped)
{
    SetPlatform("fuse");
    auto graph = MakeGraph(2, FORMAT_NHWC, DT_FLOAT16, DT_INT8);
    CustomPassContext context;
    ops::Conv2DTransposeWeightTransFusionPass pass;
    EXPECT_EQ(pass.Run(graph, context), SUCCESS);
    auto conv = FindNode(graph, "Conv2DTranspose");
    ASSERT_NE(conv, nullptr);
    auto reformat = conv->GetInDataNodesAndPortIndexs(2).first;
    ASSERT_NE(reformat, nullptr);
    auto reshapeOut = reformat->GetInDataNodesAndPortIndexs(0).first;
    ASSERT_NE(reshapeOut, nullptr);
    TensorDesc desc;
    ASSERT_EQ(reshapeOut->GetOutputDesc(0, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetShape().GetDims(), (std::vector<int64_t>{64, 32, 3, 3}));
    auto transpose = reshapeOut->GetInDataNodesAndPortIndexs(0).first;
    ASSERT_NE(transpose, nullptr);
    std::vector<int64_t> perm;
    EXPECT_EQ(transpose->GetAttr("perm", perm), GRAPH_SUCCESS);
    EXPECT_EQ(perm, (std::vector<int64_t>{0, 4, 1, 2, 3}));
}

TEST(Conv2DTransposeWeightTransFusionPassTest, OtherDtypeKeepsNhwcConversion)
{
    SetPlatform("fuse");
    auto graph = MakeGraph(1, FORMAT_NCHW, DT_FLOAT16, DT_FLOAT16);
    CustomPassContext context;
    ops::Conv2DTransposeWeightTransFusionPass pass;
    EXPECT_EQ(pass.Run(graph, context), SUCCESS);
    auto conv = FindNode(graph, "Conv2DTranspose");
    ASSERT_NE(conv, nullptr);
    TensorDesc desc;
    auto reformat = conv->GetInDataNodesAndPortIndexs(2).first;
    ASSERT_NE(reformat, nullptr);
    ASSERT_EQ(reformat->GetOutputDesc(0, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetFormat(), FORMAT_NHWC);
}

TEST(Conv2DTransposeWeightTransFusionPassTest, SplitSocDoesNotChangeGraph)
{
    SetPlatform("split");
    auto graph = MakeGraph(1, FORMAT_NCHW, DT_FLOAT16, DT_INT8);
    CustomPassContext context;
    ops::Conv2DTransposeWeightTransFusionPass pass;
    EXPECT_EQ(pass.Run(graph, context), GRAPH_NOT_CHANGED);
    EXPECT_EQ(FindNode(graph, "ReFormat"), nullptr);
}
