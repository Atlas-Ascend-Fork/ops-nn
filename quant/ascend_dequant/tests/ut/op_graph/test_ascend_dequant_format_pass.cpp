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
 * \file test_ascend_dequant_format_pass.cpp
 * \brief AscendDequantFormatPass UT：deq_scale 固定 NCHW、y 跟随 x、3D 量化场景的 format 修补
 */

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "graph/tensor.h"
#include "register/register_custom_pass.h"

namespace ops {
ge::Status AscendDequantFormatPass(ge::GraphPtr& graph, ge::CustomPassContext& context);
} // namespace ops

namespace {

using namespace ge;

constexpr int32_t OUT_PORT = 0;

// 简易图构建器：声明节点/连边/设置描述符，最后 Build 出 ge::GraphPtr。
class TestGraph {
public:
    explicit TestGraph(const std::string& name) : builder_(name.c_str())
    {
        rawGraph_ = builder_.GetCGraphBuilder()->GetGraph();
    }

    GNode& AddNode(const std::string& nodeName, const std::string& opType, int32_t inputNum)
    {
        std::vector<es::CompliantNodeBuilder::IrInputDef> inputDefs;
        for (int32_t i = 0; i < inputNum; ++i) {
            inputDefs.push_back({"in" + std::to_string(i), es::CompliantNodeBuilder::kEsIrInputRequired, ""});
        }
        GNode node = es::CompliantNodeBuilder(rawGraph_)
                         .OpType(opType.c_str())
                         .Name(nodeName.c_str())
                         .IrDefInputs(inputDefs)
                         .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        nodes_[nodeName] = node;
        return nodes_[nodeName];
    }

    void Connect(const std::string& from, int32_t fromPort, const std::string& to, int32_t toPort)
    {
        auto src = nodes_.find(from);
        auto dst = nodes_.find(to);
        if (src == nodes_.end() || dst == nodes_.end()) {
            return;
        }
        (void)es::AddEdgeAndUpdatePeerDesc(*rawGraph_, src->second, fromPort, dst->second, toPort);
    }

    void SetInDesc(const std::string& nodeName, int32_t idx, DataType dtype, Format format,
                   const std::vector<int64_t>& dims)
    {
        TensorDesc desc;
        (void)nodes_[nodeName].GetInputDesc(idx, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(format);
        desc.SetShape(Shape(dims));
        desc.SetOriginFormat(format);
        desc.SetOriginShape(Shape(dims));
        (void)nodes_[nodeName].UpdateInputDesc(idx, desc);
    }

    void SetOutDesc(const std::string& nodeName, int32_t idx, DataType dtype, Format format,
                    const std::vector<int64_t>& dims)
    {
        TensorDesc desc;
        (void)nodes_[nodeName].GetOutputDesc(idx, desc);
        desc.SetDataType(dtype);
        desc.SetFormat(format);
        desc.SetShape(Shape(dims));
        desc.SetOriginFormat(format);
        desc.SetOriginShape(Shape(dims));
        (void)nodes_[nodeName].UpdateOutputDesc(idx, desc);
    }

    void SetTfTag(const std::string& nodeName, const std::string& tag)
    {
        AscendString value(tag.c_str());
        (void)nodes_[nodeName].SetAttr(AscendString("tf_tag"), value);
    }

    GraphPtr Build(const std::string& outputNode)
    {
        auto holder = builder_.GetCGraphBuilder()->GetTensorHolderFromNode(nodes_[outputNode], OUT_PORT);
        return builder_.BuildAndReset({holder});
    }

private:
    es::EsGraphBuilder builder_;
    Graph* rawGraph_ = nullptr;
    std::map<std::string, GNode> nodes_;
};

// 从整图里按名字取节点（pass 通过 graph->GetAllNodes() 修改，重新读一遍最稳妥）
GNode FindNode(const GraphPtr& graph, const std::string& name)
{
    for (auto& node : graph->GetAllNodes()) {
        AscendString type;
        AscendString nodeName;
        (void)node.GetType(type);
        (void)node.GetName(nodeName);
        if (std::string(nodeName.GetString()) == name) {
            return node;
        }
    }
    return GNode();
}

Format NodeInFormat(const GraphPtr& graph, const std::string& name, int32_t idx)
{
    TensorDesc desc;
    (void)FindNode(graph, name).GetInputDesc(idx, desc);
    return desc.GetOriginFormat();
}

Format NodeOutFormat(const GraphPtr& graph, const std::string& name, int32_t idx)
{
    TensorDesc desc;
    (void)FindNode(graph, name).GetOutputDesc(idx, desc);
    return desc.GetOriginFormat();
}

void RunPass(GraphPtr& graph)
{
    CustomPassContext context;
    ASSERT_EQ(ops::AscendDequantFormatPass(graph, context), ge::GRAPH_SUCCESS);
}

// 非 3D：deq_scale 固定 NCHW，y 跟随 x，x 保持原格式
TEST(AscendDequantFormatPassTest, Non3DNodeFixesScaleFormatAndCopiesXToY)
{
    TestGraph g("non3d");
    g.AddNode("relu", "Relu", 1);
    g.AddNode("dequant", "AscendDequant", 2);
    g.SetOutDesc("relu", OUT_PORT, ge::DT_INT32, ge::FORMAT_ND, {2, 3});
    g.SetInDesc("dequant", 0, ge::DT_INT32, ge::FORMAT_ND, {2, 3});
    g.SetInDesc("dequant", 1, ge::DT_UINT64, ge::FORMAT_ND, {3});
    g.SetOutDesc("dequant", OUT_PORT, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3});
    g.Connect("relu", OUT_PORT, "dequant", 0);
    GraphPtr graph = g.Build("dequant");
    ASSERT_NE(graph, nullptr);

    RunPass(graph);

    EXPECT_EQ(NodeInFormat(graph, "dequant", 1), ge::FORMAT_NCHW);
    EXPECT_EQ(NodeInFormat(graph, "dequant", 0), ge::FORMAT_ND);
    EXPECT_EQ(NodeOutFormat(graph, "dequant", OUT_PORT), ge::FORMAT_ND);
}

// 5D + Conv3D(NCDHW)：x / y / deq_scale 统一为 NCDHW
TEST(AscendDequantFormatPassTest, Rank5WithConv3DNcdhwAlignsAllFormats)
{
    TestGraph g("conv3d_ncdhw");
    g.AddNode("conv3d", "Conv3D", 1);
    g.AddNode("dequant", "AscendDequant", 2);
    g.SetOutDesc("conv3d", OUT_PORT, ge::DT_INT32, ge::FORMAT_NCDHW, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 0, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 1, ge::DT_UINT64, ge::FORMAT_ND, {1});
    g.SetOutDesc("dequant", OUT_PORT, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.Connect("conv3d", OUT_PORT, "dequant", 0);
    GraphPtr graph = g.Build("dequant");
    ASSERT_NE(graph, nullptr);

    RunPass(graph);

    EXPECT_EQ(NodeInFormat(graph, "dequant", 0), ge::FORMAT_NCDHW);
    EXPECT_EQ(NodeInFormat(graph, "dequant", 1), ge::FORMAT_NCDHW);
    EXPECT_EQ(NodeOutFormat(graph, "dequant", OUT_PORT), ge::FORMAT_NCDHW);
}

// 5D + Conv3D(NDHWC)：沿用 Conv3D 的 NDHWC
TEST(AscendDequantFormatPassTest, Rank5WithConv3DNdhwcKeepsConvFormat)
{
    TestGraph g("conv3d_ndhwc");
    g.AddNode("conv3d", "Conv3D", 1);
    g.AddNode("dequant", "AscendDequant", 2);
    g.SetOutDesc("conv3d", OUT_PORT, ge::DT_INT32, ge::FORMAT_NDHWC, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 0, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 1, ge::DT_UINT64, ge::FORMAT_ND, {1});
    g.SetOutDesc("dequant", OUT_PORT, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.Connect("conv3d", OUT_PORT, "dequant", 0);
    GraphPtr graph = g.Build("dequant");
    ASSERT_NE(graph, nullptr);

    RunPass(graph);

    EXPECT_EQ(NodeInFormat(graph, "dequant", 0), ge::FORMAT_NDHWC);
    EXPECT_EQ(NodeOutFormat(graph, "dequant", OUT_PORT), ge::FORMAT_NDHWC);
}

// 5D 但输入侧不是 Conv3D 且无 tf_tag：不按 3D 处理，仅 deq_scale 改 NCHW 且 y 跟随 x
TEST(AscendDequantFormatPassTest, Rank5WithoutConv3DOnlyFixesScaleFormat)
{
    TestGraph g("rank5_no_conv3d");
    g.AddNode("relu", "Relu", 1);
    g.AddNode("dequant", "AscendDequant", 2);
    g.SetOutDesc("relu", OUT_PORT, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 0, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 1, ge::DT_UINT64, ge::FORMAT_ND, {1});
    g.SetOutDesc("dequant", OUT_PORT, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.Connect("relu", OUT_PORT, "dequant", 0);
    GraphPtr graph = g.Build("dequant");
    ASSERT_NE(graph, nullptr);

    RunPass(graph);

    EXPECT_EQ(NodeInFormat(graph, "dequant", 0), ge::FORMAT_ND);
    EXPECT_EQ(NodeInFormat(graph, "dequant", 1), ge::FORMAT_NCHW);
    EXPECT_EQ(NodeOutFormat(graph, "dequant", OUT_PORT), ge::FORMAT_ND);
}

// 5D + Conv3D 但 Conv3D 输出不是 3D 格式：无 tf_tag 时退化为 NCDHW
TEST(AscendDequantFormatPassTest, Rank5WithConv3DNon3DFormatFallsBackToNcdhw)
{
    TestGraph g("conv3d_nd");
    g.AddNode("conv3d", "Conv3D", 1);
    g.AddNode("dequant", "AscendDequant", 2);
    g.SetOutDesc("conv3d", OUT_PORT, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 0, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 1, ge::DT_UINT64, ge::FORMAT_ND, {1});
    g.SetOutDesc("dequant", OUT_PORT, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.Connect("conv3d", OUT_PORT, "dequant", 0);
    GraphPtr graph = g.Build("dequant");
    ASSERT_NE(graph, nullptr);

    RunPass(graph);

    EXPECT_EQ(NodeInFormat(graph, "dequant", 0), ge::FORMAT_NCDHW);
    EXPECT_EQ(NodeOutFormat(graph, "dequant", OUT_PORT), ge::FORMAT_NCDHW);
}

// 5D + tf_tag：隔一层（BiasAdd）找到 Conv3D，按 TF 语义回退 NDHWC
TEST(AscendDequantFormatPassTest, Rank5WithTfTagFindsConv3DOneLevelUp)
{
    TestGraph g("conv3d_tf_tag");
    g.AddNode("conv3d", "Conv3D", 1);
    g.AddNode("biasadd", "BiasAdd", 1);
    g.AddNode("dequant", "AscendDequant", 2);
    g.SetOutDesc("conv3d", OUT_PORT, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.SetOutDesc("biasadd", OUT_PORT, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 0, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.SetInDesc("dequant", 1, ge::DT_UINT64, ge::FORMAT_ND, {1});
    g.SetOutDesc("dequant", OUT_PORT, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2, 3, 4, 5});
    g.Connect("conv3d", OUT_PORT, "biasadd", 0);
    g.Connect("biasadd", OUT_PORT, "dequant", 0);
    g.SetTfTag("dequant", "tf");
    GraphPtr graph = g.Build("dequant");
    ASSERT_NE(graph, nullptr);

    RunPass(graph);

    EXPECT_EQ(NodeInFormat(graph, "dequant", 0), ge::FORMAT_NDHWC);
    EXPECT_EQ(NodeOutFormat(graph, "dequant", OUT_PORT), ge::FORMAT_NDHWC);
}

// 图中没有 AscendDequant 节点：pass 不改动任何格式
TEST(AscendDequantFormatPassTest, GraphWithoutAscendDequantIsUntouched)
{
    TestGraph g("no_dequant");
    g.AddNode("relu", "Relu", 1);
    g.SetInDesc("relu", 0, ge::DT_FLOAT16, ge::FORMAT_NHWC, {2, 3, 4, 5});
    g.SetOutDesc("relu", OUT_PORT, ge::DT_FLOAT16, ge::FORMAT_NHWC, {2, 3, 4, 5});
    GraphPtr graph = g.Build("relu");
    ASSERT_NE(graph, nullptr);

    RunPass(graph);

    EXPECT_EQ(NodeOutFormat(graph, "relu", OUT_PORT), ge::FORMAT_NHWC);
}

} // namespace
