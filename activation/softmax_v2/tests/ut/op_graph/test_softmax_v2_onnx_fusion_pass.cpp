/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "ge/compliant_node_builder.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/softmax_v2_onnx_fusion_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace es;
using namespace ops;

#if defined(CANN_MAJOR) && defined(CANN_MINOR)
#define NN_UT_HAS_V2_IR_API ((CANN_MAJOR > 9) || (CANN_MAJOR == 9 && CANN_MINOR >= 1))
#else
#define NN_UT_HAS_V2_IR_API 0
#endif

class ASoftmaxFusionPassTest : public testing::Test {
protected:
    void SetUp() override
    {
        PlatFormInfos platform_infos;
        (void)platform_infos.Init();
        std::map<std::string, std::string> version_res{{"NpuArch", "3510"}, {"Short_SoC_version", "Ascend950"}};
        platform_infos.SetPlatformRes("version", version_res);
        OptionalInfos optional_infos;
        (void)optional_infos.Init();
        optional_infos.SetSocVersion("Ascend950");
        PlatformInfoManager::Instance().platform_infos_map_["Ascend950"] = platform_infos;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional_infos);
    }

    std::shared_ptr<Graph> BuildSoftmaxV2Graph(DataType dtype, const vector<int64_t>& dims, int64_t axis,
                                               int32_t need_fusion, bool has_need_fusion = true)
    {
        EsGraphBuilder graph_builder("asoftmax_fusion_test");
        auto x = graph_builder.CreateInput(0, "x", dtype, FORMAT_ND, dims);
        auto* graph = graph_builder.GetCGraphBuilder()->GetGraph();

        TensorDesc x_out_desc;
        x.GetProducer()->GetOutputDesc(0, x_out_desc);
        x_out_desc.SetDataType(dtype);
        x_out_desc.SetShape(Shape(dims));
        x.GetProducer()->UpdateOutputDesc(0, x_out_desc);

#if NN_UT_HAS_V2_IR_API
        GNode softmax_node = CompliantNodeBuilder(graph)
                                 .OpType("SoftmaxV2")
                                 .Name("SoftmaxV2")
                                 .IrDefInputs(
                                     {{ "x",
                                        CompliantNodeBuilder::kEsIrInputRequired,
                                        "" }})
                                 .IrDefOutputs(
                                     {{ "y",
                                        CompliantNodeBuilder::kEsIrOutputRequired,
                                        "" }})
                                 .IrDefAttrs({{"axes", CompliantNodeBuilder::kEsAttrOptional, "ListInt",
                                               CreateFrom(vector<int64_t>{axis})},
                                              { "half_to_float",
                                                CompliantNodeBuilder::kEsAttrOptional,
                                                "Bool",
                                                CreateFrom(false) }})
                                 .Build();
#else
        GNode softmax_node = CompliantNodeBuilder(graph)
                                 .OpType("SoftmaxV2")
                                 .Name("SoftmaxV2")
                                 .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                 .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                 .IrDefAttrs({{"axes", CompliantNodeBuilder::kEsAttrOptional, "ListInt",
                                               CreateFrom(vector<int64_t>{axis})},
                                              {"half_to_float", CompliantNodeBuilder::kEsAttrOptional, "Bool",
                                               CreateFrom(false)}})
                                 .Build();
#endif
        if (has_need_fusion) {
            int64_t value = need_fusion;
            softmax_node.SetAttr(AscendString("need_fusion"), value);
        }
        AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), softmax_node, 0);

        TensorDesc softmax_input_desc;
        softmax_input_desc.SetDataType(dtype);
        softmax_input_desc.SetShape(Shape(dims));
        softmax_node.UpdateInputDesc(0, softmax_input_desc);

        TensorDesc y_desc;
        y_desc.SetDataType(dtype);
        y_desc.SetShape(Shape(dims));
        softmax_node.UpdateOutputDesc(0, y_desc);

        auto* y_holder = graph_builder.GetCGraphBuilder()->GetTensorHolderFromNode(softmax_node, 0);
        return graph_builder.BuildAndReset({EsTensorHolder(y_holder)});
    }

    bool HasReshapeNode(const std::shared_ptr<Graph>& graph)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == "Reshape") {
                return true;
            }
        }
        return false;
    }
};

TEST_F(ASoftmaxFusionPassTest, asoftmax_fusion_pattern_test)
{
    ops::ASoftmaxFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(ASoftmaxFusionPassTest, asoftmax_fusion_float_success)
{
    auto graph = BuildSoftmaxV2Graph(DT_FLOAT, {2, 3, 4}, 1, 1);
    CustomPassContext pass_context;
    ops::ASoftmaxFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(HasReshapeNode(graph), true);
}

TEST_F(ASoftmaxFusionPassTest, asoftmax_fusion_float16_success)
{
    auto graph = BuildSoftmaxV2Graph(DT_FLOAT16, {2, 3, 4, 5}, 1, 1);
    CustomPassContext pass_context;
    ops::ASoftmaxFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(HasReshapeNode(graph), true);
}

TEST_F(ASoftmaxFusionPassTest, asoftmax_fusion_last_axis_skip)
{
    auto graph = BuildSoftmaxV2Graph(DT_FLOAT, {2, 3, 4}, 2, 1);
    CustomPassContext pass_context;
    ops::ASoftmaxFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(HasReshapeNode(graph), false);
}

TEST_F(ASoftmaxFusionPassTest, asoftmax_fusion_missing_need_fusion_skip)
{
    auto graph = BuildSoftmaxV2Graph(DT_FLOAT, {2, 3, 4}, 1, 0, false);
    CustomPassContext pass_context;
    ops::ASoftmaxFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(HasReshapeNode(graph), false);
}

TEST_F(ASoftmaxFusionPassTest, asoftmax_fusion_2d_skip)
{
    auto graph = BuildSoftmaxV2Graph(DT_FLOAT, {2, 3}, 0, 1);
    CustomPassContext pass_context;
    ops::ASoftmaxFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(HasReshapeNode(graph), false);
}

TEST_F(ASoftmaxFusionPassTest, asoftmax_fusion_negative_axis_success)
{
    auto graph = BuildSoftmaxV2Graph(DT_FLOAT, {2, 3, 4, 5}, -2, 1);
    CustomPassContext pass_context;
    ops::ASoftmaxFusionPass pass;
    Status status = pass.Run(graph, pass_context);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(HasReshapeNode(graph), true);
}

TEST_F(ASoftmaxFusionPassTest, asoftmax_fusion_zero_need_fusion_skip)
{
    auto graph = BuildSoftmaxV2Graph(DT_FLOAT, {2, 3, 4}, 1, 0);
    CustomPassContext ctx;
    ASoftmaxFusionPass pass;
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasReshapeNode(graph));
}
