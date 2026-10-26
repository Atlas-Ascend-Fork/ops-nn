/*
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
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "ge/compliant_node_builder.h"
#include "../../../op_graph/fusion_pass/confusion_softmax_grad_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace es;
using namespace ops;

namespace {
const int32_t kExpectedNodeCount = 4; // 2 Data + 1 ConfusionSoftmaxGrad + 1 NetOutput

void SetOutDesc(const EsTensorHolder& holder, DataType dtype, const vector<int64_t>& dims)
{
    TensorDesc desc;
    holder.GetProducer()->GetOutputDesc(0, desc);
    desc.SetDataType(dtype);
    desc.SetShape(ge::Shape(dims));
    holder.GetProducer()->UpdateOutputDesc(0, desc);
}

void SetInDesc(const GNode& node, int32_t idx, DataType dtype, const vector<int64_t>& dims)
{
    TensorDesc desc;
    node.GetInputDesc(idx, desc);
    desc.SetDataType(dtype);
    desc.SetShape(ge::Shape(dims));
    const_cast<GNode&>(node).UpdateInputDesc(idx, desc);
}

vector<int64_t> ReduceShape(const vector<int64_t>& dims, int64_t axis, bool keepDims)
{
    int64_t rank = static_cast<int64_t>(dims.size());
    int64_t a = axis < 0 ? axis + rank : axis;
    vector<int64_t> out;
    for (int64_t i = 0; i < rank; ++i) {
        if (i == a) {
            if (keepDims) {
                out.push_back(1);
            }
        } else {
            out.push_back(dims[i]);
        }
    }
    return out;
}

EsTensorHolder BuildBinaryNode(EsGraphBuilder& builder, const EsTensorHolder& in0, const EsTensorHolder& in1,
                               const char* opType)
{
    auto* c_builder = builder.GetCGraphBuilder();
    auto* graph = c_builder->GetGraph();
    GNode node = CompliantNodeBuilder(graph)
                     .OpType(opType)
                     .Name(c_builder->GenerateNodeName(opType).GetString())
                     .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *in0.GetProducer(), in0.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *in1.GetProducer(), in1.GetProducerOutIndex(), node, 1);
    return EsTensorHolder(c_builder->GetTensorHolderFromNode(node, 0));
}

EsTensorHolder BuildReduceSum(EsGraphBuilder& builder, const EsTensorHolder& in0, const EsTensorHolder& axes,
                              bool keepDims)
{
    auto* c_builder = builder.GetCGraphBuilder();
    auto* graph = c_builder->GetGraph();
    GNode node = CompliantNodeBuilder(graph)
                     .OpType("ReduceSum")
                     .Name(c_builder->GenerateNodeName("ReduceSum").GetString())
                     .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                   {"axes", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                     .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                     .IrDefAttrs(
                         {{"keep_dims", CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(keepDims)},
                          {"noop_with_empty_axes", CompliantNodeBuilder::kEsAttrOptional, "Bool", CreateFrom(true)}})
                     .Build();
    AddEdgeAndUpdatePeerDesc(*graph, *in0.GetProducer(), in0.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *axes.GetProducer(), axes.GetProducerOutIndex(), node, 1);
    return EsTensorHolder(c_builder->GetTensorHolderFromNode(node, 0));
}

struct BuiltGraph {
    shared_ptr<Graph> graph;
};

// 构造 Mul -> ReduceSum(axes) -> Sub(grad, sum) 目标图（Sub 与 Mul 共享 grad），
// 手动设置全部 tensor desc（等价 InferShapeForTest）后返回。
// axesInt32 为 true 时构造 int32 axes const，否则 int64。
BuiltGraph BuildConfusionSoftmaxGradGraph(DataType dtype, const vector<int64_t>& gradDims, const vector<int64_t>& xDims,
                                          int64_t axis, bool keepDims, bool axesInt32, bool sharedAxes = false,
                                          bool runtimeAxes = false)
{
    EsGraphBuilder builder("confusion_softmax_grad_fusion_test");
    auto grad = builder.CreateInput(0, "grad", dtype, FORMAT_ND, gradDims);
    auto x = builder.CreateInput(1, "x", dtype, FORMAT_ND, xDims);
    SetOutDesc(grad, dtype, gradDims);
    SetOutDesc(x, dtype, xDims);

    auto mul = BuildBinaryNode(builder, grad, x, "Mul");
    EsTensorHolder axes;
    DataType axesDtype = DT_INT64;
    if (runtimeAxes) {
        axes = builder.CreateInput(2, "axes", DT_INT64, FORMAT_ND, {1});
    } else if (axesInt32) {
        axes = builder.CreateConst(vector<int32_t>{static_cast<int32_t>(axis)}, {1});
        axesDtype = DT_INT32;
    } else {
        axes = builder.CreateConst(vector<int64_t>{axis}, {1});
    }
    auto sum = BuildReduceSum(builder, mul, axes, keepDims);
    auto sub = BuildBinaryNode(builder, grad, sum, "Sub");

    // InferShapeForTest：手动设置所有节点 desc，保证 pattern 匹配与守卫检查可以执行。
    const vector<int64_t> mulOutDims = gradDims.size() >= xDims.size() ? gradDims : xDims;
    const vector<int64_t> reduced = ReduceShape(mulOutDims, axis, keepDims);
    SetInDesc(*mul.GetProducer(), 0, dtype, gradDims);
    SetInDesc(*mul.GetProducer(), 1, dtype, xDims);
    SetOutDesc(mul, dtype, mulOutDims);
    SetInDesc(*sum.GetProducer(), 0, dtype, mulOutDims);
    SetInDesc(*sum.GetProducer(), 1, axesDtype, {1});
    SetOutDesc(sum, dtype, reduced);
    SetInDesc(*sub.GetProducer(), 0, dtype, gradDims);
    SetInDesc(*sub.GetProducer(), 1, dtype, reduced);
    SetOutDesc(sub, dtype, mulOutDims);

    BuiltGraph ret;
    if (sharedAxes) {
        auto external = BuildReduceSum(builder, x, axes, keepDims);
        SetOutDesc(external, dtype, ReduceShape(xDims, axis, keepDims));
        ret.graph = builder.BuildAndReset({sub, external});
    } else {
        ret.graph = builder.BuildAndReset({sub});
    }
    return ret;
}

void SetPlatform(const string& soc, int32_t npu_arch = 3510)
{
    PlatformInfo platformInfo;
    OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = soc;
    optiCompilationInfo.soc_version = soc;
    PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    PlatFormInfos platform_infos;
    (void)platform_infos.Init();
    std::map<std::string, std::string> version_res;
    version_res["NpuArch"] = std::to_string(npu_arch);
    version_res["Short_SoC_version"] = soc;
    platform_infos.SetPlatformRes("version", version_res);
    OptionalInfos optional_infos;
    (void)optional_infos.Init();
    optional_infos.SetSocVersion(soc);
    PlatformInfoManager::Instance().platform_infos_map_[soc] = platform_infos;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional_infos);
}
} // namespace

class ConfusionSoftmaxGradFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { SetPlatform("Ascend950"); }
    void SetUp() override { SetPlatform("Ascend950"); }

    // 校验图中恰有 1 个 ConfusionSoftmaxGrad、无 Mul/ReduceSum/Sub 残留、
    // 节点总数为 4（2 Data + 1 ConfusionSoftmaxGrad + 1 NetOutput）、
    // 输入映射 grad/x、输入 dtype 与 shape 与预期一致。
    void ExpectFused(const shared_ptr<Graph>& graph, DataType dtype, const vector<int64_t>& dims)
    {
        bool found = false;
        int32_t nodeCount = 0;
        int32_t mulSubReduceCount = 0;
        for (auto node : graph->GetAllNodes()) {
            nodeCount++;
            AscendString type;
            node.GetType(type);
            if (type == "Mul" || type == "Sub" || type == "ReduceSum") {
                mulSubReduceCount++;
            }
            if (type == "ConfusionSoftmaxGrad") {
                found = true;
                // 校验输入映射：input0=grad, input1=x
                for (int32_t i = 0; i < 2; ++i) {
                    auto src = node.GetInDataNodesAndPortIndexs(i);
                    AscendString srcName;
                    src.first->GetName(srcName);
                    std::string expected = (i == 0) ? "grad" : "x";
                    std::string actual = srcName.GetString();
                    EXPECT_NE(actual.find(expected), std::string::npos)
                        << "input " << i << " expected " << expected << " got " << actual;
                }
                // 校验输入与输出 dtype/shape
                TensorDesc input0Desc;
                TensorDesc input1Desc;
                TensorDesc outputDesc;
                node.GetInputDesc(0, input0Desc);
                node.GetInputDesc(1, input1Desc);
                node.GetOutputDesc(0, outputDesc);
                EXPECT_EQ(input0Desc.GetDataType(), dtype);
                EXPECT_EQ(input1Desc.GetDataType(), dtype);
                EXPECT_EQ(outputDesc.GetDataType(), dtype);
                EXPECT_EQ(input0Desc.GetShape().GetDims(), dims);
                EXPECT_EQ(input1Desc.GetShape().GetDims(), dims);
                EXPECT_EQ(outputDesc.GetShape().GetDims(), dims);
            }
        }
        EXPECT_TRUE(found);
        EXPECT_EQ(mulSubReduceCount, 0);
        EXPECT_EQ(nodeCount, kExpectedNodeCount);
    }

    // 未融合用例：Run 返回 GRAPH_NOT_CHANGED 且 Mul 节点保留。
    void ExpectNotFused(const shared_ptr<Graph>& graph)
    {
        bool findMul = false;
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == "Mul") {
                findMul = true;
            }
        }
        EXPECT_TRUE(findMul);
    }
};

// fp32、axis=-1、keep_dims=true，融合成功
TEST_F(ConfusionSoftmaxGradFusionPassTest, fp32_axis_neg1_keepdims_true_success)
{
    vector<int64_t> dims{1, 8, 900, 3};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, dims, dims, -1, true, false);
    built.graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_graph_for_confusion_softmax_grad_test1");
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), SUCCESS);
    built.graph->DumpToFile(Graph::DumpFormat::kOnnx, "dump_afterpass_graph_for_confusion_softmax_grad_test1");
    ExpectFused(built.graph, DT_FLOAT, dims);
}

// axes 为 int32 const、axis=3（最后一维），融合成功
TEST_F(ConfusionSoftmaxGradFusionPassTest, fp32_int32_axes_last_dim_success)
{
    vector<int64_t> dims{1, 8, 900, 3};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, dims, dims, 3, true, true);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), SUCCESS);
    ExpectFused(built.graph, DT_FLOAT, dims);
}

// fp16，融合成功
TEST_F(ConfusionSoftmaxGradFusionPassTest, fp16_success)
{
    vector<int64_t> dims{1, 8, 900, 3};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT16, dims, dims, 3, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), SUCCESS);
    ExpectFused(built.graph, DT_FLOAT16, dims);
}

// bf16，融合成功
TEST_F(ConfusionSoftmaxGradFusionPassTest, bf16_success)
{
    vector<int64_t> dims{1, 8, 900, 3};
    auto built = BuildConfusionSoftmaxGradGraph(DT_BF16, dims, dims, 3, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), SUCCESS);
    ExpectFused(built.graph, DT_BF16, dims);
}

// grad 为动态 shape（-1），不融合
TEST_F(ConfusionSoftmaxGradFusionPassTest, dynamic_shape_grad_not_changed)
{
    vector<int64_t> dims{-1, 8, 900, 3};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, dims, dims, 3, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

// x 为动态 shape（-1），不融合
TEST_F(ConfusionSoftmaxGradFusionPassTest, dynamic_shape_x_not_changed)
{
    vector<int64_t> gradDims{1, 8, 900, 3};
    vector<int64_t> xDims{-1, 8, 900, 3};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, gradDims, xDims, 3, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

// 最后一维 40000 > 30000，不融合
TEST_F(ConfusionSoftmaxGradFusionPassTest, last_dim_more_than_30000_not_changed)
{
    vector<int64_t> dims{1, 8, 900, 40000};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, dims, dims, 3, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

// keep_dims=false，不融合
TEST_F(ConfusionSoftmaxGradFusionPassTest, keep_dims_false_not_changed)
{
    vector<int64_t> dims{900, 900};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, dims, dims, 1, false, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

// axis=0 非最后一维，不融合
TEST_F(ConfusionSoftmaxGradFusionPassTest, axis_not_last_dim_not_changed)
{
    vector<int64_t> dims{900, 900};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, dims, dims, 0, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

// int32 dtype 不支持，不融合
TEST_F(ConfusionSoftmaxGradFusionPassTest, int32_dtype_not_changed)
{
    vector<int64_t> dims{900, 900};
    auto built = BuildConfusionSoftmaxGradGraph(DT_INT32, dims, dims, 1, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

// grad 与 x shape 不一致，不融合
TEST_F(ConfusionSoftmaxGradFusionPassTest, input_shape_not_equal_not_changed)
{
    vector<int64_t> gradDims{1, 8, 900, 3};
    vector<int64_t> xDims{1, 8, 900, 4};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, gradDims, xDims, 3, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

// 非目标平台（Ascend910B），不融合（ops-nn 算子仅配置 ascend950）
TEST_F(ConfusionSoftmaxGradFusionPassTest, unsupported_platform_not_changed)
{
    SetPlatform("Ascend910B", 910);
    vector<int64_t> dims{1, 8, 900, 3};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT16, dims, dims, 3, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

// 5102 同为 regbase 但本规则当前仅在 3510 上启用，不融合
TEST_F(ConfusionSoftmaxGradFusionPassTest, npu_arch_5102_not_changed)
{
    SetPlatform("Ascend950", 5102);
    vector<int64_t> dims{1, 8, 900, 3};
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT16, dims, dims, 3, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

// Mul 与 Sub 的 input0 不共享同一节点（Sub.x1 来自 x），结构不匹配，不融合
TEST_F(ConfusionSoftmaxGradFusionPassTest, sub_input0_not_shared_not_changed)
{
    vector<int64_t> dims{2, 32, 128};
    EsGraphBuilder builder("sub_input_not_shared_test");
    auto grad = builder.CreateInput(0, "grad", DT_FLOAT16, FORMAT_ND, dims);
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_ND, dims);
    SetOutDesc(grad, DT_FLOAT16, dims);
    SetOutDesc(x, DT_FLOAT16, dims);

    auto mul = BuildBinaryNode(builder, grad, x, "Mul");
    auto axes = builder.CreateConst(vector<int64_t>{-1}, {1});
    auto sum = BuildReduceSum(builder, mul, axes, true);
    auto sub = BuildBinaryNode(builder, x, sum, "Sub"); // Sub 的 input0 来自 x 而非 grad

    SetInDesc(*mul.GetProducer(), 0, DT_FLOAT16, dims);
    SetInDesc(*mul.GetProducer(), 1, DT_FLOAT16, dims);
    SetOutDesc(mul, DT_FLOAT16, dims);
    SetInDesc(*sum.GetProducer(), 0, DT_FLOAT16, dims);
    SetInDesc(*sum.GetProducer(), 1, DT_INT64, {1});
    SetOutDesc(sum, DT_FLOAT16, {2, 32, 1});
    SetInDesc(*sub.GetProducer(), 0, DT_FLOAT16, dims);
    SetInDesc(*sub.GetProducer(), 1, DT_FLOAT16, {2, 32, 1});
    SetOutDesc(sub, DT_FLOAT16, dims);

    shared_ptr<Graph> graph = builder.BuildAndReset({sub});
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(graph);
}

TEST_F(ConfusionSoftmaxGradFusionPassTest, unknown_rank_not_changed)
{
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, {-2}, {-2}, 0, true, false);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}

TEST_F(ConfusionSoftmaxGradFusionPassTest, unknown_mul_output_not_changed)
{
    for (const auto& dims : vector<vector<int64_t>>{{-2}, {2, -1}}) {
        auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, {2, 3}, {2, 3}, -1, true, false);
        for (auto node : built.graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == "Mul") {
                TensorDesc desc;
                node.GetOutputDesc(0, desc);
                desc.SetShape(Shape(dims));
                node.UpdateOutputDesc(0, desc);
            }
        }
        CustomPassContext ctx;
        ZConfusionSoftmaxGradFusionPass pass;
        EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
        ExpectNotFused(built.graph);
    }
}

TEST_F(ConfusionSoftmaxGradFusionPassTest, shared_axes_preserves_external_consumer)
{
    for (bool int32_axes : {false, true}) {
        auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, {2, 3}, {2, 3}, -1, true, int32_axes, true);
        CustomPassContext ctx;
        ZConfusionSoftmaxGradFusionPass pass;
        ASSERT_EQ(pass.Run(built.graph, ctx), SUCCESS);
        int fused_count = 0;
        int reduce_count = 0;
        int const_count = 0;
        for (auto node : built.graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            EXPECT_NE(type, "Mul");
            EXPECT_NE(type, "Sub");
            if (type == "ConfusionSoftmaxGrad") {
                ++fused_count;
                EXPECT_EQ(node.GetInputsSize(), 2U);
                for (int i = 0; i < 2; ++i) {
                    auto source = node.GetInDataNodesAndPortIndexs(i);
                    ASSERT_NE(source.first, nullptr);
                    AscendString name;
                    source.first->GetName(name);
                    EXPECT_EQ(name, i == 0 ? "grad" : "x");
                }
            } else if (type == "ReduceSum") {
                ++reduce_count;
                auto source = node.GetInDataNodesAndPortIndexs(0);
                ASSERT_NE(source.first, nullptr);
                AscendString name;
                source.first->GetName(name);
                EXPECT_EQ(name, "x");
                Tensor axes;
                ASSERT_EQ(node.GetInputConstData(1, axes), GRAPH_SUCCESS);
                EXPECT_EQ(axes.GetDataType(), int32_axes ? DT_INT32 : DT_INT64);
                ASSERT_EQ(axes.GetSize(), int32_axes ? sizeof(int32_t) : sizeof(int64_t));
                // -1 is all-one bytes in either integer width.
                for (size_t i = 0; i < axes.GetSize(); ++i) {
                    EXPECT_EQ(axes.GetData()[i], 0xff);
                }
                EXPECT_EQ(node.GetOutDataNodesAndPortIndexs(0).size(), 1U);
            } else if (type == "Const") {
                ++const_count;
                EXPECT_EQ(node.GetOutDataNodesAndPortIndexs(0).size(), 1U);
            }
        }
        EXPECT_EQ(fused_count, 1);
        EXPECT_EQ(reduce_count, 1);
        EXPECT_EQ(const_count, 1);
        EXPECT_EQ(built.graph->GetAllNodes().size(), 6U);
        EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    }
}

TEST_F(ConfusionSoftmaxGradFusionPassTest, runtime_axes_not_changed)
{
    auto built = BuildConfusionSoftmaxGradGraph(DT_FLOAT, {2, 3}, {2, 3}, -1, true, false, false, true);
    CustomPassContext ctx;
    ZConfusionSoftmaxGradFusionPass pass;
    EXPECT_EQ(pass.Run(built.graph, ctx), GRAPH_NOT_CHANGED);
    ExpectNotFused(built.graph);
}
