/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <type_traits>
#include <utility>
#include <gtest/gtest.h>
#include "acl/acl_rt.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "ge/fusion/graph_rewriter.h"
#include "../../../../weight_quant_batch_matmul_v2/op_graph/weight_quant_batch_matmul_v2_proto.h"
#define private public
#include "platform/platform_info.h"
#undef private
#include "../../../op_graph/fusion_pass/matmul_quant_to_fixpipe_fusion_pass.h"

namespace {
using namespace ge;
using Builder = ge::es::CompliantNodeBuilder;
class TestPass : public ops::MatmulQuantToFixpipeFusionPass {
public:
    using MatmulQuantToFixpipeFusionPass::Fuse;
    using MatmulQuantToFixpipeFusionPass::HasRequiredPublicApis;
    using MatmulQuantToFixpipeFusionPass::Run;
};

template <typename T, typename = void>
struct HasOriginDataType : std::false_type {};
template <typename T>
struct HasOriginDataType<T, std::void_t<decltype(std::declval<const T&>().GetOriginDataType()),
                                        decltype(std::declval<T&>().SetOriginDataType(DT_UNDEFINED))>>
    : std::true_type {};
template <typename T>
void ExpectOriginDataType(const T& desc, DataType type)
{
    if constexpr (HasOriginDataType<T>::value) {
        EXPECT_EQ(desc.GetOriginDataType(), type);
    } else {
        FAIL() << "SDK does not provide the required public origin dtype API";
    }
}
template <typename T>
void SetPublicOriginDataType(T& desc, DataType type)
{
    if constexpr (HasOriginDataType<T>::value) {
        desc.SetOriginDataType(type);
    } else {
        FAIL() << "SDK does not provide the required public origin dtype API";
    }
}

TensorDesc Desc(DataType type, Format format = FORMAT_ND, std::vector<int64_t> shape = {16, 64})
{
    TensorDesc desc(Shape(shape), format, type);
    desc.SetOriginShape(desc.GetShape());
    desc.SetOriginFormat(format);
    return desc;
}

// Check ordering through public control connections without requiring a particular
// NoOp layout, direct-edge count, or framework-generated node name.
bool HasControlPath(const GNode& source, const GNode& target)
{
    AscendString targetName;
    if (target.GetName(targetName) != GRAPH_SUCCESS) {
        return false;
    }
    std::vector<GNode> pending = {source};
    std::set<std::string> visited;
    while (!pending.empty()) {
        const auto node = pending.back();
        pending.pop_back();
        AscendString name;
        if (node.GetName(name) != GRAPH_SUCCESS || !visited.insert(name.GetString()).second) {
            continue;
        }
        if (name == targetName) {
            return true;
        }
        for (const auto& next : node.GetOutControlNodes()) {
            if (next) {
                pending.push_back(*next);
            }
        }
    }
    return false;
}

struct Case {
    const char* type = "MatMul";
    const char* successor = "AscendRequant";
    DataType aType = DT_INT8;
    DataType bType = DT_INT8;
    DataType output = DT_INT8;
    Format aFormat = FORMAT_ND;
    Format bFormat = FORMAT_ND;
    size_t inputSlots = 3;
    bool three = false;
    bool bias = false;
    bool transA = false;
    bool transB = true;
    std::vector<uint64_t> scale = {13802860151ULL};
    float factor = 118.689f;
    float offset = -1;
};

class FixpipeFusionTest : public testing::Test {
protected:
    GraphPtr graph;
    GraphPtr attributeReferenceGraph;
    std::map<std::string, GNode> nodes;
    TestPass pass;
    decltype(fe::PlatformInfoManager::Instance().platform_info_map_) saved;
    fe::OptionalInfo savedOptional;
    bool platformSaved = false;

    void SetUp() override
    {
        if (!TestPass::HasRequiredPublicApis() &&
            std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) != "RuntimeVersionGate") {
            GTEST_SKIP() << "This pass requires the public TensorDesc origin dtype APIs";
        }
        auto& manager = fe::PlatformInfoManager::Instance();
        saved = manager.platform_info_map_;
        platformSaved = true;
        fe::PlatformInfo unused;
        (void)manager.GetPlatformInfoWithOutSocVersion(unused, savedOptional);
        fe::OptionalInfo optional;
        optional.soc_version = "fixpipe_ut";
        fe::PlatformInfo info;
        info.ai_core_intrinsic_dtype_map["Intrinsic_fix_pipe_l0c2out"] = {"s32"};
        info.ai_core_intrinsic_dtype_map["Intrinsic_mmad"] = {"s8s4"};
        manager.platform_info_map_.clear();
        manager.platform_info_map_[optional.soc_version] = info;
        manager.SetOptionalCompilationInfo(optional);
    }
    void TearDown() override
    {
        if (!platformSaved) {
            return;
        }
        auto& manager = fe::PlatformInfoManager::Instance();
        manager.platform_info_map_ = saved;
        manager.SetOptionalCompilationInfo(savedOptional);
    }
    auto& Caps()
    {
        return fe::PlatformInfoManager::Instance().platform_info_map_["fixpipe_ut"].ai_core_intrinsic_dtype_map;
    }

    GNode Make(const char* name, const char* type, const TensorDesc& output, size_t inputs = 0)
    {
        Builder builder(graph.get());
        builder.Name(name).OpType(type);
        if (inputs == 4) {
            builder.IrDefInputs({{"x1", Builder::kEsIrInputRequired, ""},
                                 {"x2", Builder::kEsIrInputRequired, ""},
                                 {"bias", Builder::kEsIrInputOptional, ""},
                                 {"offset_w", Builder::kEsIrInputOptional, ""}});
        } else if (inputs == 3) {
            builder.IrDefInputs({{"x1", Builder::kEsIrInputRequired, ""},
                                 {"x2", Builder::kEsIrInputRequired, ""},
                                 {"bias", Builder::kEsIrInputOptional, ""}});
        } else if (inputs == 2) {
            builder.IrDefInputs({{"x", Builder::kEsIrInputRequired, ""}, {"scale", Builder::kEsIrInputRequired, ""}});
        } else if (inputs == 1) {
            builder.IrDefInputs({{"x", Builder::kEsIrInputRequired, ""}});
        }
        auto node = builder.IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                        .InstanceOutputDataType("y", output.GetDataType())
                        .InstanceOutputFormat("y", output.GetFormat())
                        .InstanceOutputShape("y", output.GetShape().GetDims())
                        .Build();
        EXPECT_EQ(node.UpdateOutputDesc(0, output), GRAPH_SUCCESS);
        nodes[name] = node;
        return node;
    }
    void Link(GNode& from, GNode& to, int port = 0)
    {
        TensorDesc desc;
        ASSERT_EQ(from.GetOutputDesc(0, desc), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddDataEdge(from, 0, to, port), GRAPH_SUCCESS);
        ASSERT_EQ(to.UpdateInputDesc(port, desc), GRAPH_SUCCESS);
    }
    void Build(Case c = {})
    {
        graph = std::make_shared<Graph>("fixpipe_test");
        nodes.clear();
        auto a = Make("a", "Data", Desc(c.aType, c.aFormat));
        auto b = Make("b", "Data", Desc(c.bType, c.bFormat));
        auto m = Make("m", c.type, Desc(DT_INT32), c.inputSlots);
        Link(a, m, 0);
        Link(b, m, 1);
        const bool batch = std::string(c.type).find("Batch") != std::string::npos;
        ASSERT_EQ(m.SetAttr(batch ? "adj_x1" : "transpose_x1", c.transA), GRAPH_SUCCESS);
        ASSERT_EQ(m.SetAttr(batch ? "adj_x2" : "transpose_x2", c.transB), GRAPH_SUCCESS);
        if (c.bias) {
            auto bias = Make("bias", "Data", Desc(c.aType == DT_INT8 ? DT_INT32 : c.aType));
            Link(bias, m, 2);
        }
        auto sd = Desc(DT_UINT64, FORMAT_ND, {static_cast<int64_t>(c.scale.size())});
        auto scale = Make("scale", "Const", sd);
        Tensor value(sd, reinterpret_cast<const uint8_t*>(c.scale.data()), c.scale.size() * sizeof(uint64_t));
        ASSERT_EQ(scale.SetAttr("value", value), GRAPH_SUCCESS);
        auto q = Make("q", c.three ? "AscendDequant" : c.successor, Desc(c.three ? DT_FLOAT16 : c.output), 2);
        Link(m, q);
        Link(scale, q, 1);
        auto tail = q;
        if (c.three) {
            tail = Make("last", "AscendQuant", Desc(c.output), 1);
            Link(q, tail);
            ASSERT_EQ(tail.SetAttr("scale", c.factor), GRAPH_SUCCESS);
            ASSERT_EQ(tail.SetAttr("offset", c.offset), GRAPH_SUCCESS);
        }
        auto consumer = Make("consumer", "Identity", Desc(c.output), 1);
        Link(tail, consumer);
    }
    Status Run()
    {
        CustomPassContext context;
        // Test graph rewriting independently of other suites' process-wide ACL version stubs.
        return pass.Fuse(graph, context);
    }
    Status TextAttr(GNode& node, const char* key, const char* text)
    {
        AscendString value(text);
        return node.SetAttr(key, value);
    }
    GNodePtr FindFusedTarget()
    {
        for (const auto& node : graph->GetDirectNode()) {
            AscendString type;
            node.GetType(type);
            if (type == "QuantBatchMatmulV3" || type == "WeightQuantBatchMatmulV2") {
                return std::make_shared<GNode>(node);
            }
        }
        return nullptr;
    }
    GNodePtr Target() { return nodes.at("consumer").GetInDataNodesAndPortIndexs(0).first; }
    void BuildFrameworkAttributeReference(GNodePtr& result)
    {
        attributeReferenceGraph = std::make_shared<Graph>("attribute_reference");
        ASSERT_EQ(attributeReferenceGraph->CopyFrom(*graph), GRAPH_SUCCESS);
        auto matmul = attributeReferenceGraph->FindNodeByName("m");
        auto dequant = attributeReferenceGraph->FindNodeByName("q");
        auto tail = attributeReferenceGraph->FindNodeByName("last");
        ASSERT_NE(matmul, nullptr);
        ASSERT_NE(dequant, nullptr);
        if (!tail) {
            tail = dequant;
        }
        ge::fusion::SubgraphBoundary boundary(
            {ge::fusion::SubgraphInput({{*matmul, 0}}), ge::fusion::SubgraphInput({{*matmul, 1}}),
             ge::fusion::SubgraphInput({{*dequant, 1}})},
            {ge::fusion::SubgraphOutput({*tail, 0})});
        ge::es::EsGraphBuilder builder("attribute_replacement");
        auto replacementGraph = builder.GetCGraphBuilder()->GetGraph();
        auto input = builder.CreateInput(0);
        builder.CreateInput(1);
        builder.CreateInput(2);
        TensorDesc output;
        ASSERT_EQ(tail->GetOutputDesc(0, output), GRAPH_SUCCESS);
        auto node = Builder(replacementGraph)
                        .Name("framework_attribute_reference")
                        .OpType("Identity")
                        .IrDefInputs({{"x", Builder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                        .InstanceOutputDataType("y", output.GetDataType())
                        .InstanceOutputShape("y", output.GetShape().GetDims())
                        .InstanceOutputFormat("y", output.GetFormat())
                        .Build();
        TensorDesc inputDesc;
        ASSERT_EQ(matmul->GetInputDesc(0, inputDesc), GRAPH_SUCCESS);
        ASSERT_EQ(node.UpdateInputDesc(0, inputDesc), GRAPH_SUCCESS);
        ASSERT_EQ(replacementGraph->AddDataEdge(*input.GetProducer(), 0, node, 0), GRAPH_SUCCESS);
        auto tensor = ge::es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
        auto replacement = builder.BuildAndReset({tensor});
        ASSERT_NE(replacement, nullptr);
        ASSERT_EQ(ge::fusion::SubgraphRewriter::Replace(boundary, *replacement), GRAPH_SUCCESS);
        for (const auto& candidate : attributeReferenceGraph->GetDirectNode()) {
            AscendString name;
            ASSERT_EQ(candidate.GetName(name), GRAPH_SUCCESS);
            if (std::string(name.GetString()).find("framework_attribute_reference") == 0) {
                result = std::make_shared<GNode>(candidate);
                break;
            }
        }
        ASSERT_NE(result, nullptr);
    }

    void ExpectFrameworkCompilationAttributes(const GNode& actual, const GNode& reference)
    {
        for (const char* key : {"_stream_label", "_op_compile_strategy"}) {
            AscendString expected, value;
            const auto status = reference.GetAttr(key, expected);
            EXPECT_EQ(actual.GetAttr(key, value), status) << key;
            if (status == GRAPH_SUCCESS) {
                EXPECT_EQ(value, expected) << key;
            }
        }
        int64_t expected = 0, value = 0;
        const auto status = reference.GetAttr("_keep_dtype", expected);
        EXPECT_EQ(actual.GetAttr("_keep_dtype", value), status);
        if (status == GRAPH_SUCCESS) {
            EXPECT_EQ(value, expected);
        }
    }

    void ExpectSharedDequantFallback(const GNode& user, int32_t inputPort = 0)
    {
        auto fused = FindFusedTarget();
        ASSERT_NE(fused, nullptr);
        auto quant = Target();
        ASSERT_NE(quant, nullptr);
        AscendString type, fusedName, inputName;
        ASSERT_EQ(quant->GetType(type), GRAPH_SUCCESS);
        EXPECT_EQ(type, "AscendQuant");
        ASSERT_EQ(fused->GetName(fusedName), GRAPH_SUCCESS);
        for (const auto& input : {user.GetInDataNodesAndPortIndexs(inputPort), quant->GetInDataNodesAndPortIndexs(0)}) {
            ASSERT_NE(input.first, nullptr);
            ASSERT_EQ(input.first->GetName(inputName), GRAPH_SUCCESS);
            EXPECT_EQ(inputName, fusedName);
            EXPECT_EQ(input.second, 0);
        }
        TensorDesc output;
        ASSERT_EQ(fused->GetOutputDesc(0, output), GRAPH_SUCCESS);
        EXPECT_EQ(output.GetDataType(), DT_FLOAT16);
        EXPECT_EQ(graph->FindNodeByName("m"), nullptr);
        EXPECT_EQ(graph->FindNodeByName("q"), nullptr);
        EXPECT_NE(graph->FindNodeByName("last"), nullptr);
    }

    std::vector<uint64_t> ScaleValues()
    {
        auto target = Target();
        if (!target)
            return {};
        auto scale = target->GetInDataNodesAndPortIndexs(2).first;
        if (!scale)
            return {};
        Tensor tensor;
        EXPECT_EQ(scale->GetAttr("value", tensor), GRAPH_SUCCESS);
        std::vector<uint64_t> result(tensor.GetSize() / sizeof(uint64_t));
        if (!result.empty())
            std::memcpy(result.data(), tensor.GetData(), tensor.GetSize());
        return result;
    }
};

TEST_F(FixpipeFusionTest, AllSourceTypesAndModes)
{
    for (const auto* source : {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"}) {
        for (const auto dtype : {DT_INT8, DT_FLOAT16, DT_FLOAT}) {
            for (int mode = 0; mode < 3; ++mode) {
                Case c;
                c.type = source;
                c.aType = dtype;
                c.successor = mode == 0 ? "AscendRequant" : "AscendDequant";
                c.three = mode == 2;
                SCOPED_TRACE(std::string(source) + "/" + std::to_string(dtype) + "/" + std::to_string(mode));
                Build(c);
                ASSERT_EQ(Run(), GRAPH_SUCCESS);
                ASSERT_NE(Target(), nullptr);
                AscendString type;
                ASSERT_EQ(Target()->GetType(type), GRAPH_SUCCESS);
                EXPECT_EQ(type, dtype == DT_INT8 ? "QuantBatchMatmulV3" : "WeightQuantBatchMatmulV2");
                TensorDesc output;
                ASSERT_EQ(Target()->GetOutputDesc(0, output), GRAPH_SUCCESS);
                EXPECT_EQ(output.GetDataType(), mode == 1 ? DT_FLOAT16 : DT_INT8);
                EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
            }
        }
    }
}

TEST_F(FixpipeFusionTest, HistoricalScaleEncoding)
{
    Case c;
    c.three = true;
    Build(c);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    EXPECT_EQ(ScaleValues(), std::vector<uint64_t>{140601025200128ULL});
}

TEST_F(FixpipeFusionTest, VectorScaleLowBitsAndOffset)
{
    for (float offset : {0.0f, 0.49f, 0.5f, -0.49f, -0.5f, -1.5f, 127.0f, -128.0f}) {
        Case c;
        c.three = true;
        c.offset = offset;
        c.factor = 2.0f;
        c.scale = {0x123456783f801fffULL, 0xffffffff40001fffULL, 0x3f000123ULL};
        Build(c);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        const auto values = ScaleValues();
        ASSERT_EQ(values.size(), 3U);
        const uint64_t sign = offset < 0 ? 0x200000000000ULL : 0;
        const uint64_t bits = (static_cast<uint64_t>(static_cast<int64_t>(std::round(offset))) << 37) &
                              0x1fe000000000ULL;
        EXPECT_EQ(values[0], 0x40000000ULL | bits | sign | 0x400000000000ULL);
        EXPECT_EQ(values[1], 0x40800000ULL | bits | sign | 0x400000000000ULL);
        EXPECT_EQ(values[2], 0x3f800000ULL | bits | sign | 0x400000000000ULL);
    }
}

TEST_F(FixpipeFusionTest, TransposeAndOptionalBias)
{
    for (const auto* source : {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"}) {
        for (const bool a : {false, true})
            for (const bool b : {false, true})
                for (const bool bias : {false, true}) {
                    Case c;
                    c.type = source;
                    c.transA = a;
                    c.transB = b;
                    c.bias = bias;
                    Build(c);
                    ASSERT_EQ(Run(), GRAPH_SUCCESS);
                    bool value = false;
                    ASSERT_EQ(Target()->GetAttr("transpose_x1", value), GRAPH_SUCCESS);
                    EXPECT_EQ(value, a);
                    ASSERT_EQ(Target()->GetAttr("transpose_x2", value), GRAPH_SUCCESS);
                    EXPECT_EQ(value, b);
                    EXPECT_EQ(Target()->GetInDataNodesAndPortIndexs(4).first != nullptr, bias);
                    EXPECT_EQ(Target()->GetInDataNodesAndPortIndexs(3).first, nullptr);
                    EXPECT_EQ(Target()->GetInDataNodesAndPortIndexs(5).first, nullptr);
                }
    }
}

TEST_F(FixpipeFusionTest, WeightOnlyBiasUsesRegisteredInputName)
{
    for (const auto* source : {"MatMul", "MatMulV2", "BatchMatMul", "BatchMatMulV2"}) {
        for (const auto dtype : {DT_FLOAT16, DT_FLOAT}) {
            for (int mode = 0; mode < 3; ++mode) {
                for (const bool bias : {false, true}) {
                    Case c;
                    c.type = source;
                    c.aType = dtype;
                    c.successor = mode == 0 ? "AscendRequant" : "AscendDequant";
                    c.three = mode == 2;
                    c.bias = bias;
                    SCOPED_TRACE(std::string(source) + "/" + std::to_string(dtype) + "/" + std::to_string(mode) +
                                 "/bias=" + std::to_string(bias));
                    Build(c);
                    ASSERT_EQ(Run(), GRAPH_SUCCESS);
                    auto target = Target();
                    ASSERT_NE(target, nullptr);
                    int32_t biasPort = -1;
                    ASSERT_EQ(target->GetInputIndexByName("bias", biasPort), GRAPH_SUCCESS);
                    ASSERT_EQ(biasPort, 6);
                    const auto input = target->GetInDataNodesAndPortIndexs(biasPort);
                    ASSERT_EQ(input.first != nullptr, bias);
                    if (bias) {
                        AscendString name;
                        ASSERT_EQ(input.first->GetName(name), GRAPH_SUCCESS);
                        EXPECT_EQ(name, "bias");
                        EXPECT_EQ(input.second, 0);
                        TensorDesc desc;
                        ASSERT_EQ(target->GetInputDesc(biasPort, desc), GRAPH_SUCCESS);
                        EXPECT_EQ(desc.GetDataType(), dtype);
                        EXPECT_EQ(desc.GetShape().GetDims(), (std::vector<int64_t>{16, 64}));
                    }
                    for (const char* name : {"antiquant_offset", "quant_scale", "quant_offset"}) {
                        int32_t port = -1;
                        ASSERT_EQ(target->GetInputIndexByName(name, port), GRAPH_SUCCESS);
                        ASSERT_GE(port, 0);
                        EXPECT_EQ(target->GetInDataNodesAndPortIndexs(port).first, nullptr) << name;
                    }
                }
            }
        }
    }
}

TEST_F(FixpipeFusionTest, WeightOnlyUsesRegisteredTargetIr)
{
    Case c;
    c.aType = DT_FLOAT16;
    Build(c);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    int32_t index = -1;
    ASSERT_EQ(Target()->GetInputIndexByName("quant_scale", index), GRAPH_SUCCESS);
    EXPECT_EQ(index, 4);
    ASSERT_EQ(Target()->GetInputIndexByName("quant_offset", index), GRAPH_SUCCESS);
    EXPECT_EQ(index, 5);
    ASSERT_EQ(Target()->GetInputIndexByName("bias", index), GRAPH_SUCCESS);
    EXPECT_EQ(index, 6);
}

TEST_F(FixpipeFusionTest, DequantAlwaysProducesHalf)
{
    for (const auto dtype : {DT_FLOAT16, DT_FLOAT, DT_INT8}) {
        Case c;
        c.successor = "AscendDequant";
        c.output = dtype;
        Build(c);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        TensorDesc output;
        ASSERT_EQ(Target()->GetOutputDesc(0, output), GRAPH_SUCCESS);
        EXPECT_EQ(output.GetDataType(), DT_FLOAT16);
        ASSERT_EQ(nodes.at("consumer").GetInputDesc(0, output), GRAPH_SUCCESS);
        EXPECT_EQ(output.GetDataType(), dtype);
    }
}

TEST_F(FixpipeFusionTest, FormatWhitelist)
{
    for (const auto format : {FORMAT_ND, FORMAT_NCHW, FORMAT_NHWC, FORMAT_HWCN, FORMAT_CHWN, FORMAT_NDHWC, FORMAT_NCDHW,
                              FORMAT_DHWCN, FORMAT_DHWNC}) {
        Case c;
        c.aFormat = format;
        c.bFormat = format;
        Build(c);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        TensorDesc desc;
        ASSERT_EQ(Target()->GetInputDesc(0, desc), GRAPH_SUCCESS);
        EXPECT_EQ(desc.GetFormat(), FORMAT_ND);
        EXPECT_EQ(desc.GetOriginFormat(), FORMAT_ND);
    }
    Case c;
    c.bFormat = FORMAT_FRACTAL_NZ;
    Build(c);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    TensorDesc desc;
    ASSERT_EQ(Target()->GetInputDesc(1, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetFormat(), FORMAT_FRACTAL_NZ);
    EXPECT_EQ(desc.GetOriginFormat(), FORMAT_FRACTAL_NZ);
}

TEST_F(FixpipeFusionTest, UnsupportedDtypesAndFormats)
{
    for (const auto dtype : {DT_BF16, DT_INT4, DT_FLOAT8_E4M3FN}) {
        Case c;
        c.aType = dtype;
        Build(c);
        EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    }
    Case c;
    c.aFormat = FORMAT_FRACTAL_NZ;
    Build(c);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    c = {};
    c.aType = DT_FLOAT16;
    c.bFormat = FORMAT_FRACTAL_NZ;
    Build(c);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    c = {};
    c.bType = DT_FLOAT16;
    Build(c);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
}

TEST_F(FixpipeFusionTest, CommonPlatformConditions)
{
    const auto baseline = Caps();
    Caps().erase("Intrinsic_fix_pipe_l0c2out");
    Build();
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    Caps() = baseline;
    Caps()["Intrinsic_data_move_l0c2ub"] = {};
    Build();
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    Caps() = baseline;
    Caps()["Intrinsic_fix_pipe_l0c2ub"] = {};
    Build();
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    Caps()["Intrinsic_data_move_l12bt"] = {"fp16"};
    Build();
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    Caps()["Intrinsic_data_move_l12bt"] = {"bf16"};
    Build();
    EXPECT_EQ(Run(), GRAPH_SUCCESS);
}

TEST_F(FixpipeFusionTest, ExtendedCapabilityAndTwoNodeFallback)
{
    Caps().erase("Intrinsic_mmad");
    Build();
    EXPECT_EQ(Run(), GRAPH_SUCCESS);
    Case c;
    c.aType = DT_FLOAT16;
    Build(c);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    c = {};
    c.three = true;
    Build(c);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    AscendString type;
    ASSERT_EQ(Target()->GetType(type), GRAPH_SUCCESS);
    EXPECT_EQ(type, "AscendQuant");
    auto fused = Target()->GetInDataNodesAndPortIndexs(0).first;
    ASSERT_NE(fused, nullptr);
    ASSERT_EQ(fused->GetType(type), GRAPH_SUCCESS);
    EXPECT_EQ(type, "QuantBatchMatmulV3");
}

TEST_F(FixpipeFusionTest, MatmulMultipleEdgesRejected)
{
    Build();
    auto extra = Make("extra", "Identity", Desc(DT_INT32), 1);
    Link(nodes.at("m"), extra);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
}

TEST_F(FixpipeFusionTest, TerminalConsumersAndScaleSharing)
{
    Case c;
    c.three = true;
    Build(c);
    auto extra = Make("extra", "Identity", Desc(DT_INT8), 1);
    Link(nodes.at("last"), extra);
    auto shared = Make("shared", "Identity", Desc(DT_UINT64), 1);
    Link(nodes.at("scale"), shared);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    AscendString extraName;
    AscendString targetName;
    ASSERT_EQ(extra.GetInDataNodesAndPortIndexs(0).first->GetName(extraName), GRAPH_SUCCESS);
    ASSERT_EQ(Target()->GetName(targetName), GRAPH_SUCCESS);
    EXPECT_EQ(extraName, targetName);
    ASSERT_NE(shared.GetInDataNodesAndPortIndexs(0).first, nullptr);
    EXPECT_EQ(ScaleValues(), std::vector<uint64_t>{140601025200128ULL});
}

TEST_F(FixpipeFusionTest, OutputShapeComesFromMatmul)
{
    Build();
    auto desc = Desc(DT_INT32, FORMAT_ND, {-1, 64});
    ASSERT_EQ(nodes.at("m").UpdateOutputDesc(0, desc), GRAPH_SUCCESS);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    ASSERT_EQ(Target()->GetOutputDesc(0, desc), GRAPH_SUCCESS);
    EXPECT_EQ(desc.GetShape().GetDims(), (std::vector<int64_t>{-1, 64}));
    EXPECT_EQ(desc.GetOriginShape().GetDims(), desc.GetShape().GetDims());
}

TEST_F(FixpipeFusionTest, DoesNotMatchMatmulV3)
{
    Case c;
    c.type = "MatMulV3";
    Build(c);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
}

TEST_F(FixpipeFusionTest, EnableUncacheCopied)
{
    Build();
    int64_t enabled = 1;
    ASSERT_EQ(nodes.at("m").SetAttr("enable_uncache", enabled), GRAPH_SUCCESS);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    int64_t value = 0;
    ASSERT_EQ(Target()->GetAttr("enable_uncache", value), GRAPH_SUCCESS);
    EXPECT_EQ(value, 1);
}
TEST_F(FixpipeFusionTest, FusionAttributeCompatibility)
{
    for (const auto* key : {"_user_stream_label", "_super_kernel_scope", "_super_kernel_options", "_op_aicore_num",
                            "_op_vectorcore_num"}) {
        Build();
        ASSERT_EQ(TextAttr(nodes.at("m"), key, "first"), GRAPH_SUCCESS);
        ASSERT_EQ(TextAttr(nodes.at("q"), key, "second"), GRAPH_SUCCESS);
        EXPECT_EQ(Run(), GRAPH_NOT_CHANGED) << key;
        ASSERT_EQ(TextAttr(nodes.at("q"), key, "first"), GRAPH_SUCCESS);
        EXPECT_EQ(Run(), GRAPH_SUCCESS) << key;
    }
}

TEST_F(FixpipeFusionTest, DeterministicAttributeCompatibility)
{
    for (const auto* key : {"_deterministic", "_deterministic_level"}) {
        for (const auto* invalid : {"-1", "4", "invalid", "1x"}) {
            Build();
            ASSERT_EQ(TextAttr(nodes.at("m"), key, invalid), GRAPH_SUCCESS);
            EXPECT_EQ(Run(), GRAPH_NOT_CHANGED) << key << "/" << invalid;
        }
        Build();
        ASSERT_EQ(TextAttr(nodes.at("m"), key, "1"), GRAPH_SUCCESS);
        EXPECT_EQ(Run(), GRAPH_SUCCESS);
    }
}

TEST_F(FixpipeFusionTest, NonzeroProducerOutputPorts)
{
    Build();
    auto source = Builder(graph.get())
                      .Name("multiple_outputs")
                      .OpType("Split")
                      .IrDefOutputs(
                          {{"first", Builder::kEsIrOutputRequired, ""}, {"second", Builder::kEsIrOutputRequired, ""}})
                      .InstanceOutputDataType("first", DT_UINT64)
                      .InstanceOutputDataType("second", DT_UINT64)
                      .InstanceOutputShape("first", {1})
                      .InstanceOutputShape("second", {1})
                      .Build();
    ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddDataEdge(source, 1, nodes.at("q"), 1), GRAPH_SUCCESS);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    EXPECT_EQ(Target()->GetInDataNodesAndPortIndexs(2).second, 1);
}

TEST_F(FixpipeFusionTest, SharedDequantFallsBackToTwoNodes)
{
    Case c;
    c.three = true;
    Build(c);
    auto extra = Make("extra", "Identity", Desc(DT_FLOAT16), 1);
    Link(nodes.at("q"), extra);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    ExpectSharedDequantFallback(extra);
}

TEST_F(FixpipeFusionTest, SharedDequantWithSeveralQuantConsumersFallsBack)
{
    Case c;
    c.three = true;
    Build(c);
    auto quant = Make("second_quant", "AscendQuant", Desc(DT_INT8), 1);
    Link(nodes.at("q"), quant);
    float scale = 0.5f, offset = 2.0f;
    ASSERT_EQ(quant.SetAttr("scale", scale), GRAPH_SUCCESS);
    ASSERT_EQ(quant.SetAttr("offset", offset), GRAPH_SUCCESS);
    auto user = Make("second_consumer", "Identity", Desc(DT_INT8), 1);
    Link(quant, user);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    ExpectSharedDequantFallback(quant);
    float value = 0;
    ASSERT_EQ(quant.GetAttr("scale", value), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(value, scale);
    ASSERT_EQ(quant.GetAttr("offset", value), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(value, offset);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
}

TEST_F(FixpipeFusionTest, UnusedThreeNodeScaleIsRemoved)
{
    Case c;
    c.three = true;
    Build(c);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    for (const auto& node : graph->GetDirectNode()) {
        AscendString name;
        ASSERT_EQ(node.GetName(name), GRAPH_SUCCESS);
        EXPECT_NE(name, "scale");
    }
}

TEST_F(FixpipeFusionTest, OutgoingControlDependsOnFusedNode)
{
    Build();
    auto extra = Make("extra", "NoOp", Desc(DT_FLOAT));
    ASSERT_EQ(graph->AddControlEdge(nodes.at("m"), extra), GRAPH_SUCCESS);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    ASSERT_NE(Target(), nullptr);
    EXPECT_TRUE(HasControlPath(*Target(), extra));
}

TEST_F(FixpipeFusionTest, ConstCleanupPreservesDataAndControlUsers)
{
    for (const auto* type : {"Const", "Constant"}) {
        for (int sharing = 0; sharing < 4; ++sharing) {
            SCOPED_TRACE(::testing::Message() << type << "/" << sharing);
            Case c;
            c.three = true;
            Build(c);
            Tensor value;
            ASSERT_EQ(nodes.at("scale").GetAttr("value", value), GRAPH_SUCCESS);
            auto scale = Make("replacement_scale", type, value.GetTensorDesc());
            ASSERT_EQ(scale.SetAttr("value", value), GRAPH_SUCCESS);
            ASSERT_EQ(graph->RemoveNode(nodes.at("scale")), GRAPH_SUCCESS);
            Link(scale, nodes.at("q"), 1);
            auto dataUser = Make("scale_data_user", "Identity", value.GetTensorDesc(), 1);
            auto controlUser = Make("scale_control_user", "NoOp", Desc(DT_FLOAT));
            if ((sharing & 1) != 0) {
                Link(scale, dataUser);
            }
            if ((sharing & 2) != 0) {
                ASSERT_EQ(graph->AddControlEdge(scale, controlUser), GRAPH_SUCCESS);
            }
            ASSERT_EQ(Run(), GRAPH_SUCCESS);
            EXPECT_EQ(graph->FindNodeByName("replacement_scale") != nullptr, sharing != 0);
            if ((sharing & 1) != 0) {
                ASSERT_NE(dataUser.GetInDataNodesAndPortIndexs(0).first, nullptr);
                AscendString name;
                ASSERT_EQ(dataUser.GetInDataNodesAndPortIndexs(0).first->GetName(name), GRAPH_SUCCESS);
                EXPECT_EQ(name, "replacement_scale");
            }
            if ((sharing & 2) != 0) {
                const auto controls = controlUser.GetInControlNodes();
                ASSERT_EQ(controls.size(), 1U);
                AscendString name;
                ASSERT_EQ(controls.front()->GetName(name), GRAPH_SUCCESS);
                EXPECT_EQ(name, "replacement_scale");
            }
            EXPECT_EQ(ScaleValues(), std::vector<uint64_t>{140601025200128ULL});
        }
    }
}

TEST_F(FixpipeFusionTest, SharedDequantFallbackPreservesDataAndControlDependencies)
{
    for (int outputPort : {0, 1}) {
        for (bool existingControl : {false, true}) {
            SCOPED_TRACE(::testing::Message() << outputPort << "/" << existingControl);
            Case c;
            c.three = true;
            Build(c);
            auto source = Builder(graph.get())
                              .Name("multiple_a")
                              .OpType("Split")
                              .IrDefOutputs({{"first", Builder::kEsIrOutputRequired, ""},
                                             {"second", Builder::kEsIrOutputRequired, ""}})
                              .InstanceOutputDataType("first", DT_INT8)
                              .InstanceOutputDataType("second", DT_INT8)
                              .InstanceOutputShape("first", {16, 64})
                              .InstanceOutputShape("second", {16, 64})
                              .Build();
            ASSERT_EQ(graph->RemoveEdge(nodes.at("a"), 0, nodes.at("m"), 0), GRAPH_SUCCESS);
            ASSERT_EQ(graph->AddDataEdge(source, outputPort, nodes.at("m"), 0), GRAPH_SUCCESS);
            auto dataUser = Make("dequant_data_user", "Add", Desc(DT_FLOAT16), 2);
            Link(nodes.at("q"), dataUser, 0);
            Link(nodes.at("q"), dataUser, 1);
            auto first = Make("first_control_user", "NoOp", Desc(DT_FLOAT));
            auto second = Make("second_control_user", "NoOp", Desc(DT_FLOAT));
            for (auto* user : {&first, &second}) {
                ASSERT_EQ(graph->AddControlEdge(nodes.at("q"), *user), GRAPH_SUCCESS);
                if (existingControl) {
                    ASSERT_EQ(graph->AddControlEdge(source, *user), GRAPH_SUCCESS);
                }
            }
            ASSERT_EQ(Run(), GRAPH_SUCCESS);
            for (int port : {0, 1}) {
                ExpectSharedDequantFallback(dataUser, port);
            }
            auto fused = FindFusedTarget();
            ASSERT_NE(fused, nullptr);
            for (const auto* user : {&first, &second}) {
                EXPECT_TRUE(HasControlPath(*fused, *user));
                if (existingControl) {
                    EXPECT_TRUE(HasControlPath(source, *user));
                }
            }
        }
    }
}

TEST_F(FixpipeFusionTest, AllConstInputsPreserveControlDependencies)
{
    for (bool three : {false, true}) {
        Case c;
        c.three = three;
        Build(c);
        for (int port : {0, 1}) {
            auto input = Make(port == 0 ? "const_a" : "const_b", "Const", Desc(DT_INT8));
            auto original = nodes.at("m").GetInDataNodesAndPortIndexs(port);
            ASSERT_NE(original.first, nullptr);
            ASSERT_EQ(graph->RemoveEdge(*original.first, original.second, nodes.at("m"), port), GRAPH_SUCCESS);
            Link(input, nodes.at("m"), port);
        }
        auto before = Make("before", "NoOp", Desc(DT_FLOAT));
        auto after = Make("after", "NoOp", Desc(DT_FLOAT));
        ASSERT_EQ(graph->AddControlEdge(before, nodes.at("m")), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddControlEdge(nodes.at("m"), after), GRAPH_SUCCESS);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        ASSERT_NE(Target(), nullptr);
        EXPECT_TRUE(HasControlPath(before, *Target()));
        EXPECT_TRUE(HasControlPath(*Target(), after));
        EXPECT_NE(graph->FindNodeByName("const_a"), nullptr);
        EXPECT_NE(graph->FindNodeByName("const_b"), nullptr);
    }
}

TEST_F(FixpipeFusionTest, InheritedCompilationAttributes)
{
    for (bool three : {false, true}) {
        Case c;
        c.three = three;
        Build(c);
        ASSERT_EQ(TextAttr(nodes.at("m"), "_stream_label", "stream"), GRAPH_SUCCESS);
        ASSERT_EQ(TextAttr(nodes.at("q"), "_stream_label", "stream"), GRAPH_SUCCESS);
        if (three) {
            ASSERT_EQ(TextAttr(nodes.at("last"), "_stream_label", "stream"), GRAPH_SUCCESS);
        }
        ASSERT_EQ(TextAttr(nodes.at("q"), "_op_compile_strategy", "strategy"), GRAPH_SUCCESS);
        int64_t keep = 1;
        int64_t mode = 4;
        bool enabled = true;
        ASSERT_EQ(nodes.at("m").SetAttr("_keep_dtype", keep), GRAPH_SUCCESS);
        ASSERT_EQ(nodes.at("m").SetAttr("_op_custom_impl_mode_enum", mode), GRAPH_SUCCESS);
        ASSERT_EQ(nodes.at("m").SetAttr("_recompute", enabled), GRAPH_SUCCESS);
        GNodePtr reference;
        BuildFrameworkAttributeReference(reference);
        ASSERT_NE(reference, nullptr);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        std::vector<GNodePtr> outputs;
        outputs.reserve(2);
        outputs.push_back(Target());
        if (three)
            outputs.push_back(Target()->GetInDataNodesAndPortIndexs(2).first);
        for (const auto& node : outputs) {
            AscendString text;
            ExpectFrameworkCompilationAttributes(*node, *reference);
            ASSERT_EQ(node->GetAttr("_op_compile_strategy", text), GRAPH_SUCCESS);
            EXPECT_EQ(text, "strategy");
            ASSERT_EQ(node->GetAttr("_keep_dtype", keep), GRAPH_SUCCESS);
            EXPECT_EQ(keep, 1);
            ASSERT_EQ(node->GetAttr("_op_custom_impl_mode_enum", mode), GRAPH_SUCCESS);
            EXPECT_EQ(mode, 4);
            ASSERT_EQ(node->GetAttr("_recompute", enabled), GRAPH_SUCCESS);
            EXPECT_TRUE(enabled);
        }
    }
}

TEST_F(FixpipeFusionTest, MissingScaleProducerDoesNotFuse)
{
    Build();
    ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
}

TEST_F(FixpipeFusionTest, ThreeNodeRuntimeScaleFallsBackToDequant)
{
    Case c;
    c.three = true;
    Build(c);
    auto runtimeScale = Make("runtime_scale", "Data", Desc(DT_UINT64, FORMAT_ND, {1}));
    ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
    Link(runtimeScale, nodes.at("q"), 1);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    AscendString type;
    ASSERT_EQ(Target()->GetType(type), GRAPH_SUCCESS);
    EXPECT_EQ(type, "AscendQuant");
    auto fused = Target()->GetInDataNodesAndPortIndexs(0).first;
    ASSERT_NE(fused, nullptr);
    ASSERT_EQ(fused->GetType(type), GRAPH_SUCCESS);
    EXPECT_EQ(type, "QuantBatchMatmulV3");
}

TEST_F(FixpipeFusionTest, ExtraUnconnectedInputSlotIsNotCounted)
{
    Case c;
    c.inputSlots = 4;
    Build(c);
    EXPECT_EQ(Run(), GRAPH_SUCCESS);
}

TEST_F(FixpipeFusionTest, RuntimeVersionGate)
{
    Build();
    int32_t version = 0;
    char component[] = "ge-compiler";
    const bool supported = aclsysGetVersionNum(component, &version) == 0 && version >= 90200000 &&
                           TestPass::HasRequiredPublicApis();
    CustomPassContext context;
    const auto nodeCount = graph->GetAllNodes().size();
    EXPECT_EQ(pass.Run(graph, context), supported ? GRAPH_SUCCESS : GRAPH_NOT_CHANGED);
    if (!supported) {
        EXPECT_EQ(graph->GetAllNodes().size(), nodeCount);
        EXPECT_NE(graph->FindNodeByName("m"), nullptr);
        EXPECT_NE(graph->FindNodeByName("q"), nullptr);
    }
}

TEST_F(FixpipeFusionTest, ScaleProducerWeightsFollowLegacyLookup)
{
    for (bool nested : {false, true}) {
        Case c;
        c.three = true;
        Build(c);
        auto scaleDesc = Desc(DT_UINT64, FORMAT_ND, {1});
        auto identity = Make("scale_identity", "Identity", scaleDesc, 1);
        ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
        Link(nodes.at("scale"), identity);
        if (nested) {
            auto second = Make("scale_second_identity", "Identity", scaleDesc, 1);
            Link(identity, second);
            Link(second, nodes.at("q"), 1);
        } else {
            Link(identity, nodes.at("q"), 1);
        }
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        AscendString type;
        ASSERT_EQ(Target()->GetType(type), GRAPH_SUCCESS);
        EXPECT_EQ(type, nested ? "AscendQuant" : "QuantBatchMatmulV3");
        if (!nested) {
            EXPECT_EQ(ScaleValues(), std::vector<uint64_t>{140601025200128ULL});
        }
    }
}

TEST_F(FixpipeFusionTest, BackwardRequiresBackwardInput)
{
    for (bool three : {false, true}) {
        for (bool marked : {false, true}) {
            for (bool inputMarked : {false, true}) {
                Case c;
                c.three = three;
                Build(c);
                ASSERT_EQ(nodes.at("m").SetAttr("_backward", marked), GRAPH_SUCCESS);
                ASSERT_EQ(nodes.at("a").SetAttr("_backward", inputMarked), GRAPH_SUCCESS);
                ASSERT_EQ(Run(), GRAPH_SUCCESS);
                bool actual = false;
                (void)Target()->GetAttr("_backward", actual);
                EXPECT_EQ(actual, marked && inputMarked);
            }
        }
    }
}

TEST_F(FixpipeFusionTest, BackwardCanComeFromControlPredecessor)
{
    Build();
    bool enabled = true;
    auto control = Make("backward_control", "NoOp", Desc(DT_FLOAT));
    ASSERT_EQ(control.SetAttr("_backward", enabled), GRAPH_SUCCESS);
    ASSERT_EQ(nodes.at("m").SetAttr("_backward", enabled), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddControlEdge(control, nodes.at("m")), GRAPH_SUCCESS);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    bool actual = false;
    ASSERT_EQ(Target()->GetAttr("_backward", actual), GRAPH_SUCCESS);
    EXPECT_TRUE(actual);
}

TEST_F(FixpipeFusionTest, DumpOutputMappingIsPreserved)
{
    for (bool three : {false, true}) {
        for (bool existing : {false, true}) {
            Case c;
            c.three = three;
            Build(c);
            auto& tail = nodes.at(three ? "last" : "q");
            if (existing) {
                ASSERT_EQ(
                    tail.SetOutputAttr("_datadump_origin_name", 0, ge::es::CreateFrom(AscendString("original_output"))),
                    GRAPH_SUCCESS);
                ASSERT_EQ(tail.SetOutputAttr("_datadump_origin_output_index", 0, ge::es::CreateFrom(int64_t{7})),
                          GRAPH_SUCCESS);
                ASSERT_EQ(
                    tail.SetOutputAttr("_datadump_origin_data_type", 0, ge::es::CreateFrom(AscendString("DT_INT32"))),
                    GRAPH_SUCCESS);
                ASSERT_EQ(tail.SetOutputAttr("_datadump_origin_format", 0, ge::es::CreateFrom(AscendString("NCHW"))),
                          GRAPH_SUCCESS);
            }
            ASSERT_EQ(Run(), GRAPH_SUCCESS);
            AttrValue value;
            AscendString text;
            int64_t index = -1;
            ASSERT_EQ(Target()->GetOutputAttr("_datadump_origin_name", 0, value), GRAPH_SUCCESS);
            ASSERT_EQ(value.GetAttrValue(text), GRAPH_SUCCESS);
            EXPECT_EQ(text, existing ? "original_output" : (three ? "last" : "q"));
            ASSERT_EQ(Target()->GetOutputAttr("_datadump_origin_output_index", 0, value), GRAPH_SUCCESS);
            ASSERT_EQ(value.GetAttrValue(index), GRAPH_SUCCESS);
            EXPECT_EQ(index, existing ? 7 : 0);
            ASSERT_EQ(Target()->GetOutputAttr("_datadump_origin_format", 0, value), GRAPH_SUCCESS);
            ASSERT_EQ(value.GetAttrValue(text), GRAPH_SUCCESS);
            EXPECT_EQ(text, existing ? "NCHW" : "ND");
            ASSERT_EQ(Target()->GetOutputAttr("_datadump_origin_data_type", 0, value), GRAPH_SUCCESS);
            ASSERT_EQ(value.GetAttrValue(text), GRAPH_SUCCESS);
            EXPECT_EQ(text, existing ? "DT_INT32" : "RESERVED");
            bool multi = false;
            (void)Target()->GetAttr("_datadump_is_multiop", multi);
            EXPECT_EQ(multi, three);
        }
    }
}

TEST_F(FixpipeFusionTest, UnusedOutputDoesNotCreateDumpMapping)
{
    const std::array<const char*, 4> keys = {"_datadump_origin_name", "_datadump_origin_output_index",
                                             "_datadump_origin_format", "_datadump_origin_data_type"};
    for (int mode = 0; mode < 5; ++mode) {
        for (bool controlOnly : {false, true}) {
            for (int metadata = 0; metadata < 3; ++metadata) {
                SCOPED_TRACE(::testing::Message() << mode << "/" << controlOnly << "/" << metadata);
                Case c;
                c.three = mode == 2;
                if (mode == 0 || mode >= 3) {
                    c.successor = "AscendDequant";
                    c.output = DT_FLOAT16;
                }
                if (mode >= 3) {
                    c.aType = mode == 3 ? DT_FLOAT16 : DT_FLOAT;
                }
                Build(c);
                auto& tail = nodes.at(c.three ? "last" : "q");
                ASSERT_EQ(graph->RemoveNode(nodes.at("consumer")), GRAPH_SUCCESS);
                if (controlOnly) {
                    auto control = Make("control", "NoOp", Desc(DT_INT8));
                    ASSERT_EQ(graph->AddControlEdge(tail, control), GRAPH_SUCCESS);
                }
                if (metadata != 0) {
                    ASSERT_EQ(tail.SetOutputAttr(
                                  keys[0], 0, ge::es::CreateFrom(AscendString(metadata == 1 ? "original_output" : ""))),
                              GRAPH_SUCCESS);
                }
                if (metadata == 1) {
                    ASSERT_EQ(tail.SetOutputAttr(keys[1], 0, ge::es::CreateFrom(int64_t{7})), GRAPH_SUCCESS);
                    ASSERT_EQ(tail.SetOutputAttr(keys[2], 0, ge::es::CreateFrom(AscendString("NCHW"))), GRAPH_SUCCESS);
                    ASSERT_EQ(tail.SetOutputAttr(keys[3], 0, ge::es::CreateFrom(AscendString("DT_INT32"))),
                              GRAPH_SUCCESS);
                }
                ASSERT_EQ(Run(), GRAPH_SUCCESS);
                auto target = FindFusedTarget();
                ASSERT_NE(target, nullptr);
                // Without data consumers, legacy does not copy even existing output dump attributes.
                for (const auto* key : keys) {
                    AttrValue value;
                    EXPECT_NE(target->GetOutputAttr(key, 0, value), GRAPH_SUCCESS) << key;
                }
            }
        }
    }
}

TEST_F(FixpipeFusionTest, GraphOutputIsADataConsumerForDumpMapping)
{
    for (bool three : {false, true}) {
        for (bool shared : {false, true}) {
            SCOPED_TRACE(::testing::Message() << three << "/" << shared);
            Case c;
            c.three = three;
            Build(c);
            auto& tail = nodes.at(three ? "last" : "q");
            auto output = Builder(graph.get())
                              .Name("net_output")
                              .OpType("NetOutput")
                              .IrDefInputs({{"x", Builder::kEsIrInputRequired, ""}})
                              .Build();
            Link(tail, output);
            if (!shared) {
                ASSERT_EQ(graph->RemoveNode(nodes.at("consumer")), GRAPH_SUCCESS);
            }
            ASSERT_EQ(Run(), GRAPH_SUCCESS);
            auto target = output.GetInDataNodesAndPortIndexs(0).first;
            ASSERT_NE(target, nullptr);
            if (shared) {
                AscendString name;
                ASSERT_EQ(Target()->GetName(name), GRAPH_SUCCESS);
                EXPECT_EQ(std::string(name.GetString()).find("mquantBatchmatmulv3_"), 0U);
            }
            AttrValue value;
            AscendString name;
            ASSERT_EQ(target->GetOutputAttr("_datadump_origin_name", 0, value), GRAPH_SUCCESS);
            ASSERT_EQ(value.GetAttrValue(name), GRAPH_SUCCESS);
            EXPECT_EQ(name, three ? "last" : "q");
        }
    }
}

TEST_F(FixpipeFusionTest, OutputAndScaleDescriptors)
{
    for (bool three : {false, true}) {
        for (DataType type : {DT_INT8, DT_FLOAT16, DT_FLOAT}) {
            if (three && type != DT_INT8) {
                continue;
            }
            Case c;
            c.three = three;
            c.output = type;
            Build(c);
            ASSERT_EQ(Run(), GRAPH_SUCCESS);
            TensorDesc desc;
            ASSERT_EQ(Target()->GetOutputDesc(0, desc), GRAPH_SUCCESS);
            EXPECT_EQ(desc.GetDataType(), type);
            ExpectOriginDataType(desc, type);
            EXPECT_EQ(desc.GetFormat(), FORMAT_ND);
            EXPECT_EQ(desc.GetOriginFormat(), FORMAT_ND);
            EXPECT_EQ(desc.GetShape().GetDims(), (std::vector<int64_t>{16, 64}));
            EXPECT_EQ(desc.GetOriginShape().GetDims(), desc.GetShape().GetDims());
            if (three) {
                auto scale = Target()->GetInDataNodesAndPortIndexs(2).first;
                ASSERT_NE(scale, nullptr);
                ASSERT_EQ(scale->GetOutputDesc(0, desc), GRAPH_SUCCESS);
                EXPECT_EQ(desc.GetDataType(), DT_UINT64);
                ExpectOriginDataType(desc, DT_UINT64);
                Tensor value;
                ASSERT_EQ(scale->GetAttr("value", value), GRAPH_SUCCESS);
                EXPECT_EQ(value.GetTensorDesc().GetDataType(), DT_UINT64);
                ExpectOriginDataType(value.GetTensorDesc(), DT_UINT64);
                EXPECT_EQ(value.GetTensorDesc().GetShape().GetDims(), desc.GetShape().GetDims());
                EXPECT_EQ(value.GetSize(), c.scale.size() * sizeof(uint64_t));
            }
        }
    }
}

TEST_F(FixpipeFusionTest, ScaleWeightsFromConstantAndPlaceHolder)
{
    for (const auto* type : {"Constant", "PlaceHolder"}) {
        Case c;
        c.three = true;
        Build(c);
        auto desc = Desc(DT_UINT64, FORMAT_ND, {1});
        auto scale = Make("alternate_scale", type, desc);
        Tensor value(desc, reinterpret_cast<const uint8_t*>(c.scale.data()), sizeof(uint64_t));
        ASSERT_EQ(scale.SetAttr("value", value), GRAPH_SUCCESS);
        ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
        Link(scale, nodes.at("q"), 1);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        AscendString resultType;
        ASSERT_EQ(Target()->GetType(resultType), GRAPH_SUCCESS);
        EXPECT_EQ(resultType, "QuantBatchMatmulV3");
        EXPECT_EQ(ScaleValues(), std::vector<uint64_t>{140601025200128ULL});
    }
}

TEST_F(FixpipeFusionTest, ScaleProducerUsesFirstConstInputInPortOrder)
{
    Case c;
    c.three = true;
    Build(c);
    auto desc = Desc(DT_UINT64, FORMAT_ND, {1});
    auto producer = Make("scale_producer", "Add", desc, 2);
    auto runtime = Make("runtime_scale", "Data", desc);
    ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
    Link(runtime, producer, 0);
    Link(nodes.at("scale"), producer, 1);
    Link(producer, nodes.at("q"), 1);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    EXPECT_EQ(ScaleValues(), std::vector<uint64_t>{140601025200128ULL});
}

TEST_F(FixpipeFusionTest, InvalidAdditionalConstPreventsScaleEncoding)
{
    Case c;
    c.three = true;
    Build(c);
    auto desc = Desc(DT_UINT64, FORMAT_ND, {1});
    auto producer = Make("scale_producer", "Add", desc, 2);
    auto invalid = Make("invalid_const", "Const", desc);
    ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
    Link(nodes.at("scale"), producer, 0);
    Link(invalid, producer, 1);
    Link(producer, nodes.at("q"), 1);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    AscendString type;
    ASSERT_EQ(Target()->GetType(type), GRAPH_SUCCESS);
    EXPECT_EQ(type, "AscendQuant");
}

TEST_F(FixpipeFusionTest, ScaleProducerFindsConstAfterMissingOptionalInputs)
{
    for (const int32_t scalePort : {1, 3}) {
        Case c;
        c.three = true;
        Build(c);
        const auto desc = Desc(DT_UINT64, FORMAT_ND, {1});
        auto producer = Builder(graph.get())
                            .Name("scale_producer")
                            .OpType("OptionalScaleProducer")
                            .IrDefInputs(
                                {{"aux0", Builder::kEsIrInputOptional, ""},
                                 {"input1", scalePort == 1 ? Builder::kEsIrInputRequired : Builder::kEsIrInputOptional,
                                  ""},
                                 {"aux2", Builder::kEsIrInputOptional, ""},
                                 {"input3", scalePort == 3 ? Builder::kEsIrInputRequired : Builder::kEsIrInputOptional,
                                  ""}})
                            .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                            .InstanceOutputDataType("y", DT_UINT64)
                            .InstanceOutputShape("y", {1})
                            .Build();
        ASSERT_EQ(producer.UpdateOutputDesc(0, desc), GRAPH_SUCCESS);
        ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
        Link(nodes.at("scale"), producer, scalePort);
        Link(producer, nodes.at("q"), 1);
        ASSERT_EQ(producer.GetInputsSize(), 1U);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        EXPECT_EQ(ScaleValues(), std::vector<uint64_t>{140601025200128ULL}) << scalePort;
    }
}

TEST_F(FixpipeFusionTest, SparseScaleProducerUsesFirstConstInPortOrder)
{
    Case c;
    c.three = true;
    Build(c);
    const auto desc = Desc(DT_UINT64, FORMAT_ND, {1});
    // Create the higher-port Const first to distinguish port order from graph order.
    const uint64_t otherScale = 0x3f800000ULL;
    Tensor otherValue(desc, reinterpret_cast<const uint8_t*>(&otherScale), sizeof(otherScale));
    ASSERT_EQ(nodes.at("scale").SetAttr("value", otherValue), GRAPH_SUCCESS);
    auto first = Make("first_scale", "Const", desc);
    Tensor firstValue(desc, reinterpret_cast<const uint8_t*>(c.scale.data()), sizeof(uint64_t));
    ASSERT_EQ(first.SetAttr("value", firstValue), GRAPH_SUCCESS);
    auto producer = Builder(graph.get())
                        .Name("scale_producer")
                        .OpType("OptionalScaleProducer")
                        .IrDefInputs({{"aux0", Builder::kEsIrInputOptional, ""},
                                      {"first", Builder::kEsIrInputRequired, ""},
                                      {"aux2", Builder::kEsIrInputOptional, ""},
                                      {"second", Builder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                        .InstanceOutputDataType("y", DT_UINT64)
                        .InstanceOutputShape("y", {1})
                        .Build();
    ASSERT_EQ(producer.UpdateOutputDesc(0, desc), GRAPH_SUCCESS);
    ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
    Link(first, producer, 1);
    Link(nodes.at("scale"), producer, 3);
    Link(producer, nodes.at("q"), 1);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    EXPECT_EQ(ScaleValues(), std::vector<uint64_t>{140601025200128ULL});
}

TEST_F(FixpipeFusionTest, SparseScaleProducerValidatesLaterConst)
{
    Case c;
    c.three = true;
    Build(c);
    const auto desc = Desc(DT_UINT64, FORMAT_ND, {1});
    auto invalid = Make("invalid_const", "Const", desc);
    auto producer = Builder(graph.get())
                        .Name("scale_producer")
                        .OpType("OptionalScaleProducer")
                        .IrDefInputs({{"first", Builder::kEsIrInputRequired, ""},
                                      {"aux1", Builder::kEsIrInputOptional, ""},
                                      {"aux2", Builder::kEsIrInputOptional, ""},
                                      {"second", Builder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", Builder::kEsIrOutputRequired, ""}})
                        .InstanceOutputDataType("y", DT_UINT64)
                        .InstanceOutputShape("y", {1})
                        .Build();
    ASSERT_EQ(producer.UpdateOutputDesc(0, desc), GRAPH_SUCCESS);
    ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
    Link(nodes.at("scale"), producer, 0);
    Link(invalid, producer, 3);
    Link(producer, nodes.at("q"), 1);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    AscendString type;
    ASSERT_EQ(Target()->GetType(type), GRAPH_SUCCESS);
    EXPECT_EQ(type, "AscendQuant");
}

TEST_F(FixpipeFusionTest, DumpMappingFallsBackToOriginalNameAndFormat)
{
    Build();
    std::vector<AscendString> names = {AscendString("original_q"), AscendString("older_q")};
    ASSERT_EQ(nodes.at("q").SetAttr("_datadump_original_op_names", names), GRAPH_SUCCESS);
    ASSERT_EQ(nodes.at("q").SetOutputAttr("_datadump_origin_name", 0, ge::es::CreateFrom(AscendString(""))),
              GRAPH_SUCCESS);
    ASSERT_EQ(nodes.at("q").SetOutputAttr("_datadump_origin_format", 0, ge::es::CreateFrom(AscendString("RESERVED"))),
              GRAPH_SUCCESS);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    AttrValue value;
    AscendString text;
    ASSERT_EQ(Target()->GetOutputAttr("_datadump_origin_name", 0, value), GRAPH_SUCCESS);
    ASSERT_EQ(value.GetAttrValue(text), GRAPH_SUCCESS);
    EXPECT_EQ(text, "original_q");
    ASSERT_EQ(Target()->GetOutputAttr("_datadump_origin_format", 0, value), GRAPH_SUCCESS);
    ASSERT_EQ(value.GetAttrValue(text), GRAPH_SUCCESS);
    EXPECT_EQ(text, "ND");
}
TEST_F(FixpipeFusionTest, SubgraphScaleDataAndWhileLookupFollowLegacy)
{
    for (bool whileBody : {false, true}) {
        for (bool identity : {false, true}) {
            Case c;
            c.three = true;
            Build(c);
            auto body = graph;
            Tensor value;
            ASSERT_EQ(nodes.at("scale").GetAttr("value", value), GRAPH_SUCCESS);
            const auto desc = value.GetTensorDesc();
            auto data = Make("body_scale", "Data", desc);
            int64_t parentIndex = 0;
            ASSERT_EQ(data.SetAttr("_parent_node_index", parentIndex), GRAPH_SUCCESS);
            ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
            if (identity) {
                auto id = Make("body_scale_identity", "Identity", desc, 1);
                Link(data, id);
                Link(id, nodes.at("q"), 1);
            } else {
                Link(data, nodes.at("q"), 1);
            }
            graph = std::make_shared<Graph>("fixpipe_parent");
            auto outer = graph;
            auto scale = Make("outer_scale", "Const", desc);
            ASSERT_EQ(scale.SetAttr("value", value), GRAPH_SUCCESS);
            auto wrapper = Make("wrapper", whileBody ? "While" : "PartitionedCall", Desc(DT_INT8), 1);
            Link(scale, wrapper);
            ASSERT_EQ(wrapper.SetSubgraph("body", *body), GRAPH_SUCCESS);
            graph = body;
            ASSERT_EQ(Run(), GRAPH_SUCCESS);
            AscendString type;
            ASSERT_EQ(Target()->GetType(type), GRAPH_SUCCESS);
            EXPECT_EQ(type, whileBody && !identity ? "AscendQuant" : "QuantBatchMatmulV3");
        }
    }
}
TEST_F(FixpipeFusionTest, DumpDefaultUsesOriginalDataType)
{
    for (bool three : {false, true}) {
        Case c;
        c.three = three;
        Build(c);
        auto& tail = nodes.at(three ? "last" : "q");
        TensorDesc desc;
        ASSERT_EQ(tail.GetOutputDesc(0, desc), GRAPH_SUCCESS);
        SetPublicOriginDataType(desc, DT_FLOAT16);
        ASSERT_EQ(tail.UpdateOutputDesc(0, desc), GRAPH_SUCCESS);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        AttrValue value;
        AscendString dtype;
        ASSERT_EQ(Target()->GetOutputAttr("_datadump_origin_data_type", 0, value), GRAPH_SUCCESS);
        ASSERT_EQ(value.GetAttrValue(dtype), GRAPH_SUCCESS);
        EXPECT_EQ(dtype, "DT_FLOAT16");
    }
}

TEST_F(FixpipeFusionTest, AliasedMatmulInputsKeepBoundaryRoles)
{
    for (bool three : {false, true}) {
        for (bool bias : {false, true}) {
            Case c;
            c.three = three;
            c.bias = bias;
            Build(c);
            ASSERT_EQ(graph->RemoveEdge(nodes.at("b"), 0, nodes.at("m"), 1), GRAPH_SUCCESS);
            Link(nodes.at("a"), nodes.at("m"), 1);
            ASSERT_EQ(Run(), GRAPH_SUCCESS);
            auto first = Target()->GetInDataNodesAndPortIndexs(0).first;
            auto second = Target()->GetInDataNodesAndPortIndexs(1).first;
            ASSERT_NE(first, nullptr);
            ASSERT_NE(second, nullptr);
            AscendString a, b;
            first->GetName(a);
            second->GetName(b);
            EXPECT_EQ(a, "a");
            EXPECT_EQ(b, "a");
        }
    }
}

TEST_F(FixpipeFusionTest, SharedDequantWithConstantAFallsBackToTwoNodes)
{
    Case c;
    c.three = true;
    Build(c);
    auto constant = Make("constant_a", "Const", Desc(DT_INT8));
    ASSERT_EQ(graph->RemoveEdge(nodes.at("a"), 0, nodes.at("m"), 0), GRAPH_SUCCESS);
    Link(constant, nodes.at("m"));
    auto user = Make("shared_user", "Identity", Desc(DT_FLOAT16), 1);
    Link(nodes.at("q"), user);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    ExpectSharedDequantFallback(user);
}

TEST_F(FixpipeFusionTest, StreamLabelDifferencesDoNotBlockOtherMatches)
{
    Case c;
    c.three = true;
    Build(c);
    ASSERT_EQ(TextAttr(nodes.at("m"), "_stream_label", "first"), GRAPH_SUCCESS);
    ASSERT_EQ(TextAttr(nodes.at("q"), "_stream_label", "second"), GRAPH_SUCCESS);
    auto a = Make("good_a", "Data", Desc(DT_INT8));
    auto b = Make("good_b", "Data", Desc(DT_INT8));
    auto m = Make("good_m", "MatMul", Desc(DT_INT32), 3);
    Link(a, m, 0);
    Link(b, m, 1);
    bool transpose = false;
    ASSERT_EQ(m.SetAttr("transpose_x1", transpose), GRAPH_SUCCESS);
    ASSERT_EQ(m.SetAttr("transpose_x2", transpose), GRAPH_SUCCESS);
    auto dequant = Make("good_dequant", "AscendDequant", Desc(DT_FLOAT16), 2);
    Link(m, dequant);
    Link(nodes.at("scale"), dequant, 1);
    auto quant = Make("good_quant", "AscendQuant", Desc(DT_INT8), 1);
    Link(dequant, quant);
    ASSERT_EQ(quant.SetAttr("scale", c.factor), GRAPH_SUCCESS);
    ASSERT_EQ(quant.SetAttr("offset", c.offset), GRAPH_SUCCESS);
    auto user = Make("good_consumer", "Identity", Desc(DT_INT8), 1);
    Link(quant, user);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    size_t count = 0;
    for (const auto& node : graph->GetDirectNode()) {
        AscendString type;
        node.GetType(type);
        count += type == "QuantBatchMatmulV3";
    }
    EXPECT_EQ(count, 2U);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    Build(c);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    CustomPassContext context;
    GraphPtr invalid;
    ASSERT_EQ(pass.Fuse(invalid, context), GRAPH_NOT_CHANGED);
}

TEST_F(FixpipeFusionTest, InternalStreamLabelsUsePublicCanFuse)
{
    for (bool three : {false, true}) {
        for (int labels : {1, 2, 3}) {
            Case c;
            c.three = three;
            Build(c);
            if (labels & 1) {
                ASSERT_EQ(TextAttr(nodes.at("m"), "_stream_label", "matmul_stream"), GRAPH_SUCCESS);
            }
            if (labels & 2) {
                ASSERT_EQ(TextAttr(nodes.at("q"), "_stream_label", "dequant_stream"), GRAPH_SUCCESS);
            }
            EXPECT_EQ(Run(), GRAPH_SUCCESS);
        }
    }
}

TEST_F(FixpipeFusionTest, SeveralThreeNodeMatchesInOneGraph)
{
    Case c;
    c.three = true;
    Build(c);
    for (int index = 0; index < 3; ++index) {
        const std::string prefix = "additional_" + std::to_string(index);
        auto a = Make((prefix + "a").c_str(), "Data", Desc(DT_INT8));
        auto b = Make((prefix + "b").c_str(), "Data", Desc(DT_INT8));
        auto m = Make((prefix + "m").c_str(), "MatMul", Desc(DT_INT32), 3);
        Link(a, m, 0);
        Link(b, m, 1);
        bool transpose = false;
        ASSERT_EQ(m.SetAttr("transpose_x1", transpose), GRAPH_SUCCESS);
        ASSERT_EQ(m.SetAttr("transpose_x2", transpose), GRAPH_SUCCESS);
        auto dequant = Make((prefix + "dequant").c_str(), "AscendDequant", Desc(DT_FLOAT16), 2);
        Link(m, dequant);
        Link(nodes.at("scale"), dequant, 1);
        auto quant = Make((prefix + "quant").c_str(), "AscendQuant", Desc(DT_INT8), 1);
        Link(dequant, quant);
        float scale = 1, offset = 0;
        ASSERT_EQ(quant.SetAttr("scale", scale), GRAPH_SUCCESS);
        ASSERT_EQ(quant.SetAttr("offset", offset), GRAPH_SUCCESS);
        auto user = Make((prefix + "user").c_str(), "Identity", Desc(DT_INT8), 1);
        Link(quant, user);
    }
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    size_t count = 0;
    for (const auto& node : graph->GetDirectNode()) {
        AscendString type;
        node.GetType(type);
        count += type == "QuantBatchMatmulV3";
        EXPECT_NE(type, "AscendQuant");
        EXPECT_NE(type, "AscendDequant");
    }
    EXPECT_EQ(count, 4U);
    EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
}

TEST_F(FixpipeFusionTest, IncomingControlsAndUnrelatedParallelEdges)
{
    Build();
    for (int port : {0, 1}) {
        auto original = nodes.at("m").GetInDataNodesAndPortIndexs(port);
        auto constant = Make(port == 0 ? "const_a" : "const_b", "Const", Desc(DT_INT8));
        ASSERT_EQ(graph->RemoveEdge(*original.first, original.second, nodes.at("m"), port), GRAPH_SUCCESS);
        Link(constant, nodes.at("m"), port);
    }
    auto first = Make("created_first", "NoOp", Desc(DT_INT8));
    auto second = Make("created_second", "NoOp", Desc(DT_INT8));
    auto after = Make("after", "NoOp", Desc(DT_INT8));
    // Insertion order differs from node creation order.
    ASSERT_EQ(graph->AddControlEdge(second, nodes.at("m")), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddControlEdge(first, nodes.at("m")), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddControlEdge(nodes.at("m"), after), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddControlEdge(first, second), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddControlEdge(first, second), GRAPH_SUCCESS);
    ASSERT_EQ(Run(), GRAPH_SUCCESS);
    ASSERT_NE(Target(), nullptr);
    EXPECT_TRUE(HasControlPath(first, *Target()));
    EXPECT_TRUE(HasControlPath(second, *Target()));
    EXPECT_TRUE(HasControlPath(*Target(), after));
    EXPECT_EQ(second.GetInControlNodes().size(), 2U);
}

TEST_F(FixpipeFusionTest, FrameworkInheritsCommonExecutionAttributes)
{
    const std::vector<std::pair<const char*, const char*>> attributes = {
        {"_user_stream_label", "user_stream"}, {"_super_kernel_scope", "scope"},
        {"_super_kernel_options", "options"},  {"_op_aicore_num", "8"},
        {"_op_vectorcore_num", "4"},           {"_deterministic", "1"},
        {"_deterministic_level", "2"}};
    for (bool three : {false, true}) {
        Case c;
        c.three = three;
        Build(c);
        for (const auto& name :
             three ? std::vector<std::string>{"m", "q", "last"} : std::vector<std::string>{"m", "q"}) {
            for (const auto& attribute : attributes) {
                ASSERT_EQ(TextAttr(nodes.at(name), attribute.first, attribute.second), GRAPH_SUCCESS);
            }
        }
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        std::vector<GNodePtr> replacements = {Target()};
        ASSERT_NE(replacements.front(), nullptr);
        if (three) {
            replacements.push_back(Target()->GetInDataNodesAndPortIndexs(2).first);
        }
        for (const auto& node : replacements) {
            ASSERT_NE(node, nullptr);
            for (const auto& attribute : attributes) {
                AscendString value;
                ASSERT_EQ(node->GetAttr(attribute.first, value), GRAPH_SUCCESS);
                EXPECT_EQ(value, attribute.second);
            }
        }
    }
}

TEST_F(FixpipeFusionTest, FrameworkDefaultCompilationAttributeSelection)
{
    for (bool three : {false, true}) {
        for (int mask = 0; mask < (three ? 9 : 5); ++mask) {
            SCOPED_TRACE(std::to_string(three) + "/" + std::to_string(mask));
            Case c;
            c.three = three;
            Build(c);
            const bool same = mask == (three ? 8 : 4);
            const int selected = same ? 7 : mask;
            for (const auto& source : std::vector<std::pair<std::string, int>>{{"m", 1}, {"q", 2}, {"last", 4}}) {
                if ((!three && source.first == "last") || (selected & source.second) == 0) {
                    continue;
                }
                ASSERT_EQ(
                    TextAttr(nodes.at(source.first), "_op_compile_strategy", same ? "same" : source.first.c_str()),
                    GRAPH_SUCCESS);
                int64_t keep = same ? 1 : source.second;
                ASSERT_EQ(nodes.at(source.first).SetAttr("_keep_dtype", keep), GRAPH_SUCCESS);
            }
            bool enabled = true;
            ASSERT_EQ(nodes.at("m").SetAttr("_optimizer", enabled), GRAPH_SUCCESS);
            ASSERT_EQ(nodes.at("q").SetAttr("_op_debug_compile", enabled), GRAPH_SUCCESS);
            int64_t highPrecision = 0x4, floatExecution = 0x20;
            ASSERT_EQ(nodes.at("m").SetAttr("_op_custom_impl_mode_enum", highPrecision), GRAPH_SUCCESS);
            ASSERT_EQ(nodes.at("q").SetAttr("_op_custom_impl_mode_enum", floatExecution), GRAPH_SUCCESS);
            GNodePtr reference;
            BuildFrameworkAttributeReference(reference);
            ASSERT_NE(reference, nullptr);
            ASSERT_EQ(Run(), GRAPH_SUCCESS);
            std::vector<GNodePtr> replacements = {Target()};
            if (three) {
                replacements.push_back(Target()->GetInDataNodesAndPortIndexs(2).first);
            }
            for (const auto& node : replacements) {
                ASSERT_NE(node, nullptr);
                ExpectFrameworkCompilationAttributes(*node, *reference);
                int64_t mode = 0;
                ASSERT_EQ(node->GetAttr("_op_custom_impl_mode_enum", mode), GRAPH_SUCCESS);
                EXPECT_EQ(mode, floatExecution);
                for (const char* key : {"_optimizer", "_op_debug_compile"}) {
                    bool value = false;
                    ASSERT_EQ(node->GetAttr(key, value), GRAPH_SUCCESS);
                    EXPECT_TRUE(value);
                }
            }
        }
    }
}

TEST_F(FixpipeFusionTest, FrameworkControlInheritancePreservesUnrelatedNodes)
{
    for (bool three : {false, true}) {
        Case c;
        c.three = three;
        Build(c);
        auto before = Make("before", "NoOp", Desc(DT_INT8));
        auto after = Make("after", "NoOp", Desc(DT_INT8));
        auto first = Make("unrelated_first", "NoOp", Desc(DT_INT8));
        auto second = Make("unrelated_second", "NoOp", Desc(DT_INT8));
        auto user = Make("unrelated_user", "NoOp", Desc(DT_INT8));
        ASSERT_EQ(graph->AddControlEdge(before, nodes.at("m")), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddControlEdge(nodes.at(three ? "last" : "q"), after), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddControlEdge(nodes.at("a"), after), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddControlEdge(first, after), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddControlEdge(nodes.at("a"), after), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddControlEdge(second, user), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddControlEdge(first, user), GRAPH_SUCCESS);
        ASSERT_EQ(graph->AddControlEdge(second, user), GRAPH_SUCCESS);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        const auto unrelated = user.GetInControlNodes();
        ASSERT_EQ(unrelated.size(), 3U);
        const std::vector<std::string> expected = {"unrelated_second", "unrelated_first", "unrelated_second"};
        for (size_t index = 0; index < unrelated.size(); ++index) {
            AscendString name;
            ASSERT_EQ(unrelated[index]->GetName(name), GRAPH_SUCCESS);
            EXPECT_EQ(name.GetString(), expected[index]);
        }
        ASSERT_NE(Target(), nullptr);
        EXPECT_TRUE(HasControlPath(before, *Target()));
        EXPECT_TRUE(HasControlPath(*Target(), after));
        for (const char* existing : {"before", "after", "unrelated_first", "unrelated_second", "unrelated_user"}) {
            EXPECT_NE(graph->FindNodeByName(existing), nullptr);
        }
        EXPECT_EQ(Run(), GRAPH_NOT_CHANGED);
    }
}

TEST_F(FixpipeFusionTest, SharedDequantWithAllConstantMatmulInputs)
{
    for (bool scaleProducer : {false, true}) {
        Case c;
        c.three = true;
        Build(c);
        for (int port : {0, 1}) {
            auto original = nodes.at("m").GetInDataNodesAndPortIndexs(port);
            auto constant = Make(port == 0 ? "const_a" : "const_b", "Const", Desc(DT_INT8));
            ASSERT_EQ(graph->RemoveEdge(*original.first, original.second, nodes.at("m"), port), GRAPH_SUCCESS);
            Link(constant, nodes.at("m"), port);
        }
        if (scaleProducer) {
            auto producer = Make("scale_producer", "Identity", Desc(DT_UINT64), 1);
            Link(nodes.at("scale"), producer);
            ASSERT_EQ(graph->RemoveEdge(nodes.at("scale"), 0, nodes.at("q"), 1), GRAPH_SUCCESS);
            Link(producer, nodes.at("q"), 1);
        }
        auto user = Make("shared_user", "Identity", Desc(DT_FLOAT16), 1);
        Link(nodes.at("q"), user);
        ASSERT_EQ(Run(), GRAPH_SUCCESS);
        ExpectSharedDequantFallback(user);
    }
}

} // namespace
