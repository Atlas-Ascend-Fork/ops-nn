/** Copyright (c) 2026 Huawei Technologies Co., Ltd. See LICENSE for licensing details. */
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/add_rms_norm_quant_fusion_pass.h"

namespace ops {
namespace {
using ge::es::EsGraphBuilder;
using ge::es::EsTensorHolder;
constexpr float kEpsilon = 1e-5f;
const std::vector<int64_t> kShape{2, 64};
struct Config {
    ge::DataType dtype = ge::DT_FLOAT16;
    int branches = 1;
    bool offset = false;
    bool reshape = false;
    bool rstd_consumer = false;
};
std::string Type(const ge::GNode& node)
{
    ge::AscendString type;
    node.GetType(type);
    return type.GetString() ? type.GetString() : "";
}
std::string Name(const ge::GNode& node)
{
    ge::AscendString name;
    node.GetName(name);
    return name.GetString() ? name.GetString() : "";
}
void Describe(const EsTensorHolder& tensor, ge::DataType dtype, const std::vector<int64_t>& shape)
{
    ASSERT_EQ(tensor.GetProducer()->UpdateOutputDesc(tensor.GetProducerOutIndex(),
                                                     ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, dtype)),
              ge::GRAPH_SUCCESS);
}
ge::GraphPtr Build(const Config& config)
{
    EsGraphBuilder builder("AddRmsNormQuantFusionTest");
    int index = 0;
    auto input = [&](const std::string& name, ge::DataType dtype, const std::vector<int64_t>& shape) {
        return builder.CreateInput(index++, name.c_str(), dtype, ge::FORMAT_ND, shape);
    };
    auto x1 = input("x1", config.dtype, kShape);
    auto x2 = input("x2", config.dtype, kShape);
    auto gamma = input("gamma", config.dtype, {64});
    auto norm = ge::es::AddRmsNorm(x1, x2, gamma, kEpsilon);
    Describe(norm.y, config.dtype, kShape);
    Describe(norm.x, config.dtype, kShape);
    Describe(norm.rstd, ge::DT_FLOAT, {2, 1});
    std::vector<EsTensorHolder> outputs;
    for (int i = 0; i < config.branches; ++i) {
        auto y = norm.y;
        if (config.reshape) {
            y = ge::es::Reshape(y, builder.CreateVector(kShape));
            Describe(y, config.dtype, kShape);
        }
        auto scale = input("scale", ge::DT_FLOAT, {64});
        auto zero = config.offset ? input("zero_points", ge::DT_FLOAT, {64}) : EsTensorHolder(nullptr);
        auto quant = ge::es::Quantize(y, scale, zero, "torch.qint8", -1);
        Describe(quant, ge::DT_INT8, kShape);
        outputs.push_back(quant);
    }
    outputs.push_back(norm.x);
    if (config.rstd_consumer) {
        outputs.push_back(norm.rstd);
    }
    auto graph = builder.BuildAndReset(outputs);
    // Supply the connected input descriptors available at the pass execution stage.
    for (auto node : graph->GetAllNodes()) {
        for (int port = 0; port < static_cast<int>(node.GetInputsSize()); ++port) {
            auto source = node.GetInDataNodesAndPortIndexs(port);
            if (source.first == nullptr) {
                continue;
            }
            ge::TensorDesc desc;
            EXPECT_EQ(source.first->GetOutputDesc(source.second, desc), ge::GRAPH_SUCCESS);
            EXPECT_EQ(node.UpdateInputDesc(port, desc), ge::GRAPH_SUCCESS);
        }
    }
    return graph;
}
int Count(const ge::GraphPtr& graph, const std::string& type)
{
    int count = 0;
    for (const auto& node : graph->GetAllNodes()) {
        count += Type(node) == type;
    }
    return count;
}
class AddRmsNormQuantFusionTest : public testing::Test {
protected:
    fe::OptionalInfo saved_legacy_;
    fe::OptionalInfos saved_options_;
    void SetUp() override
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        saved_legacy_ = manager.opti_compilation_info_;
        saved_options_ = manager.opti_compilation_infos_;
        fe::PlatformInfo legacy;
        legacy.str_info.short_soc_version = "Ascend950";
        manager.platform_info_map_["AddRmsNormQuantFusionTest"] = legacy;
        fe::OptionalInfo legacy_options;
        legacy_options.soc_version = "AddRmsNormQuantFusionTest";
        manager.SetOptionalCompilationInfo(legacy_options);
        fe::PlatFormInfos platform;
        ASSERT_TRUE(platform.Init());
        std::map<std::string, std::string> version{{"NpuArch", "3510"}};
        platform.SetPlatformRes("version", version);
        manager.platform_infos_map_["AddRmsNormQuantFusionTest"] = platform;
        fe::OptionalInfos options;
        ASSERT_TRUE(options.Init());
        options.SetSocVersion("AddRmsNormQuantFusionTest");
        manager.SetOptionalCompilationInfo(options);
    }
    void TearDown() override
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        manager.opti_compilation_info_ = saved_legacy_;
        manager.opti_compilation_infos_ = saved_options_;
        manager.platform_info_map_.erase("AddRmsNormQuantFusionTest");
        manager.platform_infos_map_.erase("AddRmsNormQuantFusionTest");
    }
    void Check(const Config& config)
    {
        auto graph = Build(config);
        ASSERT_NE(graph, nullptr);
        const auto old_size = graph->GetAllNodes().size();
        std::vector<ge::TensorDesc> expected_outputs;
        for (const auto& node : graph->GetAllNodes()) {
            if (Type(node) != "NetOutput") {
                continue;
            }
            for (int port = 0; port < static_cast<int>(node.GetInputsSize()); ++port) {
                auto source = node.GetInDataNodesAndPortIndexs(port);
                ASSERT_NE(source.first, nullptr);
                ge::TensorDesc desc;
                ASSERT_EQ(source.first->GetOutputDesc(source.second, desc), ge::GRAPH_SUCCESS);
                expected_outputs.push_back(desc);
            }
        }
        ge::CustomPassContext context;
        context.SetPassName("AddRmsNormQuantFusionPass");
        AddRmsNormQuantFusionPass pass;
        const auto status = pass.Run(graph, context);
        if (config.rstd_consumer) {
            EXPECT_EQ(status, ge::GRAPH_NOT_CHANGED);
            EXPECT_EQ(Count(graph, "AddRmsNormQuant"), 0);
            EXPECT_EQ(Count(graph, "AddRmsNorm"), 1);
            EXPECT_EQ(Count(graph, "Quantize"), config.branches);
            EXPECT_EQ(graph->GetAllNodes().size(), old_size);
        } else {
            ASSERT_EQ(status, ge::SUCCESS);
            EXPECT_EQ(Count(graph, "AddRmsNormQuant"), 1);
            EXPECT_EQ(Count(graph, "AddRmsNorm"), 0);
            EXPECT_EQ(Count(graph, "Quantize"), 0);
            EXPECT_EQ(Count(graph, "Reshape"), 0);
            for (const auto& node : graph->GetAllNodes()) {
                if (Type(node) != "AddRmsNormQuant") {
                    continue;
                }
                auto scale = node.GetInDataNodesAndPortIndexs(3);
                ASSERT_NE(scale.first, nullptr);
                EXPECT_EQ(Name(*scale.first), "scale");
                auto zero = node.GetInDataNodesAndPortIndexs(5);
                if (config.offset) {
                    ASSERT_NE(zero.first, nullptr);
                    EXPECT_EQ(Name(*zero.first), "zero_points");
                } else {
                    EXPECT_EQ(zero.first, nullptr);
                }
            }
        }
        for (const auto& node : graph->GetAllNodes()) {
            if (Type(node) != "NetOutput") {
                continue;
            }
            ASSERT_EQ(node.GetInputsSize(), expected_outputs.size());
            for (int port = 0; port < static_cast<int>(node.GetInputsSize()); ++port) {
                auto source = node.GetInDataNodesAndPortIndexs(port);
                ASSERT_NE(source.first, nullptr);
                ge::TensorDesc desc;
                ASSERT_EQ(source.first->GetOutputDesc(source.second, desc), ge::GRAPH_SUCCESS);
                EXPECT_EQ(desc.GetShape().GetDims(), expected_outputs[port].GetShape().GetDims());
                EXPECT_EQ(desc.GetDataType(), expected_outputs[port].GetDataType());
                if (config.rstd_consumer && port == static_cast<int>(expected_outputs.size()) - 1) {
                    EXPECT_EQ(Type(*source.first), "AddRmsNorm");
                    EXPECT_EQ(source.second, 1);
                }
            }
        }
    }
};
TEST_F(AddRmsNormQuantFusionTest, DirectWithoutOffset)
{
    Config c;
    Check(c);
}

TEST_F(AddRmsNormQuantFusionTest, DirectWithOffsetBf16)
{
    Config c;
    c.offset = true;
    c.dtype = ge::DT_BF16;
    Check(c);
}

TEST_F(AddRmsNormQuantFusionTest, ReshapeWithoutOffset)
{
    Config c;
    c.reshape = true;
    Check(c);
}

TEST_F(AddRmsNormQuantFusionTest, ReshapeWithOffsetBf16)
{
    Config c;
    c.reshape = true;
    c.offset = true;
    c.dtype = ge::DT_BF16;
    Check(c);
}

TEST_F(AddRmsNormQuantFusionTest, RejectConsumedRstd)
{
    Config c;
    c.rstd_consumer = true;
    Check(c);
}

} // namespace
} // namespace ops
