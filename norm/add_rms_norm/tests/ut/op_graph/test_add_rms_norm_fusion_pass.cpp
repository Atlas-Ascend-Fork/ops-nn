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
#include <string>
#include <vector>
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "../../../op_graph/fusion_pass/add_rms_norm_fusion_pass.h"

namespace {
using ge::es::EsTensorHolder;
constexpr float kEpsilon = 1e-5f;
const std::vector<int64_t> kShape{2, 64};
std::string Type(const ge::GNode& node)
{
    ge::AscendString type;
    node.GetType(type);
    return type.GetString() ? type.GetString() : "";
}
int Count(const ge::GraphPtr& graph, const std::string& type)
{
    int count = 0;
    for (const auto& node : graph->GetAllNodes()) {
        count += Type(node) == type;
    }
    return count;
}
void Describe(const EsTensorHolder& tensor, ge::DataType dtype, const std::vector<int64_t>& dims)
{
    ASSERT_NE(tensor.GetProducer(), nullptr);
    ASSERT_EQ(tensor.GetProducer()->UpdateOutputDesc(tensor.GetProducerOutIndex(),
                                                     ge::TensorDesc(ge::Shape(dims), ge::FORMAT_ND, dtype)),
              ge::GRAPH_SUCCESS);
}
void InitializeInputs(const ge::GraphPtr& graph)
{
    for (auto node : graph->GetAllNodes()) {
        for (int port = 0; port < static_cast<int>(node.GetInputsSize()); ++port) {
            auto source = node.GetInDataNodesAndPortIndexs(port);
            if (!source.first) {
                continue;
            }
            ge::TensorDesc desc;
            ASSERT_EQ(source.first->GetOutputDesc(source.second, desc), ge::GRAPH_SUCCESS);
            ASSERT_EQ(node.UpdateInputDesc(port, desc), ge::GRAPH_SUCCESS);
        }
    }
}
class AddRmsNormFusionGraphPassTest : public testing::Test {
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
        manager.platform_info_map_["AddRmsNormFusionGraphPassTest"] = legacy;
        fe::OptionalInfo legacy_options;
        legacy_options.soc_version = "AddRmsNormFusionGraphPassTest";
        manager.SetOptionalCompilationInfo(legacy_options);
        fe::PlatFormInfos platform;
        ASSERT_TRUE(platform.Init());
        std::map<std::string, std::string> version{{"NpuArch", "3510"}};
        platform.SetPlatformRes("version", version);
        manager.platform_infos_map_["AddRmsNormFusionGraphPassTest"] = platform;
        fe::OptionalInfos options;
        ASSERT_TRUE(options.Init());
        options.SetSocVersion("AddRmsNormFusionGraphPassTest");
        manager.SetOptionalCompilationInfo(options);
    }
    void TearDown() override
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        manager.opti_compilation_info_ = saved_legacy_;
        manager.opti_compilation_infos_ = saved_options_;
        manager.platform_info_map_.erase("AddRmsNormFusionGraphPassTest");
        manager.platform_infos_map_.erase("AddRmsNormFusionGraphPassTest");
    }
    void RunAndCheck(ge::GraphPtr& graph, const std::string& target, bool reject)
    {
        ASSERT_NE(graph, nullptr);
        InitializeInputs(graph);
        const auto original_size = graph->GetAllNodes().size();
        std::vector<ge::TensorDesc> expected_outputs;
        for (const auto& node : graph->GetAllNodes()) {
            if (Type(node) != "NetOutput") {
                continue;
            }
            for (int i = 0; i < static_cast<int>(node.GetInputsSize()); ++i) {
                ge::TensorDesc desc;
                ASSERT_EQ(node.GetInputDesc(i, desc), ge::GRAPH_SUCCESS);
                expected_outputs.push_back(desc);
            }
        }
        ASSERT_FALSE(expected_outputs.empty());
        ge::CustomPassContext context;
        context.SetPassName("AddRmsNormFusionGraphPass");
        ops::AddRmsNormFusionGraphPass pass;
        ASSERT_EQ(pass.Run(graph, context), reject ? ge::GRAPH_NOT_CHANGED : ge::SUCCESS);
        EXPECT_EQ(Count(graph, target), reject ? 0 : 1);
        EXPECT_EQ(Count(graph, "Add"), reject ? 1 : 0);
        EXPECT_EQ(Count(graph, "RmsNorm"), reject ? 1 : 0);
        if (reject) {
            EXPECT_EQ(graph->GetAllNodes().size(), original_size);
        }
        size_t output_count = 0;
        for (const auto& node : graph->GetAllNodes()) {
            if (Type(node) == target) {
                float epsilon = 0;
                ASSERT_EQ(node.GetAttr("epsilon", epsilon), ge::GRAPH_SUCCESS);
                EXPECT_FLOAT_EQ(epsilon, kEpsilon);
            }
            if (Type(node) != "NetOutput") {
                continue;
            }
            for (int i = 0; i < static_cast<int>(node.GetInputsSize()); ++i) {
                auto source = node.GetInDataNodesAndPortIndexs(i);
                ASSERT_NE(source.first, nullptr);
                ASSERT_LT(output_count, expected_outputs.size());
                ge::TensorDesc actual;
                ASSERT_EQ(source.first->GetOutputDesc(source.second, actual), ge::GRAPH_SUCCESS);
                EXPECT_EQ(actual.GetDataType(), expected_outputs[output_count].GetDataType());
                EXPECT_EQ(actual.GetShape().GetDims(), expected_outputs[output_count].GetShape().GetDims());
                EXPECT_EQ(actual.GetFormat(), expected_outputs[output_count].GetFormat());
                if (!reject) {
                    EXPECT_EQ(Type(*source.first), target);
                    EXPECT_EQ(source.second, i);
                }
                ++output_count;
            }
        }
        EXPECT_EQ(output_count, expected_outputs.size());
    }
    ge::GraphPtr Build(ge::DataType dtype, bool residual = false, bool broadcast = false)
    {
        ge::es::EsGraphBuilder builder("AddRmsNormFusionUt");
        auto x1 = builder.CreateInput(0, "x1", dtype, ge::FORMAT_ND, kShape);
        auto x2 = builder.CreateInput(1, "x2", dtype, ge::FORMAT_ND, broadcast ? std::vector<int64_t>{1, 64} : kShape);
        auto gamma = builder.CreateInput(2, "gamma", dtype, ge::FORMAT_ND, {64});
        auto sum = ge::es::Add(x1, x2);
        auto norm = ge::es::RmsNorm(sum, gamma, kEpsilon);
        Describe(sum, dtype, kShape);
        Describe(norm.y, dtype, kShape);
        Describe(norm.rstd, ge::DT_FLOAT, {2, 1});
        float epsilon = kEpsilon;
        EXPECT_EQ(norm.y.GetProducer()->SetAttr("epsilon", epsilon), ge::GRAPH_SUCCESS);
        std::vector<EsTensorHolder> outputs{norm.y, norm.rstd};
        if (residual) {
            outputs.push_back(sum);
        }
        return builder.BuildAndReset(outputs);
    }
};
TEST_F(AddRmsNormFusionGraphPassTest, Fp16)
{
    auto graph = Build(ge::DT_FLOAT16, false, false);
    RunAndCheck(graph, "AddRmsNorm", false);
}

TEST_F(AddRmsNormFusionGraphPassTest, Bf16)
{
    auto graph = Build(ge::DT_BF16, false, false);
    RunAndCheck(graph, "AddRmsNorm", false);
}

TEST_F(AddRmsNormFusionGraphPassTest, Fp32)
{
    auto graph = Build(ge::DT_FLOAT, false, false);
    RunAndCheck(graph, "AddRmsNorm", false);
}

TEST_F(AddRmsNormFusionGraphPassTest, ResidualOutput)
{
    auto graph = Build(ge::DT_FLOAT16, true, false);
    RunAndCheck(graph, "AddRmsNorm", false);
}

TEST_F(AddRmsNormFusionGraphPassTest, RejectBroadcastInput)
{
    auto graph = Build(ge::DT_FLOAT16, false, true);
    RunAndCheck(graph, "AddRmsNorm", true);
}
} // namespace
