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

#include <functional>
#include <string>
#include <vector>

#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "../../../op_graph/fusion_pass/conv2d_transpose_fixpipe_fusion_pass.h"

using namespace ge;
using namespace es;
using namespace fe;
using namespace ops;
using namespace test_conv_fusion_framework;

namespace {

constexpr int64_t INPUT_SIZE_DIM = 4;
constexpr int32_t EXTEND_CONV_TRANSPOSE_INPUT_SIZE_INDEX = 0;
constexpr int32_t EXTEND_CONV_TRANSPOSE_X_INDEX = 1;
constexpr int32_t EXTEND_CONV_TRANSPOSE_FILTER_INDEX = 2;
constexpr int32_t EXTEND_CONV_TRANSPOSE_SCALE_INDEX = 4;
constexpr int64_t REARRANGE_REVERSE_MODE_VALUE = 1;
constexpr int32_t RELU_FUSION_MODE_VALUE = 1;
constexpr int32_t DEFAULT_FUSION_MODE_VALUE = 0;

struct Conv2DTransposeConfig : public NodeConfig {
    Conv2DTransposeConfig()
    {
        name = "Conv2DTranspose";
        opType = "Conv2DTranspose";
        inputDefs = {{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
                     {"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("strides", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("pads", std::vector<int64_t>{0, 0, 0, 0});
        SetAttr("dilations", std::vector<int64_t>{1, 1, 1, 1});
        SetAttr("groups", int64_t(1));
        SetAttr("data_format", std::string("NCHW"));
        SetAttr("output_padding", std::vector<int64_t>{0, 0, 0, 0});
        SetAttr("offset_x", int64_t(0));
    }

    static Conv2DTransposeConfig Basic(const std::string& nodeName, DataType xDtype = DT_INT8,
                                       DataType filterDtype = DT_INT8, DataType outDtype = DT_FLOAT16,
                                       Format format = FORMAT_NCHW,
                                       const std::vector<int64_t>& xShape = {1, 16, 16, 16},
                                       const std::vector<int64_t>& filterShape = {16, 32, 3, 3},
                                       const std::vector<int64_t>& outShape = {1, 32, 32, 32})
    {
        Conv2DTransposeConfig config;
        config.SetName(nodeName)
            .AddInput(DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM}, nodeName + "_input_size")
            .AddInput(xDtype, format, xShape, nodeName + "_x")
            .AddInput(filterDtype, format, filterShape, nodeName + "_filter")
            .AddOutput(outDtype, format, outShape, "y");
        return config;
    }

    Conv2DTransposeConfig& WithBias(DataType biasDtype = DT_INT32, const std::vector<int64_t>& biasShape = {32})
    {
        biasTensor = TensorInfo(biasDtype, FORMAT_ND, biasShape, name + "_bias");
        biasTensor.enabled = true;
        this->AddInput(biasTensor);
        return *this;
    }

    TensorInfo biasTensor;
};

struct TransposeConfig : public NodeConfig {
    TransposeConfig()
    {
        name = "Transpose";
        opType = "Transpose";
        inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"perm", CompliantNodeBuilder::kEsIrInputOptional, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("perm", std::vector<int64_t>{0, 2, 3, 1});
    }

    static TransposeConfig Basic(const std::string& nodeName, DataType dataType, Format format,
                                 const std::vector<int64_t>& inShape, const std::vector<int64_t>& outShape)
    {
        TransposeConfig config;
        config.SetName(nodeName)
            .AddInput(dataType, format, inShape, nodeName + "_x")
            .AddInput(DT_INT32, FORMAT_ND, {4}, nodeName + "_perm", true)
            .AddOutput(dataType, format, outShape, "y");
        return config;
    }
};

struct SetM1DequantConfig : public NodeConfig {
    SetM1DequantConfig()
    {
        name = "SetM1Dequant";
        opType = "SetM1Dequant";
        inputDefs = {{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        SetAttr("_is_single_op", true);
        SetAttr("dequantouttype", int64_t(DT_FLOAT16));
    }

    static SetM1DequantConfig Basic(const std::string& nodeName, const std::vector<int64_t>& paramShape = {1})
    {
        SetM1DequantConfig config;
        config.SetName(nodeName)
            .AddInput(DT_UINT64, FORMAT_ND, paramShape, nodeName + "_x1")
            .AddOutput(DT_UINT64, FORMAT_ND, paramShape, "y");
        return config;
    }

    SetM1DequantConfig& WithOffset(float offset)
    {
        SetAttr("offset", offset);
        return *this;
    }
};

void RunPassAndCheck(GraphPtr& graph, Status expectRes, bool expectFused, bool expectDual = false)
{
    CustomPassContext passContext;
    passContext.SetPassName("Conv2DTransposeFixpipeFusionPass");
    Conv2DTransposeFixpipeFusionPass pass({AscendString("Conv2DTranspose")});
    auto res = pass.Run(graph, passContext);
    EXPECT_EQ(res, expectRes);
    EXPECT_EQ(GraphChecker::HasNode(graph, "ExtendConvTranspose"), expectFused && !expectDual);
    EXPECT_EQ(GraphChecker::HasNode(graph, "ExtendConvTransposeV2"), expectDual);
    EXPECT_EQ(GraphChecker::HasNode(graph, "Unsqueeze"), expectFused);
    EXPECT_EQ(GraphChecker::HasNode(graph, "Squeeze"), expectFused);
    EXPECT_EQ(GraphChecker::HasNode(graph, "Conv2DTranspose"), !expectFused);
}

} // namespace

class Conv2DTransposeFixpipeFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    // the cube vector combine res is global in the PlatformInfoManager, patch it in place
    // (get the full PlatFormInfos, modify the SoCInfo section and update back) so the other
    // platform resources are kept and the previous testcase never leaks its fuse state
    static void SetCubeVectorCombine(const std::string& value)
    {
        fe::PlatFormInfos platformInfos;
        fe::OptionalInfos optionalInfos;
        if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfos, optionalInfos) !=
            GRAPH_SUCCESS) {
            return;
        }
        std::map<std::string, std::string> socInfos;
        platformInfos.GetPlatformRes("SoCInfo", socInfos);
        socInfos["cube_vector_combine"] = value;
        platformInfos.SetPlatformRes("SoCInfo", socInfos);
        PlatformInfoManager::Instance().UpdatePlatformInfos(platformInfos);
    }

    void SetUp() override
    {
        SocConfig::Ascend950().Apply();
        SetCubeVectorCombine("");
    }
};

// Test 1: patternTest - basic fusion success, 2D expand to 3D input, 3D squeeze to 2D output
TEST_F(Conv2DTransposeFixpipeFusionPassTest, patternTest)
{
    TestGraph builder("patternTest");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose"), {0, 1, 2})
                     .SetOutput("Conv2DTranspose")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
}

// Test 2: withBiasFusionSuccess - fusion with bias input
TEST_F(Conv2DTransposeFixpipeFusionPassTest, withBiasFusionSuccess)
{
    TestGraph builder("withBiasFusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose").WithBias(), {0, 1, 2, 3})
                     .SetOutput("Conv2DTranspose")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
}

// Test 3: dequantPostCubeFusionSuccess - post node fused by cube_utils, scale input connected
TEST_F(Conv2DTransposeFixpipeFusionPassTest, dequantPostCubeFusionSuccess)
{
    TestGraph builder("dequantPostCubeFusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .SetOutput("AscendDequant")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));

    // check scale input of ExtendConvTranspose comes from dequant scale input
    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    TensorDesc scaleDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_SCALE_INDEX, scaleDesc), GRAPH_SUCCESS);
    EXPECT_EQ(scaleDesc.GetDataType(), DT_UINT64);
    auto scaleProducer = extendNode.GetInDataNodesAndPortIndexs(EXTEND_CONV_TRANSPOSE_SCALE_INDEX);
    EXPECT_NE(scaleProducer.first, nullptr);
    // fusion mode is 0 when no relu in fusion op list, es api skips attr setting when value equals default
    int64_t fusionModeValue = RELU_FUSION_MODE_VALUE;
    if (extendNode.GetAttr(AscendString("fusion_mode"), fusionModeValue) != GRAPH_SUCCESS) {
        fusionModeValue = DEFAULT_FUSION_MODE_VALUE;
    }
    EXPECT_EQ(fusionModeValue, DEFAULT_FUSION_MODE_VALUE);

    // input_size keeps 4 elements so that ExtendConvTranspose takes the 2d extend 3d infershape branch
    TensorDesc inputSizeDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_INPUT_SIZE_INDEX, inputSizeDesc), GRAPH_SUCCESS);
    EXPECT_EQ(inputSizeDesc.GetShape().GetDims(), std::vector<int64_t>({INPUT_SIZE_DIM}));
    // output is NCDHW with the D dim inserted, the shape follows the postcube output shape
    TensorDesc outDesc;
    EXPECT_EQ(extendNode.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetFormat(), FORMAT_NCDHW);
    EXPECT_EQ(outDesc.GetShape().GetDims(), std::vector<int64_t>({1, 3, 1, 244, 244}));
}

// Test 4: reluPostCubeFusionSuccess - relu post node fused, fusion mode is 1
TEST_F(Conv2DTransposeFixpipeFusionPassTest, reluPostCubeFusionSuccess)
{
    TestGraph builder("reluPostCubeFusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_FLOAT16), {0, 1, 2})
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                     .Connect("Conv2DTranspose", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));

    // check fusion mode is 1 when relu in fusion op list
    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    int64_t fusionModeValue = -1;
    EXPECT_EQ(extendNode.GetAttr(AscendString("fusion_mode"), fusionModeValue), GRAPH_SUCCESS);
    EXPECT_EQ(fusionModeValue, RELU_FUSION_MODE_VALUE);
}

// Test 5: dequantReluPostCubeFusionSuccess - dequant + relu chain fused
TEST_F(Conv2DTransposeFixpipeFusionPassTest, dequantReluPostCubeFusionSuccess)
{
    TestGraph builder("dequantReluPostCubeFusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16))
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .Connect("AscendDequant", 0, "Relu", 0)
                     .SetOutput("Relu")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Relu"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));

    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    // scale input exists and fusion mode is 1
    TensorDesc scaleDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_SCALE_INDEX, scaleDesc), GRAPH_SUCCESS);
    EXPECT_EQ(scaleDesc.GetDataType(), DT_UINT64);
    int64_t fusionModeValue = -1;
    EXPECT_EQ(extendNode.GetAttr(AscendString("fusion_mode"), fusionModeValue), GRAPH_SUCCESS);
    EXPECT_EQ(fusionModeValue, RELU_FUSION_MODE_VALUE);
}

// Test 6: requantPostCubeFusionSuccess - requant post node fused
TEST_F(Conv2DTransposeFixpipeFusionPassTest, requantPostCubeFusionSuccess)
{
    TestGraph builder("requantPostCubeFusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                     .Connect("Conv2DTranspose", 0, "AscendRequant", 0)
                     .SetOutput("AscendRequant")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendRequant"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));
}

// Test 7: quantPostCubeFusionSuccess - quant post node fused
TEST_F(Conv2DTransposeFixpipeFusionPassTest, quantPostCubeFusionSuccess)
{
    TestGraph builder("quantPostCubeFusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_INT32))
                     .Connect("Conv2DTranspose", 0, "AscendQuant", 0)
                     .SetOutput("AscendQuant")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendQuant"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));
}

// Test 9: groupsNotOneSuccess - groups > 1 fusion
TEST_F(Conv2DTransposeFixpipeFusionPassTest, groupsNotOneSuccess)
{
    TestGraph builder("groupsNotOneSuccess");
    Conv2DTransposeConfig config = Conv2DTransposeConfig::Basic(
        "Conv2DTranspose", DT_INT8, DT_INT8, DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16}, {16, 4, 3, 3}, {1, 32, 32, 32});
    config.SetAttr("groups", int64_t(4));
    auto graph = builder.SetSocAscend950().AddNode(config, {0, 1, 2}).SetOutput("Conv2DTranspose").Build();
    RunPassAndCheck(graph, SUCCESS, true);
}

// Test 10: mc62SocFusionSuccess - cube vector fuse soc fusion
TEST_F(Conv2DTransposeFixpipeFusionPassTest, mc62SocFusionSuccess)
{
    TestGraph builder("mc62SocFusionSuccess");
    auto graph = builder.SetSocMC62()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose"), {0, 1, 2})
                     .SetOutput("Conv2DTranspose")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
}

// Test 11: unsupportedPlatformFail - platform not supported
TEST_F(Conv2DTransposeFixpipeFusionPassTest, unsupportedPlatformFail)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.str_info.short_soc_version = "Ascend910_93";
    optionalInfo.soc_version = "Ascend910_93";
    PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);

    TestGraph builder("unsupportedPlatformFail");
    auto graph = builder.AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose"), {0, 1, 2})
                     .SetOutput("Conv2DTranspose")
                     .Build();
    RunPassAndCheck(graph, GRAPH_NOT_CHANGED, false);
}

// Test 12: invalidShapeFail - x with invalid 3D shape
TEST_F(Conv2DTransposeFixpipeFusionPassTest, invalidShapeFail)
{
    TestGraph builder("invalidShapeFail");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_FLOAT16, FORMAT_NCHW,
                                                           {1, 16, 16}, {16, 32, 3, 3}, {1, 32, 32, 32}),
                              {0, 1, 2})
                     .SetOutput("Conv2DTranspose")
                     .Build();
    CustomPassContext passContext;
    Conv2DTransposeFixpipeFusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_NE(pass.Run(graph, passContext), SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "ExtendConvTranspose"));
}

// Test 13: noMatchedNodeNotChanged - no Conv2DTranspose node in graph
TEST_F(Conv2DTransposeFixpipeFusionPassTest, noMatchedNodeNotChanged)
{
    TestGraph builder("noMatchedNodeNotChanged");
    auto graph = builder.SetSocAscend950()
                     .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16), true)
                     .SetOutput("Relu")
                     .Build();
    CustomPassContext passContext;
    Conv2DTransposeFixpipeFusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
}

// Test 16: dualPostCubeSkip - dual postcube outputs are not supported, the node is skipped
TEST_F(Conv2DTransposeFixpipeFusionPassTest, dualPostCubeSkip)
{
    TestGraph builder("dualPostCubeSkip");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .Connect("Conv2DTranspose", 0, "AscendRequant", 0)
                     .SetOutput("AscendDequant")
                     .SetOutput("AscendRequant")
                     .Build();
    RunPassAndCheck(graph, GRAPH_NOT_CHANGED, false);
    // the skip decision is made before the FixPipe creation, no engine-less FixPipe node is
    // left in the graph (otherwise the graph compile fails with "cannot find engine")
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));
}

// Test 17: postCubeWithOtherConsumerSkip - conv2dtranspose output has both postcube and other consumer,
// the subgraph would not be self-contained, so the node is skipped
TEST_F(Conv2DTransposeFixpipeFusionPassTest, postCubeWithOtherConsumerSkip)
{
    TestGraph builder("postCubeWithOtherConsumerSkip");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .SetOutput("Conv2DTranspose")
                     .SetOutput("AscendDequant")
                     .Build();
    RunPassAndCheck(graph, GRAPH_NOT_CHANGED, false);
    // the skip decision is made before the FixPipe creation, no engine-less FixPipe node is
    // left in the graph (otherwise the graph compile fails with "cannot find engine")
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));
}

// Test 18: FP32 dequant output is converted to the FP16 ExtendConvTranspose output.
// Skipping this fusion leaves Conv2DTranspose without an engine on the fuse SoC.
TEST_F(Conv2DTransposeFixpipeFusionPassTest, fp32DequantOutputFusionSuccess)
{
    TestGraph builder("fp32DequantOutputFusionSuccess");
    SetCubeVectorCombine("fuse");
    auto graph = builder.SetSocMC62()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant", DT_FLOAT))
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .SetOutput("AscendDequant")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendDequant"));
    GNode extendNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    TensorDesc outputDesc;
    ASSERT_EQ(extendNode.GetOutputDesc(0, outputDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outputDesc.GetDataType(), DT_FLOAT16);
    GNode rebuiltSetM1;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "SetM1Dequant", rebuiltSetM1));
    int64_t dequantOutDtype = -1;
    ASSERT_EQ(rebuiltSetM1.GetAttr(AscendString("dequantouttype"), dequantOutDtype), GRAPH_SUCCESS);
    EXPECT_EQ(dequantOutDtype, static_cast<int64_t>(DT_FLOAT16));
}

// Test 19: setM1RequantOffsetFromSetM1Success - requant scenario (postcube with int8 output),
// the rebuilt SetM1Dequant keeps the offset attr carried by the original SetM1Dequant node
TEST_F(Conv2DTransposeFixpipeFusionPassTest, setM1RequantOffsetFromSetM1Success)
{
    TestGraph builder("setM1RequantOffsetFromSetM1Success");
    PostCubeConfig fixPipeConfig = PostCubeConfig::Basic("FixPipe", DT_INT8).WithScale0();
    fixPipeConfig.SetAttr("fusion_op_list", std::vector<std::string>{"AscendDequant", "Relu"});
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddNode(SetM1DequantConfig::Basic("SetM1Dequant").WithOffset(2.5f), {0})
                     .AddPostCube(fixPipeConfig, false)
                     .Connect("Conv2DTranspose", 0, "FixPipe", 0)
                     .Connect("SetM1Dequant", 0, "FixPipe",
                              Conv2DTransposeFixpipeFusion::POST_CUBE_INPUT_QUANT_SCALE_0_INDEX)
                     .SetOutput("FixPipe")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);

    // the rebuilt SetM1Dequant exists in the replacement graph with the int8 dequantouttype
    // and the float requant zero point encoded from the original offset (the SetM1Dequant
    // host op reads the offset attr as a float), the original one is pruned by the
    // subgraph replace
    GNode rebuiltSetM1;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "SetM1Dequant", rebuiltSetM1));
    int64_t outDtype = 0;
    EXPECT_EQ(rebuiltSetM1.GetAttr(AscendString("dequantouttype"), outDtype), GRAPH_SUCCESS);
    EXPECT_EQ(outDtype, static_cast<int64_t>(DT_INT8));
    float offset = -1.0f;
    EXPECT_EQ(rebuiltSetM1.GetAttr(AscendString("offset"), offset), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(offset, 2.0f);
}

// Test 20: setM1RequantOffsetMissingDefaultSymmetricSuccess - the original SetM1Dequant and
// the postcube node carry no offset in the requant scenario, the conversion proceeds with the
// symmetric int8 requant zero point (-128) and the rebuilt node carries the 9-bit encoded
// value 384 ((-128) & 0x1FF)
TEST_F(Conv2DTransposeFixpipeFusionPassTest, setM1RequantOffsetMissingDefaultSymmetricSuccess)
{
    TestGraph builder("setM1RequantOffsetMissingDefaultSymmetricSuccess");
    PostCubeConfig fixPipeConfig = PostCubeConfig::Basic("FixPipe", DT_INT8).WithScale0();
    fixPipeConfig.SetAttr("fusion_op_list", std::vector<std::string>{"AscendDequant", "Relu"});
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddNode(SetM1DequantConfig::Basic("SetM1Dequant"), {0})
                     .AddPostCube(fixPipeConfig, false)
                     .Connect("Conv2DTranspose", 0, "FixPipe", 0)
                     .Connect("SetM1Dequant", 0, "FixPipe",
                              Conv2DTransposeFixpipeFusion::POST_CUBE_INPUT_QUANT_SCALE_0_INDEX)
                     .SetOutput("FixPipe")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);

    // the rebuilt SetM1Dequant exists in the replacement graph with the int8 dequantouttype
    // and the float encoded symmetric requant zero point (-128 -> 384 as the 9-bit
    // two's-complement field value)
    GNode rebuiltSetM1;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "SetM1Dequant", rebuiltSetM1));
    int64_t outDtype = 0;
    EXPECT_EQ(rebuiltSetM1.GetAttr(AscendString("dequantouttype"), outDtype), GRAPH_SUCCESS);
    EXPECT_EQ(outDtype, static_cast<int64_t>(DT_INT8));
    float offset = -1.0f;
    EXPECT_EQ(rebuiltSetM1.GetAttr(AscendString("offset"), offset), GRAPH_SUCCESS);
    EXPECT_FLOAT_EQ(offset, 384.0f);
}

// Test 21: the old DT_FLOAT M1 attr is normalized to DT_FLOAT16 for the fused FP16 output.
// The offset attr must not be set on the rebuilt SetM1Dequant.
TEST_F(Conv2DTransposeFixpipeFusionPassTest, setM1FloatOutputWithoutOffsetSuccess)
{
    TestGraph builder("setM1FloatOutputWithoutOffsetSuccess");
    PostCubeConfig fixPipeConfig = PostCubeConfig::Basic("FixPipe", DT_FLOAT16).WithScale0();
    fixPipeConfig.SetAttr("fusion_op_list", std::vector<std::string>{"AscendDequant"});
    auto setM1Config = SetM1DequantConfig::Basic("SetM1Dequant");
    setM1Config.SetAttr("dequantouttype", int64_t(DT_FLOAT));
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddNode(setM1Config, {0})
                     .AddPostCube(fixPipeConfig, false)
                     .Connect("Conv2DTranspose", 0, "FixPipe", 0)
                     .Connect("SetM1Dequant", 0, "FixPipe",
                              Conv2DTransposeFixpipeFusion::POST_CUBE_INPUT_QUANT_SCALE_0_INDEX)
                     .SetOutput("FixPipe")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);

    // the rebuilt SetM1Dequant carries FP16 dequantouttype and no offset attr
    GNode rebuiltSetM1;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "SetM1Dequant", rebuiltSetM1));
    int64_t outDtype = 0;
    EXPECT_EQ(rebuiltSetM1.GetAttr(AscendString("dequantouttype"), outDtype), GRAPH_SUCCESS);
    EXPECT_EQ(outDtype, static_cast<int64_t>(DT_FLOAT16));
    float offset = 0.0f;
    EXPECT_NE(rebuiltSetM1.GetAttr(AscendString("offset"), offset), GRAPH_SUCCESS);
}

// Test 22: a16w8RearrangeWeightFusionSuccess - fp16 x + int8 filter (A16W8) with groups == 1,
// the int8 weight is rearranged by the RearrangeA16W8WeightFzOp host node before ExtendConvTranspose
TEST_F(Conv2DTransposeFixpipeFusionPassTest, a16w8RearrangeWeightFusionSuccess)
{
    TestGraph builder("a16w8RearrangeWeightFusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_FLOAT16, DT_INT8, DT_FLOAT16),
                              {0, 1, 2})
                     .SetOutput("Conv2DTranspose")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);

    // the rearrange node exists with the groups and reverse mode attrs of the original pass
    GNode rearrangeNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "RearrangeA16W8WeightFzOp", rearrangeNode));
    int64_t groups = 0;
    EXPECT_EQ(rearrangeNode.GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
    EXPECT_EQ(groups, 1);
    int64_t reverseMode = 0;
    EXPECT_EQ(rearrangeNode.GetAttr(AscendString("attrReverseMode"), reverseMode), GRAPH_SUCCESS);
    EXPECT_EQ(reverseMode, REARRANGE_REVERSE_MODE_VALUE);

    // the filter input of ExtendConvTranspose comes from the rearrange node with the int8 dtype
    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    std::string filterProducerType;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(extendNode, EXTEND_CONV_TRANSPOSE_FILTER_INDEX, filterProducerType));
    EXPECT_EQ(filterProducerType, "RearrangeA16W8WeightFzOp");
    TensorDesc filterDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_FILTER_INDEX, filterDesc), GRAPH_SUCCESS);
    EXPECT_EQ(filterDesc.GetDataType(), DT_INT8);
    EXPECT_EQ(filterDesc.GetShape().GetDims(), std::vector<int64_t>({16, 32, 1, 3, 3}));

    // the x input keeps the fp16 dtype and expands to the 3D NCDHW desc by Unsqueeze
    TensorDesc xDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_X_INDEX, xDesc), GRAPH_SUCCESS);
    EXPECT_EQ(xDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(xDesc.GetFormat(), FORMAT_NCDHW);
    EXPECT_EQ(xDesc.GetShape().GetDims(), std::vector<int64_t>({1, 16, 1, 16, 16}));
}

// Test 23: a16w8GroupsNotOneFusionSuccess - fp16 x + int8 filter with groups > 1, the weight is
// still rearranged for the AIC (migrated from the updated fixpipe pass: both group and
// non-group A16W8 scenarios insert the rearrange node)
TEST_F(Conv2DTransposeFixpipeFusionPassTest, a16w8GroupsNotOneFusionSuccess)
{
    TestGraph builder("a16w8GroupsNotOneFusionSuccess");
    Conv2DTransposeConfig config = Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_FLOAT16, DT_INT8, DT_FLOAT16,
                                                                FORMAT_NCHW, {1, 16, 16, 16}, {16, 4, 3, 3},
                                                                {1, 32, 32, 32});
    config.SetAttr("groups", int64_t(4));
    auto graph = builder.SetSocAscend950().AddNode(config, {0, 1, 2}).SetOutput("Conv2DTranspose").Build();
    RunPassAndCheck(graph, SUCCESS, true);

    // the multi-group A16W8 scenario also rearranges the weight now
    EXPECT_TRUE(GraphChecker::HasNode(graph, "RearrangeA16W8WeightFzOp"));
    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    std::string filterProducerType;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(extendNode, EXTEND_CONV_TRANSPOSE_FILTER_INDEX, filterProducerType));
    EXPECT_EQ(filterProducerType, "RearrangeA16W8WeightFzOp");
    TensorDesc filterDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_FILTER_INDEX, filterDesc), GRAPH_SUCCESS);
    EXPECT_EQ(filterDesc.GetDataType(), DT_INT8);
}

// Test 24: a16w8DequantPostCubeFusionSuccess - A16W8 x fused with the dequant postcube, the scale
// input and the rearrange weight node coexist in the replacement graph
TEST_F(Conv2DTransposeFixpipeFusionPassTest, a16w8DequantPostCubeFusionSuccess)
{
    TestGraph builder("a16w8DequantPostCubeFusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_FLOAT16, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .SetOutput("AscendDequant")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "RearrangeA16W8WeightFzOp"));

    // the filter input of ExtendConvTranspose comes from the dequant scale input
    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    TensorDesc scaleDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_SCALE_INDEX, scaleDesc), GRAPH_SUCCESS);
    EXPECT_EQ(scaleDesc.GetDataType(), DT_UINT64);
}

// Test 26: fractalFilterFusionSuccess - FRACTAL_Z filter keeps its layout directly without the
// Unsqueeze node, the origin format is forced to NCHW (migrated from the original fixpipe pass)
TEST_F(Conv2DTransposeFixpipeFusionPassTest, fractalFilterFusionSuccess)
{
    TestGraph builder("fractalFilterFusionSuccess");
    Conv2DTransposeConfig config;
    config.SetName("Conv2DTranspose")
        .AddInput(DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM}, "Conv2DTranspose_input_size")
        .AddInput(DT_INT8, FORMAT_NCHW, {1, 16, 16, 16}, "Conv2DTranspose_x")
        .AddInput(DT_INT8, FORMAT_FRACTAL_Z, {4, 432, 1, 16}, "Conv2DTranspose_filter")
        .AddOutput(DT_FLOAT16, FORMAT_NCHW, {1, 32, 32, 32}, "y");
    auto graph = builder.SetSocAscend950().AddNode(config, {0, 1, 2}).SetOutput("Conv2DTranspose").Build();
    RunPassAndCheck(graph, SUCCESS, true);

    // the filter input of ExtendConvTranspose comes from the boundary input directly (the
    // original filter producer) instead of the Unsqueeze node, and keeps the fractal desc
    // with the origin format forced to NCHW and the shape unchanged
    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    std::string filterProducerType;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(extendNode, EXTEND_CONV_TRANSPOSE_FILTER_INDEX, filterProducerType));
    EXPECT_NE(filterProducerType, "Unsqueeze");
    TensorDesc filterDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_FILTER_INDEX, filterDesc), GRAPH_SUCCESS);
    EXPECT_EQ(filterDesc.GetFormat(), FORMAT_FRACTAL_Z);
    EXPECT_EQ(filterDesc.GetOriginFormat(), FORMAT_NCHW);
    EXPECT_EQ(filterDesc.GetShape().GetDims(), std::vector<int64_t>({4, 432, 1, 16}));

    // the x input still expands to the 3D desc through the Unsqueeze node
    std::string xProducerType;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(extendNode, EXTEND_CONV_TRANSPOSE_X_INDEX, xProducerType));
    EXPECT_EQ(xProducerType, "Unsqueeze");
}

// Test 27: fractalFilterA16W8RearrangeSuccess - A16W8 with a FRACTAL_Z filter, the rearrange
// node consumes the fractal filter directly and its output keeps the fractal layout with the
// int8 dtype
TEST_F(Conv2DTransposeFixpipeFusionPassTest, fractalFilterA16W8RearrangeSuccess)
{
    TestGraph builder("fractalFilterA16W8RearrangeSuccess");
    Conv2DTransposeConfig config;
    config.SetName("Conv2DTranspose")
        .AddInput(DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM}, "Conv2DTranspose_input_size")
        .AddInput(DT_FLOAT16, FORMAT_NCHW, {1, 16, 16, 16}, "Conv2DTranspose_x")
        .AddInput(DT_INT8, FORMAT_FRACTAL_Z, {4, 432, 1, 16}, "Conv2DTranspose_filter")
        .AddOutput(DT_FLOAT16, FORMAT_NCHW, {1, 32, 32, 32}, "y");
    auto graph = builder.SetSocAscend950().AddNode(config, {0, 1, 2}).SetOutput("Conv2DTranspose").Build();
    RunPassAndCheck(graph, SUCCESS, true);

    // the rearrange node exists and consumes the boundary filter input directly
    GNode rearrangeNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "RearrangeA16W8WeightFzOp", rearrangeNode));
    std::string rearrangeProducerType;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(rearrangeNode, 0, rearrangeProducerType));
    EXPECT_NE(rearrangeProducerType, "Unsqueeze");

    // the filter input of ExtendConvTranspose comes from the rearrange node with the fractal
    // layout kept and the int8 dtype
    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    std::string filterProducerType;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(extendNode, EXTEND_CONV_TRANSPOSE_FILTER_INDEX, filterProducerType));
    EXPECT_EQ(filterProducerType, "RearrangeA16W8WeightFzOp");
    TensorDesc filterDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_FILTER_INDEX, filterDesc), GRAPH_SUCCESS);
    EXPECT_EQ(filterDesc.GetFormat(), FORMAT_FRACTAL_Z);
    EXPECT_EQ(filterDesc.GetDataType(), DT_INT8);
}

// Test 28: fp32XA16W8FusionSuccess - fp32 x is converted to the fp16 path, the int8 filter is
// still rearranged and the fixed shift value is derived from the int8 filter (migrated from the
// updated fixpipe pass)
TEST_F(Conv2DTransposeFixpipeFusionPassTest, fp32XA16W8FusionSuccess)
{
    TestGraph builder("fp32XA16W8FusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_FLOAT, DT_INT8, DT_FLOAT16), {0, 1, 2})
                     .SetOutput("Conv2DTranspose")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);

    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    // the x input is converted to fp16 after the Unsqueeze node
    TensorDesc xDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_X_INDEX, xDesc), GRAPH_SUCCESS);
    EXPECT_EQ(xDesc.GetDataType(), DT_FLOAT16);
    // the int8 filter is rearranged
    EXPECT_TRUE(GraphChecker::HasNode(graph, "RearrangeA16W8WeightFzOp"));
    // the fixed shift value is derived from the int8 filter dtype
    int64_t fixedShiftVal = 0;
    EXPECT_EQ(extendNode.GetAttr(ge::AscendString("fixed_shift_value"), fixedShiftVal), GRAPH_SUCCESS);
    EXPECT_EQ(fixedShiftVal, 13);
}

// Test 29: int32OutputConvertedSuccess - int32 output without postcube is converted to the int8
// path of ExtendConvTranspose (migrated from the updated fixpipe pass)
TEST_F(Conv2DTransposeFixpipeFusionPassTest, int32OutputConvertedSuccess)
{
    TestGraph builder("int32OutputConvertedSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .SetOutput("Conv2DTranspose")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);

    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    TensorDesc outDesc;
    EXPECT_EQ(extendNode.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_INT8);
    GNode squeezeNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));
    TensorDesc squeezeOutDesc;
    EXPECT_EQ(squeezeNode.GetOutputDesc(0, squeezeOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(squeezeOutDesc.GetDataType(), DT_INT8);
}

// Test 30: fp32FilterConvertedSuccess - fp32 x + fp32 filter are both converted to the fp16
// path, no rearrange node is inserted and the fixed shift value is derived from the fp32 filter,
// the fp32 bias keeps its dtype with the _keep_dtype attr
TEST_F(Conv2DTransposeFixpipeFusionPassTest, fp32FilterConvertedSuccess)
{
    TestGraph builder("fp32FilterConvertedSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_FLOAT, DT_FLOAT, DT_FLOAT)
                                  .WithBias(DT_FLOAT, {32}),
                              {0, 1, 2, 3})
                     .SetOutput("Conv2DTranspose")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);

    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    // the x and filter inputs are both converted to fp16
    TensorDesc xDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_X_INDEX, xDesc), GRAPH_SUCCESS);
    EXPECT_EQ(xDesc.GetDataType(), DT_FLOAT16);
    TensorDesc filterDesc;
    EXPECT_EQ(extendNode.GetInputDesc(EXTEND_CONV_TRANSPOSE_FILTER_INDEX, filterDesc), GRAPH_SUCCESS);
    EXPECT_EQ(filterDesc.GetDataType(), DT_FLOAT16);
    // the fp32 filter takes no rearrange node
    EXPECT_FALSE(GraphChecker::HasNode(graph, "RearrangeA16W8WeightFzOp"));
    // the fixed shift value is derived from the fp32 filter dtype
    int64_t fixedShiftVal = 0;
    EXPECT_EQ(extendNode.GetAttr(ge::AscendString("fixed_shift_value"), fixedShiftVal), GRAPH_SUCCESS);
    EXPECT_EQ(fixedShiftVal, 42);
    // the fp32 bias keeps its dtype by the _keep_dtype attr
    int64_t keepDtype = 0;
    EXPECT_EQ(extendNode.GetAttr(ge::AscendString("_keep_dtype"), keepDtype), GRAPH_SUCCESS);
    EXPECT_EQ(keepDtype, 1);
}

// Test 31: outputTransposeFusionSuccess - the single Transpose successor with a 4D NHWC output
// is removed, the ExtendConvTranspose output becomes NDHWC and the Squeeze output takes the
// removed Transpose 4D desc (migrated from the updated fixpipe pass)
TEST_F(Conv2DTransposeFixpipeFusionPassTest, outputTransposeFusionSuccess)
{
    TestGraph builder("outputTransposeFusionSuccess");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose"), {0, 1, 2})
                     .AddNode(
                         TransposeConfig::Basic("Transpose", DT_FLOAT16, FORMAT_NHWC, {1, 32, 32, 32}, {1, 32, 32, 32}))
                     .Connect("Conv2DTranspose", 0, "Transpose", 0)
                     .SetOutput("Transpose")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);

    // the Transpose node is removed by the subgraph replace
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));

    // the ExtendConvTranspose output becomes NDHWC with the D dim inserted at index 1
    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    TensorDesc outDesc;
    EXPECT_EQ(extendNode.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetFormat(), FORMAT_NDHWC);
    EXPECT_EQ(outDesc.GetShape().GetDims(), std::vector<int64_t>({1, 1, 32, 32, 32}));

    // the Squeeze output takes the removed Transpose 4D NHWC desc
    GNode squeezeNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));
    TensorDesc squeezeOutDesc;
    EXPECT_EQ(squeezeNode.GetOutputDesc(0, squeezeOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(squeezeOutDesc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(squeezeOutDesc.GetShape().GetDims(), std::vector<int64_t>({1, 32, 32, 32}));
    EXPECT_EQ(squeezeOutDesc.GetDataType(), DT_FLOAT16);
}

// Keep the Transpose output's FP32 boundary desc, but use the normalized FP16 dtype on
// the internal ExtendConvTranspose output, which has no FP32 kernel in this scenario.
TEST_F(Conv2DTransposeFixpipeFusionPassTest, fp32OutputTransposeUsesFp16ExtendOutput)
{
    TestGraph builder("fp32OutputTransposeUsesFp16ExtendOutput");
    auto graph = builder.SetSocAscend950()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_FLOAT, DT_FLOAT, DT_FLOAT), {0, 1, 2})
                     .AddNode(
                         TransposeConfig::Basic("Transpose", DT_FLOAT, FORMAT_NHWC, {1, 32, 32, 32}, {1, 32, 32, 32}))
                     .Connect("Conv2DTranspose", 0, "Transpose", 0)
                     .SetOutput("Transpose")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Transpose"));

    GNode extendNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTranspose", extendNode));
    TensorDesc extendOutDesc;
    ASSERT_EQ(extendNode.GetOutputDesc(0, extendOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(extendOutDesc.GetFormat(), FORMAT_NDHWC);
    EXPECT_EQ(extendOutDesc.GetDataType(), DT_FLOAT16);

    GNode squeezeNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Squeeze", squeezeNode));
    TensorDesc squeezeOutDesc;
    ASSERT_EQ(squeezeNode.GetOutputDesc(0, squeezeOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(squeezeOutDesc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(squeezeOutDesc.GetDataType(), DT_FLOAT16);

    GNode castNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Cast", castNode));
    TensorDesc castInDesc;
    TensorDesc castOutDesc;
    ASSERT_EQ(castNode.GetInputDesc(0, castInDesc), GRAPH_SUCCESS);
    ASSERT_EQ(castNode.GetOutputDesc(0, castOutDesc), GRAPH_SUCCESS);
    EXPECT_EQ(castInDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(castOutDesc.GetDataType(), DT_FLOAT);
    EXPECT_EQ(castOutDesc.GetFormat(), FORMAT_NHWC);
    auto castInput = castNode.GetInDataNodesAndPortIndexs(0);
    ASSERT_NE(castInput.first, nullptr);
    AscendString castInputType;
    ASSERT_EQ(castInput.first->GetType(castInputType), GRAPH_SUCCESS);
    EXPECT_EQ(castInputType, AscendString("Squeeze"));
}

// Test 32: dualFixpipeFusionSuccess - the cube vector fuse soc with two FixPipe branches
// creates ExtendConvTransposeV2 with the dual output (migrated from the updated fixpipe pass)
TEST_F(Conv2DTransposeFixpipeFusionPassTest, dualFixpipeFusionSuccess)
{
    TestGraph builder("dualFixpipeFusionSuccess");
    SetCubeVectorCombine("fuse");
    auto graph = builder.SetSocMC62()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant2"))
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .Connect("Conv2DTranspose", 0, "AscendDequant2", 0)
                     .SetOutput("AscendDequant")
                     .SetOutput("AscendDequant2")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));

    // the V2 node carries the dual output flag and the two squeeze nodes squeeze both outputs
    GNode extendNode;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTransposeV2", extendNode));
    bool dualOutput = false;
    EXPECT_EQ(extendNode.GetAttr(ge::AscendString("dual_output"), dualOutput), GRAPH_SUCCESS);
    EXPECT_TRUE(dualOutput);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 2);
    TensorDesc outDesc0;
    EXPECT_EQ(extendNode.GetOutputDesc(0, outDesc0), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc0.GetFormat(), FORMAT_NCDHW);
    TensorDesc outDesc1;
    EXPECT_EQ(extendNode.GetOutputDesc(1, outDesc1), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc1.GetFormat(), FORMAT_NCDHW);
    // both FixPipe branches take the fp16 output desc of their own AscendDequant, the int8
    // swap of the mixed dtype dual FixPipe branches is covered by the requant scenario
    EXPECT_EQ(outDesc0.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(outDesc1.GetDataType(), DT_FLOAT16);
}

// Test 33: dualFixpipeEquivalentKeepDualSuccess - the equivalent dual outputs (same dtype and
// same relu behaviour) keep the dual output form in this migration: the two V2 output paths
// carry the same value, which is functionally equivalent to the single output downgrade of the
// original fixpipe pass
TEST_F(Conv2DTransposeFixpipeFusionPassTest, dualFixpipeEquivalentKeepDualSuccess)
{
    TestGraph builder("dualFixpipeEquivalentKeepDualSuccess");
    SetCubeVectorCombine("fuse");
    auto graph = builder.SetSocMC62()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .AddAscendRequant(AscendRequantConfig::Basic("AscendRequant"))
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .Connect("Conv2DTranspose", 0, "AscendRequant", 0)
                     .SetOutput("AscendDequant")
                     .SetOutput("AscendRequant")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendRequant"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));
}

// Test 34: dualOutputPassthroughSuccess - the mixed dual output (one FixPipe branch + one
// passthrough consumer) creates ExtendConvTransposeV2, the passthrough path takes the
// conv2dtranspose original output (migrated from the updated fixpipe pass)
TEST_F(Conv2DTransposeFixpipeFusionPassTest, dualOutputPassthroughSuccess)
{
    TestGraph builder("dualOutputPassthroughSuccess");
    SetCubeVectorCombine("fuse");
    Conv2DTransposeConfig convConfig = Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32);
    auto graph = builder.SetSocMC62()
                     .AddNode(convConfig, {0, 1, 2})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .AddNode(DataConfig::Basic("PassthroughData", DT_INT32, FORMAT_NCHW, {1, 32, 32, 32}), {})
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .Connect("Conv2DTranspose", 0, "PassthroughData", 0)
                     .SetOutput("AscendDequant")
                     .SetOutput("PassthroughData")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true, true);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "AscendDequant"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "FixPipe"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Squeeze"), 2);
}

// Keep the quantized FixPipe output on y0 even when the FP16 passthrough consumer was
// encountered first. The original consumers are mapped by the replacement boundary.
TEST_F(Conv2DTransposeFixpipeFusionPassTest, dualQuantAndFp16PassthroughQuantPrimary)
{
    TestGraph builder("dualQuantAndFp16PassthroughQuantPrimary");
    SetCubeVectorCombine("fuse");
    auto graph = builder.SetSocMC62()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_FLOAT16, DT_FLOAT16, DT_FLOAT16),
                              {0, 1, 2})
                     .AddNode(DataConfig::Basic("PassthroughData", DT_FLOAT16, FORMAT_NCHW, {1, 32, 32, 32}), {})
                     .AddAscendQuant(AscendQuantConfig::Basic("AscendQuant", DT_FLOAT16))
                     .Connect("Conv2DTranspose", 0, "PassthroughData", 0)
                     .Connect("Conv2DTranspose", 0, "AscendQuant", 0)
                     .SetOutput("PassthroughData")
                     .SetOutput("AscendQuant")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true, true);
    GNode extendNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTransposeV2", extendNode));
    TensorDesc y0Desc;
    TensorDesc y1Desc;
    ASSERT_EQ(extendNode.GetOutputDesc(0, y0Desc), GRAPH_SUCCESS);
    ASSERT_EQ(extendNode.GetOutputDesc(1, y1Desc), GRAPH_SUCCESS);
    EXPECT_EQ(y0Desc.GetDataType(), DT_INT8);
    EXPECT_EQ(y1Desc.GetDataType(), DT_FLOAT16);
}

// When the passthrough output is y0, the sole FixPipe belongs to y1 and its scale
// must connect to scale1 (input 5), leaving scale0 (input 4) disconnected.
TEST_F(Conv2DTransposeFixpipeFusionPassTest, dualPassthroughPrimaryUsesScale1)
{
    TestGraph builder("dualPassthroughPrimaryUsesScale1");
    SetCubeVectorCombine("fuse");
    auto graph = builder.SetSocMC62()
                     .AddNode(Conv2DTransposeConfig::Basic("Conv2DTranspose", DT_INT8, DT_INT8, DT_INT32), {0, 1, 2})
                     .AddNode(DataConfig::Basic("PassthroughData", DT_INT32, FORMAT_NCHW, {1, 32, 32, 32}), {})
                     .AddAscendDequant(AscendDequantConfig::Basic("AscendDequant"))
                     .Connect("Conv2DTranspose", 0, "PassthroughData", 0)
                     .Connect("Conv2DTranspose", 0, "AscendDequant", 0)
                     .SetOutput("PassthroughData")
                     .SetOutput("AscendDequant")
                     .Build();
    RunPassAndCheck(graph, SUCCESS, true, true);
    GNode extendNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "ExtendConvTransposeV2", extendNode));
    EXPECT_EQ(extendNode.GetInDataNodesAndPortIndexs(4).first, nullptr);
    EXPECT_NE(extendNode.GetInDataNodesAndPortIndexs(5).first, nullptr);
}
