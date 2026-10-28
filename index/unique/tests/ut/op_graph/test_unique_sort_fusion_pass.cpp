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
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_nn_ops.h"
#include "../../../op_graph/fusion_pass/unique_sort_fusion_pass.h"
#include "register/register_custom_pass.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops;

namespace {
const std::string kSortOpType = "Sort";
const std::string kUniqueOpType = "Unique";
const std::string kUniqueWithCountsOpType = "UniqueWithCounts";
const std::string kUniqueConsecutiveOpType = "UniqueConsecutive";
const std::string kAdjacentDifferenceOpType = "AdjacentDifference";
const std::string kCumsumOpType = "Cumsum";
const std::string kScatterElementsOpType = "ScatterElements";
} // namespace

// Configure both platform maps for the given soc: the legacy PlatformInfo map
// (short_soc_version / ai_core_cnt) and the PlatFormInfos map carrying the NpuArch
// resource (the pass gates via ops::CheckNpuArch, which reads the latter). The
// soc-to-arch mapping mirrors the soc list used by the cases below.
static void SetSocVersion(const std::string& socVersion)
{
    static const std::map<std::string, std::string> kSocToNpuArch = {
        {"Ascend950", "3510"}, {"MC62CM12A", "5102"}, {"Ascend910_93", "2201"}};
    auto archIt = kSocToNpuArch.find(socVersion);
    const std::string npuArch = (archIt != kSocToNpuArch.end()) ? archIt->second : "2201";

    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = socVersion;
    optiCompilationInfo.soc_version = socVersion;
    fe::PlatformInfoManager::Instance().platform_info_map_[socVersion] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    fe::PlatFormInfos platformInfos;
    platformInfos.Init();
    std::map<std::string, std::string> version{{"NpuArch", npuArch}};
    platformInfos.SetPlatformRes("version", version);
    fe::PlatformInfoManager::Instance().platform_infos_map_[socVersion] = platformInfos;
    // The PlatFormInfos overload of GetPlatformInfoWithOutSocVersion picks the map
    // entry by OptionalInfos::GetSocVersion(), so the non-legacy OptionalInfos must
    // be set as well (mirrors InplaceAddRmsNormFusionPassTest).
    fe::OptionalInfos options;
    options.Init();
    options.SetSocVersion(socVersion);
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(options);
}

struct UniqueGraphParam {
    std::vector<int64_t> xDims{16};
    DataType xDtype = DT_FLOAT;
    DataType outIdxDtype = DT_INT32;
    bool withCounts = false;
    bool useY = true;
    bool useIdx = true;
    bool useCnt = false;
};

static GraphPtr BuildUniqueGraph(const std::string& graphName, const UniqueGraphParam& param)
{
    auto graphBuilder = es::EsGraphBuilder(graphName.c_str());
    auto x = graphBuilder.CreateInput(0, "x", param.xDtype, FORMAT_ND, param.xDims);

    es::EsTensorHolder yOut;
    es::EsTensorHolder idxOut;
    es::EsTensorHolder cntOut;
    GNode uniqueNode;
    if (param.withCounts) {
        auto uniqueOut = es::UniqueWithCounts(x, param.outIdxDtype);
        yOut = uniqueOut.y;
        idxOut = uniqueOut.idx;
        cntOut = uniqueOut.count;
        uniqueNode = *uniqueOut.y.GetProducer();
    } else {
        auto uniqueOut = es::Unique(x, param.outIdxDtype);
        yOut = uniqueOut.y;
        idxOut = uniqueOut.idx;
        uniqueNode = *uniqueOut.y.GetProducer();
    }

    // set input/output descs: x is 1-D, y/count are dynamic dim, idx has the same shape as x.
    TensorDesc xDesc(Shape(param.xDims), FORMAT_ND, param.xDtype);
    uniqueNode.UpdateInputDesc(0, xDesc);
    TensorDesc yDesc(Shape({static_cast<int64_t>(-1)}), FORMAT_ND, param.xDtype);
    uniqueNode.UpdateOutputDesc(0, yDesc);
    TensorDesc idxDesc(Shape(param.xDims), FORMAT_ND, param.outIdxDtype);
    uniqueNode.UpdateOutputDesc(1, idxDesc);
    if (param.withCounts) {
        TensorDesc cntDesc(Shape({static_cast<int64_t>(-1)}), FORMAT_ND, param.outIdxDtype);
        uniqueNode.UpdateOutputDesc(2, cntDesc);
    }

    // graph outputs decide which outputs of the unique node have downstream consumers.
    std::vector<es::EsTensorHolder> graphOutputs;
    if (param.useY) {
        graphOutputs.push_back(yOut);
    }
    if (param.useIdx) {
        graphOutputs.push_back(idxOut);
    }
    if (param.useCnt) {
        graphOutputs.push_back(cntOut);
    }
    auto graph = graphBuilder.BuildAndReset(graphOutputs);
    return GraphPtr(std::move(graph));
}

static int32_t CountNodeType(const GraphPtr& graph, const std::string& opType)
{
    int32_t count = 0;
    for (auto& node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == SUCCESS && type == opType.c_str()) {
            count++;
        }
    }
    return count;
}

static bool FindFirstNodeByType(const GraphPtr& graph, const std::string& opType, GNode& foundNode)
{
    for (auto& node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == SUCCESS && type == opType.c_str()) {
            foundNode = node;
            return true;
        }
    }
    return false;
}

// The ES builder omits attrs equal to the IR default, and the runtime falls back to the default
// value, so an absent attr is equivalent to the expected (default) value.
template <typename T>
static void ExpectAttrIsOrDefault(const GNode& node, const std::string& attrName, const T& expectedValue)
{
    T attrValue = expectedValue;
    const bool attrExists = (node.GetAttr(attrName.c_str(), attrValue) == SUCCESS);
    EXPECT_TRUE(!attrExists || attrValue == expectedValue);
}

class UniqueSortFusionPassTest : public testing::Test {
protected:
    // Restore the global platform state after every case: SetSocVersion overwrites
    // PlatformInfoManager entries (platform_info_map_ / platform_infos_map_ / optional
    // compilation info) that other opgraph UT suites in the same process may rely on,
    // so snapshot the touched keys first (presence + value) and put them back in
    // TearDown instead of leaving the last case's platform behind. Mirrors
    // InplaceAddRmsNormFusionPassTest (PR #12471).
    static constexpr const char* kTouchedSocs[] = {"Ascend950", "MC62CM12A", "Ascend910_93"};

    void SetUp() override
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        savedLegacyInfo_ = manager.opti_compilation_info_;
        savedOptionsInfo_ = manager.opti_compilation_infos_;
        for (const char* soc : kTouchedSocs) {
            infoKeyExisted_[soc] = manager.platform_info_map_.count(soc) > 0;
            if (infoKeyExisted_[soc]) {
                infoKeyValues_[soc] = manager.platform_info_map_[soc];
            }
            infosKeyExisted_[soc] = manager.platform_infos_map_.count(soc) > 0;
            if (infosKeyExisted_[soc]) {
                infosKeyValues_[soc] = manager.platform_infos_map_[soc];
            }
        }
        // Default platform is not regbase; success cases set the platform explicitly.
        SetSocVersion("Ascend910_93");
    }

    void TearDown() override
    {
        auto& manager = fe::PlatformInfoManager::Instance();
        manager.opti_compilation_info_ = savedLegacyInfo_;
        manager.opti_compilation_infos_ = savedOptionsInfo_;
        for (const char* soc : kTouchedSocs) {
            if (infoKeyExisted_[soc]) {
                manager.platform_info_map_[soc] = infoKeyValues_[soc];
            } else {
                manager.platform_info_map_.erase(soc);
            }
            if (infosKeyExisted_[soc]) {
                manager.platform_infos_map_[soc] = infosKeyValues_[soc];
            } else {
                manager.platform_infos_map_.erase(soc);
            }
        }
    }

private:
    fe::OptionalInfo savedLegacyInfo_;
    fe::OptionalInfos savedOptionsInfo_;
    std::map<std::string, bool> infoKeyExisted_;
    std::map<std::string, fe::PlatformInfo> infoKeyValues_;
    std::map<std::string, bool> infosKeyExisted_;
    std::map<std::string, fe::PlatFormInfos> infosKeyValues_;
};

// UniqueWithCounts with all outputs used: full replacement structure is built.
TEST_F(UniqueSortFusionPassTest, uniqueSortFusionAllOutputsSuccess)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.withCounts = true;
    param.useCnt = true;
    param.outIdxDtype = DT_INT64;
    GraphPtr graph = BuildUniqueGraph("unique_sort_all_outputs_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodeType(graph, kSortOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kUniqueConsecutiveOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kAdjacentDifferenceOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kCumsumOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kScatterElementsOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kUniqueOpType), 0);
    EXPECT_EQ(CountNodeType(graph, kUniqueWithCountsOpType), 0);
}

// Unique with idx used (out_idx attr = int64): inverse index structure is built.
TEST_F(UniqueSortFusionPassTest, uniqueFusionIdxInt64AttrSuccess)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.outIdxDtype = DT_INT64;
    GraphPtr graph = BuildUniqueGraph("unique_sort_idx_int64_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodeType(graph, kSortOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kUniqueConsecutiveOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kScatterElementsOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kUniqueOpType), 0);

    // scatter output keeps the original idx dtype (int64).
    GNode scatterNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, kScatterElementsOpType, scatterNode));
    TensorDesc scatterYDesc;
    EXPECT_EQ(scatterNode.GetOutputDesc(0, scatterYDesc), SUCCESS);
    EXPECT_EQ(scatterYDesc.GetDataType(), DT_INT64);
}

// UniqueWithCounts whose idx output is unused: the inverse index structure must NOT be built, while
// the count output is still mapped through the dense boundary slot.
TEST_F(UniqueSortFusionPassTest, uniqueFusionIdxUnusedNoInverseStructure)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.withCounts = true;
    param.useIdx = false;
    param.useCnt = true;
    GraphPtr graph = BuildUniqueGraph("unique_sort_idx_unused_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodeType(graph, kSortOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kUniqueConsecutiveOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kAdjacentDifferenceOpType), 0);
    EXPECT_EQ(CountNodeType(graph, kCumsumOpType), 0);
    EXPECT_EQ(CountNodeType(graph, kScatterElementsOpType), 0);
    EXPECT_EQ(CountNodeType(graph, kUniqueWithCountsOpType), 0);

    // return_counts is true only because the count output is used.
    GNode uniqueConsNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, kUniqueConsecutiveOpType, uniqueConsNode));
    bool returnCounts = false;
    EXPECT_EQ(uniqueConsNode.GetAttr("return_counts", returnCounts), SUCCESS);
    EXPECT_EQ(returnCounts, true);
}

// Unique with only y used (out_idx attr absent, desc check path): no inverse structure, no count.
TEST_F(UniqueSortFusionPassTest, uniqueFusionYOnlySuccess)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.useIdx = false;
    GraphPtr graph = BuildUniqueGraph("unique_sort_y_only_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodeType(graph, kSortOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kUniqueConsecutiveOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kAdjacentDifferenceOpType), 0);
    EXPECT_EQ(CountNodeType(graph, kCumsumOpType), 0);
    EXPECT_EQ(CountNodeType(graph, kScatterElementsOpType), 0);
    EXPECT_EQ(CountNodeType(graph, kUniqueOpType), 0);

    GNode uniqueConsNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, kUniqueConsecutiveOpType, uniqueConsNode));
    ExpectAttrIsOrDefault<bool>(uniqueConsNode, "return_counts", false);
}

// verify key attributes and dtypes of the replacement nodes.
TEST_F(UniqueSortFusionPassTest, uniqueFusionVerifyAttrsAndDtypes)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.withCounts = true;
    param.useCnt = true;
    param.outIdxDtype = DT_INT32;
    GraphPtr graph = BuildUniqueGraph("unique_sort_verify_attrs_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);

    // Sort attrs keep consistent with the original pass (attrs equal to IR defaults may be omitted).
    GNode sortNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, kSortOpType, sortNode));
    int64_t sortAxis = -1;
    EXPECT_EQ(sortNode.GetAttr("axis", sortAxis), SUCCESS);
    EXPECT_EQ(sortAxis, 0);
    ExpectAttrIsOrDefault<bool>(sortNode, "descending", false);
    bool stable = false;
    EXPECT_EQ(sortNode.GetAttr("stable", stable), SUCCESS);
    EXPECT_EQ(stable, true);

    // UniqueConsecutive attrs keep consistent with the original pass.
    GNode uniqueConsNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, kUniqueConsecutiveOpType, uniqueConsNode));
    ExpectAttrIsOrDefault<bool>(uniqueConsNode, "return_idx", false);
    // The legacy pass sets _unknown_shape_type = 3 (THRID_CLASS_DYN_TYPE) in
    // SetUniqueConsAttr and gives up the fusion on failure; the replacement must
    // carry the same attribute.
    int64_t unknownShapeType = 0;
    EXPECT_EQ(uniqueConsNode.GetAttr("_unknown_shape_type", unknownShapeType), SUCCESS);
    EXPECT_EQ(unknownShapeType, 3);
    bool returnCounts = false;
    EXPECT_EQ(uniqueConsNode.GetAttr("return_counts", returnCounts), SUCCESS);
    EXPECT_EQ(returnCounts, true);
    ExpectAttrIsOrDefault<int64_t>(uniqueConsNode, "axis", 1000);
    DataType outIdxDtype = DT_UNDEFINED;
    EXPECT_EQ(uniqueConsNode.GetAttr("out_idx", outIdxDtype), SUCCESS);
    EXPECT_EQ(outIdxDtype, DT_INT32);

    // y output dtype is preserved from the original unique node.
    TensorDesc yOutDesc;
    EXPECT_EQ(uniqueConsNode.GetOutputDesc(0, yOutDesc), SUCCESS);
    EXPECT_EQ(yOutDesc.GetDataType(), DT_FLOAT);

    // ScatterElements axis attr and idx output dtype.
    GNode scatterNode;
    ASSERT_TRUE(FindFirstNodeByType(graph, kScatterElementsOpType, scatterNode));
    ExpectAttrIsOrDefault<int64_t>(scatterNode, "axis", 0);
    TensorDesc scatterYDesc;
    EXPECT_EQ(scatterNode.GetOutputDesc(0, scatterYDesc), SUCCESS);
    EXPECT_EQ(scatterYDesc.GetDataType(), DT_INT32);
}

// bf16 input, supported by the original dtype whitelist.
TEST_F(UniqueSortFusionPassTest, uniqueSortFusionBf16Success)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.xDtype = DT_BF16;
    GraphPtr graph = BuildUniqueGraph("unique_sort_bf16_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeType(graph, kSortOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kUniqueOpType), 0);
}

// different shape: 1-D with 2560 elements, same as the original ut.
TEST_F(UniqueSortFusionPassTest, uniqueSortFusionDifferentShapeSuccess)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.xDims = {2560};
    param.xDtype = DT_FLOAT;
    param.withCounts = true;
    param.useCnt = true;
    param.outIdxDtype = DT_INT32;
    GraphPtr graph = BuildUniqueGraph("unique_sort_shape_2560_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeType(graph, kSortOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kUniqueWithCountsOpType), 0);
}

// MC62CM12A (dav 5102) is also a regbase platform.
TEST_F(UniqueSortFusionPassTest, uniqueSortFusionMc62PlatformSuccess)
{
    SetSocVersion("MC62CM12A");
    UniqueGraphParam param;
    GraphPtr graph = BuildUniqueGraph("unique_sort_mc62_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(CountNodeType(graph, kSortOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kUniqueOpType), 0);
}

// unsupported input dtype (double) returns NOT_CHANGED and keeps the original node.
TEST_F(UniqueSortFusionPassTest, unsupportedDtypeFail)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.xDtype = DT_DOUBLE;
    GraphPtr graph = BuildUniqueGraph("unique_sort_dtype_fail_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeType(graph, kUniqueOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kSortOpType), 0);
}

// non 1-D input returns NOT_CHANGED.
TEST_F(UniqueSortFusionPassTest, unsupportedDimFail)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.xDims = {16, 16};
    GraphPtr graph = BuildUniqueGraph("unique_sort_dim_fail_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeType(graph, kUniqueOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kSortOpType), 0);
}

// out_idx attr with unsupported dtype (float) returns NOT_CHANGED.
TEST_F(UniqueSortFusionPassTest, unsupportedIdxDtypeAttrFail)
{
    SetSocVersion("Ascend950");
    UniqueGraphParam param;
    param.outIdxDtype = DT_FLOAT;
    GraphPtr graph = BuildUniqueGraph("unique_sort_idx_dtype_fail_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeType(graph, kUniqueOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kSortOpType), 0);
}

// non regbase platform (Ascend910_93) returns NOT_CHANGED.
TEST_F(UniqueSortFusionPassTest, unsupportedPlatformFail)
{
    // SetUp already set Ascend910_93 which is not a regbase platform.
    UniqueGraphParam param;
    GraphPtr graph = BuildUniqueGraph("unique_sort_platform_fail_test", param);

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graph, passContext);
    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountNodeType(graph, kUniqueOpType), 1);
    EXPECT_EQ(CountNodeType(graph, kSortOpType), 0);
}

// two independent unique nodes in one graph are all fused.
TEST_F(UniqueSortFusionPassTest, multiNodeFusionSuccess)
{
    SetSocVersion("Ascend950");
    auto graphBuilder = es::EsGraphBuilder("unique_sort_multi_node_test");
    auto x1 = graphBuilder.CreateInput(0, "x1", DT_FLOAT, FORMAT_ND, {16});
    auto x2 = graphBuilder.CreateInput(1, "x2", DT_INT32, FORMAT_ND, {32});

    auto uniqueOut1 = es::Unique(x1, DT_INT32);
    auto uniqueOut2 = es::Unique(x2, DT_INT64);

    GNode uniqueNode1 = *uniqueOut1.y.GetProducer();
    GNode uniqueNode2 = *uniqueOut2.y.GetProducer();
    uniqueNode1.UpdateInputDesc(0, TensorDesc(Shape({16}), FORMAT_ND, DT_FLOAT));
    uniqueNode1.UpdateOutputDesc(0, TensorDesc(Shape({-1}), FORMAT_ND, DT_FLOAT));
    uniqueNode1.UpdateOutputDesc(1, TensorDesc(Shape({16}), FORMAT_ND, DT_INT32));
    uniqueNode2.UpdateInputDesc(0, TensorDesc(Shape({32}), FORMAT_ND, DT_INT32));
    uniqueNode2.UpdateOutputDesc(0, TensorDesc(Shape({-1}), FORMAT_ND, DT_INT32));
    uniqueNode2.UpdateOutputDesc(1, TensorDesc(Shape({32}), FORMAT_ND, DT_INT64));

    auto graph = graphBuilder.BuildAndReset({uniqueOut1.y, uniqueOut1.idx, uniqueOut2.y, uniqueOut2.idx});
    GraphPtr graphPtr(std::move(graph));

    CustomPassContext passContext;
    passContext.SetPassName("UniqueSortFusionPass");
    ops::UniqueSortFusionPass pass;
    Status status = pass.Run(graphPtr, passContext);
    EXPECT_EQ(status, SUCCESS);

    EXPECT_EQ(CountNodeType(graphPtr, kSortOpType), 2);
    EXPECT_EQ(CountNodeType(graphPtr, kUniqueConsecutiveOpType), 2);
    EXPECT_EQ(CountNodeType(graphPtr, kScatterElementsOpType), 2);
    EXPECT_EQ(CountNodeType(graphPtr, kUniqueOpType), 0);
}
