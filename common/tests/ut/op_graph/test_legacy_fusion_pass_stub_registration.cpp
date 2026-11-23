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
 * \file test_legacy_fusion_pass_stub_registration.cpp
 * \brief fe 融合 pass 桩登记基线校验 UT：校验 legacy_fusion_pass.cc 的 60 条登记
 *        （名字/类型/attr 逐条对照基线表）、排除名单不登记，以及运行期版本门槛分层行为。
 *
 * 前置条件：构建 soc 含 ascend950 且构建环境 ge_compiler >= 9.0.0 时，注册代码经
 * graph_plugin 对象编入本 UT（common/src/op_graph/CMakeLists.txt 门控与产品一致）。
 * 注册发生在静态初始化期（一次性），门槛分层场景通过 exec 自身 + 环境变量注入
 *（LEGACY_STUB_UT_GE_VER / LEGACY_STUB_UT_QUERY_FAIL，见 fe_fusion_pass_registry_stub.cpp）。
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

#include "fe_fusion_pass_registry_ut_stub.h"

namespace {

constexpr uint64_t kAttrForbiddenClose = 0x01;
constexpr uint64_t kAttrSingleSceneOpen = 0x04;

// 登记基线表：名字/类型/attr 与 legacy_fusion_pass.cc 的 60 条登记一一对应（照抄 canndev 注册体）。
// 本表是独立于实现的期望值，legacy_fusion_pass.cc 改动导致偏差时在此基线断言失败。
struct BaselineEntry {
    const char* name;
    fe::GraphFusionPassType type;
    uint64_t attr;
};

const BaselineEntry kBaseline[] = {
    // ===== 融合 pass：编译门槛 9.0 组（22 个）=====
    {"AInplaceAddFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"AInplaceSubFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"AInplaceUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"BatchMatMul2MulFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"Conv2DTransposeWeightTransFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"EinsumPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"GatherToGatherV2FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"GemmToMatmulFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"IndexByTensorStaticFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"MatmulReshapeTransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, 0},
    {"MatmulToGemmOpFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"MatMulToMatmulV3FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"MatMulUnsqueezeSqueezeFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"MaxPoolFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"MaxPoolGradWithArgmaxV3FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"MaxPoolWithArgmaxV3FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"QuantBatchMatmulV4ToV3FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"TensorScatterAddFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"TensorScatterUpdateFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"UniqueSortFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"WhereFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"ZZMatMulTOQBMMV3FusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    // ===== 分解改写 pass：编译门槛 9.0 组（9 个）=====
    {"ADeformableConv2dPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv2DBackpropFilterToV3FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv2DBackpropInputToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv2DTransposeToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv3DBackpropFilterToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv3DBackpropInputToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv3DTransposeToV2FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv3dToConv3dV2FusionPass", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"DepthwiseDfFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"DepthwiseDwMulFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"DepthwiseToConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    // ===== 融合 pass：编译门槛 9.1 组，固定 kCompatibleInherited（10 个）=====
    {"AConv2dMulFusion", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"ADepthwiseFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"ASplitConv2dConcatPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"Conv2DFixPipeToExtendConv2DFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv2DSqueezeBiasaddFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"Conv2dToConv2dV2FusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv3DDequantToQuantConv3DFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"PadConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"PadDepthwiseConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"QuantUnsqueezeConv2DFusionPass", fe::BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS, kAttrForbiddenClose},
    // ===== 融合 pass：编译门槛 9.1 组，运行判定型（12 个，运行门槛 9.1）=====
    {"AntiQuantMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"BatchMatMul2TransposeBatchMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"BatchMatMulTransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"FusedMatMulFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"MatMulBiasAddOpenFusionPass", fe::BUILT_IN_GRAPH_PASS, 0},
    {"MatMulReshapeBiasAddFusionPass", fe::BUILT_IN_GRAPH_PASS, 0},
    {"QuantBatchMatmulInplaceAddTransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"QuantBatchMatmulV3TransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"QuantBatchMatmulV3TransposeLimitFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"QuantBatchMatmulV4TransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"WeightQuantBatchMatmulV2TransposeFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"WeightQuantBatchMatmulV2TransposeNZFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    // ===== 分解改写 pass：编译门槛 9.1 组（1 个）=====
    {"Conv2DTransposeFixpipeFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    // ===== 融合 pass：编译门槛 9.3 组（5 个）=====
    {"AABiasaddConvFusion", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
    {"Conv2DQuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, kAttrSingleSceneOpen},
    {"Conv2DTQuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, kAttrSingleSceneOpen},
    {"Conv3DQuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, kAttrSingleSceneOpen},
    {"DWConv2DQuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, kAttrSingleSceneOpen},
    // ===== 融合 pass：编译门槛 9.3 组（fixpipe 公共 pass 4 个 + V2 QuantProcess 2 个）=====
    {"BatchMatmulV2QuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, kAttrSingleSceneOpen},
    {"FixPipeAbilityProcessPass", fe::BUILT_IN_GRAPH_PASS, 0x07},
    {"FIXPIPEAPREQUANTFUSIONPASS", fe::BUILT_IN_GRAPH_PASS, kAttrSingleSceneOpen},
    {"FIXPIPEARELUQUANTFUSIONPASS", fe::BUILT_IN_GRAPH_PASS, 0},
    {"FIXPIPEFUSIONPASS", fe::SECOND_ROUND_BUILT_IN_GRAPH_PASS, 0x07},
    {"MatmulV2QuantProcessFusionPass", fe::BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS, kAttrSingleSceneOpen},
    // ===== 融合 pass：编译门槛 9.2 组（1 个，运行门槛 9.2 + API 探测）=====
    {"MatmulQuantToFixpipeFusionPass", fe::BUILT_IN_GRAPH_PASS, kAttrForbiddenClose},
};

// 排除名单一：canndev 有同名注册但 nn 实现 Stage 非 kCompatibleInherited，登记会报 "pattern fusion is nullptr"
const char* kExcludedNonCompatStage[] = {"AddLayerNormDynamicQuantFusionPass",
                                         "AddLayerNormFusionPass",
                                         "AddRmsNormDynamicQuantFusionPass",
                                         "AddRmsNormDynamicQuantV2FusionPass",
                                         "AddRmsNormFusionGraphPass",
                                         "AddRmsNormQuantFusionPass",
                                         "ASoftmaxFusionPass",
                                         "AscendQuantV2ScatterFusionPass",
                                         "DynamicQuantUpdateScatterV2FusionPass",
                                         "InplaceAddRmsNormFusionPass",
                                         "LayerNormFusionPass",
                                         "LayerNormInferenceFusionPass",
                                         "LayerNormONNXFusionPass",
                                         "LayerNormONNXMULFusionPass",
                                         "MultiAddRmsNormDynamicQuantFusionPass",
                                         "PReluGradFusionPass",
                                         "QuantizeAddLayerNormPass",
                                         "ZConfusionSoftmaxGradFusionPass",
                                         "ZInplaceAddLayerNormFusionPass"};

// 排除名单二：canndev 无同名注册的 nn 新增 pass，无 fe 表名字需要接管
const char* kExcludedNoCanndevEntry[] = {
    "BucketizeFusionPass",      "FusedWeightQuantMatMulFusionPass",   "GeluV2ONNXFusionPass",
    "IndexPutV2FusionPass",     "LayerNormRemoveBroadcastFusionPass", "MaxPoolGradFusionPass",
    "SoftmaxGradExtFusionPass", "SoftmaxGradExtV2FusionPass",         "SortedSparseSegmentMeanGradFusionPass"};

// 近似名陷阱：canndev 独有的小写 d 注册（nn 侧对应大写 D），大小写不同名，不得登记
const char* kExcludedCaseTrap[] = {"Conv3dTransposeToV2FusionPass"};

std::map<std::string, std::pair<fe::GraphFusionPassType, uint64_t>> BuildActualMap()
{
    std::map<std::string, std::pair<fe::GraphFusionPassType, uint64_t>> actual;
    for (const auto& record : ut_stub_fe::GetRecords()) {
        actual[record.name] = {record.type, record.attr};
    }
    return actual;
}

// ===== 主基线用例：默认环境（版本 9.3.0）下 60 条登记逐项校验 =====
TEST(LegacyFusionPassStubRegistration, BaselineEntriesMatchExpected)
{
    const auto& records = ut_stub_fe::GetRecords();
    ASSERT_EQ(records.size(), sizeof(kBaseline) / sizeof(kBaseline[0])) << "登记条数偏离基线";

    auto actual = BuildActualMap();
    for (const auto& entry : kBaseline) {
        const auto it = actual.find(entry.name);
        ASSERT_TRUE(it != actual.end()) << "基线条目未登记: " << entry.name;
        EXPECT_EQ(it->second.first, entry.type) << "类型不匹配: " << entry.name;
        EXPECT_EQ(it->second.second, entry.attr) << "attr 不匹配: " << entry.name;
    }
    EXPECT_EQ(actual.size(), sizeof(kBaseline) / sizeof(kBaseline[0])) << "存在基线外的多余登记";
}

TEST(LegacyFusionPassStubRegistration, AllCreateFnAreNullptr)
{
    for (const auto& record : ut_stub_fe::GetRecords()) {
        EXPECT_EQ(record.create_fn, nullptr) << "工厂必须为 nullptr（创建由 REG_FUSION_PASS 承接）: " << record.name;
    }
}

TEST(LegacyFusionPassStubRegistration, ExcludedNamesNotRegistered)
{
    auto actual = BuildActualMap();
    for (const auto* name : kExcludedNonCompatStage) {
        EXPECT_TRUE(actual.find(name) == actual.end()) << "非兼容 Stage 名不得登记: " << name;
    }
    for (const auto* name : kExcludedNoCanndevEntry) {
        EXPECT_TRUE(actual.find(name) == actual.end()) << "canndev 无同名的新增名不得登记: " << name;
    }
    for (const auto* name : kExcludedCaseTrap) {
        EXPECT_TRUE(actual.find(name) == actual.end()) << "近似名陷阱（大小写不同名）不得登记: " << name;
    }
}

TEST(LegacyFusionPassStubRegistration, RegistryQueryConsistentWithRecords)
{
    size_t totalByType = 0;
    for (int type = fe::BUILT_IN_GRAPH_PASS; type <= fe::GRAPH_FUSION_PASS_TYPE_RESERVED; ++type) {
        const auto descMap = fe::FusionPassRegistry::GetInstance().GetPassDesc(
            static_cast<fe::GraphFusionPassType>(type));
        for (const auto& kv : descMap) {
            EXPECT_EQ(kv.second.create_fn, nullptr) << kv.first;
        }
        totalByType += descMap.size();
    }
    EXPECT_EQ(totalByType, ut_stub_fe::GetRecords().size());
}

// ===== 门槛分层用例：exec 自身注入环境变量，子进程断言登记条数 =====
// 子进程专用用例：读 LEGACY_STUB_UT_EXPECT_COUNT 断言登记条数
TEST(LegacyFusionPassStubGateEnv, ExpectCount)
{
    const char* expect = std::getenv("LEGACY_STUB_UT_EXPECT_COUNT");
    ASSERT_NE(expect, nullptr);
    EXPECT_EQ(ut_stub_fe::GetRecords().size(), static_cast<size_t>(std::atoi(expect)));
}

class LegacyFusionPassStubGateTest : public ::testing::TestWithParam<std::pair<int, int>> {};

// 参数：(LEGACY_STUB_UT_GE_VER, 期望登记条数)
// 门槛分布：运行门槛 9.0 × 55 条、9.1 × 12 条、9.2+API × 1 条（共 68）
TEST_P(LegacyFusionPassStubGateTest, VersionGateScenarios)
{
    const auto& param = GetParam();
    const std::string version = std::to_string(param.first);
    const std::string expect = std::to_string(param.second);

    setenv("LEGACY_STUB_UT_GE_VER", version.c_str(), 1);
    unsetenv("LEGACY_STUB_UT_QUERY_FAIL");
    setenv("LEGACY_STUB_UT_EXPECT_COUNT", expect.c_str(), 1);

    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        const int fd = open("/dev/null", O_WRONLY);
        if (fd >= 0) {
            (void)dup2(fd, STDOUT_FILENO);
            (void)dup2(fd, STDERR_FILENO);
        }
        const char* const argv[] = {"/proc/self/exe", "--gtest_filter=LegacyFusionPassStubGateEnv.*", nullptr};
        execv("/proc/self/exe", const_cast<char* const*>(argv));
        _exit(127);
    }
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status)) << "子进程异常终止, 版本场景 " << version;
    EXPECT_EQ(WEXITSTATUS(status), 0) << "版本 " << version << " 期望登记 " << expect << " 条不满足";
}

INSTANTIATE_TEST_SUITE_P(VersionGates, LegacyFusionPassStubGateTest,
                         ::testing::Values(std::make_pair(80000000, 0), // 版本不足 9.0：全部不登记
                                           std::make_pair(90000000,
                                                          55), // 9.0 门槛 55 条（含 DECOMPOSE 与 9.3 块固定 kCompat）
                                           std::make_pair(90100000, 67), // +12 条运行判定型（9.1 门槛）
                                           std::make_pair(90200000,
                                                          68))); // +MatmulQuantToFixpipe（9.2 门槛 + API 可用）

// 查询失败（弱符号返回错误码）场景：与版本不足同语义，全部不登记
TEST(LegacyFusionPassStubGateTest, QueryFailRegistersNothing)
{
    setenv("LEGACY_STUB_UT_QUERY_FAIL", "1", 1);
    setenv("LEGACY_STUB_UT_EXPECT_COUNT", "0", 1);

    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        const int fd = open("/dev/null", O_WRONLY);
        if (fd >= 0) {
            (void)dup2(fd, STDOUT_FILENO);
            (void)dup2(fd, STDERR_FILENO);
        }
        const char* const argv[] = {"/proc/self/exe", "--gtest_filter=LegacyFusionPassStubGateEnv.*", nullptr};
        execv("/proc/self/exe", const_cast<char* const*>(argv));
        _exit(127);
    }
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0) << "查询失败场景应不登记";
    unsetenv("LEGACY_STUB_UT_QUERY_FAIL");
}

} // namespace
