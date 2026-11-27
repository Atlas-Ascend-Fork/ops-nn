/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// NormalizeBatch arch35 tiling 单元测试。
//   覆盖：
//     - 注册存在性（tiling / tiling_parse）
//     - 四模板 TilingKey 选择（per_feature/all_features × SmallD/BigD）
//     - 合法域（normalize_type 两值、epsilon 正值）
//     - 非法拒绝（rank、dtype、format、seq_len 形状、normalize_type、epsilon、d=1、输出不一致）
//   平台基准：UB_SIZE=253952 / CORE_NUM=56（与 l2_normalize UT 同款 Ascend950 编译信息）。
//     BigD 判定预算 ≈ 253952 - 8192(RESERVE) - seqLen/scalar ≈ 245600B：
//     per_feature BigD 需 rowStride > 预算/2（双缓冲折算，d > ~30700）；
//     all_features BigD 需 c*rowStride*4 > 预算。

#include <gtest/gtest.h>

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/normalize_batch_tiling_arch35.h"
#include "../../../../op_kernel/arch35/normalize_batch_struct.h"

namespace {

constexpr const char* OP_TYPE = "NormalizeBatch";
constexpr const char* COMPILE_INFO = R"({
    "hardware_info": {
        "BT_SIZE": 0,
        "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 253952,
        "L2_SIZE": 33554432,
        "L1_SIZE": 524288,
        "L0A_SIZE": 65536,
        "L0B_SIZE": 65536,
        "L0C_SIZE": 131072,
        "CORE_NUM": 56,
        "socVersion": "Ascend950"
    }
})";

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t key = 0;
};

TilingResult RunTiling(const std::vector<int64_t>& xDims, const std::vector<int64_t>& seqDims,
                       const std::string& normalizeType, float epsilon = 1e-5f, ge::DataType xType = ge::DT_FLOAT,
                       ge::DataType seqType = ge::DT_INT32, ge::DataType yType = ge::DT_FLOAT,
                       ge::Format xFormat = ge::FORMAT_ND)
{
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> version = {{"NpuArch", "3510"}, {"Short_SoC_version", "ASCEND950"}};
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics, version);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::NormalizeBatchCompileInfo compileInfo;

    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    if (opImpl == nullptr || opImpl->tiling == nullptr || opImpl->tiling_parse == nullptr) {
        return {};
    }

    auto parseHolder = gert::KernelRunContextFaker()
                           .SetOpType(OP_TYPE)
                           .KernelIONum(2, 1)
                           .Inputs({const_cast<char*>(COMPILE_INFO), reinterpret_cast<void*>(&platformInfo)})
                           .Outputs({&compileInfo})
                           .Build();
    auto* parseContext = parseHolder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr || parseContext->GetPlatformInfo() == nullptr ||
        !parseContext->GetPlatformInfo()->Init()) {
        return {};
    }
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parseContext->GetPlatformInfo()->SetPlatformRes("version", version);
    if (opImpl->tiling_parse(parseHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return {};
    }

    gert::StorageShape xShape = MakeShape(xDims);
    gert::StorageShape seqShape = MakeShape(seqDims);
    gert::StorageShape yShape = MakeShape(xDims);
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (tilingData == nullptr || workspace == nullptr) {
        return {};
    }

    auto holder = gert::TilingContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1}, {1})
                      .InputShapes({&xShape, &seqShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xType, xFormat, xFormat)
                      .NodeInputTd(1, seqType, xFormat, xFormat)
                      .NodeOutputTd(0, yType, xFormat, xFormat)
                      .NodeAttrs({{"normalize_type", Ops::NN::AnyValue::CreateFrom<std::string>(normalizeType)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto* context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return {};
    }
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetPlatformInfo()->SetPlatformRes("version", version);

    TilingResult result;
    result.status = opImpl->tiling(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.key = context->GetTilingKey();
    }
    return result;
}

} // namespace

TEST(NormalizeBatchTiling, IsRegistered)
{
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    ASSERT_NE(opImpl, nullptr);
    EXPECT_NE(opImpl->tiling, nullptr);
    EXPECT_NE(opImpl->tiling_parse, nullptr);
}

// 四模板 TilingKey 选择（per_feature/all_features × SmallD/BigD）
TEST(NormalizeBatchTiling, SelectsAllFourTemplates)
{
    // per_feature + SmallD：canndev ST 用例 [8, 2, 1024]
    const auto perSmall = RunTiling({8, 2, 1024}, {8}, "per_feature");
    ASSERT_EQ(perSmall.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(perSmall.key, GET_TPL_TILING_KEY(0, 0));

    // per_feature + BigD：d=100000 → rowStride*4 ≈ 400KB > UB 预算
    const auto perBig = RunTiling({2, 2, 100000}, {2}, "per_feature");
    ASSERT_EQ(perBig.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(perBig.key, GET_TPL_TILING_KEY(0, 1));

    // all_features + SmallD：[2, 2, 1024] 平面 2*1024*4 = 8KB
    const auto allSmall = RunTiling({2, 2, 1024}, {2}, "all_features");
    ASSERT_EQ(allSmall.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(allSmall.key, GET_TPL_TILING_KEY(1, 0));

    // all_features + BigD：[4, 64, 1024] 平面 64*1024*4 = 256KB > UB 预算
    const auto allBig = RunTiling({4, 64, 1024}, {4}, "all_features");
    ASSERT_EQ(allBig.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(allBig.key, GET_TPL_TILING_KEY(1, 1));
}

TEST(NormalizeBatchTiling, AcceptsContractDomains)
{
    // 预算按双缓冲折算后（单份 ≤ 预算/2 ≈ 30708 elems），rowStride=60032 落 BigD：
    // 双缓冲实占 2*60032*4 ≈ 480KB，若误判 SmallD 将运行期 UB 溢出（检视意见回归锚点）
    const auto perBigBoundary = RunTiling({3, 2, 60000}, {3}, "per_feature");
    ASSERT_EQ(perBigBoundary.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(perBigBoundary.key, GET_TPL_TILING_KEY(0, 1));
    EXPECT_EQ(RunTiling({3, 2, 60000}, {3}, "all_features").status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({8, 2, 1024}, {8}, "per_feature", 0.5f).status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({1, 1, 2}, {1}, "per_feature").status, ge::GRAPH_SUCCESS);
}

TEST(NormalizeBatchTiling, RejectsInvalidInputs)
{
    // rank / 形状
    EXPECT_EQ(RunTiling({2, 3}, {2}, "per_feature").status, ge::GRAPH_FAILED);       // rank=2
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, {2}, "per_feature").status, ge::GRAPH_FAILED); // rank=4
    EXPECT_EQ(RunTiling({2, 3, 4}, {3}, "per_feature").status, ge::GRAPH_FAILED);    // seq_len ≠ (n,)
    EXPECT_EQ(RunTiling({2, 3, 4}, {2, 3}, "per_feature").status, ge::GRAPH_FAILED); // seq_len rank=2
    EXPECT_EQ(RunTiling({2, 3, 1}, {2}, "per_feature").status, ge::GRAPH_FAILED);    // d=1
    // dtype / format
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "per_feature", 1e-5f, ge::DT_FLOAT16).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "per_feature", 1e-5f, ge::DT_FLOAT, ge::DT_INT64).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "per_feature", 1e-5f, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16).status,
              ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "per_feature", 1e-5f, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::FORMAT_NCHW)
                  .status,
              ge::GRAPH_FAILED);
    // attr 值域
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "per_features").status, ge::GRAPH_FAILED); // 非法 normalize_type
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "").status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "per_feature", 0.0f).status, ge::GRAPH_FAILED); // epsilon ≤ 0
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "per_feature", -1.0f).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "per_feature", std::numeric_limits<float>::infinity()).status,
              ge::GRAPH_FAILED); // epsilon +Inf（isfinite 拦截）
    EXPECT_EQ(RunTiling({2, 3, 4}, {2}, "per_feature", std::numeric_limits<float>::quiet_NaN()).status,
              ge::GRAPH_FAILED); // epsilon NaN
}
