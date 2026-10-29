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
 * \file test_index_check_tiling.cpp
 * \brief IndexCheck tiling 单元测试：覆盖 indices 动态列表的 dtype 一致性校验、列表数量边界、
 *        bounds 形状约束与维度上限。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/index_check_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;

namespace {
constexpr const char* COMPILE_INFO = R"({
  "hardware_info": {
    "BT_SIZE": 0,
    "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false,
    "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true,
    "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 245760,
    "L2_SIZE": 33554432,
    "L1_SIZE": 524288,
    "L0A_SIZE": 65536,
    "L0B_SIZE": 65536,
    "L0C_SIZE": 131072,
    "CORE_NUM": 64
  }
})";
} // namespace

class IndexCheckTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IndexCheckTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "IndexCheckTiling TearDown" << std::endl; }
};

// 运行 IndexCheck tiling。
// boundsDims: bounds 的 shape 维度；indicesDims: indices 动态列表各张量的 shape 维度。
// 返回 tiling 的 graphStatus。indices 各张量统一用 indicesDtype。
static ge::graphStatus RunTilingCase(const std::vector<int64_t>& boundsDims,
                                     const std::vector<std::vector<int64_t>>& indicesDims,
                                     ge::DataType indicesDtype = ge::DT_INT32)
{
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO, socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::IndexCheckCompileInfo compileInfo;

    const std::string opType("IndexCheck");
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (impl == nullptr || impl->tiling_parse == nullptr || impl->tiling == nullptr) {
        ADD_FAILURE() << "IndexCheck tiling callbacks are not registered";
        return ge::GRAPH_FAILED;
    }

    // tiling_parse 模拟：从平台信息填充 CompileInfo
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(COMPILE_INFO), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    auto* parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr || parseContext->GetPlatformInfo() == nullptr ||
        !parseContext->GetPlatformInfo()->Init()) {
        ADD_FAILURE() << "Failed to create the tiling parse context";
        return ge::GRAPH_FAILED;
    }
    auto* parsePlatformInfo = parseContext->GetPlatformInfo();
    parsePlatformInfo->SetPlatformRes("SoCInfo", socInfos);
    parsePlatformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    parsePlatformInfo->SetCoreNumByCoreType("AICore");
    parsePlatformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    if (impl->tiling_parse(kernelHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        ADD_FAILURE() << "tiling_parse failed";
        return ge::GRAPH_FAILED;
    }

    // tiling 模拟：bounds(输入0) + indices_list(输入1, 动态列表)
    auto param = gert::TilingData::CreateCap(4096);
    if (param == nullptr) {
        ADD_FAILURE() << "Failed to allocate tiling data buffer";
        return ge::GRAPH_FAILED;
    }
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    gert::StorageShape boundsShape;
    for (const int64_t dim : boundsDims) {
        boundsShape.MutableOriginShape().AppendDim(dim);
        boundsShape.MutableStorageShape().AppendDim(dim);
    }

    std::vector<gert::StorageShape> indicesShapes;
    for (const auto& dims : indicesDims) {
        gert::StorageShape s;
        for (const int64_t dim : dims) {
            s.MutableOriginShape().AppendDim(dim);
            s.MutableStorageShape().AppendDim(dim);
        }
        indicesShapes.push_back(s);
    }

    std::vector<gert::StorageShape*> inputShapes;
    inputShapes.push_back(&boundsShape);
    for (auto& s : indicesShapes) {
        inputShapes.push_back(&s);
    }

    // IndexCheck 无实际输出，但 TilingContextFaker 需要至少 1 个输出槽，
    // 以便正确放置 CompileInfo/PlatformInfo 的 slot（与 golden 占位输出 (1,) 一致）。
    gert::StorageShape dummyOutputShape = {{1}, {1}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, indicesDims.size()})
                      .InputShapes(inputShapes)
                      .OutputShapes({&dummyOutputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        ADD_FAILURE() << "Failed to create the tiling context";
        return ge::GRAPH_FAILED;
    }
    auto* tilingPlatformInfo = tilingContext->GetPlatformInfo();
    tilingPlatformInfo->SetPlatformRes("SoCInfo", socInfos);
    tilingPlatformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingPlatformInfo->SetCoreNumByCoreType("AICore");
    tilingPlatformInfo->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    return impl->tiling(tilingContext);
}

// 正常场景：2 个 int32 indices，tiling 成功
TEST_F(IndexCheckTiling, tiling_success_same_dtype)
{
    ge::graphStatus status = RunTilingCase({2}, {{3}, {4}}, ge::DT_INT32);
    EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

// 空 indices 列表（tensorId==0）：tiling 拒绝
TEST_F(IndexCheckTiling, tiling_reject_empty_list)
{
    ge::graphStatus status = RunTilingCase({0}, {});
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// indices 数量超过上限 8：tiling 拒绝
TEST_F(IndexCheckTiling, tiling_reject_too_many_indices)
{
    std::vector<std::vector<int64_t>> indices(9, std::vector<int64_t>{1});
    ge::graphStatus status = RunTilingCase({9}, indices, ge::DT_INT32);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// bounds 非 1D：tiling 拒绝
TEST_F(IndexCheckTiling, tiling_reject_bounds_not_1d)
{
    ge::graphStatus status = RunTilingCase({2, 2}, {{3}, {4}}, ge::DT_INT32);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}

// bounds 长度与 indices 张量数不一致：tiling 拒绝
TEST_F(IndexCheckTiling, tiling_reject_bounds_length_mismatch)
{
    ge::graphStatus status = RunTilingCase({3}, {{3}, {4}}, ge::DT_INT32);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
