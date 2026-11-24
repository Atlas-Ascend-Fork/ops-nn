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
 * \file test_centralization_tiling.cpp
 * \brief Tiling UT：大 shape（单维/总元素数超 INT32_MAX、2^32、2^34）不得在 Host 侧被截成 32 位。
 */

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "ut_op_util.h"
#include "test_cube_util.h"

using namespace ut_util;
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/arch35/centralization_tiling.h"
#include "../../../op_kernel/arch35/centralization_tiling_data.h"

using namespace ge;

namespace {
static const char* kCompileInfoStr = R"({
   "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                     "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                     "CORE_NUM": 64, "socVersion": "Ascend950"}
            })";

struct TilingResult {
    uint64_t key{UINT64_MAX};
    uint32_t blocks{};
    bool ok{false};
    size_t workspace{0};
    CentralizationTilingData td{};
};

struct TilingTensor {
    std::vector<int64_t> dims;
    ge::DataType dtype;
    ge::Format format{ge::FORMAT_ND};
};

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

static TilingResult RunTilingWithDescriptors(const TilingTensor& x, const TilingTensor& y,
                                             const std::vector<int64_t>& axes, uint64_t ubSize, bool omitAxes = false)
{
    TilingResult result;
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::string compile_info_string(kCompileInfoStr);
    const std::string ubMarker = "\"UB_SIZE\": 253952";
    const size_t ubPos = compile_info_string.find(ubMarker);
    if (ubPos != std::string::npos) {
        compile_info_string.replace(ubPos, ubMarker.size(), "\"UB_SIZE\": " + std::to_string(ubSize));
    }
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::CentralizationCompileInfo compile_info;

    const std::string op_type("Centralization");
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (impl == nullptr || impl->tiling == nullptr || impl->tiling_parse == nullptr) {
        return result;
    }
    auto tiling_func = impl->tiling;
    auto tiling_parse_func = impl->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(3, 3)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    if (!kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init()) {
        return result;
    }
    auto parse_ctx = kernel_holder.GetContext<gert::TilingParseContext>();
    parse_ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return result;
    }

    auto x_shape = MakeStorageShape(x.dims);
    auto y_shape = MakeStorageShape(y.dims);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_holder.get());
    if (param == nullptr) {
        return result;
    }
    gert::TilingContextFaker faker;
    faker.SetOpType(op_type)
        .NodeIoNum(1, 1)
        .IrInstanceNum({1})
        .InputShapes({&x_shape})
        .OutputShapes({&y_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, x.dtype, x.format, x.format)
        .NodeOutputTd(0, y.dtype, y.format, y.format)
        .TilingData(param.get())
        .Workspace(ws_size);
    if (omitAxes) {
        faker.NodeAttrs({});
    } else {
        faker.NodeAttrs({{"axes", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(axes)}});
    }
    auto holder = faker.Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context == nullptr || tiling_context->GetPlatformInfo() == nullptr) {
        return result;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    if (tiling_func(tiling_context) != ge::GRAPH_SUCCESS) {
        return result;
    }
    result.key = tiling_context->GetTilingKey();
    result.blocks = tiling_context->GetBlockDim();
    if (auto* ws = tiling_context->GetWorkspaceSizes(1); ws != nullptr) {
        result.workspace = ws[0];
    }
    auto* raw = tiling_context->GetRawTilingData();
    if (raw != nullptr && raw->GetDataSize() >= sizeof(CentralizationTilingData)) {
        result.td = *reinterpret_cast<const CentralizationTilingData*>(raw->GetData());
    }
    result.ok = true;
    return result;
}

static TilingResult RunTilingWithUb(const std::vector<int64_t>& dims, ge::DataType dtype,
                                    const std::vector<int64_t>& axes, uint64_t ubSize, bool omitAxes = false)
{
    const TilingTensor tensor{dims, dtype};
    return RunTilingWithDescriptors(tensor, tensor, axes, ubSize, omitAxes);
}

static TilingResult RunTiling(const std::vector<int64_t>& dims, ge::DataType dtype, std::vector<int64_t> axes)
{
    return RunTilingWithUb(dims, dtype, axes, 253952UL);
}

static size_t ElementBytes(ge::DataType dtype) { return dtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t); }

static size_t LiveBufferBytes(const CentralizationTilingData& td, uint32_t key, ge::DataType dtype)
{
    size_t bytes = 0;
    if (key == 7030) {
        bytes = 2UL * td.largeTrailingInputBufferBytes + 2UL * td.largeTrailingOutputBufferBytes +
                td.largeTrailingCalcBufferBytes + td.largeTrailingMeanBufferBytes + 128UL;
    } else if (key == 7000) {
        const int64_t cols = td.ubFactor > 0 ? td.ubFactor : td.alignedReduce;
        const int64_t rows = td.rowsPerLoop > 1 ? td.rowsPerLoop : td.rowsPerBatch;
        bytes = 2UL * static_cast<size_t>(td.bufNum) * static_cast<size_t>(rows * cols) * ElementBytes(dtype);
    } else {
        const bool singleDepth = td.smallContiguous != 0 || td.largeContiguousSmallInner != 0;
        const size_t depth = singleDepth ? 1UL : 2UL;
        bytes = depth * (static_cast<size_t>(td.genericInputBufferBytes) +
                         static_cast<size_t>(td.genericOutputBufferBytes)) +
                static_cast<size_t>(td.genericSumBufferBytes);
    }
    return bytes;
}
} // namespace

TEST(CentralizationTiling, NormalShapeKeepsCounts)
{
    auto r = RunTiling({2, 3, 4}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, 24);
    EXPECT_EQ(r.td.reduceCount, 3);
    EXPECT_EQ(r.td.groupCount, 8);
}

TEST(CentralizationTiling, RejectIncompatibleStorageDescriptors)
{
    const TilingTensor x{{2, 3}, ge::DT_FLOAT};
    const TilingTensor blocked{{2, 3}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ};
    EXPECT_FALSE(RunTilingWithDescriptors(blocked, x, {-1}, 253952UL).ok);
    EXPECT_FALSE(RunTilingWithDescriptors(x, blocked, {-1}, 253952UL).ok);
    EXPECT_FALSE(RunTilingWithDescriptors(x, {{3}, ge::DT_FLOAT}, {-1}, 253952UL).ok);
    EXPECT_FALSE(RunTilingWithDescriptors(x, {{3, 2}, ge::DT_FLOAT}, {-1}, 253952UL).ok);
    EXPECT_FALSE(RunTilingWithDescriptors(x, {{2, 3}, ge::DT_FLOAT16}, {-1}, 253952UL).ok);
}

TEST(CentralizationTiling, RejectAxisBelowNegativeRank)
{
    EXPECT_FALSE(RunTiling({2, 3, 4}, ge::DT_FLOAT, {-4}).ok);
    EXPECT_TRUE(RunTiling({2, 3, 4}, ge::DT_FLOAT, {-3}).ok);
}

TEST(CentralizationTiling, SingleDimOverInt32NotTruncated)
{
    constexpr int64_t kOverInt32 = 3000000000L; // > 2^31 - 1
    auto r = RunTiling({kOverInt32, 1, 1, 1}, ge::DT_FLOAT, {3});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, static_cast<int64_t>(kOverInt32));
    EXPECT_EQ(r.td.groupCount, static_cast<int64_t>(kOverInt32));
    EXPECT_EQ(r.td.reduceCount, 1);
}

TEST(CentralizationTiling, Upboundary2Pow32Accepted)
{
    constexpr int64_t k2Pow32 = 4294967296L; // UINT32_MAX + 1
    auto r = RunTiling({k2Pow32, 1, 1, 1, 1, 1, 1}, ge::DT_FLOAT, {3});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, k2Pow32);
    EXPECT_EQ(r.td.groupCount, k2Pow32);
    EXPECT_EQ(r.td.reduceCount, 1);
}

TEST(CentralizationTiling, UpboundaryReduce2Pow32Accepted)
{
    constexpr int64_t k2Pow32 = 4294967296L;
    auto r = RunTiling({1, k2Pow32, 1}, ge::DT_FLOAT, {});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, k2Pow32);
    EXPECT_EQ(r.td.groupCount, k2Pow32);
    EXPECT_EQ(r.td.reduceCount, 1);
}

TEST(CentralizationTiling, FarOverUint32Accepted)
{
    constexpr int64_t k2Pow34 = 17179869184L; // 2^34
    auto r = RunTiling({1, 1, k2Pow34}, ge::DT_FLOAT, {2});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, k2Pow34);
    EXPECT_EQ(r.td.reduceCount, k2Pow34);
}

TEST(CentralizationTiling, LargeReduceWith2Pow32GroupsAccepted)
{
    constexpr int64_t k2Pow32 = 4294967296L;
    constexpr int64_t kReduce = 16384L; // > kLargeReduceThreshold，走 LargeTrailing
    auto r = RunTiling({k2Pow32, kReduce}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, k2Pow32 * kReduce);
    EXPECT_EQ(r.td.groupCount, k2Pow32);
    EXPECT_EQ(r.td.reduceCount, kReduce);
}

TEST(CentralizationTiling, UbBudgetPathMatrix)
{
    struct Case {
        std::vector<int64_t> dims;
        std::vector<int64_t> axes;
        ge::DataType dtype;
        uint64_t expectedKey;
        int64_t expectedRedTile;
        int64_t expectedRows;
        int64_t expectedSlots;
        const char* name;
    };
    const std::vector<Case> cases = {
        // Wide FP32 rows with few output groups use LargeTrailing when its buffers fit.
        {{2, 512, 1}, {1}, ge::DT_FLOAT, 7030, 0, 1, 16, "normalized_trailing_512_fp32"},
        {{2, 64, 4}, {1}, ge::DT_FLOAT, 0, 0, 1, 16, "small_mode2_fp32"},
        {{2, 300, 8}, {1}, ge::DT_FLOAT, 0, 0, 1, 16, "contiguous_fp32"},
        {{64, 20000, 8}, {1}, ge::DT_FLOAT, 0, 320, 1, 1, "lcsi_fp32"},
        {{64, 20000, 8}, {1}, ge::DT_FLOAT16, 0, 640, 1, 1, "lcsi_fp16"},
        // 布局归一后 (64,20000,1)->[K64,R20000] 尾轴 LargeTrailing（原 7020 行并行）
        {{64, 20000, 1}, {1}, ge::DT_FLOAT, 7030, 0, 1, 16, "trailing_one_lifted_to_large_trailing_fp32"},
        {{20000, 8}, {0}, ge::DT_FLOAT, 7020, 0, 1, 16, "large_7020_reduce_parallel_fp32"},
        {{4, 8, 16, 32}, {0, 2}, ge::DT_FLOAT, 0, 0, 1, 1, "irregular_fp32"},
        {{8, 20000}, {1}, ge::DT_FLOAT, 7030, 0, 1, 16, "large_trailing_fp32"},
        {{128, 20000}, {1}, ge::DT_FLOAT, 7030, 0, 1, 16, "large_trailing_row_parallel_fp32"},
        {{8, 4096}, {1}, ge::DT_FLOAT, 7030, 0, 1, 16, "large_trailing_4096_fp32"},
        {{8, 4096}, {1}, ge::DT_FLOAT16, 7000, 0, 7, 1, "fast_fp16"},
        {{2, 2000, 1}, {1}, ge::DT_FLOAT, 7030, 0, 1, 16, "normalized_trailing_2000_fp32"},
        {{2, 0, 4}, {1}, ge::DT_FLOAT, 8000, 0, 1, 1, "empty_fp32"},
    };
    for (const auto& c : cases) {
        auto r = RunTiling(c.dims, c.dtype, c.axes);
        ASSERT_TRUE(r.ok) << c.name;
        const size_t live = LiveBufferBytes(r.td, static_cast<uint32_t>(r.key), c.dtype);
        EXPECT_EQ(r.key, c.expectedKey) << c.name;
        EXPECT_LE(live, 253952UL) << c.name;
        if (r.key != 8000) {
            EXPECT_LE(live + 8192UL, 253952UL) << c.name;
        }
        EXPECT_EQ(r.td.largeContiguousReduceTile, c.expectedRedTile) << c.name;
        EXPECT_EQ(r.td.rowsPerLoop, c.expectedRows) << c.name;
        EXPECT_EQ(r.td.numAccSlots, c.expectedSlots) << c.name;
    }
}

TEST(CentralizationTiling, UbBudgetFallbackMatrix)
{
    struct Case {
        std::vector<int64_t> dims;
        std::vector<int64_t> axes;
        ge::DataType dtype;
        uint64_t ubSize;
        uint64_t expectedKey;
        int64_t expectedRedTile;
        const char* name;
    };
    const std::vector<Case> cases = {
        {{8, 4096}, {1}, ge::DT_FLOAT, 65536, 7030, 0, "fast_to_large_trailing_fp32"},
        {{8, 4096}, {1}, ge::DT_FLOAT, 32768, 7020, 0, "fast_to_generic_large_fp32"},
        {{8, 20000}, {1}, ge::DT_FLOAT, 32768, 7020, 0, "large_trailing_to_generic_fp32"},
        // 归一后只剩两个宽度为 512 的 FP32 行；LargeTrailing 缓冲区放不下时回退到 7020。
        {{2, 512, 1}, {1}, ge::DT_FLOAT, 12288, 7020, 0, "trailing_one_to_large_7020_fp32"},
        {{2, 64, 4}, {1}, ge::DT_FLOAT, 24576, 0, 0, "small_to_contiguous_fp32"},
        {{64, 20000, 8}, {1}, ge::DT_FLOAT, 16384, 0, 10, "lcsi_tiny_ub_fp32"},
        {{4, 8, 16, 32}, {0, 2}, ge::DT_FLOAT, 16384, 0, 0, "irregular_tiny_ub_fp32"},
    };
    for (const auto& c : cases) {
        auto r = RunTilingWithUb(c.dims, c.dtype, c.axes, c.ubSize);
        ASSERT_TRUE(r.ok) << c.name;
        const size_t live = LiveBufferBytes(r.td, static_cast<uint32_t>(r.key), c.dtype);
        EXPECT_EQ(r.key, c.expectedKey) << c.name;
        EXPECT_LE(live, c.ubSize) << c.name;
        if (r.key != 8000) {
            EXPECT_LE(live + 8192UL, c.ubSize) << c.name;
        }
        EXPECT_EQ(r.td.largeContiguousReduceTile, c.expectedRedTile) << c.name;
    }
}

TEST(CentralizationTiling, UbBudgetMinimumFit)
{
    // contiguous 的最小 sum buffer 放不下 -> scalar（不返回 GRAPH_FAILED）
    auto a = RunTilingWithUb({2, 300, 8}, ge::DT_FLOAT, {1}, 12288);
    ASSERT_TRUE(a.ok);
    EXPECT_EQ(a.key, 0);
    EXPECT_EQ(a.td.contiguousGeneric, 0);
    EXPECT_EQ(a.td.smallContiguous, 0);
    EXPECT_EQ(a.td.largePath, 0);

    // small mode2 放不下，contiguous 也放不下 -> scalar
    auto b = RunTilingWithUb({2, 64, 4}, ge::DT_FLOAT, {1}, 12288);
    ASSERT_TRUE(b.ok);
    EXPECT_EQ(b.key, 0);
    EXPECT_EQ(b.td.contiguousGeneric, 0);
    EXPECT_EQ(b.td.smallContiguous, 0);

    // irregular 的最小 sum buffer 放不下 -> scalar
    auto c = RunTilingWithUb({4, 8, 16, 32}, ge::DT_FLOAT, {0, 2}, 10240);
    ASSERT_TRUE(c.ok);
    EXPECT_EQ(c.key, 0);
    EXPECT_EQ(c.td.irregularVectorizable, 0);

    // lcsi 最小 tile（1 个 reduce 行）仍可放下
    auto d = RunTilingWithUb({64, 20000, 8}, ge::DT_FLOAT, {1}, 12288);
    ASSERT_TRUE(d.ok);
    EXPECT_EQ(d.key, 0);
    EXPECT_EQ(d.td.largeContiguousSmallInner, 1);
    EXPECT_EQ(d.td.largeContiguousReduceTile, 5);
    EXPECT_LE(LiveBufferBytes(d.td, static_cast<uint32_t>(d.key), ge::DT_FLOAT) + 8192UL, 12288UL);

    // UB 连预留都不够：明确失败，不产生零步长/死循环
    auto e = RunTilingWithUb({2, 512, 1}, ge::DT_FLOAT, {1}, 8192);
    EXPECT_FALSE(e.ok);
}

TEST(CentralizationTiling, PackedFp32TrailingRows)
{
    const std::vector<int64_t> shape{71, 76, 94, 76};
    const auto packed = RunTiling(shape, ge::DT_FLOAT, {3});
    ASSERT_TRUE(packed.ok);
    EXPECT_EQ(packed.key, 7000U);
    EXPECT_EQ(packed.td.groupCount, 507224);
    EXPECT_EQ(packed.td.reduceCount, 76);
    EXPECT_GE(packed.td.rowsPerLoop, 64);

    const auto limited = RunTilingWithUb(shape, ge::DT_FLOAT, {3}, 65536);
    ASSERT_TRUE(limited.ok);
    EXPECT_EQ(limited.key, 7000U);
    EXPECT_GE(limited.td.rowsPerLoop, 16);
    const auto fallback = RunTilingWithUb(shape, ge::DT_FLOAT, {3}, 32768);
    ASSERT_TRUE(fallback.ok);
    EXPECT_NE(fallback.key, 7000U);
}

TEST(CentralizationTiling, UbBudgetFallbackFlags)
{
    auto checkLargeFallback = [](const TilingResult& r, uint64_t expectedKey) {
        EXPECT_EQ(r.key, expectedKey);
        EXPECT_EQ(r.td.largePath, 1);
        EXPECT_EQ(r.td.contiguousGeneric, 0);
        EXPECT_EQ(r.td.smallContiguous, 0);
        EXPECT_EQ(r.td.largeContiguousSmallInner, 0);
        EXPECT_EQ(r.td.irregularVectorizable, 0);
        const size_t expectedWorkspace = 16UL * 1024UL * 1024UL + static_cast<size_t>(r.td.groupCount) *
                                                                      static_cast<size_t>(r.td.alignedCoresPerRow) *
                                                                      32UL;
        EXPECT_EQ(r.workspace, expectedWorkspace);
        EXPECT_GT(r.td.numAccSlots, 0);
        EXPECT_LE(r.td.numAccSlots, 16);
    };

    auto a = RunTilingWithUb({8, 4096}, ge::DT_FLOAT, {1}, 65536);
    ASSERT_TRUE(a.ok);
    checkLargeFallback(a, 7030);
    EXPECT_EQ(a.td.rowParallel, 1);
    auto b = RunTilingWithUb({8, 4096}, ge::DT_FLOAT, {1}, 32768);
    ASSERT_TRUE(b.ok);
    checkLargeFallback(b, 7020);
    EXPECT_EQ(b.td.rowParallel, 1);
    auto c = RunTilingWithUb({8, 20000}, ge::DT_FLOAT, {1}, 32768);
    ASSERT_TRUE(c.ok);
    checkLargeFallback(c, 7020);
    EXPECT_EQ(c.td.rowParallel, 0);
    EXPECT_EQ(c.td.coresPerRow, 8);

    // 12 KiB UB 下，两行宽度为 512 的 FP32 输入走 7020 行并行路径。
    auto d = RunTilingWithUb({2, 512, 1}, ge::DT_FLOAT, {1}, 12288);
    ASSERT_TRUE(d.ok);
    checkLargeFallback(d, 7020);
    EXPECT_EQ(d.td.rowParallel, 1);
}

TEST(CentralizationTiling, DuplicateAxesDeduplicated)
{
    auto r = RunTiling({2, 3, 4}, ge::DT_FLOAT, {1, 1});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 1);
    EXPECT_EQ(r.td.reduceCount, 3);
    EXPECT_EQ(r.td.groupCount, 8);
}

TEST(CentralizationTiling, LongAxisListDeduplicated)
{
    auto r = RunTiling({2, 3, 4}, ge::DT_FLOAT, {0, 1, 2, 0});
    ASSERT_TRUE(r.ok);
    // 布局归一：[R2,R3,R4] 相邻同类合并为单个 reduce 段
    EXPECT_EQ(r.td.reduceRank, 1);
    EXPECT_EQ(r.td.reduceCount, 24);
    EXPECT_EQ(r.td.groupCount, 1);
}

// ── 布局归一专项 ──────────────────────────────────────────────────────────
// 归一规则：长度 1 的维度吸收；物理连续的相邻同类维度合并；原始 shape 只用于输出。

TEST(CentralizationTiling, LayoutNormalizeTrailingOneUsesLargeTrailing)
{
    // Removing the final singleton makes the 2000-element FP32 reduction trailing.
    auto r = RunTiling({2, 2000, 1}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 1);
    EXPECT_EQ(r.td.reduceCount, 2000);
    EXPECT_EQ(r.td.groupCount, 2);
    EXPECT_EQ(r.key, 7030U);
}

TEST(CentralizationTiling, LayoutNormalizeSingletonReduceAbsorbed)
{
    // The singleton reduction is absorbed; Fast still evaluates x - x.
    auto r = RunTiling({4, 1, 8}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 0);
    EXPECT_EQ(r.td.reduceCount, 1);
    EXPECT_EQ(r.td.groupCount, 32);
    EXPECT_EQ(r.key, 7000U);
}

TEST(CentralizationTiling, LayoutNormalizeKeepOneBridgesReduceSegments)
{
    // [K,R,K1,R,K] -> [K,R,K]：桥接 1 维使两段 reduce 合并，复用 contiguous 路径
    // （原 axes {1,3} 非连续 -> irregular 路径）
    auto r = RunTiling({2, 30, 1, 50, 8}, ge::DT_FLOAT, {1, 3});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 1);
    EXPECT_EQ(r.td.reduceCount, 1500);
    EXPECT_EQ(r.td.groupCount, 16);
    EXPECT_EQ(r.td.contiguousOuterCount, 2);
    EXPECT_EQ(r.td.contiguousInnerCount, 8);
    EXPECT_EQ(r.td.contiguousGeneric, 1);
    EXPECT_EQ(r.key, 0U);
}

TEST(CentralizationTiling, LayoutNormalizeAdjacentReduceMerged)
{
    // [K,R,R,K] -> [K,R,K]：相邻同类 reduce 合并为一个 reduce 段
    auto r = RunTiling({2, 30, 50, 8}, ge::DT_FLOAT, {1, 2});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 1);
    EXPECT_EQ(r.td.reduceCount, 1500);
    EXPECT_EQ(r.td.groupCount, 16);
    EXPECT_EQ(r.td.contiguousOuterCount, 2);
    EXPECT_EQ(r.td.contiguousInnerCount, 8);
}

TEST(CentralizationTiling, LayoutNormalizeMultiSegmentStaysIrregular)
{
    // [R,K,R,K]：真多段 reduce 仍走 irregular，段描述建立在归一条目上
    auto r = RunTiling({4, 8, 16, 32}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 2);
    EXPECT_EQ(r.td.reduceCount, 64);
    EXPECT_EQ(r.td.groupCount, 256);
    EXPECT_EQ(r.td.irregularVectorInnerCount, 32);
    EXPECT_EQ(r.td.irregularKeepOuterRank, 1);
    EXPECT_EQ(r.td.irregularVectorizable, 1);
    EXPECT_EQ(r.key, 0U);
}

TEST(CentralizationTiling, SplitsUnalignedSuffixAcrossCores)
{
    auto r = RunTiling({151, 2, 159}, ge::DT_FLOAT16, {0, 2});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, 7020U);
    EXPECT_EQ(r.td.groupCount, 2);
    EXPECT_EQ(r.td.reduceSegmentCount, 151);
    EXPECT_GT(r.td.irregularReduceChunks, 1);
    EXPECT_EQ(r.td.irregularReduceChunks, r.td.coresPerRow);
    EXPECT_GT(r.workspace, 0U);
}

TEST(CentralizationTiling, RepeatingGroupsUseWholePhysicalPeriods)
{
    auto r = RunTiling({16386, 2, 4}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.ok);
    ASSERT_EQ(r.td.largePlane, 1);
    EXPECT_EQ(r.td.groupCount, 2);
    EXPECT_EQ(r.td.reduceCount, 16386 * 4);
    ASSERT_GT(r.td.contiguousInnerCount, 0);
    EXPECT_EQ(r.td.contiguousInnerCount % r.td.reduceStrides[0], 0);
    EXPECT_EQ(r.td.totalCount % r.td.contiguousInnerCount, 0);
    EXPECT_GT(r.td.coresPerRow, 1);
}

TEST(CentralizationTiling, BatchesStridedReductionAlongLongestOuterAxis)
{
    auto r = RunTiling({2, 1201, 2, 3, 2, 3}, ge::DT_FLOAT16, {1, 3, 5});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, 7020U);
    EXPECT_EQ(r.td.largePath, Centralization::kStridedSuffixPath);
    EXPECT_EQ(r.td.stridedReduceAxis, 0U);
    EXPECT_EQ(r.td.irregularReduceSuffixElements, 3);
    EXPECT_GT(r.td.coresPerRow, 1);
    EXPECT_EQ(r.blocks, r.td.groupCount * r.td.coresPerRow);

    auto inner = RunTiling({3, 2, 3001, 3, 3}, ge::DT_FLOAT, {0, 2, 4});
    ASSERT_TRUE(inner.ok);
    EXPECT_EQ(inner.td.largePath, Centralization::kStridedSuffixPath);
    EXPECT_EQ(inner.td.stridedReduceAxis, 1U);

    const auto many = RunTiling({3872, 111, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(many.ok);
    EXPECT_EQ(many.td.largePath, Centralization::kStridedSuffixPath);
    EXPECT_EQ(many.td.coresPerRow, 1);
    EXPECT_LT(many.blocks, many.td.groupCount);
    EXPECT_EQ(many.workspace, 0U);
}

TEST(CentralizationTiling, DividesLongSuffixWithinEachPhysicalSegment)
{
    auto r = RunTiling({2, 2, 131073}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.key, 7020U);
    EXPECT_EQ(r.td.largePath, Centralization::kStridedSuffixPath);
    EXPECT_EQ(r.td.stridedReduceAxis, 0U);
    EXPECT_GT(r.td.coresPerRow, r.td.reduceSegmentCount);
}

TEST(CentralizationTiling, DistributesInterleavedKeptColumns)
{
    auto r = RunTiling({2, 149, 2, 71, 2}, ge::DT_FLOAT, {1, 3});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.largePath, Centralization::kInterleavedColumnsPath);
    EXPECT_EQ(r.td.stridedReduceAxis, 1U);
    EXPECT_GT(r.td.coresPerRow, 1);
    EXPECT_EQ(r.blocks, 4 * r.td.coresPerRow);
    EXPECT_EQ(r.workspace, 16UL * 1024 * 1024 + r.blocks * 2 * 32);

    const auto many = RunTiling({128, 16254, 3}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(many.ok);
    EXPECT_EQ(many.td.largePath, Centralization::kInterleavedColumnsPath);
    EXPECT_EQ(many.td.coresPerRow, 1);
    EXPECT_EQ(many.workspace, 0U);
}

TEST(CentralizationTiling, SplitsSegmentsOutsideInterleavedColumnRange)
{
    const auto r = RunTiling({2, 149, 2, 71, 33}, ge::DT_FLOAT, {1, 3});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.irregularVectorizable, 1);
    EXPECT_GT(r.td.irregularReduceChunks, 1);
    EXPECT_EQ(r.td.useReduceOffsetTable, 0);
    EXPECT_EQ(r.td.reduceSegmentCount, 149);
    EXPECT_GE(r.td.irregularPartialStride, 2 * r.td.irregularVectorInnerCount);
    EXPECT_GT(r.workspace, 0U);
}

TEST(CentralizationTiling, EmptyAxesNoReduce)
{
    auto r = RunTiling({2, 3, 4}, ge::DT_FLOAT, {});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 0);
    EXPECT_EQ(r.td.reduceCount, 1);
    EXPECT_EQ(r.td.groupCount, 24);
    EXPECT_EQ(r.key, 7000U);
}

TEST(CentralizationTiling, Rank0Rejected)
{
    auto r = RunTiling({}, ge::DT_FLOAT, {-1});
    EXPECT_FALSE(r.ok);
}

TEST(CentralizationTiling, Rank9RejectedByToolchainLimit)
{
    auto r = RunTiling({2, 1, 1, 1, 1, 1, 1, 1, 4}, ge::DT_FLOAT, {8});
    EXPECT_FALSE(r.ok);
}

TEST(CentralizationTiling, AxesOmittedEqualsExplicitMinusOne)
{
    auto omitted = RunTilingWithUb({2, 3}, ge::DT_FLOAT, {}, 253952UL, true);
    auto explicitMinusOne = RunTilingWithUb({2, 3}, ge::DT_FLOAT, {-1}, 253952UL);
    auto empty = RunTilingWithUb({2, 3}, ge::DT_FLOAT, {}, 253952UL);

    ASSERT_TRUE(omitted.ok);
    ASSERT_TRUE(explicitMinusOne.ok);
    ASSERT_TRUE(empty.ok);

    EXPECT_EQ(omitted.td.reduceRank, 1);
    EXPECT_EQ(omitted.td.reduceCount, 3);
    EXPECT_EQ(omitted.td.groupCount, 2);
    EXPECT_EQ(omitted.key, explicitMinusOne.key);

    EXPECT_EQ(explicitMinusOne.td.reduceRank, 1);
    EXPECT_EQ(explicitMinusOne.td.reduceCount, 3);
    EXPECT_EQ(explicitMinusOne.td.groupCount, 2);

    EXPECT_EQ(empty.td.reduceRank, 0);
    EXPECT_EQ(empty.td.reduceCount, 1);
    EXPECT_EQ(empty.td.groupCount, 6);
    EXPECT_EQ(empty.key, 7000U);
}

TEST(CentralizationTiling, SingleElementReductionsUseFast)
{
    // Both layouts reduce one element per group and use the same Fast path.
    auto a = RunTiling({2, 3, 4}, ge::DT_FLOAT, {});
    ASSERT_TRUE(a.ok);
    EXPECT_EQ(a.key, 7000U);

    auto b = RunTiling({4, 1, 8}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(b.ok);
    EXPECT_EQ(b.key, 7000U);
}
