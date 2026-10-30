/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <gtest/gtest.h>

#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace {
constexpr uint32_t CORE_COUNT = 16U;
constexpr uint64_t UB_BYTES = 253952UL;
constexpr uint64_t L1_BYTES = 1048576UL;
constexpr uint64_t L0AB_BYTES = 65536UL;
constexpr uint64_t L0C_BYTES = 262144UL;
constexpr uint64_t NO_FULL_LOAD = 12UL;
constexpr uint64_t A_FULL_LOAD = 13UL;
constexpr uint64_t B_FULL_LOAD = 14UL;

struct PerGroupCase {
    int64_t m = 35;
    int64_t k = 770;
    int64_t n = 80;
    int64_t groupK = 384;
    ge::DataType weightType = ge::DT_INT8;
    bool transB = false;
    bool bias = true;
    int64_t batchA = 1;
    int64_t batchB = 1;
    uint32_t expectedBaseK = 0U;
};

uint64_t AlignTo(uint64_t value, uint64_t alignment) { return (value + alignment - 1UL) / alignment * alignment; }

gert::StorageShape MakeShape(const gert::Shape& shape)
{
    gert::StorageShape result;
    result.MutableOriginShape() = shape;
    result.MutableStorageShape() = shape;
    return result;
}

gert::StorageShape MakeWeightShape(const PerGroupCase& param, bool batched)
{
    const int64_t rows = param.transB ? param.n : param.k;
    const int64_t cols = param.transB ? param.k : param.n;
    const int64_t c0 = param.weightType == ge::DT_INT4 ? 64 : 32;
    auto result = MakeShape(batched ? gert::Shape({param.batchB, rows, cols}) : gert::Shape({rows, cols}));
    result.MutableStorageShape() = batched ?
                                       gert::Shape({param.batchB, (cols + c0 - 1) / c0, (rows + 15) / 16, 16, c0}) :
                                       gert::Shape({(cols + c0 - 1) / c0, (rows + 15) / 16, 16, c0});
    return result;
}

void InitPlatform(fe::PlatFormInfos& platform)
{
    platform.Init();
    const char* hardware = R"({"hardware_info": {
        "BT_SIZE": 4096, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true,
        "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": true, "Intrinsic_data_move_l12bt": true, "Intrinsic_mmad": true,
        "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
        "CORE_NUM": 16, "cube_core_cnt": 16, "vector_core_cnt": 16,
        "core_type_list": "CubeCore,VectorCore", "lut_type": "MTE2_QTABLE"}})";
    std::map<std::string, std::string> soc, spec, intrinsics;
    std::map<std::string, std::string> version = {
        {"SoC_version", "MC62CM12AA"}, {"Short_SoC_version", "MC62CM12AA"}, {"NpuArch", "5102"}};
    GetPlatFormInfos(hardware, soc, spec, intrinsics);
    intrinsics["Intrinsic_mmad"] = "s8s4";
    spec["cube_freq"] = "1800";
    platform.SetPlatformRes("SoCInfo", soc);
    platform.SetPlatformRes("AICoreSpec", spec);
    platform.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platform.SetPlatformRes("version", version);
    platform.SetCoreNumByCoreType("AICore");
}

void CheckResources(const PerGroupCase& param, const DequantBmm::QuantBatchMatmulV3BasicAPITilingData& tiling,
                    uint64_t kernelType)
{
    const auto& mm = tiling.matmulTiling;
    ASSERT_GT(mm.baseM, 0U);
    ASSERT_GT(mm.baseN, 0U);
    ASSERT_GT(mm.baseK, 0U);
    EXPECT_EQ(mm.baseM % 16U, 0U);
    EXPECT_EQ(mm.baseN % 16U, 0U);
    EXPECT_EQ(mm.baseK % 64U, 0U);
    EXPECT_EQ(static_cast<uint64_t>(param.groupK) % mm.baseK, 0UL);
    ASSERT_GE(mm.kAL1, mm.baseK);
    ASSERT_GE(mm.kBL1, mm.baseK);
    EXPECT_EQ(mm.kAL1 % mm.baseK, 0U);
    EXPECT_EQ(mm.kBL1 % mm.baseK, 0U);
    const uint64_t pack = param.weightType == ge::DT_INT4 ? 2UL : 1UL;
    const uint64_t c0 = 32UL * pack;
    const uint64_t tileElements = static_cast<uint64_t>(mm.baseM) * mm.baseN;
    EXPECT_LE(14UL * tileElements + (param.bias ? 6UL * mm.baseN : 0UL), UB_BYTES);
    EXPECT_LE(static_cast<uint64_t>(mm.baseM) * mm.baseK * 2UL, L0AB_BYTES);
    EXPECT_LE(static_cast<uint64_t>(mm.baseN) * mm.baseK / pack * 2UL, L0AB_BYTES);
    ASSERT_TRUE(mm.dbL0C == 1U || mm.dbL0C == 2U);
    EXPECT_LE(tileElements * sizeof(int32_t) * mm.dbL0C, L0C_BYTES);

    const uint64_t aK = kernelType == A_FULL_LOAD ? param.k : mm.kAL1;
    const uint64_t bK = kernelType == B_FULL_LOAD ? param.k : mm.kBL1;
    const uint64_t aBytes = AlignTo(mm.baseM, 16UL) * AlignTo(aK, 32UL);
    const uint64_t bBytes = AlignTo(mm.baseN, param.transB ? 16UL : c0) * AlignTo(bK, c0) / pack;
    const uint64_t scaleRowBytes = AlignTo(mm.baseN, 16UL) * sizeof(uint64_t);
    uint64_t usedL1 = 2UL * (aBytes + bBytes + scaleRowBytes);
    if (kernelType == A_FULL_LOAD || kernelType == B_FULL_LOAD) {
        const uint64_t dataBytes = kernelType == A_FULL_LOAD ? aBytes + 4UL * bBytes : 4UL * aBytes + bBytes;
        ASSERT_LT(dataBytes, L1_BYTES);
        const uint64_t groups = (param.k + param.groupK - 1) / param.groupK;
        const uint64_t scaleRows = std::min(
            {groups, std::max(1UL, 32768UL / scaleRowBytes), (L1_BYTES - dataBytes) / (2UL * scaleRowBytes)});
        ASSERT_GT(scaleRows, 0UL);
        usedL1 = dataBytes + 2UL * scaleRows * scaleRowBytes;
    }
    EXPECT_LE(usedL1, L1_BYTES);
    if (kernelType == A_FULL_LOAD) {
        EXPECT_GE(mm.kAL1, static_cast<uint64_t>(param.k));
        EXPECT_EQ(static_cast<uint64_t>(param.groupK) % mm.kBL1, 0UL);
    } else if (kernelType == B_FULL_LOAD) {
        EXPECT_GE(mm.kBL1, static_cast<uint64_t>(param.k));
        EXPECT_EQ(static_cast<uint64_t>(param.groupK) % mm.kAL1, 0UL);
    } else {
        EXPECT_EQ(mm.kAL1, mm.kBL1);
    }
}

void RunCase(const PerGroupCase& param, uint64_t kernelType, bool success = true)
{
    SCOPED_TRACE(::testing::Message() << "M=" << param.m << " K=" << param.k << " N=" << param.n
                                      << " G=" << param.groupK << " weight=" << param.weightType
                                      << " transB=" << param.transB << " kernelType=" << kernelType);
    const bool batched = param.batchA != 1 || param.batchB != 1;
    const int64_t batchC = std::max(param.batchA, param.batchB);
    auto x1 = MakeShape(batched ? gert::Shape({param.batchA, param.m, param.k}) : gert::Shape({param.m, param.k}));
    auto x2 = MakeWeightShape(param, batched);
    const int64_t groups = (param.k + param.groupK - 1) / param.groupK;
    auto scale = MakeShape(batched ? gert::Shape({param.batchB, groups, param.n}) : gert::Shape({groups, param.n}));
    auto bias = MakeShape(gert::Shape({param.n}));
    auto out = MakeShape(batched ? gert::Shape({batchC, param.m, param.n}) : gert::Shape({param.m, param.n}));
    fe::PlatFormInfos platform;
    platform.Init();
    optiling::QuantBatchMatmulV3CompileInfo compileInfo{};
    auto rawTiling = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTiling, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    ASSERT_NE(workspaceHolder, nullptr);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    const int64_t groupSize = param.groupK | (1LL << 16) | (1LL << 32);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&x1, &x2, &scale, nullptr, param.bias ? &bias : nullptr, nullptr})
                      .OutputShapes({&out})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platform))
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.weightType, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                      .NodeInputTd(2, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transB)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
                      .TilingData(rawTiling.get())
                      .Workspace(workspace)
                      .SetOpType("QuantBatchMatmulV3")
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    auto* tilingPlatform = context->GetPlatformInfo();
    ASSERT_NE(tilingPlatform, nullptr);
    // Configure the platform owned by the built context, following the existing Host UT setup.
    InitPlatform(*tilingPlatform);
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV3");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling_parse, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    std::string compileJson = "{}";
    auto parseHolder = gert::KernelRunContextFaker()
                           .KernelIONum(2, 1)
                           .Inputs({const_cast<char*>(compileJson.c_str()), reinterpret_cast<void*>(tilingPlatform)})
                           .Outputs({&compileInfo})
                           .Build();
    auto* parseContext = parseHolder.GetContext<gert::TilingParseContext>();
    ASSERT_NE(parseContext, nullptr);
    auto* parsePlatform = parseContext->GetPlatformInfo();
    ASSERT_NE(parsePlatform, nullptr);
    InitPlatform(*parsePlatform);
    std::string intrinsicMmad;
    parsePlatform->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", intrinsicMmad);
    ASSERT_NE(intrinsicMmad.find("s8s4"), std::string::npos) << "Intrinsic_mmad=" << intrinsicMmad;
    ASSERT_EQ(impl->tiling_parse(parseHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    ASSERT_TRUE(compileInfo.supportMmadS8S4) << "Intrinsic_mmad=" << intrinsicMmad;
    ASSERT_EQ(compileInfo.aicNum, CORE_COUNT);
    ASSERT_EQ(compileInfo.aivNum, CORE_COUNT);
    const auto status = impl->tiling(context);
    if (!success) {
        EXPECT_EQ(status, ge::GRAPH_FAILED);
        return;
    }
    ASSERT_EQ(status, ge::GRAPH_SUCCESS);
    // BASIC=1, WITH_BATCH=0: check the complete key independently of the construction macro.
    EXPECT_EQ(context->GetTilingKey(), (1UL << 16UL) | (kernelType << 8UL) | (param.transB ? 1UL << 2UL : 0UL));
    EXPECT_EQ(context->GetBlockDim(), CORE_COUNT);
    ASSERT_EQ(context->GetRawTilingData()->GetDataSize(), sizeof(DequantBmm::QuantBatchMatmulV3BasicAPITilingData));
    DequantBmm::QuantBatchMatmulV3BasicAPITilingData tiling{};
    std::memcpy(&tiling, context->GetRawTilingData()->GetData(), sizeof(tiling));
    EXPECT_EQ(tiling.matmulTiling.m, static_cast<uint32_t>(param.m));
    EXPECT_EQ(tiling.matmulTiling.n, static_cast<uint32_t>(param.n));
    EXPECT_EQ(tiling.matmulTiling.k, static_cast<uint32_t>(param.k));
    EXPECT_EQ(tiling.params.batchA, static_cast<uint32_t>(param.batchA));
    EXPECT_EQ(tiling.params.batchB, static_cast<uint32_t>(param.batchB));
    EXPECT_EQ(tiling.params.batchC, static_cast<uint32_t>(batchC));
    EXPECT_EQ(tiling.params.groupSizeM, 1U);
    EXPECT_EQ(tiling.params.groupSizeN, 1U);
    EXPECT_EQ(tiling.params.groupSizeK, static_cast<uint32_t>(param.groupK));
    EXPECT_EQ(tiling.params.x2QuantMode, static_cast<uint32_t>(optiling::BasicQuantMode::PERGROUP_MODE));
    if (param.expectedBaseK != 0U) {
        EXPECT_EQ(tiling.matmulTiling.baseK, param.expectedBaseK);
    }
    CheckResources(param, tiling, kernelType);
}
} // namespace

TEST(QuantBatchMatmulV3PerGroupTiling, ChecksAllLoadModesForBothWeightTypesAndTransposes)
{
    for (const auto dtype : {ge::DT_INT8, ge::DT_INT4}) {
        for (const bool transB : {false, true}) {
            PerGroupCase param;
            param.weightType = dtype;
            param.transB = transB;
            RunCase(param, NO_FULL_LOAD);
            param.m = 32;
            param.n = 8192;
            RunCase(param, A_FULL_LOAD);
            param.m = 8192;
            param.n = 32;
            RunCase(param, B_FULL_LOAD);
        }
    }
}

TEST(QuantBatchMatmulV3PerGroupTiling, ChecksGroupBoundariesAndSingleOddEvenGroupCounts)
{
    const std::pair<int64_t, int64_t> cases[] = {{64, 64}, {64, 130}, {128, 256}, {384, 770}, {65472, 65472}};
    for (const auto& gk : cases) {
        PerGroupCase param;
        param.groupK = gk.first;
        param.k = gk.second;
        param.bias = false;
        param.expectedBaseK = param.groupK <= 128 ? static_cast<uint32_t>(param.groupK) : 192U;
        RunCase(param, NO_FULL_LOAD);
    }
}

TEST(QuantBatchMatmulV3PerGroupTiling, ChecksWeightAndActivationBatchBroadcast)
{
    PerGroupCase param;
    param.weightType = ge::DT_INT4;
    param.transB = true;
    param.batchA = 3;
    RunCase(param, NO_FULL_LOAD);
    param.batchA = 1;
    param.batchB = 3;
    RunCase(param, NO_FULL_LOAD);
}

TEST(QuantBatchMatmulV3PerGroupTiling, RejectsInvalidGroupSizes)
{
    for (const int64_t groupK : {1, 63, 65, 65535}) {
        PerGroupCase param;
        param.groupK = groupK;
        RunCase(param, NO_FULL_LOAD, false);
    }
}
