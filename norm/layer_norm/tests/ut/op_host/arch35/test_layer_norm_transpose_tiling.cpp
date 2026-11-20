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
#include <array>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/layer_norm_tiling_arch35.h"

namespace {
// Expected parameter mode values in serialized transpose tiling data.
constexpr uint32_t TRANSPOSE_PARAM_SAME = 0;
constexpr uint32_t TRANSPOSE_PARAM_BRC_P = 1;
constexpr uint32_t TRANSPOSE_PARAM_TILE_PARAMS = 2;
constexpr uint32_t TRANSPOSE_PARAM_FULL_PARAMS = 3;
constexpr uint32_t CORE_COUNT = 64;
constexpr uint64_t UB_BYTES = 253952;
struct TransposeData {
    uint64_t a;
    uint64_t aBlockFactor;
    uint32_t tileA;
    uint32_t r;
    uint32_t rAlign;
    uint32_t paramAlign;
    float epsilon;
    uint32_t usedCoreNum;
    uint64_t paramRows;
    uint32_t paramR;
    uint32_t paramStride;
    uint32_t paramMode;
};
struct RouteResult {
    ge::graphStatus status;
    uint64_t key;
    uint32_t blocks;
    TransposeData transpose{};
};

RouteResult RunTiling(const std::vector<int64_t>& dims, int64_t normAxis, int64_t paramsAxis,
                      ge::DataType xType = ge::DT_FLOAT, ge::DataType paramType = ge::DT_FLOAT, bool regbase = true)
{
    const int64_t rank = dims.size();
    const int64_t norm = normAxis < 0 ? rank + normAxis : normAxis;
    const int64_t params = paramsAxis < 0 ? rank + paramsAxis : paramsAxis;
    gert::Shape xDims, paramDims, statsDims;
    for (int64_t idx = 0; idx < rank; ++idx) {
        xDims.AppendDim(dims[idx]);
        statsDims.AppendDim(idx < norm ? dims[idx] : 1);
        if (idx >= params) {
            paramDims.AppendDim(dims[idx]);
        }
    }
    gert::StorageShape x, parameters, stats;
    x.MutableOriginShape() = xDims;
    x.MutableStorageShape() = xDims;
    parameters.MutableOriginShape() = paramDims;
    parameters.MutableStorageShape() = paramDims;
    stats.MutableOriginShape() = statsDims;
    stats.MutableStorageShape() = statsDims;
    optiling::LayerNormOpInfo compileInfo{};
    auto& info = compileInfo.regbaseCompileInfo;
    info.coreNum = CORE_COUNT;
    info.ubSizePlatForm = UB_BYTES;
    info.vectorLength = 256;
    info.blockSize = 32;
    info.isRegBase = regbase;
    fe::PlatFormInfos platform;
    platform.Init();
    std::map<std::string, std::string> arch{{"NpuArch", regbase ? "3510" : "2201"}};
    platform.SetPlatformRes("SoCInfo", arch);
    auto data = gert::TilingData::CreateCap(4096);
    auto workspace = gert::ContinuousVector::Create<size_t>(16);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("LayerNorm")
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &parameters, &parameters})
                      .OutputShapes({&x, &stats, &stats})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platform))
                      .NodeInputTd(0, xType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, paramType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, paramType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, xType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, paramType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, paramType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"begin_norm_axis", Ops::NN::AnyValue::CreateFrom<int64_t>(normAxis)},
                                  {"begin_params_axis", Ops::NN::AnyValue::CreateFrom<int64_t>(paramsAxis)},
                                  {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5f)}})
                      .TilingData(data.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("LayerNorm");
    EXPECT_NE(impl, nullptr);
    if (impl == nullptr) {
        return {ge::GRAPH_FAILED, 0, 0};
    }
    const auto status = impl->tiling(context);
    RouteResult result{status, context->GetTilingKey(), context->GetBlockDim()};
    if (status == ge::GRAPH_SUCCESS && result.key >= 1000 && result.key <= 1022) {
        const auto* rawData = context->GetRawTilingData();
        EXPECT_GE(rawData->GetDataSize(), sizeof(TransposeData));
        if (rawData->GetDataSize() >= sizeof(TransposeData)) {
            std::memcpy(&result.transpose, rawData->GetData(), sizeof(TransposeData));
        }
    }
    return result;
}

struct TypeCase {
    ge::DataType x;
    ge::DataType params;
    uint64_t key;
};
const std::array<TypeCase, 5> TYPES = {{{ge::DT_FLOAT, ge::DT_FLOAT, 1000},
                                        {ge::DT_FLOAT16, ge::DT_FLOAT, 1010},
                                        {ge::DT_FLOAT16, ge::DT_FLOAT16, 1011},
                                        {ge::DT_BF16, ge::DT_FLOAT, 1020},
                                        {ge::DT_BF16, ge::DT_BF16, 1022}}};
class LayerNormTransposeTiling : public testing::Test {};

TEST_F(LayerNormTransposeTiling, TransposeAllDtypesAtBoundary)
{
    for (const auto& type : TYPES) {
        SCOPED_TRACE(std::to_string(type.key));
        auto result = RunTiling({32769, 16}, -1, -1, type.x, type.params);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.key, type.key);
        EXPECT_EQ(result.transpose.paramMode, TRANSPOSE_PARAM_SAME);
        EXPECT_GT(result.blocks, 0U);
        EXPECT_LE(result.blocks, CORE_COUNT);
        // At 64 cores and VL_FP32=64, A=32768 is excluded; A=32769 is admitted.
        const auto belowBoundary = RunTiling({32768, 16}, -1, -1, type.x, type.params);
        ASSERT_EQ(belowBoundary.status, ge::GRAPH_SUCCESS);
        EXPECT_LT(belowBoundary.key, 1000U);
    }
}

TEST_F(LayerNormTransposeTiling, TransposeMultiDimensionalAxes)
{
    auto result = RunTiling({3641, 9, 2, 3}, 2, -2);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.key, 1000U);
}

TEST_F(LayerNormTransposeTiling, TransposeLargeA)
{
    auto result = RunTiling({131073, 3}, 1, 1);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.key, 1000U);
    EXPECT_EQ(result.blocks, CORE_COUNT);
}

TEST_F(LayerNormTransposeTiling, RAboveBoundaryKeepsExistingTemplate)
{
    auto result = RunTiling({32769, 17}, 1, 1);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_LT(result.key, 1000U);
}

TEST_F(LayerNormTransposeTiling, ROneKeepsNoReduceTemplate)
{
    auto result = RunTiling({32769, 1}, -1, -1);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.key, 600U);
}

TEST_F(LayerNormTransposeTiling, DifferentParameterAxesUseTransposeModes)
{
    for (const auto& axes : {std::pair<int64_t, int64_t>{1, 2}, {2, 1}}) {
        auto result = RunTiling({32769, 2, 3}, axes.first, axes.second);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.key, 1000U);
        const auto mode = axes.first < axes.second ? TRANSPOSE_PARAM_BRC_P : TRANSPOSE_PARAM_FULL_PARAMS;
        EXPECT_EQ(result.transpose.paramMode, mode);
        EXPECT_EQ(result.transpose.paramR, 3U);
    }
}

TEST_F(LayerNormTransposeTiling, TiledParametersUseAvailableBudget)
{
    for (int64_t b : {513, 8193}) {
        const auto result = RunTiling({113, b, 16}, 2, 1);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.key, 1000U);
        const auto& td = result.transpose;
        EXPECT_EQ(td.paramMode, TRANSPOSE_PARAM_TILE_PARAMS);
        EXPECT_EQ(td.paramRows, static_cast<uint64_t>(b));
        EXPECT_EQ(td.tileA, 384U);
        EXPECT_EQ(td.paramStride, td.tileA);
        EXPECT_EQ(td.paramAlign, td.r * td.tileA);
        const uint64_t baseBytes = td.tileA * (2 * (td.r + td.rAlign) * sizeof(float) + td.rAlign * sizeof(float) +
                                               4 * sizeof(float));
        EXPECT_LE(baseBytes + 4 * td.paramAlign * sizeof(float), UB_BYTES);
    }
}

TEST_F(LayerNormTransposeTiling, ResidentParametersAtCapacityBoundary)
{
    const auto result = RunTiling({113, 512, 16}, 2, 1);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.key, 1000U);
    EXPECT_EQ(result.transpose.paramMode, TRANSPOSE_PARAM_FULL_PARAMS);
    EXPECT_EQ(result.transpose.tileA, 512U);
    EXPECT_EQ(result.transpose.paramAlign, 512U * 16);
    EXPECT_EQ(result.transpose.paramStride, 0U);
    // The same parameter period must not be loaded in full by cores using only a small slice.
    const auto noReuse = RunTiling({65536, 16}, 1, 0);
    ASSERT_EQ(noReuse.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(noReuse.key, 1000U);
    EXPECT_EQ(noReuse.transpose.paramMode, TRANSPOSE_PARAM_TILE_PARAMS);
}

TEST_F(LayerNormTransposeTiling, ParameterModesCoverAllDtypes)
{
    for (const auto& type : TYPES) {
        for (int64_t b : {2, 3, 63, 64, 65, 8193}) {
            const auto result = RunTiling({32769, b, 3}, 2, 1, type.x, type.params);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            ASSERT_EQ(result.key, type.key);
            const auto& td = result.transpose;
            const bool resident = b != 8193;
            EXPECT_EQ(td.paramMode, resident ? TRANSPOSE_PARAM_FULL_PARAMS : TRANSPOSE_PARAM_TILE_PARAMS);
            EXPECT_EQ(td.paramRows, static_cast<uint64_t>(b));
            EXPECT_GT(td.tileA, 0U);
            EXPECT_EQ(td.tileA % 64, 0U);
            EXPECT_EQ(td.paramStride, resident ? 0U : td.tileA);
            if (resident) {
                EXPECT_LE(td.paramRows, td.tileA);
            } else {
                EXPECT_EQ(td.paramAlign, td.r * td.tileA);
            }
            if (b < 64) {
                EXPECT_GT(td.tileA, td.paramRows);
            }
            const uint64_t xBytes = type.x == ge::DT_FLOAT ? 4 : 2;
            const uint64_t paramBytes = type.params == ge::DT_FLOAT ? 4 : 2;
            const uint64_t baseBytes = td.tileA * (2 * (td.r * 4 + td.rAlign * xBytes) + td.rAlign * 4 + 16);
            const uint64_t expectedElements = resident ? b * td.r : td.tileA * td.r;
            EXPECT_EQ(td.paramAlign * paramBytes, (expectedElements * paramBytes + 31) / 32 * 32);
            EXPECT_LE(baseBytes + (resident ? 2 : 4) * td.paramAlign * paramBytes, UB_BYTES);
        }
        const auto broadcast = RunTiling({32769, 2, 3}, 1, 2, type.x, type.params);
        ASSERT_EQ(broadcast.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(broadcast.key, type.key);
        EXPECT_EQ(broadcast.transpose.paramMode, TRANSPOSE_PARAM_BRC_P);
    }
}

TEST_F(LayerNormTransposeTiling, ROneWithRowParametersUsesTranspose)
{
    const auto result = RunTiling({10923, 3, 1}, 2, 1);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.key, 1000U);
    EXPECT_EQ(result.transpose.paramMode, TRANSPOSE_PARAM_FULL_PARAMS);
}

TEST_F(LayerNormTransposeTiling, UnitDimensionsKeepSameParameterMode)
{
    const auto result = RunTiling({32769, 1, 3}, 2, 1);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(result.key, 1000U);
    EXPECT_EQ(result.transpose.paramMode, TRANSPOSE_PARAM_SAME);
}

TEST_F(LayerNormTransposeTiling, EmptyReductionAllDtypes)
{
    for (const auto& type : TYPES) {
        auto result = RunTiling({17, 0}, 1, 1, type.x, type.params);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.key, 1100U);
        EXPECT_GT(result.blocks, 0U);
    }
}

TEST_F(LayerNormTransposeTiling, EmptyReductionLargeAAndDifferentAxes)
{
    for (const auto& axes : {std::pair<int64_t, int64_t>{1, 2}, {2, 1}}) {
        auto result = RunTiling({4000001, 1, 0}, axes.first, axes.second);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(result.key, 1100U);
        EXPECT_EQ(result.blocks, CORE_COUNT);
    }
}

TEST_F(LayerNormTransposeTiling, EmptyOuterAxisIsRejected)
{
    EXPECT_NE(RunTiling({0, 8}, 1, 1).status, ge::GRAPH_SUCCESS);
    EXPECT_NE(RunTiling({0, 0}, 1, 1).status, ge::GRAPH_SUCCESS);
}

TEST_F(LayerNormTransposeTiling, EmptyReductionNonRegbaseIsRejected)
{
    EXPECT_NE(RunTiling({3, 0}, 1, 1, ge::DT_FLOAT, ge::DT_FLOAT, false).status, ge::GRAPH_SUCCESS);
}
} // namespace
