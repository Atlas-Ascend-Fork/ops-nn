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
#include <algorithm>
#include <cstring>
#include "../../../../op_host/arch35/rotate_dual_level_quant_tiling.h"
#include "infershape_test_util.h"
#include "tiling_context_faker.h"
#include "ut_op_common.h"

namespace {
namespace RDLQ = RotateDualLevelQuantNS;
struct TilingCase {
    const char* name;
    uint32_t m = 4, n = 544, k = 32;
    ge::DataType target = ge::DT_FLOAT4_E2M1;
    bool grouped = false, perBlock = false;
    int64_t groupType = 0, block0 = 512, block1 = 32, alg = 0;
    float dstMax = 0;
    bool a5 = true;
    uint32_t aiv = 64;
    uint64_t ub = 253952;
    uint32_t expectedRows = 0, expectedUbDB = 0;
    ge::graphStatus expected = ge::GRAPH_SUCCESS;
};

class RotateDualLevelQuantTiling : public testing::TestWithParam<TilingCase> {};
TEST_P(RotateDualLevelQuantTiling, HostContract)
{
    const auto c = GetParam();
    optiling::RotateDualLevelQuantCompileInfo compile;
    compile.initialized = true;
    compile.isA5 = c.a5;
    compile.aicNum = 32;
    compile.aivNum = c.aiv;
    compile.l1Size = 524288;
    compile.l0ASize = compile.l0BSize = 65536;
    compile.l0CSize = 262144;
    compile.ubSize = c.ub;
    auto x = CreateStorageShape({c.m, c.n});
    std::vector<int64_t> rotDims = c.grouped ? std::vector<int64_t>{3, c.k, c.k} : std::vector<int64_t>{c.k, c.k};
    if (c.perBlock) {
        rotDims.insert(rotDims.end() - 2, c.n / c.k);
    }
    auto rot = CreateStorageShape(rotDims);
    auto groups = CreateStorageShape({3});
    auto y = CreateStorageShape({c.m, c.n});
    auto s0 = CreateStorageShape({c.m, (c.n + 511) / 512});
    auto s1 = CreateStorageShape({c.m, (c.n + 63) / 64, 2});
    auto raw = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto faker = gert::TilingContextFaker();
    faker.IrInstanceNum({1, 1, 0, c.grouped ? 1U : 0U}, {1, 1, 1});
    if (c.grouped) {
        faker.InputShapes({&x, &rot, &groups});
        faker.NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND);
    } else {
        faker.InputShapes({&x, &rot});
    }
    auto holder = faker.SetOpType("RotateDualLevelQuant")
                      .OutputShapes({&y, &s0, &s1})
                      .CompileInfo(&compile)
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, c.target, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(c.target)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(c.alg)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(c.dstMax)},
                                  {"trans", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(c.groupType)},
                                  {"level0_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(c.block0)},
                                  {"level1_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(c.block1)}})
                      .TilingData(raw.get())
                      .Workspace(workspace)
                      .Build();
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("RotateDualLevelQuant");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(impl->tiling(context), c.expected);
    if (c.expected != ge::GRAPH_SUCCESS) {
        return;
    }
    RDLQ::RotateDualLevelQuantTilingData data;
    ASSERT_EQ(context->GetRawTilingData()->GetDataSize(), sizeof(data));
    std::memcpy(&data, context->GetRawTilingData()->GetData(), sizeof(data));
    EXPECT_EQ(context->GetTilingKey(), c.grouped ? 1UL : 0UL);
    EXPECT_EQ(data.M, c.m);
    EXPECT_EQ(data.N, c.n);
    EXPECT_EQ(data.rotK, c.k);
    EXPECT_EQ(data.rotationMatricesPerGroup, c.perBlock ? c.n / c.k : 1);
    EXPECT_EQ(data.level0ScalesPerRow, (c.n + 511) / 512);
    EXPECT_EQ(data.level1ScalePairsPerRow, (c.n + 63) / 64);
    EXPECT_EQ(data.lastLevel0BlockSize, c.n == 0 ? 0 : (c.n - 1) % 512 + 1);
    const float maxValue = c.target == ge::DT_FLOAT4_E2M1 ? 6.0f :
                                                            (c.target == ge::DT_FLOAT8_E4M3FN ? 448.0f : 57344.0f);
    EXPECT_FLOAT_EQ(data.level0MaxValue, maxValue);
    EXPECT_FLOAT_EQ(data.level0InvMaxValue, 1.0f / maxValue);
    EXPECT_EQ(data.userWorkspaceSize, 0);
    if (c.m == 0 || c.n == 0) {
        EXPECT_EQ(context->GetWorkspaceSizes(1)[0], 0);
        return;
    }
    EXPECT_EQ(context->GetWorkspaceSizes(1)[0], 16UL * 1024 * 1024);
    EXPECT_EQ(data.aivNumRatio, 2);
    EXPECT_EQ(data.splitM, 1);
    EXPECT_EQ(data.mL1 % 16, 0);
    EXPECT_EQ(data.maxRowsPerAiv, (data.mL1 + 1) / 2);
    EXPECT_EQ(data.ubPitchGran, 16);
    const uint64_t pitch = (static_cast<uint64_t>(data.baseN) + 15) / 16 * 16;
    const uint64_t receiveBytes = (data.maxRowsPerAiv * pitch * 2 + 512 + 31) / 32 * 32;
    const uint64_t workspaceBytes = 55296 + static_cast<uint64_t>(data.maxRowsPerAiv) * 512 * 2 + 512 +
                                    data.maxRowsPerAiv * 32;
    const uint64_t privateEnd = receiveBytes + workspaceBytes;
    EXPECT_EQ(data.fixpipeReceiveBytes, receiveBytes);
    EXPECT_EQ(data.quantWorkspaceBytes, workspaceBytes);
    EXPECT_LE(data.totalUbBytes, c.ub);
    EXPECT_LE(privateEnd, data.ubDB == 2 ? c.ub / 2 : c.ub);
    EXPECT_EQ(data.totalUbBytes, data.ubDB == 2 ? std::max(privateEnd, c.ub / 2 + receiveBytes) : privateEnd);
    EXPECT_EQ(data.collectUbOffset, data.mxWorkspaceBytes);
    EXPECT_GE(data.collectUbBytes, data.maxRowsPerAiv * 512 * 2 + 512);
    EXPECT_GE(data.level0ScaleUbOffset, data.collectUbOffset + data.collectUbBytes);
    EXPECT_EQ(data.level0ScaleUbOffset + data.level0ScaleUbBytes, data.quantWorkspaceBytes);
    EXPECT_LE(2UL * data.baseK * data.baseN * 2, 65536UL);
    const uint64_t tasks = ((c.m + data.mL1 - 1) / data.mL1) * static_cast<uint64_t>((c.n + 511) / 512);
    EXPECT_EQ(data.taskCount, c.grouped ? 0 : tasks);
    EXPECT_EQ(context->GetBlockDim(), data.usedCoreNum);
    EXPECT_EQ(data.ubDB, (c.grouped || c.n > c.k || tasks > data.usedCoreNum) && privateEnd <= c.ub / 2 ? 2 : 1);
    if (c.expectedRows != 0) {
        EXPECT_EQ(data.mL1, c.expectedRows);
    }
    if (c.expectedUbDB != 0) {
        EXPECT_EQ(data.ubDB, c.expectedUbDB);
    }
}

std::vector<TilingCase> Cases()
{
    std::vector<TilingCase> cases;
    for (auto target : {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2}) {
        for (uint32_t k : {32, 64, 128, 256}) {
            TilingCase c{"normal"};
            c.target = target;
            c.k = k;
            c.n = 3 * k;
            cases.push_back(c);
        }
    }
    TilingCase c{"grouped_per_column"};
    c.grouped = true;
    c.perBlock = true;
    c.groupType = 1;
    cases.push_back(c);
    c = TilingCase{"many_rounds"};
    c.m = 4097;
    c.n = 1024;
    cases.push_back(c);
    c = TilingCase{"single_fixpipe"};
    c.m = 1;
    c.n = 32;
    c.expectedUbDB = 1;
    cases.push_back(c);
    c = TilingCase{"one_task_sixteen_fixpipe_results"};
    c.m = 1;
    c.n = 512;
    c.expectedUbDB = 2;
    cases.push_back(c);
    c = TilingCase{"single_slot_exact_capacity"};
    c.m = 1;
    c.n = 512;
    c.ub = 65280;
    c.expectedRows = 16;
    c.expectedUbDB = 1;
    cases.push_back(c);
    c.name = "single_slot_missing_receive_space";
    c.ub = 65248;
    c.expected = ge::GRAPH_FAILED;
    cases.push_back(c);
    c = TilingCase{"double_slot_exact_capacity"};
    c.m = 1;
    c.n = 512;
    c.ub = 130560;
    c.expectedRows = 16;
    c.expectedUbDB = 2;
    cases.push_back(c);
    c.name = "double_slot_falls_back_without_shrinking_rows";
    c.ub = 130528;
    c.expectedUbDB = 1;
    cases.push_back(c);
    c = TilingCase{"receive_space_limits_rows"};
    c.m = 4097;
    c.n = 1024;
    c.k = 256;
    c.ub = 98304;
    c.expectedRows = 48;
    c.expectedUbDB = 1;
    cases.push_back(c);
    c = TilingCase{"empty_m"};
    c.m = 0;
    cases.push_back(c);
    c = TilingCase{"empty_n"};
    c.n = 0;
    cases.push_back(c);
    c = TilingCase{"block0"};
    c.block0 = 256;
    c.expected = ge::GRAPH_FAILED;
    cases.push_back(c);
    c = TilingCase{"block1"};
    c.block1 = 64;
    c.expected = ge::GRAPH_FAILED;
    cases.push_back(c);
    c = TilingCase{"non_a5"};
    c.a5 = false;
    c.expected = ge::GRAPH_FAILED;
    cases.push_back(c);
    c = TilingCase{"bad_ratio"};
    c.aiv = 32;
    c.expected = ge::GRAPH_FAILED;
    cases.push_back(c);
    c = TilingCase{"ub_short"};
    c.ub = 1024;
    c.expected = ge::GRAPH_FAILED;
    cases.push_back(c);
    c = TilingCase{"fp4_alg2"};
    c.alg = 2;
    c.dstMax = 7;
    cases.push_back(c);
    c = TilingCase{"fp8_alg1"};
    c.target = ge::DT_FLOAT8_E4M3FN;
    c.alg = 1;
    cases.push_back(c);
    return cases;
}
INSTANTIATE_TEST_SUITE_P(A5, RotateDualLevelQuantTiling, testing::ValuesIn(Cases()));
} // namespace
