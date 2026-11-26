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
#include <cstring>
#include <string>
#include <vector>
#include "tiling_context_faker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include <map>
#include "register/op_impl_registry.h"
#include "op_impl_registry.h"
#include "../../../op_host/situ_glu_tiling.h"

namespace {
// Mirror of the serialized kernel ABI layout written by SaveToBuffer().
// The host-side SituGluTilingData struct carries per-field reserve buffers, so its
// sizeof() is much larger than the bytes actually serialized into the tiling buffer.
struct SituGluTilingDataMirror {
    int64_t coreNumAll;
    int64_t dimBatchSize;
    int64_t dim2H;
    int64_t isLongH;
    int64_t ubMaxPair;
    float beta;
    float linearBeta;
    int64_t activateLeft;
};

struct TilingCase {
    const char* name;
    int64_t rows;
    int64_t columns;
    int64_t outRows;
    int64_t outColumns;
    int64_t dim;
    ge::DataType dtype;
    ge::DataType otherDtype;
    bool success;
    const char* shortSocVersion;
    const char* npuArch;
    int64_t expectedIsLongH;
};
class SituGluTilingTest : public testing::TestWithParam<TilingCase> {};

TEST_P(SituGluTilingTest, ValidatesInputsAndWritesTilingData)
{
    const auto& test = GetParam();
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("SituGlu");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> coreSpec;
    std::map<std::string, std::string> intrinsics;
    const char* hardware = R"({"hardware_info": {
        "UB_SIZE":196608,"L2_SIZE":33554432,"L1_SIZE":524288,
        "L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":131072,"CORE_NUM":40}})";
    GetPlatFormInfos(hardware, socInfos, coreSpec, intrinsics);
    fe::PlatFormInfos platform;
    ASSERT_TRUE(platform.Init());
    std::map<std::string, std::string> version = {{"Short_SoC_version", test.shortSocVersion},
                                                  {"NpuArch", test.npuArch}};
    platform.SetPlatformRes("version", version);
    platform.SetPlatformRes("SoCInfo", socInfos);
    platform.SetPlatformRes("AICoreSpec", coreSpec);
    platform.SetCoreNumByCoreType("AICore");
    platform.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    gert::StorageShape xShape = {{test.rows, test.columns}, {test.rows, test.columns}};
    gert::StorageShape otherShape = {{test.outRows, test.outColumns}, {test.outRows, test.outColumns}};
    optiling::SituGluCompileInfo compileInfo{};
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);
    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(workspace, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("SituGlu")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&otherShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(&platform)
                      .NodeInputTd(0, test.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, test.otherDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(test.dim)},
                                  {"beta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"linear_beta", Ops::NN::AnyValue::CreateFrom<float>(0.0f)},
                                  {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(tilingData.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    const auto status = impl->tiling(context);
    EXPECT_EQ(status, test.success ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED);
    if (test.success) {
        ASSERT_NE(context->GetRawTilingData(), nullptr);
        EXPECT_GT(context->GetRawTilingData()->GetDataSize(), 0U);
        ASSERT_GE(context->GetRawTilingData()->GetDataSize(), sizeof(SituGluTilingDataMirror));
        SituGluTilingDataMirror rawTiling{};
        std::memcpy(&rawTiling, context->GetRawTilingData()->GetData(), sizeof(rawTiling));
        EXPECT_EQ(rawTiling.isLongH, test.expectedIsLongH);
    }
}

INSTANTIATE_TEST_SUITE_P(
    InputValidation, SituGluTilingTest,
    testing::Values(
        TilingCase{"Valid", 4, 8, 4, 4, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, true, "Ascend910B", "2201", 1},
        TilingCase{"ValidDimZero", 4, 8, 2, 8, 0, ge::DT_FLOAT16, ge::DT_FLOAT16, true, "Ascend910B", "2201", 0},
        TilingCase{"EmptyXEmptyOut", 0, 8, 0, 4, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, true, "Ascend910B", "2201", 1},
        TilingCase{"DimOutOfRange", 4, 8, 4, 4, 2, ge::DT_FLOAT16, ge::DT_FLOAT16, false, "Ascend910B", "2201", 0},
        TilingCase{"OddSplit", 4, 7, 4, 3, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, false, "Ascend910B", "2201", 0},
        TilingCase{"UnsupportedInputDtype", 4, 8, 4, 4, -1, ge::DT_INT32, ge::DT_INT32, false, "Ascend910B", "2201", 0},
        TilingCase{"OutputDtypeMismatch", 4, 8, 4, 4, -1, ge::DT_FLOAT16, ge::DT_FLOAT, false, "Ascend910B", "2201", 0},
        TilingCase{"OutputSplitMismatch", 4, 8, 4, 3, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, false, "Ascend910B", "2201",
                   0},
        TilingCase{"NonSplitDimMismatchSmaller", 4, 8, 3, 4, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, false, "Ascend910B",
                   "2201", 0},
        TilingCase{"NonSplitDimMismatchBigger", 4, 8, 5, 4, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, false, "Ascend910B",
                   "2201", 0},
        TilingCase{"NonSplitDimMismatchDimZero", 4, 8, 2, 7, 0, ge::DT_FLOAT16, ge::DT_FLOAT16, false, "Ascend910B",
                   "2201", 0},
        TilingCase{"EmptyXNonEmptyOut", 0, 8, 1, 8, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, false, "Ascend910B", "2201", 0},

        TilingCase{"Valid960DT", 4, 8, 4, 4, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, true, "Ascend960DT", "9201", 1},
        TilingCase{"Valid960DTBf16", 4, 8, 4, 4, -1, ge::DT_BF16, ge::DT_BF16, true, "Ascend960DT", "9201", 1},
        TilingCase{"Valid960DTLongHUnaligned", 2, 250, 2, 125, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, true, "Ascend960DT",
                   "9201", 1},
        TilingCase{"Valid960DTLongHUbLimit", 2, 20000, 2, 10000, -1, ge::DT_FLOAT16, ge::DT_FLOAT16, true,
                   "Ascend960DT", "9201", 1}),
    [](const testing::TestParamInfo<TilingCase>& info) { return info.param.name; });
} // namespace
