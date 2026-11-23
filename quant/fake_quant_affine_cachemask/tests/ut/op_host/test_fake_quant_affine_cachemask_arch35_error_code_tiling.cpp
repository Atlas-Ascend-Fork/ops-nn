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
 * \file test_fake_quant_affine_cachemask_arch35_error_code_tiling.cpp
 * \brief FakeQuantAffineCachemask arch35 Tiling 错误码整改 UT。
 *
 * 平台信息按 NpuArch=3510（Ascend950）配置，使 Tiling 入口经 IsRegbaseSocVersion 分派到
 * arch35 GetShapeAttrsInfo，从而覆盖该文件的正式错误码上报点。
 * 每个用例断言：返回 ge::GRAPH_FAILED、期望正式码恰好 1 条、无内部兜底码
 * （外层聚合日志已改 OP_LOGE_WITHOUT_REPORT，不得追加内部兜底码）。
 */

#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "fake_quant_error_code_ut_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "ut_op_common.h"
#include "ut_op_util.h"

namespace {
constexpr const char* kOpType = "FakeQuantAffineCachemask";
constexpr const char* kSuiteName = "FakeQuantAffineCachemaskErrorCodeTiling";

struct FakeQuantArch35CompileInfo {
    int32_t totalCoreNum = 48;
    uint64_t ubSizePlatForm = 262144;
};

struct TilingCaseParam {
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType scaleDtype = ge::DT_FLOAT;
    ge::DataType zpDtype = ge::DT_INT32;
    ge::Format xFormat = ge::FORMAT_ND;
    int64_t axis = 0;
    int64_t quantMin = 0;
    int64_t quantMax = 255;
};

// Ascend950 / NpuArch=3510 → IsRegbaseSocVersion 为真，Tiling 走 arch35 分支
void InitArch35PlatformRes(fe::PlatFormInfos* platformInfo)
{
    if (platformInfo == nullptr) {
        return;
    }
    std::map<std::string, std::string> socInfos;
    socInfos["ai_core_cnt"] = "48";
    socInfos["cube_core_cnt"] = "24";
    socInfos["vector_core_cnt"] = "48";
    socInfos["core_type_list"] = "AICore";
    socInfos["l2_size"] = "33554432";
    platformInfo->SetPlatformRes("SoCInfo", socInfos);

    std::map<std::string, std::string> aicoreSpec;
    aicoreSpec["ub_size"] = "262144";
    aicoreSpec["l0_a_size"] = "65536";
    aicoreSpec["l0_b_size"] = "65536";
    aicoreSpec["l0_c_size"] = "131072";
    aicoreSpec["l1_size"] = "524288";
    platformInfo->SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo->SetCoreNumByCoreType("AICore");

    std::map<std::string, std::string> socVersion;
    socVersion["Short_SoC_version"] = "Ascend950";
    socVersion["NpuArch"] = "3510";
    platformInfo->SetPlatformRes("version", socVersion);
}

ge::graphStatus RunArch35TilingCase(gert::StorageShape& xShape, gert::StorageShape& scaleShape,
                                    gert::StorageShape& zpShape, gert::StorageShape& yShape,
                                    gert::StorageShape& maskShape, const TilingCaseParam& param)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    FakeQuantArch35CompileInfo compileInfo;
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(16);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (tilingData == nullptr || wsSize == nullptr) {
        ADD_FAILURE() << "failed to allocate tiling data/workspace";
        return ge::GRAPH_FAILED;
    }

    auto holder = gert::TilingContextFaker()
                      .SetOpType(kOpType)
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &scaleShape, &zpShape})
                      .OutputShapes({&yShape, &maskShape})
                      .CompileInfo(&compileInfo)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(param.axis)},
                                  {"quant_min", Ops::NN::AnyValue::CreateFrom<int64_t>(param.quantMin)},
                                  {"quant_max", Ops::NN::AnyValue::CreateFrom<int64_t>(param.quantMax)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, param.xDtype, param.xFormat, param.xFormat)
                      .NodeInputTd(1, param.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, param.zpDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr) {
        ADD_FAILURE() << "failed to build tiling context";
        return ge::GRAPH_FAILED;
    }
    fe::PlatFormInfos* platformInfoPtr = tilingContext->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        ADD_FAILURE() << "failed to get tiling platform info";
        return ge::GRAPH_FAILED;
    }
    InitArch35PlatformRes(platformInfoPtr);
    return opImpl->tiling(tilingContext);
}

const auto g_tilingListenerInstalled = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new fake_quant_ut::NoEz9999Listener(kSuiteName));
    return 0;
}();
} // namespace

class FakeQuantAffineCachemaskErrorCodeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FakeQuantAffineCachemaskErrorCodeTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FakeQuantAffineCachemaskErrorCodeTiling TearDown" << std::endl; }
};

// y shape 与 x 不一致 → EZ0010（秩不同这一支）
TEST_F(FakeQuantAffineCachemaskErrorCodeTiling, arch35_y_shape_mismatch_ez0010)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape zpShape = {{1}, {1}};
    gert::StorageShape yShape = {{6}, {6}};
    gert::StorageShape maskShape = {{2, 3}, {2, 3}};
    TilingCaseParam param;
    EXPECT_EQ(RunArch35TilingCase(xShape, scaleShape, zpShape, yShape, maskShape, param), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0010", "arch35_y_shape_mismatch");
}

// mask shape 与 x 不一致 → EZ0010（秩相同、元素数不同这一支）
TEST_F(FakeQuantAffineCachemaskErrorCodeTiling, arch35_mask_shape_mismatch_ez0010)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape zpShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 3}, {2, 3}};
    gert::StorageShape maskShape = {{2, 2}, {2, 2}};
    TilingCaseParam param;
    EXPECT_EQ(RunArch35TilingCase(xShape, scaleShape, zpShape, yShape, maskShape, param), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0010", "arch35_mask_shape_mismatch");
}

// x format 非 ND → EZ0035
TEST_F(FakeQuantAffineCachemaskErrorCodeTiling, arch35_x_format_not_nd_ez0035)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape zpShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 3}, {2, 3}};
    gert::StorageShape maskShape = {{2, 3}, {2, 3}};
    TilingCaseParam param;
    param.xFormat = ge::FORMAT_FRACTAL_NZ;
    EXPECT_EQ(RunArch35TilingCase(xShape, scaleShape, zpShape, yShape, maskShape, param), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0035", "arch35_x_format_not_nd");
}

// x dtype 不支持 → EZ0020
TEST_F(FakeQuantAffineCachemaskErrorCodeTiling, arch35_x_dtype_unsupported_ez0020)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape zpShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 3}, {2, 3}};
    gert::StorageShape maskShape = {{2, 3}, {2, 3}};
    TilingCaseParam param;
    param.xDtype = ge::DT_DOUBLE;
    param.scaleDtype = ge::DT_DOUBLE;
    EXPECT_EQ(RunArch35TilingCase(xShape, scaleShape, zpShape, yShape, maskShape, param), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0020", "arch35_x_dtype_unsupported");
}

// scale 与 x dtype 不一致 → EZ0021
TEST_F(FakeQuantAffineCachemaskErrorCodeTiling, arch35_scale_dtype_mismatch_ez0021)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape zpShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 3}, {2, 3}};
    gert::StorageShape maskShape = {{2, 3}, {2, 3}};
    TilingCaseParam param;
    param.scaleDtype = ge::DT_FLOAT16;
    EXPECT_EQ(RunArch35TilingCase(xShape, scaleShape, zpShape, yShape, maskShape, param), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0021", "arch35_scale_dtype_mismatch");
}

// axis 越界 → EZ0026
TEST_F(FakeQuantAffineCachemaskErrorCodeTiling, arch35_axis_out_of_range_ez0026)
{
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape zpShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 3}, {2, 3}};
    gert::StorageShape maskShape = {{2, 3}, {2, 3}};
    TilingCaseParam param;
    param.axis = 2;
    EXPECT_EQ(RunArch35TilingCase(xShape, scaleShape, zpShape, yShape, maskShape, param), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0026", "arch35_axis_out_of_range");
}

// scale size 与 axis 维长度不一致 → EZ0015
TEST_F(FakeQuantAffineCachemaskErrorCodeTiling, arch35_scale_size_mismatch_ez0015)
{
    gert::StorageShape xShape = {{4, 3}, {4, 3}};
    gert::StorageShape scaleShape = {{2}, {2}};
    gert::StorageShape zpShape = {{1}, {1}};
    gert::StorageShape yShape = {{4, 3}, {4, 3}};
    gert::StorageShape maskShape = {{4, 3}, {4, 3}};
    TilingCaseParam param;
    param.axis = 1;
    EXPECT_EQ(RunArch35TilingCase(xShape, scaleShape, zpShape, yShape, maskShape, param), ge::GRAPH_FAILED);
    fake_quant_ut::ExpectSingleFormalCode("EZ0015", "arch35_scale_size_mismatch");
}
