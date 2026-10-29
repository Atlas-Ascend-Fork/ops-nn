/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <array>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/ge_glu_grad_v2_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

class GeGluGradV2Tiling : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        cout << "GeGluGradV2Tiling SetUpTestSuite" << endl;
        compileInfoStr = R"({"hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 196608,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 48
                                        }})";

        GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
        platformInfo.Init();

        string opType("GeGluGradV2");
        ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
        tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
        tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    }

    static void TearDownTestSuite() { cout << "GeGluGradV2Tiling TearDownTestSuite" << endl; }

protected:
    static map<string, string> socInfos;
    static map<string, string> aicoreSpec;
    static map<string, string> intrinsics;
    static string compileInfoStr;

    // platform info
    static fe::PlatFormInfos platformInfo;
    // compile info
    static optiling::GeGluGradV2CompileInfo compileInfo;
    static gert::OpImplKernelRegistry::KernelFunc tilingParseFunc;
    static gert::OpImplKernelRegistry::TilingKernelFunc tilingFunc;
};

map<string, string> GeGluGradV2Tiling::socInfos;
map<string, string> GeGluGradV2Tiling::aicoreSpec;
map<string, string> GeGluGradV2Tiling::intrinsics;
string GeGluGradV2Tiling::compileInfoStr;
// platform info
fe::PlatFormInfos GeGluGradV2Tiling::platformInfo;
// compile info
optiling::GeGluGradV2CompileInfo GeGluGradV2Tiling::compileInfo;
gert::OpImplKernelRegistry::KernelFunc GeGluGradV2Tiling::tilingParseFunc = nullptr;
gert::OpImplKernelRegistry::TilingKernelFunc GeGluGradV2Tiling::tilingFunc = nullptr;

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_parse_func)
{
    // tilingParseFunc simulate
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);

    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(GeGluGradV2Tiling, regbase_balanced_tiles_do_not_create_an_extra_tail)
{
    struct SplitCase {
        ge::DataType dtype;
        int64_t approximate;
        int64_t m;
        int64_t tile;
        int64_t groups;
    };
    // Capacity+1 is particularly important: rounding down the balanced tile
    // incorrectly creates a third task with a tiny tail.
    const std::vector<SplitCase> cases = {
        {ge::DT_FLOAT16, 1, 5761, 2896, 2}, {ge::DT_BF16, 1, 4881, 2448, 2},   {ge::DT_FLOAT, 0, 3337, 1672, 2},
        {ge::DT_FLOAT, 1, 3729, 1872, 2},   {ge::DT_FLOAT, 1, 11008, 3672, 3},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(item.m);
        gert::StorageShape dy = {{65, item.m}, {65, item.m}};
        gert::StorageShape x = {{65, 2 * item.m}, {65, 2 * item.m}};
        optiling::GeGluGradV2CompileInfo info = {64, 253952, NpuArch::DAV_3510, true};
        auto data = gert::TilingData::CreateCap(4096);
        auto workspace = gert::ContinuousVector::Create<size_t>(4096);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 1)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&dy, &x, &dy})
                          .OutputShapes({&x})
                          .CompileInfo(&info)
                          .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                          .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                      {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(item.approximate)},
                                      {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                          .NodeInputTd(0, item.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, item.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, item.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, item.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .TilingData(data.get())
                          .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                          .Build();
        auto context = holder.GetContext<gert::TilingContext>();
        ASSERT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);
        // Wire layout: two int32 attrs followed by eight int64 fields.
        std::array<int64_t, 9> result{};
        ASSERT_EQ(context->GetRawTilingData()->GetDataSize(), sizeof(result));
        std::memcpy(result.data(), context->GetRawTilingData()->GetData(), sizeof(result));
        EXPECT_EQ(result[1], item.tile);   // maxProcCount
        EXPECT_EQ(result[8], item.groups); // groupNum
        EXPECT_EQ(context->GetBlockDim(), 64);
        EXPECT_GT(item.m - (item.groups - 1) * item.tile, 0);
    }
}

TEST_F(GeGluGradV2Tiling, regbase_direct_group_uses_per_core_budget)
{
    struct GroupCase {
        ge::DataType dtype;
        int64_t approximate;
        int64_t n;
        int64_t m;
        int64_t expectedGroup;
        int64_t expectedBlockDim;
    };
    const std::vector<GroupCase> cases = {
        // One wave fits in the UB tile: ceil(192 * 160 / 64) = 480 elements,
        // so three 160-element rows are assigned to each core.
        {ge::DT_FLOAT16, 1, 192, 160, 3, 64},
        // Include row padding: M=23 occupies 32 elements, but all 16 rows
        // assigned to a core still fit in one tile.
        {ge::DT_FLOAT16, 1, 1024, 23, 16, 64},
        {ge::DT_FLOAT, 0, 192, 160, 3, 64},
        // Capacity-limited cases balance the minimum number of waves.
        {ge::DT_FLOAT, 0, 4096, 129, 22, 64},
        {ge::DT_BF16, 0, 16384, 129, 32, 64},
        {ge::DT_FLOAT, 1, 65, 129, 2, 33},
        {ge::DT_FLOAT16, 1, 1, 23, 1, 1},
        // FP32 erf retains two waves at 2048 padded elements per core.
        {ge::DT_FLOAT, 0, 1024, 120, 16, 64},
        {ge::DT_FLOAT, 0, 1024, 121, 8, 64},
        {ge::DT_FLOAT, 0, 1024, 129, 8, 64},
        {ge::DT_FLOAT, 1, 512, 248, 8, 64},
        {ge::DT_FLOAT, 1, 512, 249, 8, 64},
        {ge::DT_FLOAT, 0, 512, 249, 4, 64},
        {ge::DT_FLOAT, 1, 128, 1024, 2, 64},
        {ge::DT_FLOAT, 0, 128, 1024, 1, 64},
        {ge::DT_FLOAT, 0, 64, 2048, 1, 64},
        {ge::DT_BF16, 0, 1024, 129, 16, 64},
    };

    for (const auto& item : cases) {
        SCOPED_TRACE(item.m);
        gert::StorageShape dy = {{item.n, item.m}, {item.n, item.m}};
        gert::StorageShape x = {{item.n, 2 * item.m}, {item.n, 2 * item.m}};
        optiling::GeGluGradV2CompileInfo info = {64, 253952, NpuArch::DAV_3510, true};
        auto data = gert::TilingData::CreateCap(4096);
        auto workspace = gert::ContinuousVector::Create<size_t>(4096);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 1)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&dy, &x, &dy})
                          .OutputShapes({&x})
                          .CompileInfo(&info)
                          .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                          .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                                      {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(item.approximate)},
                                      {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                          .NodeInputTd(0, item.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, item.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, item.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, item.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                          .TilingData(data.get())
                          .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                          .Build();
        auto context = holder.GetContext<gert::TilingContext>();
        ASSERT_EQ(tilingFunc(context), ge::GRAPH_SUCCESS);

        std::array<int64_t, 9> result{};
        ASSERT_EQ(context->GetRawTilingData()->GetDataSize(), sizeof(result));
        std::memcpy(result.data(), context->GetRawTilingData()->GetData(), sizeof(result));
        EXPECT_EQ(result[8], item.expectedGroup);
        EXPECT_EQ(context->GetBlockDim(), item.expectedBlockDim);
        const int64_t blockElements = 32 / ge::GetSizeByDataType(item.dtype);
        const int64_t alignedM = (item.m + blockElements - 1) / blockElements * blockElements;
        EXPECT_LE(result[8] * alignedM, result[1]);
        // Full core loops, extra full tasks and the final partial task cover N.
        EXPECT_EQ((result[5] * result[4] + result[6]) * result[8] + result[7], item.n);
    }
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_001)
{
    gert::StorageShape dyShape = {{23, 12, 1024}, {23, 12, 1024}};
    gert::StorageShape xShape = {{23, 12, 1024 * 2}, {23, 12, 1024 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 101);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_002)
{
    gert::StorageShape dyShape = {{23, 12, 5457}, {23, 12, 5457}};
    gert::StorageShape xShape = {{23, 12, 5457 * 2}, {23, 12, 5457 * 2}};

    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 102);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_003)
{
    gert::StorageShape dyShape = {{1132, 71, 7}, {1132, 12, 7}};
    gert::StorageShape xShape = {{1132, 71, 7 * 2}, {1132, 12, 7 * 2}};

    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 103);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_004)
{
    gert::StorageShape dyShape = {{23, 12, 1024}, {23, 12, 1024}};
    gert::StorageShape xShape = {{23, 12, 1024 * 2}, {23, 12, 1024 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 701);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp16_001)
{
    gert::StorageShape dyShape = {{24, 576, 5120}, {24, 576, 5120}};
    gert::StorageShape xShape = {{24, 576, 5120 * 2}, {24, 576, 5120 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 201);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp16_002)
{
    gert::StorageShape dyShape = {{24, 576, 6145}, {24, 576, 6145}};
    gert::StorageShape xShape = {{24, 576, 6145 * 2}, {24, 576, 6145 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 202);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp16_003)
{
    gert::StorageShape dyShape = {{24, 576, 13}, {24, 576, 13}};
    gert::StorageShape xShape = {{24, 576, 13 * 2}, {24, 576, 13 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 203);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp16_004)
{
    gert::StorageShape dyShape = {{24, 576, 1033}, {24, 576, 1033}};
    gert::StorageShape xShape = {{24, 576, 1033 * 2}, {24, 576, 1033 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 801);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_001)
{
    gert::StorageShape dyShape = {{24, 576, 3120}, {24, 576, 3120}};
    gert::StorageShape xShape = {{24, 576, 3120 * 2}, {24, 576, 3120 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 301);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_002)
{
    gert::StorageShape dyShape = {{24, 576, 4465}, {24, 576, 4465}};
    gert::StorageShape xShape = {{24, 576, 4465 * 2}, {24, 576, 4465 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 302);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_003)
{
    gert::StorageShape dyShape = {{24, 576, 13}, {24, 576, 13}};
    gert::StorageShape xShape = {{24, 576, 13 * 2}, {24, 576, 13 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 303);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_004)
{
    gert::StorageShape dyShape = {{24, 576, 1033}, {24, 576, 1033}};
    gert::StorageShape xShape = {{24, 576, 1033 * 2}, {24, 576, 1033 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2201, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 901);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_fp32_005)
{
    gert::StorageShape dyShape = {{24, 576, 1033}, {24, 576, 1033}};
    gert::StorageShape xShape = {{24, 576, 1033 * 2}, {24, 576, 1033 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {64, 262144, NpuArch::DAV_3510, true};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 901);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_310p_erf)
{
    gert::StorageShape dyShape = {{24, 576, 1033}, {24, 576, 1033}};
    gert::StorageShape xShape = {{24, 576, 1033 * 2}, {24, 576, 1033 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 0;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2002, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}

TEST_F(GeGluGradV2Tiling, ge_glu_grad_v2_tiling_bfp16_310p)
{
    gert::StorageShape dyShape = {{23, 12, 1024}, {23, 12, 1024}};
    gert::StorageShape xShape = {{23, 12, 1024 * 2}, {23, 12, 1024 * 2}};
    int64_t dimAttr = -1;
    int64_t approximateAttr = 1;
    bool activateLeftAttr = false;
    compileInfo = {48, 196608, NpuArch::DAV_2002, false};

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto tilingHolder = gert::TilingContextFaker()
                            .NodeIoNum(3, 1)
                            .IrInstanceNum({1, 1, 1})
                            .InputShapes({&dyShape, &xShape, &dyShape})
                            .OutputShapes({&xShape})
                            .CompileInfo(&compileInfo)
                            .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                            .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dimAttr)},
                                        {"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(approximateAttr)},
                                        {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activateLeftAttr)}})
                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .TilingData(param.get())
                            .Workspace(workspaceSize)
                            .Build();

    gert::TilingContext* tilingContext = tilingHolder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                      intrinsics);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
}
