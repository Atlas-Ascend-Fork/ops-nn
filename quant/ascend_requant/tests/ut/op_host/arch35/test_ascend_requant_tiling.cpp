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
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "base/err_mgr.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_kernel/arch35/ascend_requant_flat_route.h"
#include "../../../../op_kernel/arch35/ascend_requant_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {

constexpr char kCompileInfo[] = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                    "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                    "CORE_NUM": 24}
})";

constexpr int64_t kUtCoreNum = 24;

struct AscendRequantCompileInfoForTest {};

void SetupPlatform(fe::PlatFormInfos& platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                   map<string, string>& intrinsics)
{
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

void SetupKernelHolder(fe::PlatFormInfos& platformInfo, AscendRequantCompileInfoForTest& compileInfo,
                       const map<string, string>& socInfos, const map<string, string>& aicoreSpec,
                       const map<string, string>& intrinsics, gert::KernelRunContextHolder& kernelHolder)
{
    std::string compileInfoStr(kCompileInfo);
    kernelHolder = gert::KernelRunContextFaker()
                       .KernelIONum(2, 1)
                       .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                       .Outputs({&compileInfo})
                       .Build();
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", const_cast<map<string, string>&>(socInfos));
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", const_cast<map<string, string>&>(aicoreSpec));
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", const_cast<map<string, string>&>(intrinsics));
}

template <int64_t R>
AscendRequantTilingData<R>* FetchTilingData(gert::TilingContext* tilingContext)
{
    auto* tilingData = tilingContext->GetTilingData<AscendRequantTilingData<R>>();
    EXPECT_NE(tilingData, nullptr);
    return tilingData;
}

void CheckFlatTilingInvariants(const FlatRouteTiling& flat)
{
    EXPECT_EQ(flat.tileElems % ASCEND_REQUANT_FLAT_ALIGN_ELEMS, 0);
    EXPECT_EQ(flat.totalTiles, (flat.flatElems + flat.tileElems - 1) / flat.tileElems);
    EXPECT_EQ(flat.numCores, std::min(flat.totalTiles, kUtCoreNum));
    EXPECT_EQ(flat.tilesMain, flat.totalTiles / flat.numCores);
    EXPECT_EQ(flat.coresTail, flat.totalTiles % flat.numCores);
    if (flat.route == ASCEND_REQUANT_ROUTE_PERIODIC) {
        EXPECT_LE(flat.scalePeriod, ASCEND_REQUANT_FLAT_VEC_ELEMS);
        EXPECT_EQ(ASCEND_REQUANT_FLAT_VEC_ELEMS % flat.scalePeriod, 0);
        EXPECT_LE(flat.scaleInitElems, flat.scalePeriod);
    } else if (flat.route == ASCEND_REQUANT_ROUTE_CONTIG) {
        EXPECT_LE(flat.tileElems, flat.scalePeriod);
        EXPECT_EQ((flat.scalePeriod * ASCEND_REQUANT_FLAT_S_BYTES) % ASCEND_REQUANT_FLAT_MTE_ALIGN_BYTES, 0);
        EXPECT_EQ(flat.scaleInitElems, 0);
    }
}

} // namespace

class AscendRequantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AscendRequantTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AscendRequantTilingTest TearDown" << std::endl; }
};

TEST_F(AscendRequantTilingTest, tiling_rank1_scalar_broadcast)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{16}, {16}};
    gert::StorageShape sShape = {{1}, {1}};
    gert::StorageShape yShape = {{16}, {16}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank2_per_channel)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4, 8}, {4, 8}};
    gert::StorageShape sShape = {{8}, {8}};
    gert::StorageShape yShape = {{4, 8}, {4, 8}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank4_per_channel)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 5, 5}, {2, 3, 5, 5}};
    gert::StorageShape sShape = {{1, 3, 1, 1}, {1, 3, 1, 1}};
    gert::StorageShape yShape = {{2, 3, 5, 5}, {2, 3, 5, 5}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_relu_true)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 4}, {2, 4}};
    gert::StorageShape sShape = {{4}, {4}};
    gert::StorageShape yShape = {{2, 4}, {2, 4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_empty_tensor)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{0, 3}, {0, 3}};
    gert::StorageShape sShape = {{3}, {3}};
    gert::StorageShape yShape = {{0, 3}, {0, 3}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank5_scalar_broadcast)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape sShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank8_per_channel)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 4, 5, 6, 7, 8, 9}, {2, 3, 4, 5, 6, 7, 8, 9}};
    gert::StorageShape sShape = {{1, 1, 1, 1, 1, 1, 1, 9}, {1, 1, 1, 1, 1, 1, 1, 9}};
    gert::StorageShape yShape = {{2, 3, 4, 5, 6, 7, 8, 9}, {2, 3, 4, 5, 6, 7, 8, 9}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_non_aligned_tail)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4, 7, 3, 5}, {4, 7, 3, 5}};
    gert::StorageShape sShape = {{1, 7, 1, 1}, {1, 7, 1, 1}};
    gert::StorageShape yShape = {{4, 7, 3, 5}, {4, 7, 3, 5}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_full_shape_no_broadcast)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{3, 4}, {3, 4}};
    gert::StorageShape sShape = {{3, 4}, {3, 4}};
    gert::StorageShape yShape = {{3, 4}, {3, 4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_rank3_per_channel)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape sShape = {{4}, {4}};
    gert::StorageShape yShape = {{2, 3, 4}, {2, 3, 4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_dtype_not_supported)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4}, {4}};
    gert::StorageShape sShape = {{1}, {1}};
    gert::StorageShape yShape = {{4}, {4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    ASSERT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
    EXPECT_NE(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    const auto messages = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(messages.size(), 1U);
    const auto& message = messages.front();
    ASSERT_NE(message.error_id, nullptr);
    EXPECT_EQ(std::string(message.error_id.get()), "EZ0021");
    ASSERT_EQ(message.args_key.size(), 4U);
    ASSERT_EQ(message.args_value.size(), 4U);
    std::map<std::string, std::string> arguments;
    for (size_t i = 0; i < message.args_key.size(); ++i) {
        ASSERT_NE(message.args_key[i], nullptr);
        ASSERT_NE(message.args_value[i], nullptr);
        EXPECT_TRUE(arguments.emplace(message.args_key[i].get(), message.args_value[i].get()).second);
    }
    const std::map<std::string, std::string> expected = {
        {"param_names", "x, req_scale"},
        {"op_name", tilingContext->GetNodeName()},
        {"incorrect_dtypes", "DT_FLOAT, DT_UINT64"},
        {"reason", "The dtype of x must be INT32 and the dtype of req_scale must be UINT64"},
    };
    EXPECT_EQ(arguments, expected);
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST_F(AscendRequantTilingTest, tiling_shape_mismatch)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape sShape = {{2, 4}, {2, 4}};
    gert::StorageShape yShape = {{2, 3}, {2, 3}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_NE(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_broadcast_amplification_rejected)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{1, 4}, {1, 4}};
    gert::StorageShape sShape = {{3, 1}, {3, 1}};
    gert::StorageShape yShape = {{1, 4}, {1, 4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_NE(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_output_shape_mismatch_rejected)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape sShape = {{2, 3}, {2, 3}};
    gert::StorageShape yShape = {{2, 4}, {2, 4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_NE(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_total_elements_overflow_rejected)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4294967296LL, 4294967296LL, 4294967296LL, 10},
                                 {4294967296LL, 4294967296LL, 4294967296LL, 10}};
    gert::StorageShape sShape = {{1}, {1}};
    gert::StorageShape yShape = {{4294967296LL, 4294967296LL, 4294967296LL, 10},
                                 {4294967296LL, 4294967296LL, 4294967296LL, 10}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_NE(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(AscendRequantTilingTest, tiling_flat_scalar_route_rank8)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 64, 64, 64, 8, 2, 2, 2}, {2, 64, 64, 64, 8, 2, 2, 2}};
    gert::StorageShape sShape = {{1}, {1}};
    gert::StorageShape yShape = {{2, 64, 64, 64, 8, 2, 2, 2}, {2, 64, 64, 64, 8, 2, 2, 2}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    auto* tilingData = FetchTilingData<8>(tilingContext);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->rank, 8);
    EXPECT_EQ(tilingData->flat.route, ASCEND_REQUANT_ROUTE_SCALAR);
    EXPECT_EQ(tilingData->flat.flatElems, 33554432);
    EXPECT_EQ(tilingData->flat.tileElems, 51840);
    EXPECT_EQ(tilingData->flat.totalTiles, 648);
    EXPECT_EQ(tilingData->flat.numCores, 24);
    EXPECT_EQ(tilingData->flat.tilesMain, 27);
    EXPECT_EQ(tilingData->flat.coresTail, 0);
    EXPECT_EQ(tilingData->flat.scalePeriod, 1);
    EXPECT_EQ(tilingData->flat.scaleInitElems, 1);
    EXPECT_EQ(tilingContext->GetBlockDim(), static_cast<uint32_t>(kUtCoreNum));
    CheckFlatTilingInvariants(tilingData->flat);
}

TEST_F(AscendRequantTilingTest, tiling_flat_legacy_route_rank4)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{8, 16, 16, 16}, {8, 16, 16, 16}};
    gert::StorageShape sShape = {{8, 1, 16, 1}, {8, 1, 16, 1}};
    gert::StorageShape yShape = {{8, 16, 16, 16}, {8, 16, 16, 16}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    auto* tilingData = FetchTilingData<4>(tilingContext);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->rank, 4);
    EXPECT_EQ(tilingData->flat.route, ASCEND_REQUANT_ROUTE_LEGACY);
    EXPECT_EQ(tilingData->flat.flatElems, 0);
    EXPECT_EQ(tilingData->flat.tileElems, 0);
    EXPECT_EQ(tilingData->flat.totalTiles, 0);
    EXPECT_EQ(tilingData->flat.numCores, 0);
    EXPECT_EQ(tilingData->flat.tilesMain, 0);
    EXPECT_EQ(tilingData->flat.coresTail, 0);
    EXPECT_EQ(tilingData->flat.scalePeriod, 0);
    EXPECT_EQ(tilingData->flat.scaleInitElems, 0);
    EXPECT_GE(tilingData->multicore.numCores, 1);
    EXPECT_EQ(tilingContext->GetBlockDim(), static_cast<uint32_t>(tilingData->multicore.numCores));
}

TEST_F(AscendRequantTilingTest, tiling_flat_periodic_route_rank3)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{1000000, 2, 4}, {1000000, 2, 4}};
    gert::StorageShape sShape = {{1, 2, 1}, {1, 2, 1}};
    gert::StorageShape yShape = {{1000000, 2, 4}, {1000000, 2, 4}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    auto* tilingData = FetchTilingData<4>(tilingContext);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->rank, 3);
    EXPECT_EQ(tilingData->flat.route, ASCEND_REQUANT_ROUTE_PERIODIC);
    EXPECT_EQ(tilingData->flat.flatElems, 8000000);
    EXPECT_EQ(tilingData->flat.tileElems, 47680);
    EXPECT_EQ(tilingData->flat.totalTiles, 168);
    EXPECT_EQ(tilingData->flat.numCores, 24);
    EXPECT_EQ(tilingData->flat.tilesMain, 7);
    EXPECT_EQ(tilingData->flat.coresTail, 0);
    EXPECT_EQ(tilingData->flat.scalePeriod, 8);
    EXPECT_EQ(tilingData->flat.scaleInitElems, 2);
    EXPECT_EQ(tilingContext->GetBlockDim(), static_cast<uint32_t>(kUtCoreNum));
    CheckFlatTilingInvariants(tilingData->flat);
}

TEST_F(AscendRequantTilingTest, tiling_flat_contig_route_rank2)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    AscendRequantCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{48, 8192}, {48, 8192}};
    gert::StorageShape sShape = {{1, 8192}, {1, 8192}};
    gert::StorageShape yShape = {{48, 8192}, {48, 8192}};

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendRequant")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool reluFlag = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AscendRequant")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &sShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"relu_flag", Ops::NN::AnyValue::CreateFrom<bool>(reluFlag)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    auto* tilingData = FetchTilingData<4>(tilingContext);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->rank, 2);
    EXPECT_EQ(tilingData->flat.route, ASCEND_REQUANT_ROUTE_CONTIG);
    EXPECT_EQ(tilingData->flat.flatElems, 393216);
    EXPECT_EQ(tilingData->flat.tileElems, 8192);
    EXPECT_EQ(tilingData->flat.totalTiles, 48);
    EXPECT_EQ(tilingData->flat.numCores, 24);
    EXPECT_EQ(tilingData->flat.tilesMain, 2);
    EXPECT_EQ(tilingData->flat.coresTail, 0);
    EXPECT_EQ(tilingData->flat.scalePeriod, 8192);
    EXPECT_EQ(tilingData->flat.scaleInitElems, 0);
    EXPECT_EQ(tilingContext->GetBlockDim(), static_cast<uint32_t>(kUtCoreNum));
    CheckFlatTilingInvariants(tilingData->flat);
}
