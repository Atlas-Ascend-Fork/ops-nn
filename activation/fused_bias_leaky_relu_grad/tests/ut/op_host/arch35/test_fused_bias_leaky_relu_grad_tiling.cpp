/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "tests/ut/common/ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
using namespace ut_util;

namespace optiling {
struct FusedBiasLeakyReluGradCompileInfo {};
} // namespace optiling

class FusedBiasLeakyReluGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FusedBiasLeakyReluGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FusedBiasLeakyReluGradTilingTest TearDown" << std::endl; }
};

// 通用 tiling 用例：y_grad/features 两输入 + x_grad 一输出
// expectedStatus: 期望的 tiling 返回值
static void FusedBiasLeakyReluGradTilingTestCase(std::initializer_list<int64_t> yGradShape,
                                                 std::initializer_list<int64_t> featuresShape,
                                                 std::initializer_list<int64_t> xGradShape, ge::DataType yGradDtype,
                                                 ge::DataType featuresDtype, float negativeSlope, float scale,
                                                 ge::graphStatus expectedStatus)
{
    gert::StorageShape yGradShapeTensor = {yGradShape, yGradShape};
    gert::StorageShape featuresShapeTensor = {featuresShape, featuresShape};
    gert::StorageShape xGradShapeTensor = {xGradShape, xGradShape};

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version_infos);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::FusedBiasLeakyReluGradCompileInfo compile_info;

    std::string op_type("FusedBiasLeakyReluGrad");
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    auto param_holder = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param_holder, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&yGradShapeTensor, &featuresShapeTensor})
                      .OutputShapes({&xGradShapeTensor})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, yGradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, featuresDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yGradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"negative_slope", Ops::NN::AnyValue::CreateFrom<float>(negativeSlope)},
                                  {"scale", Ops::NN::AnyValue::CreateFrom<float>(scale)}})
                      .TilingData(param_holder.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    EXPECT_EQ(tiling_func(tiling_context), expectedStatus);
}

// 常规 1D float32，rank<=4 走 RANK_4 分支
TEST_F(FusedBiasLeakyReluGradTilingTest, float32_1d)
{
    FusedBiasLeakyReluGradTilingTestCase({1024}, {1024}, {1024}, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f, 1.414213562373f,
                                         ge::GRAPH_SUCCESS);
}

// 2D float32
TEST_F(FusedBiasLeakyReluGradTilingTest, float32_2d)
{
    FusedBiasLeakyReluGradTilingTestCase({8, 1024}, {8, 1024}, {8, 1024}, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f,
                                         1.414213562373f, ge::GRAPH_SUCCESS);
}

// 4D float32：RANK_4 分支上边界
TEST_F(FusedBiasLeakyReluGradTilingTest, float32_4d)
{
    FusedBiasLeakyReluGradTilingTestCase({2, 4, 8, 16}, {2, 4, 8, 16}, {2, 4, 8, 16}, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f,
                                         1.414213562373f, ge::GRAPH_SUCCESS);
}

// 5D float32：rank>4 走 RANK_8 分支
TEST_F(FusedBiasLeakyReluGradTilingTest, float32_5d)
{
    FusedBiasLeakyReluGradTilingTestCase({2, 2, 4, 8, 16}, {2, 2, 4, 8, 16}, {2, 2, 4, 8, 16}, ge::DT_FLOAT,
                                         ge::DT_FLOAT, 0.2f, 1.414213562373f, ge::GRAPH_SUCCESS);
}

// 8D float32：RANK_8 分支上边界
TEST_F(FusedBiasLeakyReluGradTilingTest, float32_8d)
{
    FusedBiasLeakyReluGradTilingTestCase({2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2},
                                         ge::DT_FLOAT, ge::DT_FLOAT, 0.2f, 1.414213562373f, ge::GRAPH_SUCCESS);
}

// float16
TEST_F(FusedBiasLeakyReluGradTilingTest, float16_1d)
{
    FusedBiasLeakyReluGradTilingTestCase({8192}, {8192}, {8192}, ge::DT_FLOAT16, ge::DT_FLOAT16, 0.2f, 1.414213562373f,
                                         ge::GRAPH_SUCCESS);
}

// 广播：features 比 y_grad 少的维度按 1 广播
TEST_F(FusedBiasLeakyReluGradTilingTest, broadcast_features_scalar_per_row)
{
    FusedBiasLeakyReluGradTilingTestCase({8, 1024}, {8, 1}, {8, 1024}, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f,
                                         1.414213562373f, ge::GRAPH_SUCCESS);
}

// 广播：y_grad 低 rank，features 高 rank
TEST_F(FusedBiasLeakyReluGradTilingTest, broadcast_ygrad_low_rank)
{
    FusedBiasLeakyReluGradTilingTestCase({1024}, {8, 1024}, {8, 1024}, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f,
                                         1.414213562373f, ge::GRAPH_SUCCESS);
}

// 大 shape：触发 UB 切分 + 多核切分
TEST_F(FusedBiasLeakyReluGradTilingTest, float32_large)
{
    FusedBiasLeakyReluGradTilingTestCase({8192, 1024}, {8192, 1024}, {8192, 1024}, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f,
                                         1.414213562373f, ge::GRAPH_SUCCESS);
}

// 空张量：走 HandleEmptyTensor
TEST_F(FusedBiasLeakyReluGradTilingTest, empty_tensor)
{
    FusedBiasLeakyReluGradTilingTestCase({0}, {0}, {0}, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f, 1.414213562373f,
                                         ge::GRAPH_SUCCESS);
}

// negative_slope=0 / scale=1 边界属性
TEST_F(FusedBiasLeakyReluGradTilingTest, negative_slope_zero_scale_one)
{
    FusedBiasLeakyReluGradTilingTestCase({1024}, {1024}, {1024}, ge::DT_FLOAT, ge::DT_FLOAT, 0.0f, 1.0f,
                                         ge::GRAPH_SUCCESS);
}

// 输入 dtype 不一致：应失败
TEST_F(FusedBiasLeakyReluGradTilingTest, dtype_mismatch_rejected)
{
    FusedBiasLeakyReluGradTilingTestCase({1024}, {1024}, {1024}, ge::DT_FLOAT, ge::DT_FLOAT16, 0.2f, 1.414213562373f,
                                         ge::GRAPH_FAILED);
}

// 不支持的 dtype（int32）：应失败
TEST_F(FusedBiasLeakyReluGradTilingTest, dtype_unsupported_rejected)
{
    FusedBiasLeakyReluGradTilingTestCase({1024}, {1024}, {1024}, ge::DT_INT32, ge::DT_INT32, 0.2f, 1.414213562373f,
                                         ge::GRAPH_FAILED);
}

// 广播不兼容：同轴且都不是 1：应失败
TEST_F(FusedBiasLeakyReluGradTilingTest, broadcast_incompatible_rejected)
{
    FusedBiasLeakyReluGradTilingTestCase({2, 6}, {2, 8}, {2, 6}, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f, 1.414213562373f,
                                         ge::GRAPH_FAILED);
}

// rank 超过 8：应失败
TEST_F(FusedBiasLeakyReluGradTilingTest, rank_exceed_max_rejected)
{
    FusedBiasLeakyReluGradTilingTestCase({2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2},
                                         {2, 2, 2, 2, 2, 2, 2, 2, 2}, ge::DT_FLOAT, ge::DT_FLOAT, 0.2f, 1.414213562373f,
                                         ge::GRAPH_FAILED);
}
