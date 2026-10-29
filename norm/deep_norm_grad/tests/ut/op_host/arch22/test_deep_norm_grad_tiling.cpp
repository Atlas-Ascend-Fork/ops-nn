/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class DeepNormGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DeepNormGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DeepNormGradTiling TearDown" << std::endl; }
};
struct DeepNormGradCompileInfo {};

// 与 op_host/arch22 生成的 DeepNormGradTilingData 内存布局一致，用于读取 tiling 结果
#pragma pack(1)
struct DeepNormGradTilingDataMirror {
    uint32_t useCoreNum;
    uint32_t nDimNum;
    uint32_t dDimNum;
    uint32_t nDealPerCore;
    uint32_t nDealLastCore;
    uint32_t mergeNCount;
    uint32_t cutDTime;
    uint32_t cutDPerTime;
    uint32_t cutDLastTime;
    uint32_t alpha;
    uint32_t fixedOutputFlag;
};
#pragma pack()

inline void runTest(gert::StorageShape& input_shape, gert::StorageShape& input1_shape, gert::StorageShape& input2_shape,
                    gert::StorageShape& input3_shape, gert::StorageShape& input4_shape,
                    gert::StorageShape& input5_shape, gert::StorageShape& out_shape, gert::StorageShape& out1_shape,
                    gert::StorageShape& out2_shape, gert::StorageShape& out3_shape, ge::DataType data_type,
                    uint32_t tiling_key_expect, ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS,
                    uint32_t cutDTimeExpect = 0, uint32_t cutDPerTimeExpect = 0, uint32_t cutDLastTimeExpect = 0)
{
    // dlog_setlevel(0, 0, 0);

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct DeepNormCompileInfo {};
    DeepNormGradCompileInfo compile_info;

    std::string op_type("DeepNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({6})
                      .InputShapes(
                          {&input_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape, &input5_shape})
                      .OutputShapes({&out_shape, &out1_shape, &out2_shape, &out3_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(0.3)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), expectedStatus);

    // check tiling result only when tiling success
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, tiling_key_expect);
        if (cutDPerTimeExpect != 0) {
            auto* tilingData = reinterpret_cast<DeepNormGradTilingDataMirror*>(
                tiling_context->GetRawTilingData()->GetData());
            EXPECT_EQ(tilingData->cutDTime, cutDTimeExpect);
            EXPECT_EQ(tilingData->cutDPerTime, cutDPerTimeExpect);
            EXPECT_EQ(tilingData->cutDLastTime, cutDLastTimeExpect);
        }
    }
    // dlog_setlevel(0, 3, 0);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_0)
{ // tiling key 0
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_1)
{ // tiling key 1
    gert::StorageShape input_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input2_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 1);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_2)
{ // tiling key 2
    gert::StorageShape input_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape input1_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape input2_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape input3_shape = {{9}, {9}};
    gert::StorageShape input4_shape = {{1, 10, 1}, {1, 10, 1}};
    gert::StorageShape input5_shape = {{1, 10, 1}, {1, 10, 1}};
    gert::StorageShape out_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape out1_shape = {{1, 10, 9}, {1, 10, 9}};
    gert::StorageShape out2_shape = {{9}, {9}};
    gert::StorageShape out3_shape = {{9}, {9}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 2);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_10)
{ // tiling key 10
    gert::StorageShape input_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input1_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input2_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape out1_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT16, 10);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_11)
{ // tiling key 11
    gert::StorageShape input_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input2_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT16, 11);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_12)
{ // tiling key 12
    gert::StorageShape input_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input1_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input2_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input3_shape = {{133}, {133}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape out1_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape out2_shape = {{133}, {133}};
    gert::StorageShape out3_shape = {{133}, {133}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT16, 12);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_20)
{ // tiling key 20
    gert::StorageShape input_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input1_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input2_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape out1_shape = {{40, 1, 1024}, {40, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_BF16, 20);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_21)
{ // tiling key 21
    gert::StorageShape input_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input2_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_BF16, 21);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_22)
{ // tiling key 22
    gert::StorageShape input_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input1_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input2_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input3_shape = {{133}, {133}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape out1_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape out2_shape = {{133}, {133}};
    gert::StorageShape out3_shape = {{133}, {133}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_BF16, 22);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_dy_dimnum_too_large)
{ // dy dim num > 8, invalid
    gert::StorageShape input_shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_dy_dimnum_too_small)
{ // dy dim num < 2, invalid
    gert::StorageShape input_shape = {{1024}, {1024}};
    gert::StorageShape input1_shape = {{1024}, {1024}};
    gert::StorageShape input2_shape = {{1024}, {1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{1}, {1}};
    gert::StorageShape input5_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{1024}, {1024}};
    gert::StorageShape out1_shape = {{1024}, {1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_gamma_dimnum_too_large)
{ // gamma dim num > 7, invalid
    gert::StorageShape input_shape = {{2, 3, 4, 5, 6, 7, 1024}, {2, 3, 4, 5, 6, 7, 1024}};
    gert::StorageShape input1_shape = {{2, 3, 4, 5, 6, 7, 1024}, {2, 3, 4, 5, 6, 7, 1024}};
    gert::StorageShape input2_shape = {{2, 3, 4, 5, 6, 7, 1024}, {2, 3, 4, 5, 6, 7, 1024}};
    gert::StorageShape input3_shape = {{1, 1, 1, 1, 1, 1, 1, 1024}, {1, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape input4_shape = {{2, 3, 4, 5, 6, 7, 1}, {2, 3, 4, 5, 6, 7, 1}};
    gert::StorageShape input5_shape = {{2, 3, 4, 5, 6, 7, 1}, {2, 3, 4, 5, 6, 7, 1}};
    gert::StorageShape out_shape = {{2, 3, 4, 5, 6, 7, 1024}, {2, 3, 4, 5, 6, 7, 1024}};
    gert::StorageShape out1_shape = {{2, 3, 4, 5, 6, 7, 1024}, {2, 3, 4, 5, 6, 7, 1024}};
    gert::StorageShape out2_shape = {{1, 1, 1, 1, 1, 1, 1, 1024}, {1, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape out3_shape = {{1, 1, 1, 1, 1, 1, 1, 1024}, {1, 1, 1, 1, 1, 1, 1, 1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_x_dimnum_not_equal)
{ // x dim num != dy dim num, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1024}, {2, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_mean_dimnum_not_equal)
{ // mean dim num != dy dim num, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1}, {2, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_dgamma_dimnum_not_equal)
{ // dgamma dim num != gamma dim num, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1, 1024}, {1, 1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_dy_dimnum_leq_gamma)
{ // dy dim num <= gamma dim num, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out3_shape = {{2, 1, 1024}, {2, 1, 1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_dy_shape_zero)
{ // dy shape dim value is 0, invalid
    gert::StorageShape input_shape = {{2, 0, 1024}, {2, 0, 1024}};
    gert::StorageShape input1_shape = {{2, 0, 1024}, {2, 0, 1024}};
    gert::StorageShape input2_shape = {{2, 0, 1024}, {2, 0, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 0, 1}, {2, 0, 1}};
    gert::StorageShape input5_shape = {{2, 0, 1}, {2, 0, 1}};
    gert::StorageShape out_shape = {{2, 0, 1024}, {2, 0, 1024}};
    gert::StorageShape out1_shape = {{2, 0, 1024}, {2, 0, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_x_shape_not_equal_dy)
{ // x shape != dy shape, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 2, 1024}, {2, 2, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_dx_shape_not_equal_x)
{ // dx shape != x shape, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 512}, {2, 1, 512}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_mean_shape_not_equal_dy)
{ // mean leading shape != dy leading shape, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{3, 1, 1}, {3, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_gamma_shape_not_equal_dy)
{ // gamma shape != dy last dim, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{512}, {512}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{512}, {512}};
    gert::StorageShape out3_shape = {{512}, {512}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_dgamma_shape_not_equal_gamma)
{ // dgamma shape != gamma shape, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{512}, {512}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_dbeta_shape_not_equal_dy)
{ // dbeta shape != dy last dim, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{512}, {512}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_failed_dtype_not_support)
{ // input dtype not support, invalid
    gert::StorageShape input_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1024}, {2, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_INT32, 0, ge::GRAPH_FAILED);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_d500_boundary)
{ // fp32 D = 500, not less than SMALL_D_STAGE(500), tiling key 0
    gert::StorageShape input_shape = {{2, 1, 500}, {2, 1, 500}};
    gert::StorageShape input1_shape = {{2, 1, 500}, {2, 1, 500}};
    gert::StorageShape input2_shape = {{2, 1, 500}, {2, 1, 500}};
    gert::StorageShape input3_shape = {{500}, {500}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 500}, {2, 1, 500}};
    gert::StorageShape out1_shape = {{2, 1, 500}, {2, 1, 500}};
    gert::StorageShape out2_shape = {{500}, {500}};
    gert::StorageShape out3_shape = {{500}, {500}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_d499_boundary)
{ // fp32 D = 499, less than SMALL_D_STAGE(500), tiling key 2
    gert::StorageShape input_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape input1_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape input2_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape input3_shape = {{499}, {499}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape out1_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape out2_shape = {{499}, {499}};
    gert::StorageShape out3_shape = {{499}, {499}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 2);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_d499_fp16_boundary)
{ // fp16 D = 499, less than SMALL_D_STAGE(500), tiling key 12
    gert::StorageShape input_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape input1_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape input2_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape input3_shape = {{499}, {499}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape out1_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape out2_shape = {{499}, {499}};
    gert::StorageShape out3_shape = {{499}, {499}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT16, 12);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_d499_bf16_boundary)
{ // bf16 D = 499, less than SMALL_D_STAGE(500), tiling key 22
    gert::StorageShape input_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape input1_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape input2_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape input3_shape = {{499}, {499}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape out1_shape = {{2, 1, 499}, {2, 1, 499}};
    gert::StorageShape out2_shape = {{499}, {499}};
    gert::StorageShape out3_shape = {{499}, {499}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_BF16, 22);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_n_over_core_num)
{ // N = 100 > core num 40, nDealPerCore > 1, tiling key 0
    gert::StorageShape input_shape = {{100, 1, 1024}, {100, 1, 1024}};
    gert::StorageShape input1_shape = {{100, 1, 1024}, {100, 1, 1024}};
    gert::StorageShape input2_shape = {{100, 1, 1024}, {100, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{100, 1, 1}, {100, 1, 1}};
    gert::StorageShape input5_shape = {{100, 1, 1}, {100, 1, 1}};
    gert::StorageShape out_shape = {{100, 1, 1024}, {100, 1, 1024}};
    gert::StorageShape out1_shape = {{100, 1, 1024}, {100, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_8d_input)
{ // dy 8 dim (max dim num), gamma 1 dim, tiling key 0
    gert::StorageShape input_shape = {{2, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape input1_shape = {{2, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape input2_shape = {{2, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape input3_shape = {{1024}, {1024}};
    gert::StorageShape input4_shape = {{2, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1, 1, 1, 1, 1, 1}, {2, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape out1_shape = {{2, 1, 1, 1, 1, 1, 1, 1024}, {2, 1, 1, 1, 1, 1, 1, 1024}};
    gert::StorageShape out2_shape = {{1024}, {1024}};
    gert::StorageShape out3_shape = {{1024}, {1024}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0);
}

TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_gamma_2d)
{ // gamma 2 dim, dDimNum = 4 * 512 = 2048, tiling key 0
    gert::StorageShape input_shape = {{2, 4, 512}, {2, 4, 512}};
    gert::StorageShape input1_shape = {{2, 4, 512}, {2, 4, 512}};
    gert::StorageShape input2_shape = {{2, 4, 512}, {2, 4, 512}};
    gert::StorageShape input3_shape = {{4, 512}, {4, 512}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 4, 512}, {2, 4, 512}};
    gert::StorageShape out1_shape = {{2, 4, 512}, {2, 4, 512}};
    gert::StorageShape out2_shape = {{4, 512}, {4, 512}};
    gert::StorageShape out3_shape = {{4, 512}, {4, 512}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 0);
}

// UT 平台 UB=196608 时 fp32 cutD 单次容量为 6112。D=6121 原始尾块为 9，
// 触发均匀重分配：(6112, 9) -> (3064, 3057)，避免小尾块
TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_cut_d_small_tail_fp32)
{
    gert::StorageShape input_shape = {{2, 1, 6121}, {2, 1, 6121}};
    gert::StorageShape input1_shape = {{2, 1, 6121}, {2, 1, 6121}};
    gert::StorageShape input2_shape = {{2, 1, 6121}, {2, 1, 6121}};
    gert::StorageShape input3_shape = {{6121}, {6121}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 6121}, {2, 1, 6121}};
    gert::StorageShape out1_shape = {{2, 1, 6121}, {2, 1, 6121}};
    gert::StorageShape out2_shape = {{6121}, {6121}};
    gert::StorageShape out3_shape = {{6121}, {6121}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 1, ge::GRAPH_SUCCESS, 2, 3064, 3057);
}

// fp32 三分块小尾块：D=12249 原始 (6112, 6112, 25) -> (4088, 4088, 4073)
TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_cut_d_small_tail_fp32_three_chunks)
{
    gert::StorageShape input_shape = {{2, 1, 12249}, {2, 1, 12249}};
    gert::StorageShape input1_shape = {{2, 1, 12249}, {2, 1, 12249}};
    gert::StorageShape input2_shape = {{2, 1, 12249}, {2, 1, 12249}};
    gert::StorageShape input3_shape = {{12249}, {12249}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 12249}, {2, 1, 12249}};
    gert::StorageShape out1_shape = {{2, 1, 12249}, {2, 1, 12249}};
    gert::StorageShape out2_shape = {{12249}, {12249}};
    gert::StorageShape out3_shape = {{12249}, {12249}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 1, ge::GRAPH_SUCCESS, 3, 4088, 4073);
}

// fp32 尾块足够大（D=8192，尾块 2080 >= 2048）时不重分配，保持原切分
TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_cut_d_large_tail_fp32_unchanged)
{
    gert::StorageShape input_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input2_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 1, ge::GRAPH_SUCCESS, 2, 6112, 2080);
}

// UT 平台 UB=196608 时 fp16 cutD 单次容量为 4432。D=4449 原始尾块为 17，
// 触发均匀重分配：(4432, 17) -> (2240, 2209)
TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_cut_d_small_tail_fp16)
{
    gert::StorageShape input_shape = {{40, 1, 4449}, {40, 1, 4449}};
    gert::StorageShape input1_shape = {{40, 1, 4449}, {40, 1, 4449}};
    gert::StorageShape input2_shape = {{40, 1, 4449}, {40, 1, 4449}};
    gert::StorageShape input3_shape = {{4449}, {4449}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 4449}, {40, 1, 4449}};
    gert::StorageShape out1_shape = {{40, 1, 4449}, {40, 1, 4449}};
    gert::StorageShape out2_shape = {{4449}, {4449}};
    gert::StorageShape out3_shape = {{4449}, {4449}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT16, 11, ge::GRAPH_SUCCESS, 2, 2240, 2209);
}

// fp16 尾块足够大（D=8192，尾块 3760 >= 2048）时不重分配，保持原切分
TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_cut_d_large_tail_fp16_unchanged)
{
    gert::StorageShape input_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input2_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input5_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape out_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out1_shape = {{40, 1, 8192}, {40, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT16, 11, ge::GRAPH_SUCCESS, 2, 4432, 3760);
}

// mean 尾部维（对应 gamma 维度）不为 1 时应被拒绝：kernel 按行扁平读取 mean 会取错数据
TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_mean_trailing_dim_not_one_rejected)
{
    gert::StorageShape input_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input2_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{2, 1, 3}, {2, 1, 3}};
    gert::StorageShape input5_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape out_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 1, ge::GRAPH_FAILED);
}

// rstd 尾部维为 0（空维）时应被拒绝：修复前该形状通过校验导致 kernel 越界读
TEST_F(DeepNormGradTiling, deep_norm_grad_tiling_rstd_trailing_dim_zero_rejected)
{
    gert::StorageShape input_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input2_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape input4_shape = {{2, 1, 1}, {2, 1, 1}};
    gert::StorageShape input5_shape = {{2, 1, 0}, {2, 1, 0}};
    gert::StorageShape out_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out1_shape = {{2, 1, 8192}, {2, 1, 8192}};
    gert::StorageShape out2_shape = {{8192}, {8192}};
    gert::StorageShape out3_shape = {{8192}, {8192}};

    runTest(input_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape, out_shape, out1_shape,
            out2_shape, out3_shape, ge::DT_FLOAT, 1, ge::GRAPH_FAILED);
}
