/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quantized_batch_norm_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "array_ops.h"

using namespace ut_util;
using namespace std;
using namespace ge;

struct QuantizedBatchNormCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class QuantizedBatchNormTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "QuantizedBatchNormTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "QuantizedBatchNormTiling TearDown" << std::endl; }
};

TEST_F(QuantizedBatchNormTiling, quantized_batch_norm_1000)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{1, 256, 13, 982}, {1, 256, 13, 982}};
    gert::StorageShape weight_shape = {{256}, {256}};
    gert::StorageShape bias_shape = {{256}, {256}};
    gert::StorageShape in_scale_shape = {{1}, {1}};
    gert::StorageShape in_zero_point_shape = {{1}, {1}};
    gert::StorageShape out_scale_shape = {{1}, {1}};
    gert::StorageShape out_zero_point_shape = {{1}, {1}};
    gert::StorageShape mean_shape = {{256}, {256}};
    gert::StorageShape variance_shape = {{256}, {256}};
    gert::StorageShape y_shape = {{1, 256, 13, 982}, {1, 256, 13, 982}};
    float eps = 1e-5;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    QuantizedBatchNormCompileInfo compile_info;

    std::string op_type("QuantizedBatchNorm");
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
                      .SetOpType(op_type)
                      .NodeIoNum(9, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &mean_shape, &variance_shape, &in_scale_shape, &in_zero_point_shape,
                                    &out_scale_shape, &out_zero_point_shape, &weight_shape, &bias_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "1 0 12766 0 256 0 6 0 4 0 8 0 6 0 1 0 1 0 4 0 2 0 12144 0 622 0 1 0 1 0 12768 0 1 0 925353388 0 ");
    // dlog_setlevel(0, 3, 0);
}

TEST_F(QuantizedBatchNormTiling, quantized_batch_norm_1001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{1, 256, 1, 9000}, {1, 256, 1, 9000}};
    gert::StorageShape weight_shape = {{256}, {256}};
    gert::StorageShape bias_shape = {{256}, {256}};
    gert::StorageShape in_scale_shape = {{1}, {1}};
    gert::StorageShape in_zero_point_shape = {{1}, {1}};
    gert::StorageShape out_scale_shape = {{1}, {1}};
    gert::StorageShape out_zero_point_shape = {{1}, {1}};
    gert::StorageShape mean_shape = {{256}, {256}};
    gert::StorageShape variance_shape = {{256}, {256}};
    gert::StorageShape y_shape = {{1, 256, 1, 9000}, {1, 256, 1, 9000}};
    float eps = 1e-5;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    QuantizedBatchNormCompileInfo compile_info;

    std::string op_type("QuantizedBatchNorm");
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
                      .SetOpType(op_type)
                      .NodeIoNum(9, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &mean_shape, &variance_shape, &in_scale_shape, &in_zero_point_shape,
                                    &out_scale_shape, &out_zero_point_shape, &weight_shape, &bias_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1001);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "1 0 9000 0 256 0 6 0 4 0 8 0 6 0 1 0 1 0 4 0 1 0 12144 0 9000 0 1 0 1 0 9000 0 1 0 925353388 0 ");
    // dlog_setlevel(0, 3, 0);
}

TEST_F(QuantizedBatchNormTiling, quantized_batch_norm_1002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{256, 256, 14, 14}, {256, 256, 14, 14}};
    gert::StorageShape weight_shape = {{256}, {256}};
    gert::StorageShape bias_shape = {{256}, {256}};
    gert::StorageShape in_scale_shape = {{1}, {1}};
    gert::StorageShape in_zero_point_shape = {{1}, {1}};
    gert::StorageShape out_scale_shape = {{1}, {1}};
    gert::StorageShape out_zero_point_shape = {{1}, {1}};
    gert::StorageShape mean_shape = {{256}, {256}};
    gert::StorageShape variance_shape = {{256}, {256}};
    gert::StorageShape y_shape = {{256, 256, 14, 14}, {256, 256, 14, 14}};
    float eps = 1e-5;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    QuantizedBatchNormCompileInfo compile_info;

    std::string op_type("QuantizedBatchNorm");
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
                      .SetOpType(op_type)
                      .NodeIoNum(9, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &mean_shape, &variance_shape, &in_scale_shape, &in_zero_point_shape,
                                    &out_scale_shape, &out_zero_point_shape, &weight_shape, &bias_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1002);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "256 0 196 0 256 0 6 0 4 0 8 0 6 0 1 0 1 0 4 0 1 0 12144 0 196 0 60 0 5 0 200 0 16 0 925353388 0 ");
    // dlog_setlevel(0, 3, 0);
}

TEST_F(QuantizedBatchNormTiling, quantized_batch_norm_1003)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{2, 256, 13, 15}, {2, 256, 13, 15}};
    gert::StorageShape weight_shape = {{256}, {256}};
    gert::StorageShape bias_shape = {{256}, {256}};
    gert::StorageShape in_scale_shape = {{1}, {1}};
    gert::StorageShape in_zero_point_shape = {{1}, {1}};
    gert::StorageShape out_scale_shape = {{1}, {1}};
    gert::StorageShape out_zero_point_shape = {{1}, {1}};
    gert::StorageShape mean_shape = {{256}, {256}};
    gert::StorageShape variance_shape = {{256}, {256}};
    gert::StorageShape y_shape = {{2, 256, 13, 15}, {2, 256, 13, 15}};
    float eps = 1e-5;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    QuantizedBatchNormCompileInfo compile_info;

    std::string op_type("QuantizedBatchNorm");
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
                      .SetOpType(op_type)
                      .NodeIoNum(9, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &mean_shape, &variance_shape, &in_scale_shape, &in_zero_point_shape,
                                    &out_scale_shape, &out_zero_point_shape, &weight_shape, &bias_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1003);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "2 0 195 0 256 0 6 0 4 0 8 0 6 0 1 0 1 0 4 0 1 0 12144 0 195 0 2 0 1 0 200 0 2 0 925353388 0 ");
    // dlog_setlevel(0, 3, 0);
}

TEST_F(QuantizedBatchNormTiling, quantized_batch_norm_1012)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{256, 256, 14, 16}, {256, 256, 14, 16}};
    gert::StorageShape weight_shape = {{256}, {256}};
    gert::StorageShape bias_shape = {{256}, {256}};
    gert::StorageShape in_scale_shape = {{1}, {1}};
    gert::StorageShape in_zero_point_shape = {{1}, {1}};
    gert::StorageShape out_scale_shape = {{1}, {1}};
    gert::StorageShape out_zero_point_shape = {{1}, {1}};
    gert::StorageShape mean_shape = {{256}, {256}};
    gert::StorageShape variance_shape = {{256}, {256}};
    gert::StorageShape y_shape = {{256, 256, 14, 16}, {256, 256, 14, 16}};
    float eps = 1e-5;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    QuantizedBatchNormCompileInfo compile_info;

    std::string op_type("QuantizedBatchNorm");
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
                      .SetOpType(op_type)
                      .NodeIoNum(9, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &mean_shape, &variance_shape, &in_scale_shape, &in_zero_point_shape,
                                    &out_scale_shape, &out_zero_point_shape, &weight_shape, &bias_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1012);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "256 0 224 0 256 0 6 0 4 0 8 0 6 0 1 0 1 0 4 0 1 0 12144 0 224 0 54 0 5 0 224 0 40 0 925353388 0 ");
    // dlog_setlevel(0, 3, 0);
}

TEST_F(QuantizedBatchNormTiling, quantized_batch_norm_1013)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{2, 256, 15, 16}, {2, 256, 15, 16}};
    gert::StorageShape weight_shape = {{256}, {256}};
    gert::StorageShape bias_shape = {{256}, {256}};
    gert::StorageShape in_scale_shape = {{1}, {1}};
    gert::StorageShape in_zero_point_shape = {{1}, {1}};
    gert::StorageShape out_scale_shape = {{1}, {1}};
    gert::StorageShape out_zero_point_shape = {{1}, {1}};
    gert::StorageShape mean_shape = {{256}, {256}};
    gert::StorageShape variance_shape = {{256}, {256}};
    gert::StorageShape y_shape = {{2, 256, 15, 16}, {2, 256, 15, 16}};
    float eps = 1e-5;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    QuantizedBatchNormCompileInfo compile_info;

    std::string op_type("QuantizedBatchNorm");
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
                      .SetOpType(op_type)
                      .NodeIoNum(9, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &mean_shape, &variance_shape, &in_scale_shape, &in_zero_point_shape,
                                    &out_scale_shape, &out_zero_point_shape, &weight_shape, &bias_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1013);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "2 0 240 0 256 0 6 0 4 0 8 0 6 0 1 0 1 0 4 0 1 0 12144 0 240 0 2 0 1 0 240 0 2 0 925353388 0 ");
    // dlog_setlevel(0, 3, 0);
}

TEST_F(QuantizedBatchNormTiling, quantized_batch_norm_1002_full_reduce_not_support)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{4800, 1344, 1, 1}, {4800, 1344, 1, 1}};
    gert::StorageShape weight_shape = {{1344}, {1344}};
    gert::StorageShape bias_shape = {{1344}, {1344}};
    gert::StorageShape in_scale_shape = {{1}, {1}};
    gert::StorageShape in_zero_point_shape = {{1}, {1}};
    gert::StorageShape out_scale_shape = {{1}, {1}};
    gert::StorageShape out_zero_point_shape = {{1}, {1}};
    gert::StorageShape mean_shape = {{1344}, {1344}};
    gert::StorageShape variance_shape = {{1344}, {1344}};
    gert::StorageShape y_shape = {{4800, 1344, 1, 1}, {4800, 1344, 1, 1}};
    float eps = 1e-5;

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    QuantizedBatchNormCompileInfo compile_info;

    std::string op_type("QuantizedBatchNorm");
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
                      .SetOpType(op_type)
                      .NodeIoNum(9, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x_shape, &mean_shape, &variance_shape, &in_scale_shape, &in_zero_point_shape,
                                    &out_scale_shape, &out_zero_point_shape, &weight_shape, &bias_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1002);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
    // EXPECT_EQ(to_string<int32_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "4800 0 1 0 1344 0 28 0 28 0 32 0 28 0 1 0 1 0 28 0 1 0 12120 0 1 0 1515 0 4 0 8 0 255 0 925353388 0
    //           ");
    // dlog_setlevel(0, 3, 0);
}
// ============================================================================
// Error code verification UT for E*9*** remediation (2026-10-21)
// Captures ErrorManager output via weak-symbol override, asserts formal codes.
// ============================================================================

namespace error_message {
struct CapturedError {
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};
static thread_local CapturedError g_last_error;
static thread_local int g_report_count = 0;
static thread_local std::vector<std::string> g_all_codes; // 记录用例全程上报过的全部错误码

__attribute__((weak)) void ResetCapture()
{
    g_last_error.code.clear();
    g_last_error.keys.clear();
    g_last_error.values.clear();
    g_report_count = 0;
    g_all_codes.clear();
}
__attribute__((weak)) std::string GetLastError() { return g_last_error.code; }
__attribute__((weak)) int GetReportCount() { return g_report_count; }
__attribute__((weak)) bool HasCode(const std::string& c)
{
    for (const auto& code : g_all_codes) {
        if (code == c) {
            return true;
        }
    }
    return false;
}

__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code, const std::vector<const char*>& key,
                                                     const std::vector<const char*>& value)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_last_error.keys.clear();
    g_last_error.values.clear();
    for (const auto& k : key)
        g_last_error.keys.push_back(k ? k : "");
    for (const auto& v : value)
        g_last_error.values.push_back(v ? v : "");
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportInnerErrMsg(const char* file_name, const char* func, uint32_t line,
                                                const char* error_code, const char* format, ...)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_report_count++;
    return 0;
}
} // namespace error_message

// 验证弱符号截获通道在本二进制内生效: 若链接格局变化导致强定义压制弱定义, 该用例显式失败,
// 避免EZ9999守护用例静默失效
TEST_F(QuantizedBatchNormTiling, error_capture_channel_self_check)
{
    error_message::ResetCapture();
    const std::vector<const char*> selfKey = {"self_check"};
    const std::vector<const char*> selfValue = {"capture channel liveness"};
    error_message::ReportPredefinedErrMsg("EZ0000", selfKey, selfValue);
    EXPECT_TRUE(error_message::HasCode("EZ0000")) << "capture channel is dead: weak override suppressed";
    EXPECT_EQ(error_message::GetReportCount(), 1);
    error_message::ResetCapture();
}
#define EXPECT_ERROR_CODE(expected)                                                 \
    EXPECT_EQ(error_message::GetLastError(), expected)                              \
        << "Expected " << expected << " but got [" << error_message::GetLastError() \
        << "] count=" << error_message::GetReportCount()
#define EXPECT_NO_EZ9999()                             \
    EXPECT_NE(error_message::GetLastError(), "EZ9999") \
        << "EZ9999 still generated! Got [" << error_message::GetLastError() << "]"
#define EXPECT_ERROR_REPORTED() EXPECT_GT(error_message::GetReportCount(), 0) << "No error reported"

class QuantizedBatchNormErrorTest : public testing::Test {
protected:
    void SetUp() override { error_message::ResetCapture(); }
};

static ge::graphStatus RunTilingWithErr(ge::DataType xDtype, ge::Format xFormat, gert::StorageShape& xShape,
                                        gert::StorageShape& weightShape, gert::StorageShape& inScaleShape,
                                        ge::DataType weightDtype, ge::DataType biasDtype)
{
    gert::StorageShape bias_shape = {{256}, {256}};
    gert::StorageShape in_zero_point_shape = {{1}, {1}};
    gert::StorageShape out_scale_shape = {{1}, {1}};
    gert::StorageShape out_zero_point_shape = {{1}, {1}};
    gert::StorageShape mean_shape = {{256}, {256}};
    gert::StorageShape variance_shape = {{256}, {256}};
    gert::StorageShape y_shape = xShape;
    QuantizedBatchNormCompileInfo compile_info = {48, 196608};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    std::string op_type("QuantizedBatchNorm");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    if (tiling_func == nullptr)
        return ge::GRAPH_FAILED;
    auto param = gert::TilingData::CreateCap(4096);
    auto ws_size = gert::ContinuousVector::Create<size_t>(4096);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(9, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&xShape, &mean_shape, &variance_shape, &inScaleShape, &in_zero_point_shape,
                                    &out_scale_shape, &out_zero_point_shape, &weightShape, &bias_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, xFormat, xFormat)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, xFormat, xFormat)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)}})
                      .TilingData(param.get())
                      .Workspace(reinterpret_cast<gert::ContinuousVector*>(ws_size.get()))
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    return tiling_func(tiling_context);
}

TEST_F(QuantizedBatchNormErrorTest, err_dtype_x_ez0020)
{
    gert::StorageShape x = {{1, 256, 13, 982}, {1, 256, 13, 982}};
    gert::StorageShape w = {{256}, {256}};
    gert::StorageShape s = {{1}, {1}};
    auto ret = RunTilingWithErr(ge::DT_BOOL, ge::FORMAT_NCHW, x, w, s, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
    EXPECT_ERROR_REPORTED();
    EXPECT_ERROR_CODE("EZ0020");
    EXPECT_NO_EZ9999();
}

TEST_F(QuantizedBatchNormErrorTest, err_format_x_ez0017)
{
    gert::StorageShape x = {{1, 256, 13, 982}, {1, 256, 13, 982}};
    gert::StorageShape w = {{256}, {256}};
    gert::StorageShape s = {{1}, {1}};
    auto ret = RunTilingWithErr(ge::DT_INT32, ge::FORMAT_NC1HWC0, x, w, s, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
    EXPECT_ERROR_REPORTED();
    EXPECT_ERROR_CODE("EZ0017");
    EXPECT_NO_EZ9999();
}

TEST_F(QuantizedBatchNormErrorTest, err_rank_x_ez0011)
{
    gert::StorageShape x = {{1, 256, 982}, {1, 256, 982}};
    gert::StorageShape w = {{256}, {256}};
    gert::StorageShape s = {{1}, {1}};
    auto ret = RunTilingWithErr(ge::DT_INT32, ge::FORMAT_NCHW, x, w, s, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
    EXPECT_ERROR_REPORTED();
    EXPECT_ERROR_CODE("EZ0011");
    EXPECT_NO_EZ9999();
}

TEST_F(QuantizedBatchNormErrorTest, err_weight_shape_ez0010)
{
    gert::StorageShape x = {{1, 256, 13, 982}, {1, 256, 13, 982}};
    gert::StorageShape w = {{128}, {128}};
    gert::StorageShape s = {{1}, {1}};
    auto ret = RunTilingWithErr(ge::DT_INT32, ge::FORMAT_NCHW, x, w, s, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
    EXPECT_ERROR_REPORTED();
    EXPECT_ERROR_CODE("EZ0010");
    EXPECT_NO_EZ9999();
}

TEST_F(QuantizedBatchNormErrorTest, err_scale_size_ez0014)
{
    gert::StorageShape x = {{1, 256, 13, 982}, {1, 256, 13, 982}};
    gert::StorageShape w = {{256}, {256}};
    gert::StorageShape s = {{2}, {2}};
    auto ret = RunTilingWithErr(ge::DT_INT32, ge::FORMAT_NCHW, x, w, s, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
    EXPECT_ERROR_REPORTED();
    EXPECT_ERROR_CODE("EZ0014");
    EXPECT_NO_EZ9999();
}

TEST_F(QuantizedBatchNormErrorTest, err_dim_zero_ez0009)
{
    gert::StorageShape x = {{1, 256, 0, 982}, {1, 256, 0, 982}};
    gert::StorageShape w = {{256}, {256}};
    gert::StorageShape s = {{1}, {1}};
    auto ret = RunTilingWithErr(ge::DT_INT32, ge::FORMAT_NCHW, x, w, s, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
    EXPECT_ERROR_REPORTED();
    EXPECT_ERROR_CODE("EZ0009");
    EXPECT_NO_EZ9999();
}

TEST_F(QuantizedBatchNormErrorTest, err_bias_dtype_ez0021)
{
    gert::StorageShape x = {{1, 256, 13, 982}, {1, 256, 13, 982}};
    gert::StorageShape w = {{256}, {256}};
    gert::StorageShape s = {{1}, {1}};
    auto ret = RunTilingWithErr(ge::DT_INT32, ge::FORMAT_NCHW, x, w, s, ge::DT_FLOAT, ge::DT_FLOAT16);
    EXPECT_NE(ret, ge::GRAPH_SUCCESS);
    EXPECT_ERROR_REPORTED();
    EXPECT_ERROR_CODE("EZ0021");
    EXPECT_NO_EZ9999();
}
