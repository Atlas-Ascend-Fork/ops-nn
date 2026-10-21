/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "quant/anti_hi_f4_quant/op_host/arch35/anti_hi_f4_quant_tiling_arch35.h"
#include "any_value.h"

using namespace std;

class AntiHiF4QuantTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AntiHiF4QuantTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AntiHiF4QuantTiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tilingData)
{
    auto data = tilingData->GetData();
    string result;
    for (size_t i = 0; i < tilingData->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(data)[i / sizeof(int64_t)]));
        result += " ";
    }
    return result;
}

static void ExecuteTestCase(ge::DataType inXDtype, ge::DataType inScaleDtype, ge::DataType outYDtype,
                            int64_t dstTypeAttr, gert::StorageShape xShape, gert::StorageShape scaleShape,
                            gert::StorageShape yShape, string expectTilingData,
                            ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend960DT"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::AntiHiF4QuantCompileInfo compile_info;

    std::string op_type("AntiHiF4Quant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 2)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &scaleShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inXDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outYDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom((int64_t)outYDtype)}})
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

// PER_HEAD mode (quantMode=1): scale[-2] == x[-1], scale[-3] == x[-2]/64
// x: [64, 128], scale: [1, 128, 1] → scale[-2]=128==x[-1]=128, scale[-3]=1==x[-2]/64=1
TEST_F(AntiHiF4QuantTiling, AntiHiF4Quant_tiling_per_head_bf16)
{
    gert::StorageShape xShape = {{64, 128}, {64, 128}};
    gert::StorageShape scaleShape = {{1, 128, 1}, {1, 128, 1}};
    gert::StorageShape yShape = {{64, 128}, {64, 128}};
    string expectTilingData = "1 64 1 27 128 64 1 64 128 128 64 1 1 1 1 0 1 1 1 ";

    ExecuteTestCase(ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, ge::DT_BF16, 27, xShape, scaleShape, yShape,
                    expectTilingData);
}

// PER_CHANNEL mode (quantMode=0): scale[-2] == x[-1]/64, scale[-3] == x[-2]
// x: [64, 128], scale: [64, 2, 1] → scale[-2]=2==x[-1]/64=2, scale[-3]=64==x[-2]=64
TEST_F(AntiHiF4QuantTiling, AntiHiF4Quant_tiling_per_channel_bf16)
{
    gert::StorageShape xShape = {{64, 128}, {64, 128}};
    gert::StorageShape scaleShape = {{64, 2, 1}, {64, 2, 1}};
    gert::StorageShape yShape = {{64, 128}, {64, 128}};
    string expectTilingData = "64 4 0 32 520 1 1 32 32 65 4 4 260 16 16 27 ";

    ExecuteTestCase(ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, ge::DT_BF16, 27, xShape, scaleShape, yShape,
                    expectTilingData);
}

// PER_HEAD mode with FP16 output
TEST_F(AntiHiF4QuantTiling, AntiHiF4Quant_tiling_per_head_fp16)
{
    gert::StorageShape xShape = {{64, 128}, {64, 128}};
    gert::StorageShape scaleShape = {{1, 128, 1}, {1, 128, 1}};
    gert::StorageShape yShape = {{64, 128}, {64, 128}};
    string expectTilingData = "1 64 1 1 128 64 1 64 128 128 64 1 1 1 1 0 1 1 1 ";

    ExecuteTestCase(ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, ge::DT_FLOAT16, 1, xShape, scaleShape, yShape,
                    expectTilingData);
}

// PER_CHANNEL mode with larger shape
TEST_F(AntiHiF4QuantTiling, AntiHiF4Quant_tiling_per_channel_large)
{
    gert::StorageShape xShape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{128, 4, 1}, {128, 4, 1}};
    gert::StorageShape yShape = {{128, 256}, {128, 256}};
    string expectTilingData = "64 16 0 32 520 1 1 32 32 65 4 4 260 16 16 27 ";

    ExecuteTestCase(ge::DT_HIFLOAT4, ge::DT_HIFLOAT4_SCALE, ge::DT_BF16, 27, xShape, scaleShape, yShape,
                    expectTilingData);
}
