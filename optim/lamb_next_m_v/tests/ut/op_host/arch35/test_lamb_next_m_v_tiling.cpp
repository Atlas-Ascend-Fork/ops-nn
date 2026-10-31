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
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "optim/lamb_next_m_v/op_host/arch35/lamb_next_m_v_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class LambNextMVTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

// Drive CheckLambNextMVBrcBounds through the real tiling entry. Keeping the context
// construction here avoids duplicating the 13-input/4-output boilerplate per case.
static void RunLambNextMVBrcTilingCase(const std::vector<std::vector<int64_t>>& inputDims,
                                       const std::vector<int64_t>& outputDims, ge::DataType dtype,
                                       ge::graphStatus expectedStatus, uint64_t* totalNumOut = nullptr,
                                       uint32_t* collapsedRankOut = nullptr)
{
    constexpr size_t inputNum = 13;
    constexpr size_t outputNum = 4;
    ASSERT_EQ(inputDims.size(), inputNum);

    std::vector<gert::StorageShape> inputShapes;
    inputShapes.reserve(inputNum);
    for (const auto& dims : inputDims) {
        inputShapes.push_back(MakeStorageShape(dims));
    }
    std::vector<gert::StorageShape> outputShapes;
    outputShapes.reserve(outputNum);
    for (size_t i = 0; i < outputNum; ++i) {
        outputShapes.push_back(MakeStorageShape(outputDims));
    }
    std::vector<gert::StorageShape*> inputPtrs;
    inputPtrs.reserve(inputNum);
    for (auto& shape : inputShapes) {
        inputPtrs.push_back(&shape);
    }
    std::vector<gert::StorageShape*> outputPtrs;
    outputPtrs.reserve(outputNum);
    for (auto& shape : outputShapes) {
        outputPtrs.push_back(&shape);
    }

    string compile_info_string =
        R"({"hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::LambNextMVCompileInfo compile_info;
    string op_type("LambNextMV");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(opImpl, nullptr);
    auto tiling_func = opImpl->tiling;
    auto tiling_parse_func = opImpl->tiling_parse;
    ASSERT_NE(tiling_func, nullptr);
    ASSERT_NE(tiling_parse_func, nullptr);

    string parse_str = R"({})";
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(parse_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_NE(kernel_holder.GetContext<gert::TilingParseContext>(), nullptr);
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    ASSERT_NE(ws_size, nullptr);

    std::vector<uint32_t> instanceNum(inputNum, 1);
    gert::TilingContextFaker faker;
    faker.SetOpType(op_type)
        .NodeIoNum(inputNum, outputNum)
        .IrInstanceNum(instanceNum)
        .InputShapes(inputPtrs)
        .OutputShapes(outputPtrs)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info));
    for (size_t i = 0; i < inputNum; ++i) {
        faker.NodeInputTd(static_cast<int32_t>(i), dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < outputNum; ++i) {
        faker.NodeOutputTd(static_cast<int32_t>(i), dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.TilingData(param.get()).Workspace(ws_size).Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ge::graphStatus status = tiling_func(tiling_context);
    EXPECT_EQ(status, expectedStatus);
    if (status == ge::GRAPH_SUCCESS) {
        auto* rawTilingData = reinterpret_cast<gert::TilingData*>(param.get());
        const auto* plan = reinterpret_cast<const ::LambBrcTilingData<13, 4>*>(rawTilingData->GetData());
        if (totalNumOut != nullptr) {
            *totalNumOut = plan->totalNum;
        }
        if (collapsedRankOut != nullptr) {
            *collapsedRankOut = plan->collapsedRank;
        }
    }
}

TEST_F(LambNextMVTilingTest, lamb_next_mv_tiling_test_fp32)
{
    gert::StorageShape in0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in2_shape = {{1}, {1}};
    gert::StorageShape in3_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in4_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in5_shape = {{1}, {1}};
    gert::StorageShape in6_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in7_shape = {{1}, {1}};
    gert::StorageShape in8_shape = {{1}, {1}};
    gert::StorageShape in9_shape = {{1}, {1}};
    gert::StorageShape in10_shape = {{1}, {1}};
    gert::StorageShape in11_shape = {{1}, {1}};
    gert::StorageShape in12_shape = {{1}, {1}};
    gert::StorageShape out0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out2_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out3_shape = {{16, 8, 375}, {16, 8, 375}};
    string compile_info_string =
        R"({"hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::LambNextMVCompileInfo compile_info;
    string op_type("LambNextMV");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string parse_str = R"({})";
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(parse_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&in0_shape, &in1_shape, &in2_shape, &in3_shape, &in4_shape, &in5_shape, &in6_shape,
                                    &in7_shape, &in8_shape, &in9_shape, &in10_shape, &in11_shape, &in12_shape})
                      .OutputShapes({&out0_shape, &out1_shape, &out2_shape, &out3_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
TEST_F(LambNextMVTilingTest, lamb_next_mv_tiling_test_fp16)
{
    gert::StorageShape in0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in2_shape = {{1}, {1}};
    gert::StorageShape in3_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in4_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in5_shape = {{1}, {1}};
    gert::StorageShape in6_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in7_shape = {{1}, {1}};
    gert::StorageShape in8_shape = {{1}, {1}};
    gert::StorageShape in9_shape = {{1}, {1}};
    gert::StorageShape in10_shape = {{1}, {1}};
    gert::StorageShape in11_shape = {{1}, {1}};
    gert::StorageShape in12_shape = {{1}, {1}};
    gert::StorageShape out0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out2_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out3_shape = {{16, 8, 375}, {16, 8, 375}};
    string compile_info_string =
        R"({"hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::LambNextMVCompileInfo compile_info;
    string op_type("LambNextMV");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string parse_str = R"({})";
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(parse_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&in0_shape, &in1_shape, &in2_shape, &in3_shape, &in4_shape, &in5_shape, &in6_shape,
                                    &in7_shape, &in8_shape, &in9_shape, &in10_shape, &in11_shape, &in12_shape})
                      .OutputShapes({&out0_shape, &out1_shape, &out2_shape, &out3_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
TEST_F(LambNextMVTilingTest, lamb_next_mv_tiling_test_dtype_mismatch)
{
    gert::StorageShape in0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in2_shape = {{1}, {1}};
    gert::StorageShape in3_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in4_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in5_shape = {{1}, {1}};
    gert::StorageShape in6_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in7_shape = {{1}, {1}};
    gert::StorageShape in8_shape = {{1}, {1}};
    gert::StorageShape in9_shape = {{1}, {1}};
    gert::StorageShape in10_shape = {{1}, {1}};
    gert::StorageShape in11_shape = {{1}, {1}};
    gert::StorageShape in12_shape = {{1}, {1}};
    gert::StorageShape out0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out2_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out3_shape = {{16, 8, 375}, {16, 8, 375}};
    string compile_info_string =
        R"({"hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::LambNextMVCompileInfo compile_info;
    string op_type("LambNextMV");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string parse_str = R"({})";
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(parse_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&in0_shape, &in1_shape, &in2_shape, &in3_shape, &in4_shape, &in5_shape, &in6_shape,
                                    &in7_shape, &in8_shape, &in9_shape, &in10_shape, &in11_shape, &in12_shape})
                      .OutputShapes({&out0_shape, &out1_shape, &out2_shape, &out3_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(LambNextMVTilingTest, lamb_next_mv_tiling_test_invalid_dtype)
{
    gert::StorageShape in0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in2_shape = {{1}, {1}};
    gert::StorageShape in3_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in4_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in5_shape = {{1}, {1}};
    gert::StorageShape in6_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape in7_shape = {{1}, {1}};
    gert::StorageShape in8_shape = {{1}, {1}};
    gert::StorageShape in9_shape = {{1}, {1}};
    gert::StorageShape in10_shape = {{1}, {1}};
    gert::StorageShape in11_shape = {{1}, {1}};
    gert::StorageShape in12_shape = {{1}, {1}};
    gert::StorageShape out0_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out1_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out2_shape = {{16, 8, 375}, {16, 8, 375}};
    gert::StorageShape out3_shape = {{16, 8, 375}, {16, 8, 375}};
    string compile_info_string =
        R"({"hardware_info": {"UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::LambNextMVCompileInfo compile_info;
    string op_type("LambNextMV");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    string parse_str = R"({})";
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(parse_str.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&in0_shape, &in1_shape, &in2_shape, &in3_shape, &in4_shape, &in5_shape, &in6_shape,
                                    &in7_shape, &in8_shape, &in9_shape, &in10_shape, &in11_shape, &in12_shape})
                      .OutputShapes({&out0_shape, &out1_shape, &out2_shape, &out3_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

// CheckLambNextMVBrcBounds coverage: normal collapse path must produce the expected plan.
TEST_F(LambNextMVTilingTest, lamb_next_mv_brc_bounds_success_collapsed_shape)
{
    std::vector<std::vector<int64_t>> inputDims(13, std::vector<int64_t>{1});
    for (size_t idx : {0, 1, 3, 4, 6}) {
        inputDims[idx] = {16, 8, 375};
    }
    uint64_t totalNum = 0;
    uint32_t collapsedRank = 0;
    RunLambNextMVBrcTilingCase(inputDims, {16, 8, 375}, ge::DT_FLOAT, ge::GRAPH_SUCCESS, &totalNum, &collapsedRank);
    EXPECT_EQ(totalNum, 16ULL * 8ULL * 375ULL);
    EXPECT_EQ(collapsedRank, 1U);
}

// CheckLambNextMVBrcBounds coverage: output rank above LAMB_BRC_MAX_DIM must be rejected.
TEST_F(LambNextMVTilingTest, lamb_next_mv_brc_bounds_reject_rank_over_limit)
{
    std::vector<std::vector<int64_t>> inputDims(13, std::vector<int64_t>{1});
    RunLambNextMVBrcTilingCase(inputDims, std::vector<int64_t>(9, 2), ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// CheckLambNextMVBrcBounds coverage: a single dim above UINT32_MAX must be rejected.
TEST_F(LambNextMVTilingTest, lamb_next_mv_brc_bounds_reject_dim_over_uint32)
{
    std::vector<std::vector<int64_t>> inputDims(13, std::vector<int64_t>{1});
    RunLambNextMVBrcTilingCase(inputDims, {4294967296LL}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// CheckLambNextMVBrcBounds coverage: collapse multiplication above INT64_MAX must be rejected.
TEST_F(LambNextMVTilingTest, lamb_next_mv_brc_bounds_reject_collapse_overflow)
{
    std::vector<std::vector<int64_t>> inputDims(13, std::vector<int64_t>{1});
    RunLambNextMVBrcTilingCase(inputDims, {4294967295LL, 4294967295LL}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

// CheckLambNextMVBrcBounds coverage: total element count above UINT64_MAX must be rejected
// without collapsing the axes (input 0 broadcasts on the middle axis to keep them separate).
TEST_F(LambNextMVTilingTest, lamb_next_mv_brc_bounds_reject_total_overflow)
{
    std::vector<std::vector<int64_t>> inputDims(13, std::vector<int64_t>{1});
    inputDims[0] = {4294967295LL, 1, 2};
    RunLambNextMVBrcTilingCase(inputDims, {4294967295LL, 4294967295LL, 2}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}
