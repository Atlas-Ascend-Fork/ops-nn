/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_reverse_v2_tiling.cpp
 * \brief ReverseV2 arch35 Tiling 错误码负向单元测试
 *
 * 覆盖错误码整改后的外部错误上报路径（扫描表 13419/13424/13428/13429/13436，推荐码 EZ0026/EZ0009/EZ0020）：
 *   - axes 非 const 输入            -> EZ0026 (GetAxisToReverseRunInfo, reverse_v2_tiling.cpp)
 *   - 空 tensor（含0维）           -> EZ0009 (Tiling4ReverseV2)
 *   - TilingParse 平台核数为 0     -> EZ0026 (TilingPrepareReverseV2ForAscendC, regbase 分支)
 *   - TilingParse 平台 UB 为 0     -> EZ0026 (TilingPrepareReverseV2ForAscendC, regbase 分支)
 *   - compile info 缺 dtype_rate   -> EZ0020 (TilingPrepare4ReverseV2, legacy 分支)
 *
 * 断言口径：公开返回值（GRAPH_FAILED）+ ErrorManager 正式首报码（GetRawErrorMessages）
 *          + 消息关键字段（参数名/实际值/reason）+ 同一路径首错唯一（仅 1 条上报，无内部泛化码残留）。
 */

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "log/log.h"
#include "base/err_msg.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "kernel_run_context_facker.h"
#include "tiling_context_faker.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/runtime_tensor.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/reverse_v2_tiling_arch35.h"

using namespace ge;
using namespace optiling;
using namespace std;

// 断言辅助：读取/清空当前线程 ErrorManager 原始上报（读取即清空）
static vector<error_message::ErrorItem> PopRawErrorMessages()
{
    return ErrorManager::GetInstance().GetRawErrorMessages();
}

static void ClearRawErrorMessages() { (void)ErrorManager::GetInstance().GetRawErrorMessages(); }

static void DumpRawErrorMessages(const vector<error_message::ErrorItem>& items)
{
    for (const auto& item : items) {
        cout << "[ut-dump] error_id=" << item.error_id << " message=" << item.error_message << endl;
    }
}

class ReverseV2ErrCodeTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReverseV2ErrCodeTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReverseV2ErrCodeTiling TearDown" << std::endl; }
};

template <typename T>
void SetConstInput(size_t const_index, ge::DataType dtype, T* const_data, int64_t data_size,
                   vector<pair<size_t, unique_ptr<uint8_t[]>>>& const_tensors)
{
    unique_ptr<uint8_t[]> input_tensor_holder = unique_ptr<uint8_t[]>(
        new uint8_t[sizeof(gert::Tensor) + sizeof(T) * data_size]);
    auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
    gert::Tensor tensor({{data_size}, {data_size}},         // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        dtype,                              // dtype
                        nullptr);
    memcpy(input_tensor, &tensor, sizeof(gert::Tensor));
    auto tensor_data = reinterpret_cast<T*>(input_tensor + 1);
    for (int64_t i = 0; i < data_size; i++) {
        tensor_data[i] = const_data[i];
    }
    input_tensor->SetData(gert::TensorData{tensor_data});
    auto pair = std::make_pair(const_index, std::move(input_tensor_holder));
    const_tensors.push_back(std::move(pair));
}

// axes 为非 const tensor（GetAddr 为空且 shape 非空）：推荐码 EZ0026（13419）
TEST_F(ReverseV2ErrCodeTiling, test_tiling_axes_not_const)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48, "version": "Ascend950"}
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    ReverseV2CompileInfo compile_info;

    std::string op_type("ReverseV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{100, 100}, {100, 100}};
    gert::StorageShape axes = {{1}, {1}};
    gert::StorageShape y = {{100, 100}, {100, 100}};
    // 不给 axes 注入 ConstInput 数据 -> GetAddr() 为空且 shape 非空 -> IsConstTensor 为 false
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x, &axes})
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ClearRawErrorMessages();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    auto errs = PopRawErrorMessages();
    DumpRawErrorMessages(errs);
    ASSERT_EQ(errs.size(), 1UL); // 首错唯一，无内部泛化码残留
    EXPECT_EQ(errs[0].error_id, "EZ0026");
    EXPECT_NE(errs[0].error_message.find("axes"), string::npos);
    EXPECT_NE(errs[0].error_message.find("const"), string::npos);
}

// x 含 0 维（空 tensor）：推荐码 EZ0009（13424）
TEST_F(ReverseV2ErrCodeTiling, test_tiling_empty_tensor)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48, "version": "Ascend950"}
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    ReverseV2CompileInfo compile_info;

    std::string op_type("ReverseV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{100, 0}, {100, 0}}; // 含 0 维
    gert::StorageShape axes = {{1}, {1}};
    gert::StorageShape y = {{100, 0}, {100, 0}};
    int32_t axes_value[] = {0};
    vector<pair<size_t, unique_ptr<uint8_t[]>>> const_tensors;
    SetConstInput(1, DT_INT32, axes_value, 1, const_tensors); // axes 必须为 const 才能越过 axes 检查
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x, &axes})
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(const_tensors)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ClearRawErrorMessages();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    auto errs = PopRawErrorMessages();
    DumpRawErrorMessages(errs);
    ASSERT_EQ(errs.size(), 1UL); // 首错唯一
    EXPECT_EQ(errs[0].error_id, "EZ0009");
    EXPECT_NE(errs[0].error_message.find("x"), string::npos);
    EXPECT_NE(errs[0].error_message.find("empty"), string::npos);
}

// regbase 分支：平台核数为 0：推荐码 EZ0026（13428）
TEST_F(ReverseV2ErrCodeTiling, test_tiling_parse_regbase_core_num_zero)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 0, "version": "Ascend950"}
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ReverseV2CompileInfo compile_info;

    std::string op_type("ReverseV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_NE(tiling_parse_func, nullptr);

    char compile_json[] = "{}";
    auto parse_holder = gert::TilingParseContextFaker()
                            .SetOpType(op_type)
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compile_json), reinterpret_cast<void*>(&platform_info)})
                            .Outputs({&compile_info})
                            .Build();
    gert::TilingParseContext* parse_context = parse_holder.GetContext<gert::TilingParseContext>();
    ASSERT_NE(parse_context, nullptr);
    ASSERT_TRUE(parse_context->GetPlatformInfo()->Init());
    parse_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // regbase 分支：version.NpuArch = 3510（DAV_3510 属 regbase 集合）
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    parse_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    ClearRawErrorMessages();
    EXPECT_EQ(tiling_parse_func(parse_context), ge::GRAPH_FAILED);
    auto errs = PopRawErrorMessages();
    DumpRawErrorMessages(errs);
    ASSERT_EQ(errs.size(), 1UL);
    EXPECT_EQ(errs[0].error_id, "EZ0026");
    EXPECT_NE(errs[0].error_message.find("totalCoreNum"), string::npos);
    EXPECT_NE(errs[0].error_message.find("core num"), string::npos);
}

// regbase 分支：平台 UB 为 0：推荐码 EZ0026（13429）
TEST_F(ReverseV2ErrCodeTiling, test_tiling_parse_regbase_ub_size_zero)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 0, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48, "version": "Ascend950"}
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ReverseV2CompileInfo compile_info;

    std::string op_type("ReverseV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_NE(tiling_parse_func, nullptr);

    char compile_json[] = "{}";
    auto parse_holder = gert::TilingParseContextFaker()
                            .SetOpType(op_type)
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compile_json), reinterpret_cast<void*>(&platform_info)})
                            .Outputs({&compile_info})
                            .Build();
    gert::TilingParseContext* parse_context = parse_holder.GetContext<gert::TilingParseContext>();
    ASSERT_NE(parse_context, nullptr);
    ASSERT_TRUE(parse_context->GetPlatformInfo()->Init());
    parse_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    parse_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    ClearRawErrorMessages();
    EXPECT_EQ(tiling_parse_func(parse_context), ge::GRAPH_FAILED);
    auto errs = PopRawErrorMessages();
    DumpRawErrorMessages(errs);
    ASSERT_EQ(errs.size(), 1UL);
    EXPECT_EQ(errs[0].error_id, "EZ0026");
    EXPECT_NE(errs[0].error_message.find("ubSize"), string::npos);
    EXPECT_NE(errs[0].error_message.find("UB size"), string::npos);
}

// legacy 分支：compile info vars 缺 dtype_rate：推荐码 EZ0020（13436）
TEST_F(ReverseV2ErrCodeTiling, test_tiling_parse_dtype_rate_missing)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 32, "version": "Ascend950"}
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ReverseV2CompileInfo compile_info;

    std::string op_type("ReverseV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_NE(tiling_parse_func, nullptr);

    // vars 不含 dtype_rate，但含前置键（max_elements / max_elements_last_large_size），
    // 确保失败点恰好落在 dtype_rate 行（13436）而不是更早的保留行
    char compile_json[] =
        R"({"vars": {"core_num": 32, "max_elements": 2048, "max_elements_last_large_size": 1024, "topk_threshold": 64, "is_vconcat": 0}})";
    auto parse_holder = gert::TilingParseContextFaker()
                            .SetOpType(op_type)
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compile_json), reinterpret_cast<void*>(&platform_info)})
                            .Outputs({&compile_info})
                            .Build();
    gert::TilingParseContext* parse_context = parse_holder.GetContext<gert::TilingParseContext>();
    ASSERT_NE(parse_context, nullptr);
    ASSERT_TRUE(parse_context->GetPlatformInfo()->Init());
    parse_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 不设置 version -> 非 regbase -> 走 legacy 分支

    ClearRawErrorMessages();
    EXPECT_EQ(tiling_parse_func(parse_context), ge::GRAPH_FAILED);
    auto errs = PopRawErrorMessages();
    DumpRawErrorMessages(errs);
    ASSERT_EQ(errs.size(), 1UL);
    EXPECT_EQ(errs[0].error_id, "EZ0020");
    EXPECT_NE(errs[0].error_message.find("x"), string::npos);
    EXPECT_NE(errs[0].error_message.find("dtype_rate"), string::npos);
}
