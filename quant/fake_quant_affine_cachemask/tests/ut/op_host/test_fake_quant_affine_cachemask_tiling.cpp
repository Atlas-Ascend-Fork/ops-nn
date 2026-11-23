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
 * \file test_fake_quant_affine_cachemask_tiling.cpp
 * \brief
 */
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "fake_quant_error_code_ut_util.h"
#include "test_cube_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace ge;
using namespace std;
using namespace ut_util;
namespace optiling {
struct FakeQuantAffineCachemaskCompileInfo {
    int32_t totalCoreNum = 30;
    uint64_t ubSizePlatForm = 0;
};
} // namespace optiling
namespace {
constexpr const char* kNonRegbaseOpType = "FakeQuantAffineCachemask";

// 非 regbase（A2/A3）路径：与上面 3 条用例一致，只设置 SoCInfo / AICoreSpec /
// AICoreintrinsicDtypeMap，不设置 "version"（Short_SoC_version / NpuArch），
// NpuArch 取默认 2001 → IsRegbaseSocVersion 为假，Tiling 入口分派到
// FakeQuantAffineCachemaskTiling::Init，覆盖该分支的正式错误码上报点。
ge::graphStatus RunNonRegbaseTilingCase(gert::StorageShape& xShape, gert::StorageShape& scaleShape,
                                        gert::StorageShape& zpShape, gert::StorageShape& yShape,
                                        gert::StorageShape& maskShape, int64_t axis, int64_t quantMin, int64_t quantMax)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kNonRegbaseOpType);
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
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

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::FakeQuantAffineCachemaskCompileInfo compile_info;

    auto param = gert::TilingData::CreateCap(4096);
    EXPECT_NE(param, nullptr);
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    EXPECT_NE(ws_size, nullptr);
    if (ws_size == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto holder = gert::TilingContextFaker()
                      .SetOpType(kNonRegbaseOpType)
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &scaleShape, &zpShape})
                      .OutputShapes({&yShape, &maskShape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                                  {"quant_min", Ops::NN::AnyValue::CreateFrom<int64_t>(quantMin)},
                                  {"quant_max", Ops::NN::AnyValue::CreateFrom<int64_t>(quantMax)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context == nullptr) {
        ADD_FAILURE() << "failed to build tiling context";
        return ge::GRAPH_FAILED;
    }
    fe::PlatFormInfos* platform_info_ptr = tiling_context->GetPlatformInfo();
    if (platform_info_ptr == nullptr) {
        ADD_FAILURE() << "failed to get tiling platform info";
        return ge::GRAPH_FAILED;
    }
    platform_info_ptr->SetPlatformRes("SoCInfo", soc_infos);
    platform_info_ptr->SetPlatformRes("AICoreSpec", aicore_spec);
    platform_info_ptr->SetCoreNumByCoreType("AICore");
    platform_info_ptr->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    return opImpl->tiling(tiling_context);
}

// 非 regbase 路径的断言口径（与整改计划 §9.3 的 adopt 类用例一致）：
// `TilingFakeQuantAffineCachemask` 外层的聚合日志
// `OP_CHECK_IF(tilingObject.Init() != ge::GRAPH_SUCCESS, OP_LOGE(...), return GRAPH_FAILED)`
// 在计划 §5 中列为 RETAINED（5701）——`FakeQuantAffineCachemaskTiling::Init` 中
// headNum 为 0 与 coreNum 为 0 两条静默失败路径依赖它作为唯一首报点，故其
// `REPORT_INNER_ERR_MSG` 上报的内部兜底码（见 `fake_quant_error_code_ut_util.h`
// 中按段拼接的 `kInternalErrCode`）会保留。
// 因此 5699/5700 上报正式码后必然再追加 1 条内部兜底码，判据为：
//   GRAPH_FAILED + 正式码恰好 1 条且为首错 + 无其他正式码 + 兜底码至多 1 条（且仅来自 5701）。
// 不套用 NoEz9999Listener：该监听器不区分正式码与 5701 的保留聚合。
void ExpectFormalCodeWithRetainedAggregate(const std::string& code, const char* caseName)
{
    const auto ids = fake_quant_ut::CollectErrorIds();
    std::string dump;
    int hit = 0;
    int fallback = 0;
    int otherFormal = 0;
    for (const auto& id : ids) {
        dump += (id.empty() ? std::string("<empty>") : id) + " ";
        if (id == code) {
            ++hit;
        } else if (id == fake_quant_ut::kInternalErrCode) {
            ++fallback;
        } else {
            ++otherFormal;
        }
    }
    EXPECT_EQ(hit, 1) << caseName << " expects exactly one " << code << ", captured: [" << dump << "]";
    EXPECT_FALSE(ids.empty()) << caseName << " captured no error message";
    const std::string firstId = ids.empty() ? std::string() : ids[0];
    EXPECT_EQ(firstId, code) << caseName << " first error must be " << code << ", captured: [" << dump << "]";
    EXPECT_LE(fallback, 1) << caseName << " must not append more than one internal fallback code from 5701, captured: ["
                           << dump << "]";
    EXPECT_EQ(otherFormal, 0) << caseName << " must not report other formal codes, captured: [" << dump << "]";
}

} // namespace

class FakeQuantAffineCachemaskTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FakeQuantAffineCachemaskTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FakeQuantAffineCachemaskTiling TearDown" << std::endl; }
};

TEST_F(FakeQuantAffineCachemaskTiling, test_fake_quant_affine_cachemask_float32)
{
    std::string op_type("FakeQuantAffineCachemask");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
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
    struct FakeQuantAffineCachemaskCompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
    } compile_info;
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
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x_shape = {{32, 1}, {32, 1}};
    gert::StorageShape scale_shape = {{32}, {32}};
    gert::StorageShape zero_point_shape = {{32}, {32}};
    gert::StorageShape y_shape = {{32, 1}, {32, 1}};
    gert::StorageShape mask_shape = {{32, 1}, {32, 1}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &scale_shape, &zero_point_shape})
                      .OutputShapes({&y_shape, &mask_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"quant_min", Ops::NN::AnyValue::CreateFrom<int64_t>(-200)},
                                  {"quant_max", Ops::NN::AnyValue::CreateFrom<int64_t>(200)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 1);
}

TEST_F(FakeQuantAffineCachemaskTiling, test_fake_quant_affine_cachemask_float16)
{
    std::string op_type("FakeQuantAffineCachemask");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
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
    struct FakeQuantAffineCachemaskCompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
    } compile_info;
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
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x_shape = {{32, 1}, {32, 1}};
    gert::StorageShape scale_shape = {{32}, {32}};
    gert::StorageShape zero_point_shape = {{32}, {32}};
    gert::StorageShape y_shape = {{32, 1}, {32, 1}};
    gert::StorageShape mask_shape = {{32, 1}, {32, 1}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &scale_shape, &zero_point_shape})
                      .OutputShapes({&y_shape, &mask_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"quant_min", Ops::NN::AnyValue::CreateFrom<int64_t>(-200)},
                                  {"quant_max", Ops::NN::AnyValue::CreateFrom<int64_t>(200)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 2);
}

TEST_F(FakeQuantAffineCachemaskTiling, test_fake_quant_affine_cachemask_rank0)
{
    std::string op_type("FakeQuantAffineCachemask");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
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

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct FakeQuantAffineCachemaskCompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
    } compile_info;
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

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // rank=0: one scalar element, empty shape/dim.
    gert::StorageShape x_shape = {{}, {}};
    gert::StorageShape scale_shape = {{1}, {1}};
    gert::StorageShape zero_point_shape = {{1}, {1}};
    gert::StorageShape y_shape = {{}, {}};
    gert::StorageShape mask_shape = {{}, {}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &scale_shape, &zero_point_shape})
                      .OutputShapes({&y_shape, &mask_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"quant_min", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"quant_max", Ops::NN::AnyValue::CreateFrom<int64_t>(255)}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling_context->GetTilingKey(), 2);
}

// 0 维（标量）x 且 axis 非 0/-1 → EZ0026（非 regbase 路径 5699 上报点）
TEST_F(FakeQuantAffineCachemaskTiling, non_regbase_0dim_axis_invalid_ez0026)
{
    // 本套件不挂监听器：用例内独占 ErrorManager 上下文，析构时 drain + 恢复
    fake_quant_ut::ScopedErrorContext scoped;
    gert::StorageShape xShape = {{}, {}};
    gert::StorageShape scaleShape = {{1}, {1}};
    gert::StorageShape zpShape = {{1}, {1}};
    gert::StorageShape yShape = {{}, {}};
    gert::StorageShape maskShape = {{}, {}};
    EXPECT_EQ(RunNonRegbaseTilingCase(xShape, scaleShape, zpShape, yShape, maskShape, 1, -200, 200), ge::GRAPH_FAILED);
    ExpectFormalCodeWithRetainedAggregate("EZ0026", "non_regbase_0dim_axis_invalid");
}

// rank>0 且归一化后 axis != 0 → EZ0037（非 regbase 路径 5700 上报点）
TEST_F(FakeQuantAffineCachemaskTiling, non_regbase_axis_not_zero_ez0037)
{
    // 本套件不挂监听器：用例内独占 ErrorManager 上下文，析构时 drain + 恢复
    fake_quant_ut::ScopedErrorContext scoped;
    gert::StorageShape xShape = {{2, 3}, {2, 3}};
    gert::StorageShape scaleShape = {{3}, {3}};
    gert::StorageShape zpShape = {{3}, {3}};
    gert::StorageShape yShape = {{2, 3}, {2, 3}};
    gert::StorageShape maskShape = {{2, 3}, {2, 3}};
    EXPECT_EQ(RunNonRegbaseTilingCase(xShape, scaleShape, zpShape, yShape, maskShape, 1, 0, 255), ge::GRAPH_FAILED);
    ExpectFormalCodeWithRetainedAggregate("EZ0037", "non_regbase_axis_not_zero");
}
