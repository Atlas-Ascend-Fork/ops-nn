/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../../../../../common/inc/op_host/op_const_def_nn.h"
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_common.h"
#include "../../../../op_host/arch35/rotate_quant_tiling.h"
#include "../../../../../../common/inc/op_host/op_const_def_nn.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"

using namespace std;

static bool IsDav3510CompileInfo(const string& compileInfo)
{
    return compileInfo.find(R"("npuArch": "dav3510")") != string::npos ||
           compileInfo.find(R"("npuArch":"dav3510")") != string::npos;
}

static void GetPlatFormInfosDavAware(const string& compileInfo, map<string, string>& socInfos,
                                     map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                                     map<string, string>& socVersion)
{
    GetPlatFormInfos(compileInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);
    if (IsDav3510CompileInfo(compileInfo)) {
        socVersion.clear();
        socVersion["NpuArch"] = "3510";
    } else if (compileInfo.find(R"("npuArch": "dav9201")") != string::npos ||
               compileInfo.find(R"("npuArch":"dav9201")") != string::npos) {
        socVersion["NpuArch"] = "9201";
    } else if (compileInfo.find(R"("npuArch": "dav9202")") != string::npos ||
               compileInfo.find(R"("npuArch":"dav9202")") != string::npos) {
        socVersion["NpuArch"] = "9202";
    }
}
constexpr size_t ROTATE_QUANT_SYS_WORKSPACE_BYTES = 16UL * 1024UL * 1024UL;

struct RotateQuantArch35Data {
    string case_name;
    string compile_info;

    gert::StorageShape x_shape;
    gert::StorageShape rot_shape;
    gert::StorageShape y_shape;
    gert::StorageShape scale_shape;

    ge::DataType xDataType{ge::DT_BF16};
    ge::DataType yDataType{ge::DT_FLOAT8_E4M3FN};
    ge::DataType scaleDataType{ge::DT_FLOAT8_E8M0};

    int64_t axis{-1};
    string roundMode{"rint"};
    int64_t scaleAlg{0};
    float dstTypeMax{0.0f};
    bool trans{false};

    ge::graphStatus expect_status{ge::GRAPH_SUCCESS};
    uint32_t expectedAivRatio{2};
    uint32_t expectedML1{0};
    uint32_t expectedTailML1{0};
    uint32_t expectedUbDB{0};
    uint32_t expectedBlockDim{32};
    uint32_t expectedBaseM{0};
};

class TilingRotateQuantArch35 : public ::testing::TestWithParam<RotateQuantArch35Data> {
protected:
    void SetUp() override { std::cout << "TilingRotateQuantArch35 SetUp" << std::endl; }
    void TearDown() override { std::cout << "TilingRotateQuantArch35 TearDown" << std::endl; }
};

TEST_P(TilingRotateQuantArch35, rotate_quant_arch35_tiling)
{
    auto test_params = GetParam();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfosDavAware(test_params.compile_info, soc_infos, aicore_spec, intrinsics, soc_version);
    if (test_params.compile_info.find("\"socVersion\": \"Ascend350\"") != string::npos) {
        soc_version["NpuArch"] = "3510";
    }

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    Ops::NN::RotateQuant::RotateQuantAptCompileInfo compile_info;

    string op_type("RotateQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(test_params.compile_info.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("RotateQuant")
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&test_params.x_shape, &test_params.rot_shape})
                      .OutputShapes({&test_params.y_shape, &test_params.scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, test_params.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, test_params.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, test_params.yDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, test_params.scaleDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.yDataType)},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.axis)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(test_params.roundMode)},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.scaleAlg)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(test_params.dstTypeMax)},
                                  {"trans", Ops::NN::AnyValue::CreateFrom<bool>(test_params.trans)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ge::graphStatus actual_status = tiling_func(tiling_context);
    EXPECT_EQ(actual_status, test_params.expect_status) << test_params.case_name;
    if (actual_status == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(tiling_context->GetTilingKey(), 0UL);
        auto data = reinterpret_cast<const RotateQuantAptOpt::RotateQuantAptTilingData*>(
            tiling_context->GetRawTilingData()->GetData());
        EXPECT_EQ(data->aivNumRatio, test_params.expectedAivRatio);
        if (test_params.expectedML1 != 0) {
            EXPECT_EQ(data->mL1, test_params.expectedML1);
            EXPECT_EQ(data->tailML1, test_params.expectedTailML1);
            EXPECT_EQ(tiling_context->GetBlockDim(), test_params.expectedBlockDim);
        }
        if (test_params.expectedBaseM != 0) {
            EXPECT_EQ(data->baseM, test_params.expectedBaseM);
        }
        if (test_params.expectedUbDB != 0) {
            EXPECT_EQ(data->ubDB, test_params.expectedUbDB) << test_params.case_name;
        }
        const bool hif4 = test_params.yDataType == Ops::NN::DT_HIFLOAT4;
        auto workspaceSizes = tiling_context->GetWorkspaceSizes(1);
        ASSERT_NE(workspaceSizes, nullptr);
        const size_t expectedWorkspace = ROTATE_QUANT_SYS_WORKSPACE_BYTES +
                                         (hif4 ? static_cast<size_t>(data->M) * data->N * sizeof(float) : 0);
        EXPECT_EQ(workspaceSizes[0], expectedWorkspace);
        const uint64_t alignedM = (data->mL1 + 15UL) / 16 * 16;
        const uint64_t rotationBytes = static_cast<uint64_t>(data->K) * data->K * 2;
        if (hif4) {
            EXPECT_EQ(data->ubDB, 1);
            EXPECT_EQ(data->baseK, data->K);
            EXPECT_LE(alignedM * data->K * 2 + rotationBytes, stoull(aicore_spec.at("l1_size")));
            EXPECT_LE(alignedM * data->baseK * 2, stoull(aicore_spec.at("l0_a_size")));
            EXPECT_LE(alignedM * data->K * 4, stoull(aicore_spec.at("l0_c_size")));
        } else {
            EXPECT_LE(data->mL1 * data->K, 16384);
            EXPECT_LE(2 * (alignedM * data->K * 2 + rotationBytes), stoull(aicore_spec.at("l1_size")));
            EXPECT_LE(2 * alignedM * data->baseK * 2, stoull(aicore_spec.at("l0_a_size")));
            EXPECT_LE(2 * alignedM * data->K * 4, stoull(aicore_spec.at("l0_c_size")));
            EXPECT_GE(data->ubDB, 1);
            EXPECT_LE(data->ubDB, 2);
            const uint64_t ubWorkspace = 52736UL + static_cast<uint64_t>(data->mL1) * 32;
            EXPECT_LE(data->ubDB * ubWorkspace, stoull(aicore_spec.at("ub_size")));
            EXPECT_LE(static_cast<uint64_t>(data->baseK) * data->baseN * 2, stoull(aicore_spec.at("l0_b_size")) / 2);
        }
        if (data->K == 256) {
            EXPECT_EQ(data->baseK, 64);
            EXPECT_LE(data->mL1, 64);
        }
        if (data->M >= 32769) {
            const uint64_t effectiveM = static_cast<uint64_t>(data->M) * (data->B == 1 ? data->nKRatio : 1);
            const uint64_t tileNum = (effectiveM + data->mL1 - 1) / data->mL1 * data->B;
            EXPECT_GT(tileNum, tiling_context->GetBlockDim());
        }
    }
}

static const string COMPILE_INFO_DAV_3510 = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "npuArch": "dav3510",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 64}
})";

// DAV_3510 with non-1:2 aic:aiv ratio (cube=32, vector=48)
static const string COMPILE_INFO_DAV_3510_BAD_RATIO = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "npuArch": "dav3510",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 48}
})";

static string TestNpuArch(const string& soc)
{
    if (soc == "Ascend960DT" || soc == "dav9201") {
        return "dav9201";
    }
    if (soc == "Ascend960PR" || soc == "dav9202") {
        return "dav9202";
    }
    return "dav3510";
}

static string MakeCompileInfoForTest(const string& arch, uint32_t vectorCores)
{
    auto info = COMPILE_INFO_DAV_3510;
    const string oldArch = "dav3510";
    info.replace(info.find(oldArch), oldArch.size(), arch);
    const string oldCores = "\"vector_core_cnt\": 64";
    info.replace(info.find(oldCores), oldCores.size(), "\"vector_core_cnt\": " + to_string(vectorCores));
    return info;
}

// Exercise A6's 1:1 MX Tensor API and HiFloat4 Basic API paths.
static RotateQuantArch35Data MakeDav920SeriesCase(const string& soc, bool hif4, bool multiRot, uint32_t vectorCores = 0)
{
    RotateQuantArch35Data data;
    data.case_name = soc + (hif4 ? "_hif4" : multiRot ? "_mx_multi_rot" : "_mx_shared");
    const auto arch = TestNpuArch(soc);
    data.expectedAivRatio = arch == "dav3510" ? 2 : 1;
    const uint32_t expectedVectorCores = 32 * data.expectedAivRatio;
    data.compile_info = MakeCompileInfoForTest(arch, vectorCores == 0 ? expectedVectorCores : vectorCores);
    data.x_shape = {{1025, 384}, {1025, 384}};
    data.rot_shape = multiRot ? gert::StorageShape{{3, 128, 128}, {3, 128, 128}} :
                                gert::StorageShape{{128, 128}, {128, 128}};
    data.y_shape = data.x_shape;
    data.scale_shape = hif4 ? gert::StorageShape{{1025, 6, 1}, {1025, 6, 1}} :
                              gert::StorageShape{{1025, 6, 2}, {1025, 6, 2}};
    if (hif4) {
        data.xDataType = ge::DT_FLOAT16;
        data.yDataType = Ops::NN::DT_HIFLOAT4;
        data.scaleDataType = Ops::NN::DT_HIFLOAT4_SCALE;
    }
    if (vectorCores != 0 && vectorCores != expectedVectorCores) {
        data.case_name += "_bad_ratio";
        data.expect_status = ge::GRAPH_FAILED;
    }
    return data;
}

// v121 dav9201 platform: 32 AIC + 32 AIV (aicNum:aivNum = 1:1)
static const string COMPILE_INFO_DAV9201 = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "npuArch": "dav9201",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 32}
})";

static RotateQuantArch35Data rotate_quant_arch35_cases[] = {
    MakeDav920SeriesCase("dav9201", false, false),
    MakeDav920SeriesCase("dav9201", false, true),
    MakeDav920SeriesCase("dav9201", true, false),
    MakeDav920SeriesCase("dav9202", false, false),
    MakeDav920SeriesCase("dav9202", false, true),
    MakeDav920SeriesCase("dav9202", true, false),
    MakeDav920SeriesCase("dav9201", false, false, 64),
    MakeDav920SeriesCase("dav9202", false, false, 64),
    {"bf16_fp8_k256_shared",
     COMPILE_INFO_DAV_3510,
     {{1025, 768}, {1025, 768}},
     {{256, 256}, {256, 256}},
     {{1025, 768}, {1025, 768}},
     {{1025, 12, 2}, {1025, 12, 2}}},
    {"fp16_fp4_k256_multi_rot",
     COMPILE_INFO_DAV_3510,
     {{1025, 768}, {1025, 768}},
     {{3, 256, 256}, {3, 256, 256}},
     {{1025, 768}, {1025, 768}},
     {{1025, 12, 2}, {1025, 12, 2}},
     ge::DT_FLOAT16,
     ge::DT_FLOAT4_E2M1},
    {"bf16_fp8_k256_single_row",
     COMPILE_INFO_DAV_3510,
     {{1, 256}, {1, 256}},
     {{256, 256}, {256, 256}},
     {{1, 256}, {1, 256}},
     {{1, 4, 2}, {1, 4, 2}}},
    // 0: BF16 + FLOAT8_E4M3FN, dav_3510 with 1:2 core ratio -> 控核通过, tiling成功
    // scale shape按契约推导: x=[128,128] -> [128, CeilDiv(128,64)=2, 2]
    {"bf16_fp8e4m3_dav3510_ratio_1to2",
     COMPILE_INFO_DAV_3510,
     {{128, 128}, {128, 128}},
     {{64, 64}, {64, 64}},
     {{128, 128}, {128, 128}},
     {{128, 2, 2}, {128, 2, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT8_E4M3FN,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_SUCCESS},
    // 1: same shape/dtype but aic:aiv not 1:2 (cube=32, vector=48) -> 控核校验失败 (IsCapable false,
    // arch注册表最终返回GRAPH_FAILED)
    {"bf16_fp8e4m3_dav3510_ratio_not_1to2_fail",
     COMPILE_INFO_DAV_3510_BAD_RATIO,
     {{128, 128}, {128, 128}},
     {{64, 64}, {64, 64}},
     {{128, 128}, {128, 128}},
     {{128, 2, 2}, {128, 2, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT8_E4M3FN,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
    // 2: 扁平一维scale（总元素数128*2*2=512与推导一致）-> 放行, tiling成功
    {"bf16_fp8e4m3_dav3510_flat_scale_ok",
     COMPILE_INFO_DAV_3510,
     {{128, 128}, {128, 128}},
     {{64, 64}, {64, 64}},
     {{128, 128}, {128, 128}},
     {{512}, {512}},
     ge::DT_BF16,
     ge::DT_FLOAT8_E4M3FN,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_SUCCESS},
    // 3: 扁平一维scale但总元素数(500)与推导(512)不符 -> 校验失败
    {"bf16_fp8e4m3_dav3510_flat_scale_wrong_num_fail",
     COMPILE_INFO_DAV_3510,
     {{128, 128}, {128, 128}},
     {{64, 64}, {64, 64}},
     {{128, 128}, {128, 128}},
     {{500}, {500}},
     ge::DT_BF16,
     ge::DT_FLOAT8_E4M3FN,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},

    // ==================== dav9201 (v121) hif4 success cases ====================
    // 4: hif4 bf16 x, K=64, scale shape (M, CeilDiv(N,64), 1)
    {"v121_hif4_bf16_k64",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 1}, {4, 1, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_SUCCESS,
     1}, // DAV_920系列 AIC:AIV=1:1 -> aivNumRatio=1
    // 5: hif4 fp16 x, K=128, scale shape (4, 2, 1)
    {"v121_hif4_fp16_k128",
     COMPILE_INFO_DAV9201,
     {{4, 128}, {4, 128}},
     {{128, 128}, {128, 128}},
     {{4, 128}, {4, 128}},
     {{4, 2, 1}, {4, 2, 1}},
     ge::DT_FLOAT16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_SUCCESS,
     1}, // DAV_920系列 AIC:AIV=1:1 -> aivNumRatio=1
    // 6: hif4 bf16 x, K=64, N=256 (blocks4), scale shape (8, 4, 1)
    {"v121_hif4_bf16_k64_n256_blocks4",
     COMPILE_INFO_DAV9201,
     {{8, 256}, {8, 256}},
     {{64, 64}, {64, 64}},
     {{8, 256}, {8, 256}},
     {{8, 4, 1}, {8, 4, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_SUCCESS,
     1}, // DAV_920系列 AIC:AIV=1:1 -> aivNumRatio=1
    // 7: hif4 3D rot (nKRatio>1) -> tiling校验失败 (hif4 only supports 2D rotation)
    {"v121_hif4_bf16_k64_3d_rot_fail",
     COMPILE_INFO_DAV9201,
     {{4, 128}, {4, 128}},
     {{2, 64, 64}, {2, 64, 64}},
     {{4, 128}, {4, 128}},
     {{4, 2, 1}, {4, 2, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},

    // ==================== dav9201 (v121) mxfp success cases ====================
    // 8: mxfp4 bf16 x, K=64, scaleAlg=0, scale shape (4, 1, 2)
    {"v121_mxfp4_bf16_k64_alg0",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 2}, {4, 1, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT4_E2M1,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_SUCCESS,
     1}, // DAV_920系列 AIC:AIV=1:1 -> aivNumRatio=1
    // 9: mxfp4 fp16 x, K=128, scaleAlg=2, dstTypeMax=8.0, scale shape (4, 2, 2)
    {"v121_mxfp4_fp16_k128_alg2",
     COMPILE_INFO_DAV9201,
     {{4, 128}, {4, 128}},
     {{128, 128}, {128, 128}},
     {{4, 128}, {4, 128}},
     {{4, 2, 2}, {4, 2, 2}},
     ge::DT_FLOAT16,
     ge::DT_FLOAT4_E2M1,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     2,
     8.0f,
     false,
     ge::GRAPH_SUCCESS,
     1}, // DAV_920系列 AIC:AIV=1:1 -> aivNumRatio=1
    // 10: mxfp4 bf16 x, K=32, N=96 (blocks3), scale shape (8, 2, 2)
    {"v121_mxfp4_bf16_k32_n96_blocks3",
     COMPILE_INFO_DAV9201,
     {{8, 96}, {8, 96}},
     {{32, 32}, {32, 32}},
     {{8, 96}, {8, 96}},
     {{8, 2, 2}, {8, 2, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT4_E2M1,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_SUCCESS,
     1}, // DAV_920系列 AIC:AIV=1:1 -> aivNumRatio=1
    // 11: mxfp8 e4m3 bf16 x, K=64, scaleAlg=0
    {"v121_mxfp8_e4m3_bf16_k64_alg0",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 2}, {4, 1, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT8_E4M3FN,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_SUCCESS,
     1}, // DAV_920系列 AIC:AIV=1:1 -> aivNumRatio=1
    // 12: mxfp8 e5m2 fp16 x, K=128, scaleAlg=1
    {"v121_mxfp8_e5m2_fp16_k128_alg1",
     COMPILE_INFO_DAV9201,
     {{4, 128}, {4, 128}},
     {{128, 128}, {128, 128}},
     {{4, 128}, {4, 128}},
     {{4, 2, 2}, {4, 2, 2}},
     ge::DT_FLOAT16,
     ge::DT_FLOAT8_E5M2,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     1,
     0.0f,
     false,
     ge::GRAPH_SUCCESS,
     1}, // DAV_920系列 AIC:AIV=1:1 -> aivNumRatio=1

    // ==================== dav9201 (v121) hif4 fail cases ====================
    // 13: hif4 K=32 not allowed (must be 64 or 128)
    {"v121_hif4_k32_fail",
     COMPILE_INFO_DAV9201,
     {{4, 32}, {4, 32}},
     {{32, 32}, {32, 32}},
     {{4, 32}, {4, 32}},
     {{4, 1, 1}, {4, 1, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
    // 14: hif4 scaleAlg=2 not allowed (must be 0)
    {"v121_hif4_scaleAlg2_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 1}, {4, 1, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     2,
     0.0f,
     false,
     ge::GRAPH_FAILED},
    // 15: hif4 roundMode="floor" not allowed (must be "rint")
    {"v121_hif4_roundMode_floor_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 1}, {4, 1, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "floor",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
    // 16: hif4 scale dtype mismatch (FLOAT8_E8M0 instead of HIFLOAT4_SCALE)
    {"v121_hif4_scale_dtype_e8m0_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 1}, {4, 1, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
    // 17: hif4 dstTypeMax != 0.0
    {"v121_hif4_dstTypeMax_nonzero_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 1}, {4, 1, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     8.0f,
     false,
     ge::GRAPH_FAILED},

    // ==================== dav9201 (v121) mxfp fail cases ====================
    // 18: mxfp4 K=48 not in {32, 64, 128}
    {"v121_mxfp4_k48_fail",
     COMPILE_INFO_DAV9201,
     {{4, 48}, {4, 48}},
     {{48, 48}, {48, 48}},
     {{4, 48}, {4, 48}},
     {{4, 1, 2}, {4, 1, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT4_E2M1,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
    // 19: mxfp4 scaleAlg=2 but dstTypeMax=3.0 out of [6.0, 12.0]
    {"v121_mxfp4_alg2_dstTypeMax_3_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 2}, {4, 1, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT4_E2M1,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     2,
     3.0f,
     false,
     ge::GRAPH_FAILED},
    // 20: mxfp4 scale dtype mismatch (HIFLOAT4_SCALE instead of FLOAT8_E8M0)
    {"v121_mxfp4_scale_hif4scale_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 2}, {4, 1, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT4_E2M1,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
    // 21: mxfp8 roundMode="round" not allowed (must be "rint")
    {"v121_mxfp8_roundMode_round_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 2}, {4, 1, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT8_E4M3FN,
     ge::DT_FLOAT8_E8M0,
     -1,
     "round",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
    // 22: mxfp8 scaleAlg=2 not allowed for fp8 (must be 0 or 1)
    {"v121_mxfp8_scaleAlg2_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 2}, {4, 1, 2}},
     ge::DT_BF16,
     ge::DT_FLOAT8_E4M3FN,
     ge::DT_FLOAT8_E8M0,
     -1,
     "rint",
     2,
     0.0f,
     false,
     ge::GRAPH_FAILED},

    // ==================== dav9201 (v121) common fail cases ====================
    // 23: x dtype unsupported (FP32)
    {"v121_x_fp32_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 1}, {4, 1, 1}},
     ge::DT_FLOAT,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
    // 24: trans=true not supported
    {"v121_trans_true_fail",
     COMPILE_INFO_DAV9201,
     {{4, 64}, {4, 64}},
     {{64, 64}, {64, 64}},
     {{4, 64}, {4, 64}},
     {{4, 1, 1}, {4, 1, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     0.0f,
     true,
     ge::GRAPH_FAILED},
    // 25: N not divisible by K
    {"v121_hif4_n100_k64_not_divisible_fail",
     COMPILE_INFO_DAV9201,
     {{4, 100}, {4, 100}},
     {{64, 64}, {64, 64}},
     {{4, 100}, {4, 100}},
     {{4, 2, 1}, {4, 2, 1}},
     ge::DT_BF16,
     Ops::NN::DT_HIFLOAT4,
     Ops::NN::DT_HIFLOAT4_SCALE,
     -1,
     "rint",
     0,
     0.0f,
     false,
     ge::GRAPH_FAILED},
};

INSTANTIATE_TEST_SUITE_P(RotateQuantArch35TilingCases, TilingRotateQuantArch35,
                         testing::ValuesIn(rotate_quant_arch35_cases));

static vector<RotateQuantArch35Data> MakeEffectiveMOverflowCases()
{
    struct BoundaryCase {
        const char* name;
        int64_t m;
        int64_t n;
        ge::graphStatus status;
        uint32_t tailOffset;
    };
    const BoundaryCase boundaries[] = {
        {"below_uint32_limit", 1073741823, 128, ge::GRAPH_SUCCESS, 4},
        {"at_uint32_limit", 1431655765, 96, ge::GRAPH_SUCCESS, 1},
        {"at_two_to_32", 1073741824, 128, ge::GRAPH_FAILED, 0},
        {"above_uint32_limit", 2000000000, 96, ge::GRAPH_FAILED, 0},
    };
    vector<RotateQuantArch35Data> cases;
    for (const char* soc : {"Ascend950", "Ascend350", "dav9201", "dav9202"}) {
        for (const auto& boundary : boundaries) {
            RotateQuantArch35Data data;
            data.case_name = string(soc) + "_shared_effective_m_" + boundary.name;
            const auto arch = TestNpuArch(soc);
            data.expectedAivRatio = arch == "dav3510" ? 2 : 1;
            data.compile_info = MakeCompileInfoForTest(arch, 32 * data.expectedAivRatio);
            data.x_shape = {{boundary.m, boundary.n}, {boundary.m, boundary.n}};
            data.rot_shape = {{32, 32}, {32, 32}};
            data.y_shape = data.x_shape;
            const int64_t scalePairs = (boundary.n + 63) / 64;
            data.scale_shape = {{boundary.m, scalePairs, 2}, {boundary.m, scalePairs, 2}};
            data.expect_status = boundary.status;
            if (boundary.status == ge::GRAPH_SUCCESS) {
                data.expectedML1 = 512;
                data.expectedTailML1 = data.expectedML1 - boundary.tailOffset;
            }
            cases.push_back(data);
        }
    }
    return cases;
}

static const auto effectiveMOverflowCases = MakeEffectiveMOverflowCases();
INSTANTIATE_TEST_SUITE_P(RotateQuantEffectiveMOverflowCases, TilingRotateQuantArch35,
                         testing::ValuesIn(effectiveMOverflowCases));

static vector<RotateQuantArch35Data> MakeDav920SeriesRegressionCases()
{
    vector<RotateQuantArch35Data> cases;
    for (const char* soc : {"dav9201", "dav9202"}) {
        for (ge::DataType inputType : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (ge::DataType outputType :
                 {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, Ops::NN::DT_HIFLOAT4}) {
                const bool hif4 = outputType == Ops::NN::DT_HIFLOAT4;
                for (int64_t k : {32, 64, 128}) {
                    if (hif4 && k == 32) {
                        continue;
                    }
                    for (bool multiRot : {false, true}) {
                        if (hif4 && multiRot) {
                            continue;
                        }
                        for (int64_t m : {1, 17, 32769}) {
                            auto data = MakeDav920SeriesCase(soc, hif4, multiRot);
                            const int64_t n = 3 * k;
                            data.case_name += "_x" + to_string(static_cast<int>(inputType)) + "_y" +
                                              to_string(static_cast<int>(outputType)) + "_k" + to_string(k) + "_m" +
                                              to_string(m);
                            data.xDataType = inputType;
                            data.yDataType = outputType;
                            data.x_shape = {{m, n}, {m, n}};
                            data.y_shape = data.x_shape;
                            data.rot_shape = multiRot ? gert::StorageShape{{3, k, k}, {3, k, k}} :
                                                        gert::StorageShape{{k, k}, {k, k}};
                            const int64_t scaleLastDim = hif4 ? 1 : 2;
                            data.scale_shape = {{m, (n + 63) / 64, scaleLastDim}, {m, (n + 63) / 64, scaleLastDim}};
                            if (hif4 && m == 32769) {
                                data.expectedML1 = k == 64 ? 512 : 256;
                                data.expectedTailML1 = 3;
                                data.expectedUbDB = 1;
                            }
                            cases.push_back(data);
                        }
                    }
                }
            }
        }
    }
    for (const char* soc : {"Ascend950", "Ascend350"}) {
        auto data = MakeDav920SeriesCase(soc, true, false);
        data.case_name = string(soc) + "_reject_hif4";
        data.expect_status = ge::GRAPH_FAILED;
        cases.push_back(data);
    }
    for (const char* soc : {"dav9201", "dav9202"}) {
        auto data = MakeDav920SeriesCase(soc, false, false);
        data.case_name += "_k256_supported";
        data.x_shape = {{17, 768}, {17, 768}};
        data.y_shape = data.x_shape;
        data.rot_shape = {{256, 256}, {256, 256}};
        data.scale_shape = {{17, 12, 2}, {17, 12, 2}};
        data.expect_status = ge::GRAPH_SUCCESS;
        cases.push_back(data);

        data = MakeDav920SeriesCase(soc, true, true);
        data.case_name += "_reject_multi_rot";
        data.expect_status = ge::GRAPH_FAILED;
        cases.push_back(data);

        data = MakeDav920SeriesCase(soc, true, false);
        data.case_name += "_reject_k32";
        data.rot_shape = {{32, 32}, {32, 32}};
        data.expect_status = ge::GRAPH_FAILED;
        cases.push_back(data);
    }
    return cases;
}

static const auto dav920SeriesRegressionCases = MakeDav920SeriesRegressionCases();
INSTANTIATE_TEST_SUITE_P(RotateQuantDav920SeriesRegressionCases, TilingRotateQuantArch35,
                         testing::ValuesIn(dav920SeriesRegressionCases));

static vector<RotateQuantArch35Data> MakeMxResourceBoundaryCases()
{
    struct ResourceBoundary {
        const char* resource;
        uint64_t bytes;
        bool succeeds;
        bool minimumRows;
    };
    const ResourceBoundary boundaries[] = {
        {"UB_SIZE", 53248, true, true},    {"UB_SIZE", 53247, false, false}, {"L1_SIZE", 73728, true, true},
        {"L1_SIZE", 73727, false, false},  {"L0A_SIZE", 8192, true, true},   {"L0A_SIZE", 8191, false, false},
        {"L0B_SIZE", 8192, true, false},   {"L0B_SIZE", 8191, false, false}, {"L0C_SIZE", 16384, true, true},
        {"L0C_SIZE", 16383, false, false},
    };
    vector<RotateQuantArch35Data> cases;
    for (const char* soc : {"Ascend960DT", "Ascend960PR"}) {
        for (const auto& boundary : boundaries) {
            auto data = MakeDav920SeriesCase(soc, false, false);
            data.case_name += "_" + string(boundary.resource) + "_" + to_string(boundary.bytes);
            const string key = "\"" + string(boundary.resource) + "\": ";
            const size_t begin = data.compile_info.find(key) + key.size();
            const size_t end = data.compile_info.find_first_of(",}", begin);
            data.compile_info.replace(begin, end - begin, to_string(boundary.bytes));
            data.expect_status = boundary.succeeds ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
            if (boundary.minimumRows) {
                data.expectedML1 = 16;
                data.expectedTailML1 = 3;
            }
            if (string(boundary.resource) == "UB_SIZE" && boundary.succeeds) {
                data.expectedUbDB = 1;
            }
            cases.push_back(data);
        }
    }
    return cases;
}

static const auto mxResourceBoundaryCases = MakeMxResourceBoundaryCases();
INSTANTIATE_TEST_SUITE_P(RotateQuantMxResourceBoundaryCases, TilingRotateQuantArch35,
                         testing::ValuesIn(mxResourceBoundaryCases));

static vector<RotateQuantArch35Data> MakeMxWorkspacePipelineCases()
{
    vector<RotateQuantArch35Data> cases;
    for (const char* soc : {"Ascend950", "Ascend960DT", "Ascend960PR"}) {
        for (int64_t k : {32, 64, 128, 256}) {
            const uint32_t rows = 16384 / k;
            for (bool multiRot : {false, true}) {
                auto data = MakeDav920SeriesCase(soc, false, multiRot);
                data.case_name = string(soc) + (multiRot ? "_mx_multi_pipeline" : "_mx_shared_pipeline") + "_k" +
                                 to_string(k);
                // Shared: 33 tiles. Independent rotations: 11 M tiles * 3 rotations, not ceil(3*M/rows).
                const int64_t m = (multiRot ? 10 : 32) * rows + 1;
                const int64_t n = (multiRot ? 3 : 1) * k;
                data.x_shape = {{m, n}, {m, n}};
                data.y_shape = data.x_shape;
                data.rot_shape = multiRot ? gert::StorageShape{{3, k, k}, {3, k, k}} :
                                            gert::StorageShape{{k, k}, {k, k}};
                data.scale_shape = {{m, (n + 63) / 64, 2}, {m, (n + 63) / 64, 2}};
                data.expectedML1 = rows;
                data.expectedTailML1 = 1;
                data.expectedUbDB = 2;
                cases.push_back(data);

                if (k != 32) {
                    continue;
                }
                const uint64_t twoWorkspaces = 2 * (52736UL + rows * 32UL);
                for (uint64_t ubBytes : {twoWorkspaces, twoWorkspaces - 1}) {
                    auto boundary = data;
                    boundary.case_name += "_ub" + to_string(ubBytes);
                    const string key = "\"UB_SIZE\": ";
                    const size_t begin = boundary.compile_info.find(key) + key.size();
                    const size_t end = boundary.compile_info.find_first_of(",}", begin);
                    boundary.compile_info.replace(begin, end - begin, to_string(ubBytes));
                    boundary.expectedUbDB = ubBytes == twoWorkspaces ? 2 : 1;
                    // One byte below the double-workspace boundary must keep the same 512-row tile.
                    cases.push_back(boundary);
                }

                auto oneWave = data;
                oneWave.case_name += "_one_wave";
                oneWave.x_shape = {{m - 1, n}, {m - 1, n}};
                oneWave.y_shape = oneWave.x_shape;
                oneWave.scale_shape = {{m - 1, (n + 63) / 64, 2}, {m - 1, (n + 63) / 64, 2}};
                oneWave.expectedTailML1 = rows;
                oneWave.expectedUbDB = 1;
                oneWave.expectedBlockDim = multiRot ? 30 : 32;
                cases.push_back(oneWave);
            }
        }
    }
    return cases;
}

static const auto mxWorkspacePipelineCases = MakeMxWorkspacePipelineCases();
INSTANTIATE_TEST_SUITE_P(RotateQuantMxWorkspacePipelineCases, TilingRotateQuantArch35,
                         testing::ValuesIn(mxWorkspacePipelineCases));

static vector<RotateQuantArch35Data> MakeHif4BasicApiCases()
{
    struct MainlineCase {
        int64_t k;
        int64_t m;
        int64_t segments;
        uint32_t baseM;
        uint32_t mL1;
        uint32_t tailM;
        uint32_t cores;
    };
    // Expected values come from the mainline A6 single-buffer Basic API tiling.
    const MainlineCase baselines[] = {
        {64, 1, 1, 512, 16, 1, 1},       {64, 17, 1, 512, 16, 1, 2},      {64, 16385, 1, 512, 512, 1, 32},
        {64, 16385, 3, 512, 512, 3, 32}, {128, 1, 1, 256, 16, 1, 1},      {128, 17, 1, 256, 16, 1, 2},
        {128, 8193, 1, 256, 256, 1, 32}, {128, 8193, 3, 256, 256, 3, 32},
    };
    vector<RotateQuantArch35Data> cases;
    for (const char* soc : {"Ascend960DT", "Ascend960PR"}) {
        for (ge::DataType inputType : {ge::DT_FLOAT16, ge::DT_BF16}) {
            for (const auto& baseline : baselines) {
                auto data = MakeDav920SeriesCase(soc, true, false);
                data.case_name = string(soc) + "_hif4_basic_api_k" + to_string(baseline.k) + "_m" +
                                 to_string(baseline.m) + "_segments" + to_string(baseline.segments) + "_dtype" +
                                 to_string(static_cast<int32_t>(inputType));
                const int64_t n = baseline.k * baseline.segments;
                data.x_shape = {{baseline.m, n}, {baseline.m, n}};
                data.y_shape = data.x_shape;
                data.rot_shape = {{baseline.k, baseline.k}, {baseline.k, baseline.k}};
                data.scale_shape = {{baseline.m, n / 64, 1}, {baseline.m, n / 64, 1}};
                data.xDataType = inputType;
                data.expectedBaseM = baseline.baseM;
                data.expectedML1 = baseline.mL1;
                data.expectedTailML1 = baseline.tailM;
                data.expectedBlockDim = baseline.cores;
                data.expectedUbDB = 1;
                cases.push_back(data);

                if (baseline.mL1 != baseline.baseM) {
                    continue;
                }
                auto reducedL0 = data;
                reducedL0.case_name += "_half_l0a";
                const string oldL0 = "\"L0A_SIZE\": 65536";
                reducedL0.compile_info.replace(reducedL0.compile_info.find(oldL0), oldL0.size(), "\"L0A_SIZE\": 32768");
                reducedL0.expectedBaseM /= 2;
                reducedL0.expectedML1 /= 2;
                cases.push_back(reducedL0);
            }
        }
    }
    return cases;
}

static const auto hif4BasicApiCases = MakeHif4BasicApiCases();
INSTANTIATE_TEST_SUITE_P(RotateQuantHif4BasicApiCases, TilingRotateQuantArch35, testing::ValuesIn(hif4BasicApiCases));

static void CheckGroupedTiling(int64_t groupNum, int64_t rotNum, int64_t groupType, ge::DataType groupDtype,
                               ge::graphStatus expected, bool hasAlpha = true, const string& socVersion = "Ascend950",
                               int64_t k = 32, int64_t rotSlices = 0, int64_t ratio = 3,
                               const map<string, uint64_t>& resources = map<string, uint64_t>(), int64_t m = 29,
                               uint32_t expectedUbDB = 2, uint32_t expectedML1 = 0)
{
    RotateQuantArch35Data test_params;
    const auto arch = TestNpuArch(socVersion);
    const uint32_t expectedAivRatio = arch == "dav3510" ? 2 : 1;
    test_params.compile_info = MakeCompileInfoForTest(arch, 32 * expectedAivRatio);
    for (const auto& resource : resources) {
        const string key = "\"" + resource.first + "\": ";
        const size_t begin = test_params.compile_info.find(key) + key.size();
        const size_t end = test_params.compile_info.find_first_of(",}", begin);
        test_params.compile_info.replace(begin, end - begin, to_string(resource.second));
    }
    const int64_t n = ratio * k;
    test_params.x_shape = gert::StorageShape({m, n}, {m, n});
    test_params.rot_shape = gert::StorageShape({rotNum, k, k}, {rotNum, k, k});
    if (rotSlices != 0) {
        test_params.rot_shape = gert::StorageShape({rotNum, rotSlices, k, k}, {rotNum, rotSlices, k, k});
    }
    test_params.y_shape = gert::StorageShape({m, n}, {m, n});
    test_params.scale_shape = gert::StorageShape({m, (n + 63) / 64, 2}, {m, (n + 63) / 64, 2});
    test_params.expect_status = expected;
    gert::StorageShape alpha_shape = {{1}, {1}};
    gert::StorageShape group_shape = {{groupNum}, {groupNum}};

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfosDavAware(test_params.compile_info, soc_infos, aicore_spec, intrinsics, soc_version);
    soc_version["Short_SoC_version"] = socVersion;

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    Ops::NN::RotateQuant::RotateQuantAptCompileInfo compile_info;

    string op_type("RotateQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(test_params.compile_info.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::TilingContextFaker faker;
    faker.SetOpType("RotateQuant")
        .NodeIoNum(hasAlpha ? 4 : 3, 2)

        .OutputShapes({&test_params.y_shape, &test_params.scale_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, test_params.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, test_params.xDataType, ge::FORMAT_ND, ge::FORMAT_ND)

        .NodeOutputTd(0, test_params.yDataType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, test_params.scaleDataType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"y_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.yDataType)},
                    {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.axis)},
                    {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(test_params.roundMode)},
                    {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(test_params.scaleAlg)},
                    {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(test_params.dstTypeMax)},
                    {"trans", Ops::NN::AnyValue::CreateFrom<bool>(test_params.trans)},
                    {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(groupType)}})
        .TilingData(param.get())
        .Workspace(ws_size);
    if (hasAlpha) {
        faker.IrInstanceNum({1, 1, 1, 1})
            .InputShapes({&test_params.x_shape, &test_params.rot_shape, &alpha_shape, &group_shape})
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, groupDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    } else {
        faker.IrInstanceNum({1, 1, 0, 1})
            .InputShapes({&test_params.x_shape, &test_params.rot_shape, &group_shape})
            .NodeInputTd(2, groupDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ge::graphStatus actual_status = tiling_func(tiling_context);
    ASSERT_EQ(actual_status, test_params.expect_status);
    if (actual_status == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(tiling_context->GetTilingKey(), 1UL);
        auto data = reinterpret_cast<const RotateQuantAptOpt::RotateQuantAptTilingData*>(
            tiling_context->GetRawTilingData()->GetData());
        EXPECT_EQ(data->groupMode, 1);
        auto workspaceSizes = tiling_context->GetWorkspaceSizes(1);
        ASSERT_NE(workspaceSizes, nullptr);
        EXPECT_EQ(workspaceSizes[0], ROTATE_QUANT_SYS_WORKSPACE_BYTES);
        EXPECT_EQ(data->ubDB, expectedUbDB);
        EXPECT_EQ(data->aivNumRatio, expectedAivRatio);
        EXPECT_EQ(data->groupNum, groupNum);
        EXPECT_EQ(data->groupListType, groupType);
        EXPECT_EQ(data->B, rotSlices != 0 ? ratio : 1);
        EXPECT_EQ(data->hasAlpha, hasAlpha);
        EXPECT_EQ(data->nKRatio, ratio);
        EXPECT_GT(data->mL1, 0);
        if (expectedML1 != 0) {
            EXPECT_EQ(data->mL1, expectedML1);
        }
        EXPECT_LE(data->mL1 * data->K, 16384);
        EXPECT_LE(data->mL1, data->baseM);
        EXPECT_EQ(data->nL1, data->K);
        EXPECT_EQ(data->kL1, data->K);
        EXPECT_EQ(data->baseN, data->K);
        EXPECT_GT(data->baseK, 0);
        EXPECT_LE(data->baseK, data->K);
        EXPECT_EQ(data->baseK % 16, 0);
        const uint64_t alignedM = (data->mL1 + 15UL) / 16 * 16;
        const uint64_t alignedBaseM = (data->baseM + 15UL) / 16 * 16;
        EXPECT_LE(2 * (alignedM * data->K * 2 + static_cast<uint64_t>(data->K) * data->K * 2),
                  stoull(aicore_spec.at("l1_size")));
        EXPECT_LE(2 * alignedBaseM * data->baseK * 2, stoull(aicore_spec.at("l0_a_size")));
        EXPECT_LE(2 * alignedBaseM * data->K * 4, stoull(aicore_spec.at("l0_c_size")));
        EXPECT_LE(static_cast<uint64_t>(data->baseK) * data->baseN * 2, stoull(aicore_spec.at("l0_b_size")) / 2);
        EXPECT_LE(data->ubDB * (52736UL + static_cast<uint64_t>(data->mL1) * 32), stoull(aicore_spec.at("ub_size")));
        if (k == 256) {
            if (resources.count("L0B_SIZE") == 0) {
                EXPECT_EQ(data->baseK, 64);
            }
            EXPECT_LE(data->mL1, 64);
        }
        EXPECT_EQ(tiling_context->GetBlockDim(), 32);
    }
}

TEST(RotateQuantGroupedTiling, GroupMetadataAndFixedBufferLimits)
{
    CheckGroupedTiling(7, 7, 0, ge::DT_INT64, ge::GRAPH_SUCCESS);
    CheckGroupedTiling(7, 7, 1, ge::DT_INT64, ge::GRAPH_SUCCESS, false);
    CheckGroupedTiling(1024, 1024, 1, ge::DT_INT64, ge::GRAPH_SUCCESS);
    CheckGroupedTiling(7, 3, 0, ge::DT_INT64, ge::GRAPH_FAILED);
    CheckGroupedTiling(1025, 1025, 0, ge::DT_INT64, ge::GRAPH_FAILED);
    CheckGroupedTiling(0, 0, 0, ge::DT_INT64, ge::GRAPH_FAILED);
    CheckGroupedTiling(7, 7, 2, ge::DT_INT64, ge::GRAPH_FAILED);
    CheckGroupedTiling(7, 7, 0, ge::DT_INT32, ge::GRAPH_FAILED);
}

TEST(RotateQuantGroupedTiling, MissingAlphaDoesNotAliasGroupList)
{
    for (int64_t groupNum : {1, 7, 1024}) {
        for (int64_t groupType : {0, 1}) {
            CheckGroupedTiling(groupNum, groupNum, groupType, ge::DT_INT64, ge::GRAPH_SUCCESS, false);
        }
    }
}

TEST(RotateQuantGroupedTiling, TensorApiWorkspaceAndResourceBoundaries)
{
    struct ResourceBoundary {
        const char* name;
        uint64_t bytes;
        bool succeeds;
        uint32_t stages;
    };
    for (const char* soc : {"Ascend950", "Ascend960DT", "Ascend960PR"}) {
        for (int64_t k : {32, 64, 128, 256}) {
            for (bool multiRot : {false, true}) {
                const int64_t slices = multiRot ? 3 : 0;
                const ResourceBoundary ubBoundaries[] = {
                    {"UB_SIZE", 53248, true, 1},
                    {"UB_SIZE", 53247, false, 0},
                    {"UB_SIZE", 106496, true, 2},
                    {"UB_SIZE", 106495, true, 1},
                };
                for (const auto& boundary : ubBoundaries) {
                    CheckGroupedTiling(7, 7, 1, ge::DT_INT64, boundary.succeeds ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED,
                                       false, soc, k, slices, 3, {{boundary.name, boundary.bytes}}, 29, boundary.stages,
                                       16);
                }
                // Large groups keep the capacity-limited tile and launch all Cube cores.
                CheckGroupedTiling(7, 7, 1, ge::DT_INT64, ge::GRAPH_SUCCESS, false, soc, k, slices, 3, {}, 32769, 2,
                                   16384 / k);
                if (k != 128) {
                    continue;
                }
                const ResourceBoundary cubeBoundaries[] = {
                    {"L1_SIZE", 73728, true, 2},  {"L1_SIZE", 73727, false, 0},  {"L0A_SIZE", 8192, true, 2},
                    {"L0A_SIZE", 8191, false, 0}, {"L0B_SIZE", 8192, true, 2},   {"L0B_SIZE", 8191, false, 0},
                    {"L0C_SIZE", 16384, true, 2}, {"L0C_SIZE", 16383, false, 0},
                };
                for (const auto& boundary : cubeBoundaries) {
                    CheckGroupedTiling(7, 7, 1, ge::DT_INT64, boundary.succeeds ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED,
                                       false, soc, k, slices, 3, {{boundary.name, boundary.bytes}}, 29, boundary.stages,
                                       16);
                }
            }
        }
    }
}

TEST(RotateQuantGroupedTiling, StageCapacityDoesNotRequireExactGroupTaskCount)
{
    // Even this one-row case keeps the capacity-selected pipeline and full-core launch.
    for (const char* soc : {"Ascend950", "Ascend960DT", "Ascend960PR"}) {
        CheckGroupedTiling(1, 1, 1, ge::DT_INT64, ge::GRAPH_SUCCESS, false, soc, 32, 0, 1, {}, 1, 2, 16);
    }
}

TEST(RotateQuantGroupedTiling, HostShapeBoundsKeepDeviceOffsetsWithinInt64)
{
    constexpr int64_t maxDim = (int64_t{1} << 31) - 1;
    // Shape metadata only: no allocation of these large input/output tensors.
    for (int64_t k : {32, 256}) {
        const int64_t ratio = maxDim / k;
        for (bool segmented : {false, true}) {
            CheckGroupedTiling(1024, 1024, 1, ge::DT_INT64, ge::GRAPH_SUCCESS, false, "Ascend950", k,
                               segmented ? ratio : 0, ratio, {}, maxDim);
            CheckGroupedTiling(1024, 1024, 1, ge::DT_INT64, ge::GRAPH_FAILED, false, "Ascend950", k,
                               segmented ? ratio : 0, ratio, {}, maxDim + 1);
            CheckGroupedTiling(1024, 1024, 1, ge::DT_INT64, ge::GRAPH_FAILED, false, "Ascend950", k,
                               segmented ? ratio + 1 : 0, ratio + 1, {}, maxDim);
        }
    }
}

TEST(RotateQuantGroupedTiling, RejectAscend350EvenWithArch35Resources)
{
    CheckGroupedTiling(7, 7, 1, ge::DT_INT64, ge::GRAPH_FAILED, false, "Ascend350");
}

TEST(RotateQuantGroupedTiling, Dav920SeriesUsesSingleAiv)
{
    for (const char* soc : {"Ascend960DT", "Ascend960PR"}) {
        CheckGroupedTiling(7, 7, 0, ge::DT_INT64, ge::GRAPH_SUCCESS, false, soc);
        CheckGroupedTiling(7, 7, 1, ge::DT_INT64, ge::GRAPH_SUCCESS, false, soc, 64, 3);
    }
}

TEST(RotateQuantGroupedTiling, K256BothGroupRepresentations)
{
    for (int64_t type : {0, 1}) {
        CheckGroupedTiling(7, 7, type, ge::DT_INT64, ge::GRAPH_SUCCESS, true, "Ascend950", 256);
        CheckGroupedTiling(7, 7, type, ge::DT_INT64, ge::GRAPH_SUCCESS, false, "Ascend950", 256);
    }
    CheckGroupedTiling(7, 7, 1, ge::DT_INT64, ge::GRAPH_FAILED, false, "Ascend950", 512);
    CheckGroupedTiling(7, 7, 1, ge::DT_INT64, ge::GRAPH_FAILED, false, "Ascend350", 256);
}

TEST(RotateQuantGroupedTiling, PerNRotationAndShapeValidation)
{
    for (int64_t k : {32, 64, 128, 256}) {
        for (int64_t type : {0, 1}) {
            CheckGroupedTiling(7, 7, type, ge::DT_INT64, ge::GRAPH_SUCCESS, true, "Ascend950", k, 3);
            CheckGroupedTiling(7, 7, type, ge::DT_INT64, ge::GRAPH_SUCCESS, false, "Ascend950", k, 3);
            CheckGroupedTiling(7, 7, type, ge::DT_INT64, ge::GRAPH_SUCCESS, false, "Ascend950", k, 1, 1);
        }
    }
    CheckGroupedTiling(7, 7, 1, ge::DT_INT64, ge::GRAPH_FAILED, false, "Ascend950", 32, 2);
    CheckGroupedTiling(1024, 1024, 1, ge::DT_INT64, ge::GRAPH_SUCCESS, false, "Ascend950", 32, 3);
    CheckGroupedTiling(7, 3, 1, ge::DT_INT64, ge::GRAPH_FAILED, false, "Ascend950", 32, 3);
    CheckGroupedTiling(7, 7, 1, ge::DT_INT64, ge::GRAPH_FAILED, false, "Ascend350", 32, 3);
}
