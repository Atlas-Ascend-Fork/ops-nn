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
#include <limits>
#include <map>
#include <string>
#include <vector>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/situ_glu_group_grad_tiling_arch35.h"

namespace {
struct SituGluGroupGradTilingCase {
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType weightDtype = ge::DT_FLOAT;
    ge::DataType groupIndexDtype = ge::DT_INT64;
    gert::StorageShape xShape = {{8, 128, 8192}, {8, 128, 8192}};
    gert::StorageShape weightShape = {{1024}, {1024}};
    gert::StorageShape groupIndexShape = {{8}, {8}};
    int64_t dim = -1;
    float beta = 1.0f;
    float linearBeta = 0.0f;
    bool activateLeft = true;
    // 0 = cumsum (group_index[-1] is the total row count), 1 = count (sum(group_index) is the total).
    int64_t groupListType = 0;
    bool hasWeight = false;
    bool hasGroupIndex = false;
    // Empty override -> derived from xShape (grad_x == x). Set to inject an invalid grad_x shape.
    gert::StorageShape gradXShapeOverride = {{}, {}};
    // < 0 -> derived (bs when weight present, else 0). Set to inject an invalid grad_weight numel.
    int64_t gradWeightNumelOverride = -1;
    ge::graphStatus status = ge::GRAPH_SUCCESS;
};

class SituGluGroupGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SituGluGroupGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SituGluGroupGradTilingTest TearDown" << std::endl; }
};

void ExecuteTilingCase(const SituGluGroupGradTilingCase& tc)
{
    const std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::SituGluGroupGradCompileInfo compileInfo;

    const std::string opType("SituGluGroupGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    auto* parsePlatform = kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo();
    ASSERT_TRUE(parsePlatform->Init());
    parsePlatform->SetPlatformRes("SoCInfo", socInfos);
    parsePlatform->SetPlatformRes("AICoreSpec", aicoreSpec);
    parsePlatform->SetCoreNumByCoreType("AICore");
    parsePlatform->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parsePlatform->SetPlatformRes("version", socVersions);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* workspaceSizes = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(tilingData, nullptr);

    gert::StorageShape xShape = tc.xShape;
    gert::StorageShape weightShape = tc.weightShape;
    gert::StorageShape groupIndexShape = tc.groupIndexShape;

    // Derive the cut axis, bs (rows before the axis) and the grad_y / grad_x / grad_weight shapes.
    const gert::Shape& xStorage = xShape.GetStorageShape();
    const int64_t rank = static_cast<int64_t>(xStorage.GetDimNum());
    int64_t axis = tc.dim < 0 ? tc.dim + rank : tc.dim;
    if (axis < 0) {
        axis = 0;
    }
    int64_t bs = 1;
    for (int64_t i = 0; i < axis && i < rank; ++i) {
        bs *= xStorage.GetDim(i);
    }
    // grad_y equals x with the cut axis halved; both the origin and storage shapes are updated so
    // the faker sees a consistent descriptor.
    gert::StorageShape gradYShape = xShape;
    if (axis < rank && xStorage.GetDim(axis) > 0) {
        gradYShape.MutableOriginShape().SetDim(axis, xStorage.GetDim(axis) / 2);
        gradYShape.MutableStorageShape().SetDim(axis, xStorage.GetDim(axis) / 2);
    }
    gert::StorageShape gradXShape = (tc.gradXShapeOverride.GetStorageShape().GetDimNum() > 0) ? tc.gradXShapeOverride :
                                                                                                xShape;
    int64_t gradWeightNumel = tc.hasWeight ? bs : 0;
    if (tc.gradWeightNumelOverride >= 0) {
        gradWeightNumel = tc.gradWeightNumelOverride;
    }
    gert::StorageShape gradWeightShape = {{gradWeightNumel}, {gradWeightNumel}};

    std::vector<uint32_t> inputInstanceNum = {1, 1, tc.hasWeight ? 1U : 0U, tc.hasGroupIndex ? 1U : 0U};
    std::vector<gert::StorageShape*> inputShapes = {&gradYShape, &xShape};
    if (tc.hasWeight) {
        inputShapes.emplace_back(&weightShape);
    }
    if (tc.hasGroupIndex) {
        inputShapes.emplace_back(&groupIndexShape);
    }

    gert::TilingContextFaker contextFaker;
    contextFaker.SetOpType(opType)
        .NodeIoNum(4, 2)
        .IrInstanceNum(inputInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes({&gradXShape, &gradWeightShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, tc.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, tc.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, tc.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(tc.dim)},
                    {"beta", Ops::NN::AnyValue::CreateFrom<float>(tc.beta)},
                    {"linear_beta", Ops::NN::AnyValue::CreateFrom<float>(tc.linearBeta)},
                    {"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(tc.activateLeft)},
                    {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(tc.groupListType)}})
        .TilingData(tilingData.get())
        .Workspace(workspaceSizes);
    int32_t inputTdIndex = 2;
    if (tc.hasWeight) {
        contextFaker.NodeInputTd(inputTdIndex++, tc.weightDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    if (tc.hasGroupIndex) {
        contextFaker.NodeInputTd(inputTdIndex++, tc.groupIndexDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }

    auto holder = contextFaker.Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersions);

    EXPECT_EQ(tilingFunc(tilingContext), tc.status);
}

TEST_F(SituGluGroupGradTilingTest, tiling_fp16)
{
    SituGluGroupGradTilingCase tc;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_bf16)
{
    SituGluGroupGradTilingCase tc;
    tc.xDtype = ge::DT_BF16;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_f32)
{
    SituGluGroupGradTilingCase tc;
    tc.xDtype = ge::DT_FLOAT;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_weight_valid)
{
    SituGluGroupGradTilingCase tc;
    tc.hasWeight = true;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_group_index_valid)
{
    SituGluGroupGradTilingCase tc;
    tc.hasGroupIndex = true;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_weight_and_group_index)
{
    SituGluGroupGradTilingCase tc;
    tc.hasWeight = true;
    tc.hasGroupIndex = true;
    ExecuteTilingCase(tc);
}

// group_index above one UB chunk: exercises the multi-chunk search and the 16-element
// down-alignment branch of CalcGroupIndexTiling (groupIndexLoop > 1). Count mode only: cumsum mode
// reads the last element on device, so no chunked reduce tiling is computed there.
TEST_F(SituGluGroupGradTilingTest, tiling_group_index_multichunk)
{
    SituGluGroupGradTilingCase tc;
    tc.hasGroupIndex = true;
    tc.groupListType = 1;
    tc.groupIndexShape = {{20001}, {20001}};
    ExecuteTilingCase(tc);
}

// The same oversized group_index in the default cumsum mode: no chunk search, tiling must succeed.
TEST_F(SituGluGroupGradTilingTest, tiling_group_index_multichunk_cumsum)
{
    SituGluGroupGradTilingCase tc;
    tc.hasGroupIndex = true;
    tc.groupIndexShape = {{20001}, {20001}};
    ExecuteTilingCase(tc);
}

// group_list_type 1 (count mode) with a single-chunk group_index.
TEST_F(SituGluGroupGradTilingTest, tiling_group_index_count_mode)
{
    SituGluGroupGradTilingCase tc;
    tc.hasGroupIndex = true;
    tc.groupListType = 1;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_group_list_type_too_large)
{
    // group_list_type must be 0 (cumsum) or 1 (count).
    SituGluGroupGradTilingCase tc;
    tc.hasGroupIndex = true;
    tc.groupListType = 2;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_group_list_type_negative)
{
    SituGluGroupGradTilingCase tc;
    tc.groupListType = -1;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_empty_group_index)
{
    SituGluGroupGradTilingCase tc;
    tc.hasGroupIndex = true;
    tc.groupIndexShape = {{0}, {0}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_dim_non_last)
{
    // dim=1: rows = 8, d = 128*8192, weight element count = 8.
    SituGluGroupGradTilingCase tc;
    tc.dim = 1;
    tc.hasWeight = true;
    tc.weightShape = {{8}, {8}};
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_invalid_cut_dim)
{
    // The cut dim of x must be divisible by 2; 1023 is invalid.
    SituGluGroupGradTilingCase tc;
    tc.xShape = {{4, 64, 1023}, {4, 64, 1023}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_invalid_x_dtype)
{
    SituGluGroupGradTilingCase tc;
    tc.xDtype = ge::DT_INT32;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_invalid_weight_dtype)
{
    SituGluGroupGradTilingCase tc;
    tc.hasWeight = true;
    tc.weightDtype = ge::DT_FLOAT16;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_invalid_weight_shape)
{
    // weight element count must equal the product of x dims before the cut dim (1024 here).
    SituGluGroupGradTilingCase tc;
    tc.hasWeight = true;
    tc.weightShape = {{1023}, {1023}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_invalid_group_index_dtype)
{
    SituGluGroupGradTilingCase tc;
    tc.hasGroupIndex = true;
    tc.groupIndexDtype = ge::DT_INT32;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_invalid_group_index_rank)
{
    // group_index must be 1D; 2D group_index is invalid.
    SituGluGroupGradTilingCase tc;
    tc.hasGroupIndex = true;
    tc.groupIndexShape = {{2, 4}, {2, 4}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_invalid_grad_x_shape)
{
    // grad_x must have the same shape as x; a halved cut dim is invalid here.
    SituGluGroupGradTilingCase tc;
    tc.gradXShapeOverride = {{8, 128, 4096}, {8, 128, 4096}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_grad_weight_nonempty_without_weight)
{
    // Without weight, grad_weight must be empty (numel 0); a non-empty grad_weight is invalid.
    SituGluGroupGradTilingCase tc;
    tc.gradWeightNumelOverride = 1024;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_grad_weight_mismatch_with_weight)
{
    // With weight, grad_weight numel must equal bs (1024); 1023 is invalid.
    SituGluGroupGradTilingCase tc;
    tc.hasWeight = true;
    tc.gradWeightNumelOverride = 1023;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_zero_beta)
{
    // beta is a divisor in the SiTU formula, so 0 must be rejected by tiling.
    SituGluGroupGradTilingCase tc;
    tc.beta = 0.0f;
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_large_legal_no_false_reject)
{
    // Large but non-overflowing shape must not be rejected by the overflow guards.
    SituGluGroupGradTilingCase tc;
    tc.xShape = {{8, 65536}, {8, 65536}};
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_bs_product_overflow)
{
    // dim=-1 on 3D: bs = dim0 * dim1 = 2^40 * 2^23 = 2^63 overflows int64.
    constexpr int64_t dim2pow40 = 1LL << 40;
    constexpr int64_t dim2pow23 = 1LL << 23;
    SituGluGroupGradTilingCase tc;
    tc.xShape = {{dim2pow40, dim2pow23, 8}, {dim2pow40, dim2pow23, 8}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_total_element_overflow)
{
    // bs and d each fit int64, but bs * d = 2^40 * 2^23 = 2^63 overflows.
    constexpr int64_t dim2pow40 = 1LL << 40;
    constexpr int64_t dim2pow23 = 1LL << 23;
    SituGluGroupGradTilingCase tc;
    tc.xShape = {{dim2pow40, dim2pow23}, {dim2pow40, dim2pow23}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

TEST_F(SituGluGroupGradTilingTest, tiling_error_weight_element_overflow)
{
    // weight (up to 8D) element-count product = 2^40 * 2^23 = 2^63 overflows int64.
    constexpr int64_t dim2pow40 = 1LL << 40;
    constexpr int64_t dim2pow23 = 1LL << 23;
    SituGluGroupGradTilingCase tc;
    tc.hasWeight = true;
    tc.weightShape = {{dim2pow40, dim2pow23}, {dim2pow40, dim2pow23}};
    tc.status = ge::GRAPH_FAILED;
    ExecuteTilingCase(tc);
}

} // namespace
