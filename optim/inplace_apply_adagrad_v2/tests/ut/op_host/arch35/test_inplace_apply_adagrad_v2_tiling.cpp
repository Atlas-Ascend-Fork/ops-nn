/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <iostream>
#include <string>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;
using namespace ut_util;

namespace optiling {
struct InplaceApplyAdagradV2CompileInfo {};
} // namespace optiling

class InplaceApplyAdagradV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InplaceApplyAdagradV2TilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "InplaceApplyAdagradV2TilingTest TearDown" << std::endl; }
};

struct AdagradV2TilingOverrides {
    int32_t dtypeInputIndex = -1;
    ge::DataType inputDtype = ge::DT_UNDEFINED;
    int32_t shapeInputIndex = -1;
    gert::StorageShape* inputShape = nullptr;
};

static void DoTilingTest(ge::DataType varDtype, gert::StorageShape& varShape, gert::StorageShape& scalarShape,
                         bool updateSlots, ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS,
                         ge::DataType accumDtype = ge::DT_UNDEFINED, gert::StorageShape* accumShape = nullptr,
                         const AdagradV2TilingOverrides& overrides = {})
{
    std::string opType("InplaceApplyAdagradV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64}
    })";
    map<string, string> socInfos, aicoreSpec, intrinsics;
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::InplaceApplyAdagradV2CompileInfo compileInfo;

    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(32);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    ge::DataType effectiveAccumDtype = accumDtype == ge::DT_UNDEFINED ? varDtype : accumDtype;
    gert::StorageShape* effectiveAccumShape = accumShape == nullptr ? &varShape : accumShape;
    std::array<ge::DataType, 5> inputDtypes = {varDtype, effectiveAccumDtype, ge::DT_FLOAT, ge::DT_FLOAT, varDtype};
    std::array<gert::StorageShape*, 5> inputShapes = {&varShape, effectiveAccumShape, &scalarShape, &scalarShape,
                                                      &varShape};
    std::array<ge::Format, 5> inputFormats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                              ge::FORMAT_ND};
    std::array<ge::Format, 2> outputFormats = {ge::FORMAT_ND, ge::FORMAT_ND};
    if (overrides.dtypeInputIndex >= 0) {
        ASSERT_LT(overrides.dtypeInputIndex, static_cast<int32_t>(inputDtypes.size()));
        inputDtypes[static_cast<size_t>(overrides.dtypeInputIndex)] = overrides.inputDtype;
    }
    if (overrides.shapeInputIndex >= 0) {
        ASSERT_LT(overrides.shapeInputIndex, static_cast<int32_t>(inputShapes.size()));
        ASSERT_NE(overrides.inputShape, nullptr);
        inputShapes[static_cast<size_t>(overrides.shapeInputIndex)] = overrides.inputShape;
    }

    // 属性按索引序构造：update_slots(0)——faker 属性索引随 NodeAttrs 传入序
    // （epsilon 已迁输入 tensor#4，不再是属性）
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> nodeAttrs;
    nodeAttrs.emplace_back("update_slots", Ops::NN::AnyValue::CreateFrom<bool>(updateSlots));

    // V2D: 5 inputs (var/accum/grad = varDtype, lr/epsilon = DT_FLOAT), 2 outputs (var, accum)
    // 输入顺序：var(0), accum(1), lr(2), epsilon(3), grad(4)
    // 属性顺序：update_slots(0)
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(5, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({inputShapes[0], inputShapes[1], inputShapes[2], inputShapes[3], inputShapes[4]})
                      .OutputShapes({&varShape, &varShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, inputDtypes[0], inputFormats[0], inputFormats[0])
                      .NodeInputTd(1, inputDtypes[1], inputFormats[1], inputFormats[1])
                      .NodeInputTd(2, inputDtypes[2], inputFormats[2], inputFormats[2])
                      .NodeInputTd(3, inputDtypes[3], inputFormats[3], inputFormats[3])
                      .NodeInputTd(4, inputDtypes[4], inputFormats[4], inputFormats[4])
                      .NodeOutputTd(0, varDtype, outputFormats[0], outputFormats[0])
                      .NodeOutputTd(1, varDtype, outputFormats[1], outputFormats[1])
                      .NodeAttrs(nodeAttrs)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* ctx = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(ctx, nullptr);
    ASSERT_NE(ctx->GetPlatformInfo(), nullptr);
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(ctx), expectedStatus);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_fp32_1d_update_slots_true)
{
    gert::StorageShape varShape = {{1024}, {1024}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_fp32_1d_update_slots_false)
{
    gert::StorageShape varShape = {{256}, {256}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, false);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_empty_tensor)
{
    gert::StorageShape varShape = {{0}, {0}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_unsupported_dtype_rejected)
{
    gert::StorageShape varShape = {{16}, {16}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT16, varShape, scalarShape, true, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_accum_dtype_mismatch_rejected)
{
    gert::StorageShape varShape = {{4, 8}, {4, 8}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true, ge::GRAPH_FAILED, ge::DT_FLOAT16);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_exact_shape_mismatch_rejected)
{
    gert::StorageShape varShape = {{2, 6}, {2, 6}};
    gert::StorageShape accumShape = {{3, 4}, {3, 4}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true, ge::GRAPH_FAILED, ge::DT_UNDEFINED, &accumShape);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_single_element_lr_1x1_accepted)
{
    // lr 为 (1,1) 单元素 tensor（元素总数为 1）：接受
    gert::StorageShape varShape = {{16}, {16}};
    gert::StorageShape scalarShape = {{1, 1}, {1, 1}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_single_element_lr_scalar_accepted)
{
    // lr 为 rank-0 标量 tensor（StorageShape{{},{}}，元素总数为 1）：接受
    gert::StorageShape varShape = {{16}, {16}};
    gert::StorageShape scalarShape = {{}, {}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_multi_element_lr_rejected)
{
    // lr 为 (1,2) 两元素 tensor（元素总数 != 1）：拒绝
    gert::StorageShape varShape = {{16}, {16}};
    gert::StorageShape scalarShape = {{1, 2}, {1, 2}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true, ge::GRAPH_FAILED);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_rank_9_accepted)
{
    // rank-9 var：elewise 展平计算，rank 不参与校验
    gert::StorageShape varShape = {{1, 1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 2}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_each_input_unsupported_dtype_rejected)
{
    gert::StorageShape varShape = {{4}, {4}};
    gert::StorageShape scalarShape = {{1}, {1}};
    for (int32_t i = 0; i < 5; ++i) {
        SCOPED_TRACE("input index " + std::to_string(i));
        AdagradV2TilingOverrides overrides;
        overrides.dtypeInputIndex = i;
        overrides.inputDtype = ge::DT_INT8;
        DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true, ge::GRAPH_FAILED, ge::DT_UNDEFINED, nullptr, overrides);
    }
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_each_tensor_shape_mismatch_rejected)
{
    gert::StorageShape varShape = {{4}, {4}};
    gert::StorageShape mismatchShape = {{2, 2}, {2, 2}};
    gert::StorageShape scalarShape = {{1}, {1}};
    for (const int32_t i : {1, 4}) {
        SCOPED_TRACE("input index " + std::to_string(i));
        AdagradV2TilingOverrides overrides;
        overrides.shapeInputIndex = i;
        overrides.inputShape = &mismatchShape;
        DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true, ge::GRAPH_FAILED, ge::DT_UNDEFINED, nullptr, overrides);
    }
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_single_element_epsilon_1x1_accepted)
{
    // epsilon 为 (1,1) 单元素 tensor（元素总数为 1）：接受
    gert::StorageShape varShape = {{16}, {16}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape epsShape = {{1, 1}, {1, 1}};
    AdagradV2TilingOverrides overrides;
    overrides.shapeInputIndex = 3;
    overrides.inputShape = &epsShape;
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true, ge::GRAPH_SUCCESS, ge::DT_UNDEFINED, nullptr, overrides);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_single_element_epsilon_scalar_accepted)
{
    // epsilon 为 rank-0 标量 tensor（StorageShape{{},{}}，元素总数为 1）：接受
    gert::StorageShape varShape = {{16}, {16}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape epsShape = {{}, {}};
    AdagradV2TilingOverrides overrides;
    overrides.shapeInputIndex = 3;
    overrides.inputShape = &epsShape;
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true, ge::GRAPH_SUCCESS, ge::DT_UNDEFINED, nullptr, overrides);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_multi_element_epsilon_rejected)
{
    // epsilon 为 (1,2) 两元素 tensor（元素总数 != 1）：拒绝
    gert::StorageShape varShape = {{16}, {16}};
    gert::StorageShape scalarShape = {{1}, {1}};
    gert::StorageShape epsShape = {{1, 2}, {1, 2}};
    AdagradV2TilingOverrides overrides;
    overrides.shapeInputIndex = 3;
    overrides.inputShape = &epsShape;
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true, ge::GRAPH_FAILED, ge::DT_UNDEFINED, nullptr, overrides);
}

TEST_F(InplaceApplyAdagradV2TilingTest, tiling_scalar_var_0d)
{
    gert::StorageShape varShape = {{}, {}};
    gert::StorageShape scalarShape = {{1}, {1}};
    DoTilingTest(ge::DT_FLOAT, varShape, scalarShape, true);
}
