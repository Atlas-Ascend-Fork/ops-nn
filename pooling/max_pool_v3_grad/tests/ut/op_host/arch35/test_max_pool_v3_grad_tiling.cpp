/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_max_pool_v3_grad_tiling.cpp
 * \brief Tiling UT for max_pool_v3_grad operator (nn TilingContextFaker pattern).
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
#include "../../../../op_host/arch35/max_pool_v3_grad_tiling.h"
#include "../../../../../pool_grad_common/op_kernel/arch35/max_pool_grad_with_argmax_struct_common.h"

using namespace ut_util;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData;
using namespace std;
using namespace ge;
using MaxPoolV3GradNHWCNameSpace::MaxPoolV3GradNHWCTilingData;

class MaxPoolV3GradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolV3GradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaxPoolV3GradTilingTest TearDown" << std::endl; }
};

// Tiling 测试参数：输入输出 shape、属性、期望返回值和期望 tiling key
struct TilingTestCase {
    std::vector<int64_t> origInput{1, 1, 4, 4};
    std::vector<int64_t> origOutput{1, 1, 2, 2};
    std::vector<int64_t> grad{1, 1, 2, 2};
    std::vector<int64_t> outGrad{1, 1, 4, 4};
    std::vector<int64_t> ksize{1, 1, 2, 2};
    std::vector<int64_t> strides{1, 1, 2, 2};
    std::string paddingMode = "CALCULATED";
    std::vector<int64_t> pads{0, 0, 0, 0};
    std::string dataFormat = "NCHW";
    bool globalPooling = false;
    bool ceilMode = false;
    ge::graphStatus expectResult = ge::GRAPH_SUCCESS;
    uint64_t expectTilingKey = 768;
    // 大于 0 时覆盖 UT 平台信息中的 AIV 核数，使切分路径可确定。
    int64_t overrideCoreNum = 0;
    bool checkNchwTiling = false;
    int64_t expectIsPad = -1;
    bool expectWholeSpatialTile = false;
    int64_t expectNchwGatherBufferSizeLE = -1;
    bool checkSimtTiling = false;
    // NHWC 快速通道 tiling data 校验项, -1 表示不校验
    int64_t expectIsBigKernel = -1;
    int64_t expectCdim = -1;
    bool expectNhwcAlignedC = false;
    bool expectNhwcStreamedWindow = false;
    int64_t expectPadH = -1;
    int64_t expectPadHBack = -1;
    int64_t expectPadW = -1;
    int64_t expectPadWBack = -1;
    // 修复回归: 各 tile 缓冲占用上限(可用 UB 内), -1 表示不校验
    int64_t expectBufferSizeLE = -1;
};

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableShape().SetDim(i, dims[i]);
    }
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

static void VerifyNchwTiling(gert::TilingContext* context, const TilingTestCase& tc, uint64_t tilingKey)
{
    auto* tilingData = context->GetTilingData<MaxPoolGradWithArgmaxNCHWTilingCommonData>();
    ASSERT_NE(tilingData, nullptr);
    ASSERT_EQ(tc.origInput.size(), 4U);
    ASSERT_EQ(tc.origOutput.size(), 4U);

    EXPECT_EQ(tilingData->hArgmax, tc.origOutput[2]);
    EXPECT_EQ(tilingData->wArgmax, tc.origOutput[3]);
    EXPECT_EQ(tilingData->hOutput, tc.origInput[2]);
    EXPECT_EQ(tilingData->wOutput, tc.origInput[3]);
    EXPECT_EQ(tilingData->hKernel, tc.ksize[2]);
    EXPECT_EQ(tilingData->wKernel, tc.ksize[3]);
    EXPECT_EQ(tilingData->hStride, tc.strides[2]);
    EXPECT_EQ(tilingData->wStride, tc.strides[3]);
    EXPECT_EQ(tilingData->dilationH, 1);
    EXPECT_EQ(tilingData->dilationW, 1);
    EXPECT_EQ(tilingData->tilingKey, static_cast<int64_t>(tilingKey));
    EXPECT_EQ(tilingData->usedCoreNum, static_cast<int64_t>(context->GetBlockDim()));
    EXPECT_GT(tilingData->usedCoreNum, 0);
    EXPECT_GT(tilingData->highAxisInner, 0);
    EXPECT_GT(tilingData->hOutputInner, 0);
    EXPECT_GT(tilingData->wOutputInner, 0);
    EXPECT_GT(tilingData->inputBufferSize, 0);
    EXPECT_GT(tilingData->outputBufferSize, 0);
    EXPECT_GT(tilingData->gradBufferSize, 0);
    EXPECT_GT(tilingData->argmaxBufferSize, 0);
    if (tc.expectIsPad >= 0) {
        EXPECT_EQ(tilingData->isPad, tc.expectIsPad);
    }
    if (tc.expectWholeSpatialTile) {
        EXPECT_EQ(tilingData->hOutputOuter, 1);
        EXPECT_EQ(tilingData->wOutputOuter, 1);
    }

    if (tc.expectNchwGatherBufferSizeLE >= 0) {
        EXPECT_LE(tilingData->gradBufferSize, tc.expectNchwGatherBufferSizeLE);
        EXPECT_LE(tilingData->outputBufferSize, tc.expectNchwGatherBufferSizeLE);
    }

    size_t* workspace = context->GetWorkspaceSizes(1);
    ASSERT_NE(workspace, nullptr);
    EXPECT_EQ(workspace[0], 0U);
}

static void VerifySimtTiling(gert::TilingContext* context, const TilingTestCase& tc, uint64_t tilingKey)
{
    auto* tilingData = context->GetTilingData<MaxPoolGradWithArgmaxSimtTilingCommonData>();
    ASSERT_NE(tilingData, nullptr);
    ASSERT_EQ(tc.origInput.size(), 4U);
    ASSERT_EQ(tc.origOutput.size(), 4U);

    const bool isNhwc = tc.dataFormat == "NHWC";
    const size_t cIndex = isNhwc ? 3U : 1U;
    const size_t hIndex = isNhwc ? 1U : 2U;
    const size_t wIndex = isNhwc ? 2U : 3U;

    EXPECT_EQ(tilingData->nDim, tc.origInput[0]);
    EXPECT_EQ(tilingData->cDim, tc.origInput[cIndex]);
    EXPECT_EQ(tilingData->hInDim, tc.origInput[hIndex]);
    EXPECT_EQ(tilingData->wInDim, tc.origInput[wIndex]);
    EXPECT_EQ(tilingData->hOutDim, tc.origOutput[hIndex]);
    EXPECT_EQ(tilingData->wOutDim, tc.origOutput[wIndex]);
    const int64_t expectKernelH = tc.globalPooling ? tc.origInput[hIndex] : tc.ksize[hIndex];
    const int64_t expectKernelW = tc.globalPooling ? tc.origInput[wIndex] : tc.ksize[wIndex];
    EXPECT_EQ(tilingData->kSizeH, expectKernelH);
    EXPECT_EQ(tilingData->kSizeW, expectKernelW);
    EXPECT_EQ(tilingData->stridesH, tc.strides[hIndex]);
    EXPECT_EQ(tilingData->stridesW, tc.strides[wIndex]);
    EXPECT_EQ(tilingData->ceilMode, tc.ceilMode ? 1 : 0);
    EXPECT_GT(context->GetBlockDim(), 0U);

    size_t* workspace = context->GetWorkspaceSizes(1);
    ASSERT_NE(workspace, nullptr);
    EXPECT_GT(workspace[0], 0U);
    (void)tilingKey;
}

static void DoTilingTest(ge::DataType dt, const TilingTestCase& tc)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 262144,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 64
                                        }
                                    })";
    if (tc.overrideCoreNum > 0) {
        const string coreKey = "\"CORE_NUM\": 64";
        auto pos = compile_info_string.find(coreKey);
        ASSERT_NE(pos, string::npos);
        const string coreValue = "\"CORE_NUM\": " + std::to_string(tc.overrideCoreNum) +
                                 ", \"vector_core_cnt\": " + std::to_string(tc.overrideCoreNum) +
                                 ", \"cube_core_cnt\": " + std::to_string(tc.overrideCoreNum);
        compile_info_string.replace(pos, coreKey.size(), coreValue);
    }
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct MaxPoolV3GradCompileInfo {
    } compile_info;

    std::string op_type("MaxPoolV3Grad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(4, 2)
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
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape origInputShape = MakeStorageShape(tc.origInput);
    gert::StorageShape origOutputShape = MakeStorageShape(tc.origOutput);
    gert::StorageShape gradShape = MakeStorageShape(tc.grad);
    gert::StorageShape outGradShape = MakeStorageShape(tc.outGrad);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&origInputShape, &origOutputShape, &gradShape})
                      .OutputShapes({&outGradShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(tc.ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(tc.strides)},
                                  {"padding_mode", Ops::NN::AnyValue::CreateFrom<std::string>(tc.paddingMode)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(tc.pads)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(tc.dataFormat)},
                                  {"global_pooling", Ops::NN::AnyValue::CreateFrom<bool>(tc.globalPooling)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(tc.ceilMode)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    EXPECT_EQ(tiling_func(tiling_context), tc.expectResult);
    if (tc.expectResult != ge::GRAPH_SUCCESS) {
        return;
    }

    auto tiling_key = tiling_context->GetTilingKey();
    std::cout << "tiling_key: " << tiling_key << std::endl;
    ASSERT_EQ(tiling_key, tc.expectTilingKey);

    if (tc.checkNchwTiling) {
        VerifyNchwTiling(tiling_context, tc, tiling_key);
    }
    if (tc.checkSimtTiling) {
        VerifySimtTiling(tiling_context, tc, tiling_key);
    }

    if (tc.expectIsBigKernel < 0) {
        return;
    }
    const auto* td = tiling_context->GetTilingData<MaxPoolV3GradNHWCTilingData>();
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->isBigKernel, tc.expectIsBigKernel);
    if (tc.expectCdim >= 0) {
        EXPECT_EQ(td->cDim, tc.expectCdim);
    }
    if (tc.expectPadH >= 0) {
        EXPECT_EQ(td->base.padH, tc.expectPadH);
    }
    if (tc.expectPadHBack >= 0) {
        EXPECT_EQ(td->base.padHBack, tc.expectPadHBack);
    }
    if (tc.expectPadW >= 0) {
        EXPECT_EQ(td->base.padW, tc.expectPadW);
    }
    if (tc.expectPadWBack >= 0) {
        EXPECT_EQ(td->base.padWBack, tc.expectPadWBack);
    }
    EXPECT_GT(td->base.usedCoreNum, 0);
    EXPECT_GT(td->base.argmaxBufferSize, 0);
    EXPECT_GT(td->base.inputBufferSize, 0);
    EXPECT_GT(td->base.gradBufferSize, 0);
    EXPECT_GT(td->base.outputBufferSize, 0);
    if (dt == ge::DT_INT64) {
        const int64_t alignedC = (td->cOutputInner + 7) / 8 * 8;
        EXPECT_GE(td->base.outputBufferSize,
                  td->base.highAxisInner * td->base.hOutputInner * td->base.wOutputInner * alignedC * 8);
    }
    EXPECT_GT(td->cOutputInner, 0);
    if (tc.expectNhwcAlignedC) {
        EXPECT_EQ(td->base.hOutputInner, tc.strides[1]);
        EXPECT_EQ(td->base.wOutputInner, tc.strides[2]);
        EXPECT_GT(td->cOutputOuter, 1);
    }
    if (tc.expectNhwcStreamedWindow) {
        const int64_t inputBytes = ge::GetSizeByDataType(dt);
        const int64_t blockElements = 32 / inputBytes;
        const int64_t alignedC = (td->cOutputInner + blockElements - 1) / blockElements * blockElements;
        const int64_t windowBytes = tc.ksize[1] * tc.ksize[2] * alignedC * inputBytes;
        EXPECT_LT(td->base.inputBufferSize, windowBytes);
    }
    if (tc.expectBufferSizeLE >= 0) {
        EXPECT_LE(td->base.argmaxBufferSize, tc.expectBufferSizeLE);
        EXPECT_LE(td->base.inputBufferSize, tc.expectBufferSizeLE);
        EXPECT_LE(td->base.gradBufferSize, tc.expectBufferSizeLE);
        EXPECT_LE(td->base.outputBufferSize, tc.expectBufferSizeLE);
    }
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_float32)
{
    TilingTestCase tc;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_float16)
{
    TilingTestCase tc;
    DoTilingTest(ge::DT_FLOAT16, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_float32)
{
    TilingTestCase tc;
    tc.origInput = {1, 4, 4, 1};
    tc.origOutput = {1, 2, 2, 1};
    tc.grad = {1, 2, 2, 1};
    tc.outGrad = {1, 4, 4, 1};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    // NHWC 快速通道: kv=4 < 128 走小 kernel 分支; 小 shape 无法对齐填满核 -> isCheckRange=1
    tc.expectTilingKey = 787;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_big_kernel)
{
    // kv = 16*16 = 256 >= 128 -> 小C大kernel分支(共用反向 BackwardBase)
    TilingTestCase tc;
    tc.origInput = {1, 64, 64, 4};
    tc.origOutput = {1, 4, 4, 4};
    tc.grad = {1, 4, 4, 4};
    tc.outGrad = {1, 64, 64, 4};
    tc.ksize = {1, 16, 16, 1};
    tc.strides = {1, 16, 16, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 1;
    tc.expectCdim = 4;
    tc.expectPadH = 0;
    tc.expectPadHBack = 0;
    tc.expectPadW = 0;
    tc.expectPadWBack = 0;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_big_kernel_multi_n)
{
    // N=64 可整轴填满核 -> TrySplitN 成功, isCheckRange=0
    TilingTestCase tc;
    tc.origInput = {64, 16, 16, 4};
    tc.origOutput = {64, 1, 1, 4};
    tc.grad = {64, 1, 1, 4};
    tc.outGrad = {64, 16, 16, 4};
    tc.ksize = {1, 16, 16, 1};
    tc.strides = {1, 16, 16, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 1;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_big_c)
{
    // C=512: kernel 侧 isBigC(C*4B > VREG/2) 走大C标量通道; tiling 侧 C 轴切分
    TilingTestCase tc;
    tc.origInput = {1, 8, 8, 512};
    tc.origOutput = {1, 4, 4, 512};
    tc.grad = {1, 4, 4, 512};
    tc.outGrad = {1, 8, 8, 512};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 512;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_same_pad)
{
    // SAME 模式: padNeed = (4-1)*2+3-7 = 2 -> padH/padHBack = 1/1
    TilingTestCase tc;
    tc.origInput = {1, 7, 7, 3};
    tc.origOutput = {1, 4, 4, 3};
    tc.grad = {1, 4, 4, 3};
    tc.outGrad = {1, 7, 7, 3};
    tc.ksize = {1, 3, 3, 1};
    tc.strides = {1, 2, 2, 1};
    tc.paddingMode = "SAME";
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 3;
    tc.expectPadH = 1;
    tc.expectPadHBack = 1;
    tc.expectPadW = 1;
    tc.expectPadWBack = 1;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_calculated_asym_pad)
{
    // CALCULATED 非对称 pad: pads = {0, 1, 0, 2}
    TilingTestCase tc;
    tc.origInput = {1, 6, 6, 2};
    tc.origOutput = {1, 3, 3, 2};
    tc.grad = {1, 3, 3, 2};
    tc.outGrad = {1, 6, 6, 2};
    tc.ksize = {1, 3, 3, 1};
    tc.strides = {1, 2, 2, 1};
    tc.paddingMode = "CALCULATED";
    tc.pads = {0, 1, 0, 2};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 2;
    tc.expectPadH = 0;
    tc.expectPadHBack = 1;
    tc.expectPadW = 0;
    tc.expectPadWBack = 2;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_int64)
{
    // int64 的值占 8 字节，argmax 仍采用 int32。
    TilingTestCase tc;
    tc.origInput = {1, 4, 4, 1};
    tc.origOutput = {1, 2, 2, 1};
    tc.grad = {1, 2, 2, 1};
    tc.outGrad = {1, 4, 4, 1};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    DoTilingTest(ge::DT_INT64, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_integer_native)
{
    // 覆盖全部 8/16/32/64 位整数类型，验证 NHWC 快速路径及原生类型的 UB 分配。
    TilingTestCase tc;
    tc.origInput = {1, 4, 4, 1};
    tc.origOutput = {1, 2, 2, 1};
    tc.grad = {1, 2, 2, 1};
    tc.outGrad = {1, 4, 4, 1};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 1;
    for (auto dtype : {ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_INT64}) {
        SCOPED_TRACE(static_cast<int32_t>(dtype));
        DoTilingTest(dtype, tc);
    }
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_global_pooling_true)
{
    TilingTestCase tc;
    tc.origOutput = {1, 1, 1, 1};
    tc.grad = {1, 1, 1, 1};
    tc.strides = {1, 1, 1, 1};
    tc.globalPooling = true;
    tc.expectTilingKey = 258;
    tc.checkSimtTiling = true;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_ceil_mode_true)
{
    TilingTestCase tc;
    tc.ksize = {1, 1, 3, 3};
    tc.strides = {1, 1, 2, 2};
    tc.ceilMode = true;
    tc.expectTilingKey = 258;
    tc.checkSimtTiling = true;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_pads_length2)
{
    TilingTestCase tc;
    tc.pads = {0, 0};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}

// ==================== NHWC 快速通道分支泛化用例 ====================
// 注: 期望值与 Ascend950(UB 256KB/64核) 平台参数绑定

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_n_split_single_row_gather)
{
    // N 切分填满核, W 整轴保留: 前向 SingleRowGather(wAct*C > VREG/4), 反向 SingleLine
    TilingTestCase tc;
    tc.origInput = {64, 4, 256, 2};
    tc.origOutput = {64, 2, 128, 2};
    tc.grad = {64, 2, 128, 2};
    tc.outGrad = {64, 4, 256, 2};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 2;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_n_split_multi_nc_gather)
{
    // 小 tile(hwc <= VREG/4): 前向 MultiNcGather 多级 N,H,W,C 打包
    TilingTestCase tc;
    tc.origInput = {64, 4, 4, 3};
    tc.origOutput = {64, 2, 2, 3};
    tc.grad = {64, 2, 2, 3};
    tc.outGrad = {64, 4, 4, 3};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.paddingMode = "SAME";
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 3;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_n_split_multi_row_gather)
{
    // 中 tile(wc <= VREG/4 < hwc): 前向 MultiRowGather, 反向 SingleLine(overlap 但 W 批不足)
    TilingTestCase tc;
    tc.origInput = {64, 16, 16, 4};
    tc.origOutput = {64, 8, 8, 4};
    tc.grad = {64, 8, 8, 4};
    tc.outGrad = {64, 16, 16, 4};
    tc.ksize = {1, 3, 3, 1};
    tc.strides = {1, 2, 2, 1};
    tc.paddingMode = "SAME";
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_backward_multiple_line_w)
{
    // W 轴 33 列 overlap(>16 并发): 反向 MultipleLineWProcessVF
    TilingTestCase tc;
    tc.origInput = {64, 2, 66, 4};
    tc.origOutput = {64, 1, 33, 4};
    tc.grad = {64, 1, 33, 4};
    tc.outGrad = {64, 2, 66, 4};
    tc.ksize = {1, 3, 3, 1};
    tc.strides = {1, 2, 2, 1};
    tc.paddingMode = "SAME";
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_backward_multiple_line_hw)
{
    // H+W 双 overlap: 反向 MultipleLineHwProcessVF
    TilingTestCase tc;
    tc.origInput = {64, 8, 66, 4};
    tc.origOutput = {64, 4, 33, 4};
    tc.grad = {64, 4, 33, 4};
    tc.outGrad = {64, 8, 66, 4};
    tc.ksize = {1, 3, 3, 1};
    tc.strides = {1, 2, 2, 1};
    tc.paddingMode = "SAME";
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_big_kernel_split_merge)
{
    // kv=2304(48x48) x C=32: 整窗 288KB 超 UB, 大 kernel 流式分批载入+merge 路径
    TilingTestCase tc;
    tc.origInput = {1, 192, 192, 32};
    tc.origOutput = {1, 10, 10, 32};
    tc.grad = {1, 10, 10, 32};
    tc.outGrad = {1, 192, 192, 32};
    tc.ksize = {1, 48, 48, 1};
    tc.strides = {1, 16, 16, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 1;
    tc.expectCdim = 32;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_no_argmax_tiles)
{
    // pad+stride 间隙行: 部分 tile 无任何重叠窗口 -> kernel 侧 ProcessNoArgmaxBlock 置零直出
    TilingTestCase tc;
    tc.origInput = {1, 17, 17, 2};
    tc.origOutput = {1, 4, 4, 2};
    tc.grad = {1, 4, 4, 2};
    tc.outGrad = {1, 17, 17, 2};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 5, 5, 1};
    tc.paddingMode = "CALCULATED";
    tc.pads = {1, 1, 0, 0};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 2;
    tc.expectPadH = 1;
    tc.expectPadHBack = 1;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_gap_stride_c_split)
{
    // stride > kernel 无重叠 + C 轴切分到 moveDataNumCacheLine
    TilingTestCase tc;
    tc.origInput = {1, 64, 64, 8};
    tc.origOutput = {1, 16, 16, 8};
    tc.grad = {1, 16, 16, 8};
    tc.outGrad = {1, 64, 64, 8};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 4, 4, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 8;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_multicore_adjust_aligned)
{
    // 大 shape: isCheckRange=0 对齐路径下 N/H/W 轴 SearchOuterSingleAligned 多核调整
    TilingTestCase tc;
    tc.origInput = {8, 1024, 1024, 4};
    tc.origOutput = {8, 512, 512, 4};
    tc.grad = {8, 512, 512, 4};
    tc.outGrad = {8, 1024, 1024, 4};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.paddingMode = "SAME";
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_ceil_mode_last_window)
{
    // ceil_mode 末窗口回退: CeilDiv(4,3)+1=3, 末窗 start=2*3=6>=5 回退为 2
    TilingTestCase tc;
    tc.origInput = {1, 5, 5, 4};
    tc.origOutput = {1, 2, 2, 4};
    tc.grad = {1, 2, 2, 4};
    tc.outGrad = {1, 5, 5, 4};
    tc.ksize = {1, 1, 1, 1};
    tc.strides = {1, 3, 3, 1};
    tc.paddingMode = "CALCULATED";
    tc.ceilMode = true;
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_global_pooling)
{
    // NHWC 全局池化: kernel=整幅, 大 kernel 分支 + N 切分
    TilingTestCase tc;
    tc.origInput = {2, 12, 12, 16};
    tc.origOutput = {2, 1, 1, 16};
    tc.grad = {2, 1, 1, 16};
    tc.outGrad = {2, 12, 12, 16};
    tc.strides = {1, 12, 12, 1};
    tc.globalPooling = true;
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 1;
    tc.expectCdim = 16;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_big_c_bf16)
{
    // bf16 大C(80*2 > VREG/2): 大C标量通道 + C 轴切分, dtype 走 NHWC 快速通道
    TilingTestCase tc;
    tc.origInput = {1, 8, 8, 80};
    tc.origOutput = {1, 4, 4, 80};
    tc.grad = {1, 4, 4, 80};
    tc.outGrad = {1, 8, 8, 80};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 80;
    DoTilingTest(ge::DT_BF16, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_try_split_align_w)
{
    // 超宽 W: H/N 均无法填满核 -> TrySplitAlignW 胜出, W 轴 stride 对齐切分(isCheckRange=0)
    TilingTestCase tc;
    tc.origInput = {1, 8, 4096, 16};
    tc.origOutput = {1, 4, 2048, 16};
    tc.grad = {1, 4, 2048, 16};
    tc.outGrad = {1, 8, 4096, 16};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 16;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_try_split_align_c)
{
    // 极小空间维 + 超大 C(1024): H/W 对齐切分填不满核 -> TrySplitAlignC 胜出, C 轴切分
    TilingTestCase tc;
    tc.origInput = {1, 4, 4, 1024};
    tc.origOutput = {1, 2, 2, 1024};
    tc.grad = {1, 2, 2, 1024};
    tc.outGrad = {1, 4, 4, 1024};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 1024;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_big_kernel_aligned_c)
{
    TilingTestCase tc;
    tc.origInput = {1, 64, 64, 192};
    tc.origOutput = {1, 4, 8, 192};
    tc.grad = tc.origOutput;
    tc.outGrad = tc.origInput;
    tc.ksize = {1, 16, 8, 1};
    tc.strides = {1, 16, 8, 1};
    tc.paddingMode = "VALID";
    tc.dataFormat = "NHWC";
    tc.overrideCoreNum = 56;
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 1;
    tc.expectCdim = 192;
    tc.expectNhwcAlignedC = true;
    for (auto dt : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16,
                    ge::DT_INT32, ge::DT_INT64}) {
        DoTilingTest(dt, tc);
    }
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_big_kernel_streamed_aligned_c)
{
    TilingTestCase tc;
    tc.origInput = {1, 192, 192, 96};
    tc.origOutput = {1, 8, 8, 96};
    tc.grad = tc.origOutput;
    tc.outGrad = tc.origInput;
    tc.ksize = {1, 24, 24, 1};
    tc.strides = {1, 24, 24, 1};
    tc.paddingMode = "VALID";
    tc.dataFormat = "NHWC";
    tc.overrideCoreNum = 56;
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 1;
    tc.expectCdim = 96;
    tc.expectNhwcAlignedC = true;
    tc.expectNhwcStreamedWindow = true;
    DoTilingTest(ge::DT_FLOAT, tc);
    DoTilingTest(ge::DT_INT64, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_kernel1_stride1)
{
    // k=1 s=1: 窗口单点, SearchOuterSingleAligned 的 kernel==1 快路径(step=1)
    TilingTestCase tc;
    tc.origInput = {1, 64, 64, 4};
    tc.origOutput = {1, 64, 64, 4};
    tc.grad = {1, 64, 64, 4};
    tc.outGrad = {1, 64, 64, 4};
    tc.ksize = {1, 1, 1, 1};
    tc.strides = {1, 1, 1, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_stride1_overlap)
{
    // s=1 < k=3: 纯 overlap 无 pad -> SplitUnalignHWC 兜底(isCheckRange=1), hProBatchSize=3
    TilingTestCase tc;
    tc.origInput = {1, 64, 64, 8};
    tc.origOutput = {1, 62, 62, 8};
    tc.grad = {1, 62, 62, 8};
    tc.outGrad = {1, 64, 64, 8};
    tc.ksize = {1, 3, 3, 1};
    tc.strides = {1, 1, 1, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 8;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_try_split_n_binary_search)
{
    // N=65(>核数) 且轴对齐切分不满足: TrySplitN 首猜失败 -> nOutputInner=1 二分搜索,
    // 随后多核调整进入 SearchOuterSingleAligned 对齐 H/W 收缩回退路径
    TilingTestCase tc;
    tc.origInput = {65, 32, 32, 4};
    tc.origOutput = {65, 16, 16, 4};
    tc.grad = {65, 16, 16, 4};
    tc.outGrad = {65, 32, 32, 4};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_dynamic_adjust_w_shrink)
{
    // H=1 + pad + W>64: SplitUnalignHWC 循环走 DynamicAdjustmentHW 的 w 轴收缩分支
    TilingTestCase tc;
    tc.origInput = {1, 1, 100, 4};
    tc.origOutput = {1, 1, 50, 4};
    tc.grad = {1, 1, 50, 4};
    tc.outGrad = {1, 1, 100, 4};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.paddingMode = "CALCULATED";
    tc.pads = {1, 1, 0, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    tc.expectPadH = 1;
    tc.expectPadHBack = 1;
    tc.expectPadW = 0;
    tc.expectPadWBack = 1;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_empty_dim_fallback_simt)
{
    // 空间维为 0: NHWC IsCapable 拒绝 -> SIMT 兜底(NHWC 格式 key 274)
    TilingTestCase tc;
    tc.origInput = {1, 0, 4, 4};
    tc.origOutput = {1, 0, 2, 4};
    tc.grad = {1, 0, 2, 4};
    tc.outGrad = {1, 0, 4, 4};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 274;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_batch_limit_fallback_simt)
{
    // k=255 s=1 -> hPro*wPro=65025 >= 256: IsCapable batch 限制 -> SIMT 兜底
    TilingTestCase tc;
    tc.origInput = {1, 300, 300, 2};
    tc.origOutput = {1, 46, 46, 2};
    tc.grad = {1, 46, 46, 2};
    tc.outGrad = {1, 300, 300, 2};
    tc.ksize = {1, 255, 255, 1};
    tc.strides = {1, 1, 1, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 274;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_int32_overflow_fallback_simt)
{
    // inputDataCount > INT32_MAX: IsCapable 拒绝 -> SIMT 兜底
    TilingTestCase tc;
    tc.origInput = {2, 3, 65536, 8192};
    tc.origOutput = {2, 1, 32768, 8192};
    tc.grad = {2, 1, 32768, 8192};
    tc.outGrad = {2, 3, 65536, 8192};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 274;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_ksize_length2)
{
    TilingTestCase tc;
    tc.ksize = {2, 2};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_strides_length2)
{
    TilingTestCase tc;
    tc.strides = {2, 2};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_align_adjust_kernel1)
{
    // hOuter=65 略超核数: 多核调整进入对齐分支, kernel==1 走 SearchOuterSingle(step=1) 快路径
    TilingTestCase tc;
    tc.origInput = {1, 130, 64, 4};
    tc.origOutput = {1, 130, 64, 4};
    tc.grad = {1, 130, 64, 4};
    tc.outGrad = {1, 130, 64, 4};
    tc.ksize = {1, 1, 1, 1};
    tc.strides = {1, 1, 1, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_align_adjust_unaligned_h)
{
    // TrySplitN 胜出 + hX=33 未按 stride 对齐: SearchOuterSingleAligned 先向下对齐 stride 网格
    TilingTestCase tc;
    tc.origInput = {65, 33, 32, 4};
    tc.origOutput = {65, 16, 16, 4};
    tc.grad = {65, 16, 16, 4};
    tc.outGrad = {65, 33, 32, 4};
    tc.ksize = {1, 2, 2, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 0;
    tc.expectCdim = 4;
    DoTilingTest(ge::DT_FLOAT, tc);
}

// ==================== 缺陷修复回归(板上 VEC_ERROR 实证, 见验证报告) ====================

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_fix_bigk_overlap_ub)
{
    // 修复回归: 原 SplitUnalignHWC 在 hInner==1 且 wInner<=proDataNumInOneBeat 时退出但 UB 未复检,
    // bigK 大 kH/小 sH 场景(kH=127, sH=1, hIn=CeilDiv(kH,sH)=127)下发的 argmax 缓冲远超可用 UB
    // (修复前 argmaxBufferSize=2113280B, 板上 VEC_ERROR)。修复后 W 轴继续收缩至 UB 满足。
    TilingTestCase tc;
    tc.origInput = {1, 130, 64, 64};
    tc.origOutput = {1, 4, 63, 64};
    tc.grad = {1, 4, 63, 64};
    tc.outGrad = {1, 130, 64, 64};
    tc.ksize = {1, 127, 2, 1};
    tc.strides = {1, 1, 1, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 1;
    tc.expectCdim = 64;
    tc.expectBufferSizeLE = 262144 - 5120;
    DoTilingTest(ge::DT_BF16, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_fix_alignc_cinner_overwrite)
{
    // 修复回归: 原 TrySplitAlignC 二分全败时无条件回写 bestSplit(默认1)*moveDataNumCacheLine,
    // cX < moveDataNumCacheLine 时 cInner 被抬到超过 cX(32 > 4, 板上 VEC_ERROR)。
    // 无成功候选时保留入口已通过校验的最小取值；大窗口也可以使用该对齐路径。
    TilingTestCase tc;
    tc.origInput = {1, 504, 504, 4};
    tc.origOutput = {1, 8, 8, 4};
    tc.grad = {1, 8, 8, 4};
    tc.outGrad = {1, 504, 504, 4};
    tc.ksize = {1, 63, 63, 1};
    tc.strides = {1, 63, 63, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 1;
    tc.expectCdim = 4;
    tc.expectBufferSizeLE = 262144 - 5120;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_fix_alignc_cinner_overwrite_f16)
{
    // 同上, fp16: moveDataNumCacheLine=64, cX=4 < 64 场景
    TilingTestCase tc;
    tc.origInput = {1, 504, 504, 4};
    tc.origOutput = {1, 8, 8, 4};
    tc.grad = {1, 8, 8, 4};
    tc.outGrad = {1, 504, 504, 4};
    tc.ksize = {1, 63, 63, 1};
    tc.strides = {1, 63, 63, 1};
    tc.dataFormat = "NHWC";
    tc.expectTilingKey = 275;
    tc.expectIsBigKernel = 1;
    tc.expectCdim = 4;
    tc.expectBufferSizeLE = 262144 - 5120;
    DoTilingTest(ge::DT_FLOAT16, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_ceil_partial_window)
{
    // ceil retains a last window at (2, 2), extending past H/W=4 by one.
    // Its trailing implicit padding must reserve the padded input buffer.
    TilingTestCase tc;
    tc.origInput = {1, 4, 4, 3};
    tc.origOutput = {1, 2, 2, 3};
    tc.grad = tc.origOutput;
    tc.outGrad = tc.origInput;
    tc.ksize = {1, 3, 3, 1};
    tc.strides = {1, 2, 2, 1};
    tc.dataFormat = "NHWC";
    tc.ceilMode = true;
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectPadH = 0;
    tc.expectPadHBack = 1;
    tc.expectPadW = 0;
    tc.expectPadWBack = 1;
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        DoTilingTest(dtype, tc);
    }
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nhwc_valid_ignores_ceil)
{
    TilingTestCase tc;
    tc.origInput = {1, 4, 4, 3};
    tc.origOutput = {1, 1, 1, 3};
    tc.grad = tc.origOutput;
    tc.outGrad = tc.origInput;
    tc.ksize = {1, 3, 3, 1};
    tc.strides = {1, 2, 2, 1};
    tc.paddingMode = "VALID";
    tc.dataFormat = "NHWC";
    tc.ceilMode = true;
    tc.expectTilingKey = 787;
    tc.expectIsBigKernel = 0;
    tc.expectPadHBack = 0;
    tc.expectPadWBack = 0;
    DoTilingTest(ge::DT_FLOAT, tc);
}

TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_supported_dtypes)
{
    const std::vector<ge::DataType> dtypes = {ge::DT_BF16,   ge::DT_INT8,  ge::DT_UINT8, ge::DT_INT16,
                                              ge::DT_UINT16, ge::DT_INT32, ge::DT_INT64};
    for (auto dt : dtypes) {
        SCOPED_TRACE(testing::Message() << "dtype=" << static_cast<int32_t>(dt));
        TilingTestCase tc;
        tc.checkNchwTiling = true;
        tc.expectIsPad = 0;
        DoTilingTest(dt, tc);
    }
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_same_pad)
{
    TilingTestCase tc;
    tc.origInput = {64, 1, 8, 8};
    tc.origOutput = {64, 1, 4, 4};
    tc.grad = {64, 1, 4, 4};
    tc.outGrad = {64, 1, 8, 8};
    tc.ksize = {1, 1, 3, 3};
    tc.strides = {1, 1, 2, 2};
    tc.paddingMode = "SAME";
    // Whole H/W tiles keep every selected index in range even with SAME padding.
    tc.expectTilingKey = 256;
    tc.checkNchwTiling = true;
    tc.expectIsPad = 1;
    tc.expectWholeSpatialTile = true;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_overlap_check_range)
{
    TilingTestCase tc;
    tc.origInput = {1, 1, 7, 7};
    tc.origOutput = {1, 1, 3, 3};
    tc.grad = {1, 1, 3, 3};
    tc.outGrad = {1, 1, 7, 7};
    tc.ksize = {1, 1, 3, 3};
    tc.strides = {1, 1, 2, 2};
    tc.paddingMode = "VALID";
    tc.expectTilingKey = 768;
    tc.overrideCoreNum = 64;
    tc.checkNchwTiling = true;
    tc.expectIsPad = 0;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_pad_check_range)
{
    TilingTestCase tc;
    tc.origInput = {1, 1, 8, 8};
    tc.origOutput = {1, 1, 4, 4};
    tc.grad = {1, 1, 4, 4};
    tc.outGrad = {1, 1, 8, 8};
    tc.ksize = {1, 1, 3, 3};
    tc.strides = {1, 1, 2, 2};
    tc.paddingMode = "SAME";
    tc.expectTilingKey = 768;
    tc.overrideCoreNum = 64;
    tc.checkNchwTiling = true;
    tc.expectIsPad = 1;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_big_kernel_large_kernel)
{
    // kv=256 >= KSIZE_THRESHOLD(128)，走 NCHW big-kernel；非对齐兜底切分 isCheckRange=1
    TilingTestCase tc;
    tc.origInput = {1, 1, 64, 64};
    tc.origOutput = {1, 1, 4, 4};
    tc.grad = {1, 1, 4, 4};
    tc.outGrad = {1, 1, 64, 64};
    tc.ksize = {1, 1, 16, 16};
    tc.strides = {1, 1, 16, 16};
    tc.paddingMode = "VALID";
    tc.expectTilingKey = 769;
    tc.checkNchwTiling = true;
    tc.expectIsPad = 0;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_big_kernel_byte_gather_limit)
{
    // This fits UB with eleven NC planes, but its byte offsets exceed uint16_t.
    // Both signed and unsigned byte kernels must split before packing indices.
    TilingTestCase tc;
    tc.origInput = {6, 287, 86, 95};
    tc.origOutput = {6, 287, 3, 2};
    tc.grad = tc.origOutput;
    tc.outGrad = tc.origInput;
    tc.ksize = {1, 1, 30, 30};
    tc.strides = {1, 1, 42, 52};
    tc.pads = {8, 20, 0, 21};
    tc.ceilMode = true;
    tc.expectTilingKey = 257;
    tc.checkNchwTiling = true;
    tc.expectWholeSpatialTile = true;
    tc.expectNchwGatherBufferSizeLE = 65536;
    tc.overrideCoreNum = 56;
    DoTilingTest(ge::DT_INT8, tc);
    DoTilingTest(ge::DT_UINT8, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_nchw_small_kernel_byte_output_gather_limit)
{
    // Large strides make the output tile exceed the forward input buffer size.
    TilingTestCase tc;
    tc.origInput = {10, 397, 41, 145};
    tc.origOutput = {10, 397, 1, 4};
    tc.grad = tc.origOutput;
    tc.outGrad = tc.origInput;
    tc.ksize = {1, 1, 5, 8};
    tc.strides = {1, 1, 48, 42};
    tc.pads = {2, 1, 1, 7};
    tc.ceilMode = true;
    tc.expectTilingKey = 256;
    tc.checkNchwTiling = true;
    tc.expectWholeSpatialTile = true;
    tc.expectNchwGatherBufferSizeLE = 65536;
    tc.overrideCoreNum = 56;
    DoTilingTest(ge::DT_INT8, tc);
    DoTilingTest(ge::DT_UINT8, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_data_format)
{
    TilingTestCase tc;
    tc.dataFormat = "ND";
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_padding_mode)
{
    TilingTestCase tc;
    tc.paddingMode = "INVALID";
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_stride_nc)
{
    TilingTestCase tc;
    tc.strides = {2, 1, 2, 2};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_stride_range)
{
    TilingTestCase tc;
    tc.strides = {1, 1, 64, 2};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_ksize_nc)
{
    TilingTestCase tc;
    tc.ksize = {2, 1, 2, 2};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_ksize_range)
{
    TilingTestCase tc;
    tc.ksize = {1, 1, 256, 2};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_pads_negative)
{
    TilingTestCase tc;
    tc.paddingMode = "CALCULATED";
    tc.pads = {-1, 0, 0, 0};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_pads_ge_ksize)
{
    TilingTestCase tc;
    tc.paddingMode = "CALCULATED";
    tc.pads = {2, 0, 0, 0};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_grad_shape_mismatch)
{
    TilingTestCase tc;
    tc.grad = {1, 1, 2, 3};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_out_grad_shape_mismatch)
{
    TilingTestCase tc;
    tc.outGrad = {1, 1, 4, 3};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_negative_dim)
{
    TilingTestCase tc;
    tc.origInput = {1, 1, -1, 4};
    tc.outGrad = {1, 1, -1, 4};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_3d_input)
{
    TilingTestCase tc;
    tc.origInput = {1, 4, 4};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
TEST_F(MaxPoolV3GradTilingTest, max_pool_v3_grad_tiling_fail_output_spatial_mismatch)
{
    TilingTestCase tc;
    tc.origOutput = {1, 1, 3, 3};
    tc.grad = {1, 1, 3, 3};
    tc.expectResult = ge::GRAPH_FAILED;
    DoTilingTest(ge::DT_FLOAT, tc);
}
