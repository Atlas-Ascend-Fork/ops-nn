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
 * \file test_max_pool3d_grad_ndhwc_tiling.cpp
 * \brief NDHWC small/big kernel tiling UT：
 *  1) 缓冲口径回归（inputBufferSize 期望口径、argmax/gradBufferSize 重算数口径）
 *  2) isCheckRange 在 pad 路径置 1（tiling key 位）
 *  3) bigKernel（kv >= 128）分派
 */

#include <gtest/gtest.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../../../op_host/arch35/max_pool3d_grad_tiling.h"
#include "test_max_pool3d_grad_partition_verify.h"

using namespace ge;
using namespace ut_util;
using namespace Pool3DGradNameSpace;

namespace {

const char* const COMPILE_INFO_950 = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 64}})";

struct NdhwcTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    Pool3DGradNDHWCTilingData td{};
};

struct NcdhwTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    Pool3DGradNCDHWTilingData td{};
};

void RunNdhwcTiling(const gert::StorageShape& xShape, const gert::StorageShape& yInShape,
                    const gert::StorageShape& gradShape, ge::DataType dtype, const std::vector<int64_t>& ksize,
                    const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& pads,
                    NdhwcTilingResult& result)
{
    gert::StorageShape xShapeVar = xShape;
    gert::StorageShape yInShapeVar = yInShape;
    gert::StorageShape gradShapeVar = gradShape;
    gert::StorageShape yShapeVar = xShape;

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_950, socInfos, aicoreSpec, intrinsics);
    std::map<std::string, std::string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::Tiling4Pool3DGradCompileInfo compileInfo;

    std::string opType("MaxPool3DGrad");
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFunc, nullptr);
    ASSERT_NE(tilingParseFunc, nullptr);

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(COMPILE_INFO_950), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShapeVar, &yInShapeVar, &gradShapeVar})
                      .OutputShapes({&yShapeVar})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dtype, ge::FORMAT_NDHWC, ge::FORMAT_RESERVED)
                      .NodeInputTd(1, dtype, ge::FORMAT_NDHWC, ge::FORMAT_RESERVED)
                      .NodeInputTd(2, dtype, ge::FORMAT_NDHWC, ge::FORMAT_RESERVED)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NDHWC, ge::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NDHWC")}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);

    result.status = tilingFunc(tilingContext);
    result.tilingKey = tilingContext->GetTilingKey();
    auto raw = tilingContext->GetRawTilingData();
    if (raw != nullptr && raw->GetDataSize() >= sizeof(Pool3DGradNDHWCTilingData)) {
        errno_t ret = memcpy_s(&result.td, sizeof(Pool3DGradNDHWCTilingData), raw->GetData(),
                               sizeof(Pool3DGradNDHWCTilingData));
        EXPECT_EQ(ret, EOK);
    }
}

// NCDHW 版 runner（结构体为 Pool3DGradNCDHWTilingData，shape 顺序 [N,C,D,H,W]）
void RunNcdhwTiling(const gert::StorageShape& xShape, const gert::StorageShape& yInShape,
                    const gert::StorageShape& gradShape, ge::DataType dtype, const std::vector<int64_t>& ksize,
                    const std::vector<int64_t>& strides, const std::string& padding, const std::vector<int64_t>& pads,
                    NcdhwTilingResult& result)
{
    gert::StorageShape xShapeVar = xShape;
    gert::StorageShape yInShapeVar = yInShape;
    gert::StorageShape gradShapeVar = gradShape;
    gert::StorageShape yShapeVar = xShape;

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_950, socInfos, aicoreSpec, intrinsics);
    std::map<std::string, std::string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::Tiling4Pool3DGradCompileInfo compileInfo;

    std::string opType("MaxPool3DGrad");
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFunc, nullptr);
    ASSERT_NE(tilingParseFunc, nullptr);

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(COMPILE_INFO_950), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShapeVar, &yInShapeVar, &gradShapeVar})
                      .OutputShapes({&yShapeVar})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                      .NodeInputTd(1, dtype, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                      .NodeInputTd(2, dtype, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);

    result.status = tilingFunc(tilingContext);
    result.tilingKey = tilingContext->GetTilingKey();
    auto raw = tilingContext->GetRawTilingData();
    if (raw != nullptr && raw->GetDataSize() >= sizeof(Pool3DGradNCDHWTilingData)) {
        errno_t ret = memcpy_s(&result.td, sizeof(Pool3DGradNCDHWTilingData), raw->GetData(),
                               sizeof(Pool3DGradNCDHWTilingData));
        EXPECT_EQ(ret, EOK);
    }
}

class MaxPool3DGradNdhwcTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPool3DGradNdhwcTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MaxPool3DGradNdhwcTiling TearDown" << std::endl; }
};

// VALID 无 pad：NDHWC 小 kernel 分派 + 期望/重算数缓冲口径回归
TEST_F(MaxPool3DGradNdhwcTiling, ndhwc_valid_fp32_buffer_and_dispatch)
{
    gert::StorageShape xShape = {{2, 4, 8, 8, 16}, {2, 4, 8, 8, 16}};
    gert::StorageShape yInShape = {{2, 2, 4, 4, 16}, {2, 2, 4, 4, 16}};
    gert::StorageShape gradShape = {{2, 2, 4, 4, 16}, {2, 2, 4, 4, 16}};
    NdhwcTilingResult r;
    RunNdhwcTiling(xShape, yInShape, gradShape, ge::DT_FLOAT, {1, 2, 2, 2, 1}, {1, 2, 2, 2, 1}, "VALID",
                   {0, 0, 0, 0, 0, 0}, r);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.td.isBigKernel, 0);

    // inputBufferSize 期望口径：(N + (k-1)*dil*2) 上界，k=2/dil=1 → inner + 2
    const int64_t cAligned = (r.td.cOutputInner + 7) / 8 * 8;
    const int64_t expectInput = r.td.base.highAxisInner * (r.td.base.dOutputInner + 2) * (r.td.base.hOutputInner + 2) *
                                (r.td.base.wOutputInner + 2) * cAligned * 4;
    EXPECT_EQ(r.td.base.inputBufferSize, expectInput);

    // argmax/gradBufferSize 重算数口径：d' = CeilDiv(dOutputInner + kD - 1, dStride)
    const int64_t dIn = (r.td.base.dOutputInner + 2) / 2;
    const int64_t hIn = (r.td.base.hOutputInner + 2) / 2;
    const int64_t wIn = (r.td.base.wOutputInner + 2) / 2;
    const int64_t expectArgmax = r.td.base.highAxisInner * r.td.cOutputInner * dIn * hIn * wIn * 4;
    EXPECT_EQ(r.td.base.argmaxBufferSize, expectArgmax);
    const int64_t expectGrad = r.td.base.highAxisInner * dIn * hIn * wIn * cAligned * 4;
    EXPECT_EQ(r.td.base.gradBufferSize, expectGrad);

    // tiling key：NDHWC 小 kernel（isChannelLast=1），isCheckRange 视对齐路径为 0/1
    const uint64_t keyCk0 = GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 0, 0);
    const uint64_t keyCk1 = GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0);
    EXPECT_TRUE(r.tilingKey == keyCk0 || r.tilingKey == keyCk1);
}

// pad 场景：走 SplitUnalignDHWC → isCheckRange=1
TEST_F(MaxPool3DGradNdhwcTiling, ndhwc_pad_calculated_is_check_range)
{
    gert::StorageShape xShape = {{2, 8, 8, 8, 16}, {2, 8, 8, 8, 16}};
    gert::StorageShape yInShape = {{2, 5, 5, 5, 16}, {2, 5, 5, 5, 16}};
    gert::StorageShape gradShape = {{2, 5, 5, 5, 16}, {2, 5, 5, 5, 16}};
    NdhwcTilingResult r;
    RunNdhwcTiling(xShape, yInShape, gradShape, ge::DT_FLOAT, {1, 2, 2, 2, 1}, {1, 2, 2, 2, 1}, "CALCULATED",
                   {1, 1, 1, 1, 1, 1}, r);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.td.isBigKernel, 0);
    EXPECT_EQ(r.td.base.padD, 1);
    EXPECT_EQ(r.td.base.padH, 1);
    EXPECT_EQ(r.td.base.padW, 1);

    // pad 路径 isCheckRange 恒为 1
    const uint64_t keyCk1 = GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0);
    EXPECT_EQ(r.tilingKey, keyCk1);

    // pad 路径仍为期望口径
    const int64_t cAligned = (r.td.cOutputInner + 7) / 8 * 8;
    const int64_t expectInput = r.td.base.highAxisInner * (r.td.base.dOutputInner + 2) * (r.td.base.hOutputInner + 2) *
                                (r.td.base.wOutputInner + 2) * cAligned * 4;
    EXPECT_EQ(r.td.base.inputBufferSize, expectInput);
}

// kv >= 128 且非大C：bigKernel 分派 + 单窗口 input 缓冲口径
TEST_F(MaxPool3DGradNdhwcTiling, ndhwc_big_kernel_dispatch)
{
    gert::StorageShape xShape = {{1, 16, 16, 4, 16}, {1, 16, 16, 4, 16}};
    gert::StorageShape yInShape = {{1, 5, 5, 2, 16}, {1, 5, 5, 2, 16}};
    gert::StorageShape gradShape = {{1, 5, 5, 2, 16}, {1, 5, 5, 2, 16}};
    NdhwcTilingResult r;
    RunNdhwcTiling(xShape, yInShape, gradShape, ge::DT_FLOAT, {1, 8, 8, 2, 1}, {1, 2, 2, 2, 1}, "VALID",
                   {0, 0, 0, 0, 0, 0}, r);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.td.isBigKernel, 1);

    // bigKernel 单窗口口径: inputBufferSize = kv×cAligned×4 = 8*8*2*cAligned*4
    const int64_t cAligned = (r.td.cOutputInner + 7) / 8 * 8;
    EXPECT_EQ(r.td.base.inputBufferSize, 8 * 8 * 2 * cAligned * 4);

    // argmax 仍为重算数口径：d' = CeilDiv(dOutputInner + kD - 1, dStride) = CeilDiv(dOutIn + 7, 2)
    const int64_t dIn = (r.td.base.dOutputInner + 8) / 2;
    const int64_t hIn = (r.td.base.hOutputInner + 8) / 2;
    const int64_t wIn = (r.td.base.wOutputInner + 2) / 2;
    const int64_t expectArgmax = r.td.base.highAxisInner * r.td.cOutputInner * dIn * hIn * wIn * 4;
    EXPECT_EQ(r.td.base.argmaxBufferSize, expectArgmax);
}

} // namespace

// kv >= 128 且无重叠（k == s）：P0-44 后大 kernel tiling 认领非重叠形状（大 kernel
// 数学对 k<=s 天然成立, g1 系列 silicon 已验证）→ isBigKernel=1 + 单窗口 input 口径。
// 原断言（小 kernel 兜底 + 整 tile 口径）基于 P0-44 前的 overlap-only 门控, 随大 kernel
// tiling 移植同步更新; 小 kernel 整 tile 口径公式回归由 ndhwc_valid_fp32 覆盖
TEST_F(MaxPool3DGradNdhwcTiling, ndhwc_nonoverlap_big_kv_big_kernel_dispatch)
{
    // N=1, D=H=W=8, C=16, k=s=(4,4,8)：kv=128 无重叠，cX*4=64 <= vReg/2
    gert::StorageShape xShape = {{1, 8, 8, 8, 16}, {1, 8, 8, 8, 16}};
    gert::StorageShape yInShape = {{1, 2, 2, 1, 16}, {1, 2, 2, 1, 16}};
    gert::StorageShape gradShape = {{1, 2, 2, 1, 16}, {1, 2, 2, 1, 16}};
    NdhwcTilingResult r;
    RunNdhwcTiling(xShape, yInShape, gradShape, ge::DT_FLOAT, {1, 4, 4, 8, 1}, {1, 4, 4, 8, 1}, "VALID",
                   {0, 0, 0, 0, 0, 0}, r);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.td.isBigKernel, 1);

    // bigKernel 单窗口口径: inputBufferSize = kv×cAligned×4 = 4*4*8*cAligned*4
    const int64_t cAligned = (r.td.cOutputInner + 7) / 8 * 8;
    EXPECT_EQ(r.td.base.inputBufferSize, 4 * 4 * 8 * cAligned * 4);
}

// NCDHW kv >= 128 回归：isBigKernel=1 下发（NCDHW 大 kernel 前向已实现）,
// inputBufferSize 按单窗口口径 kv × align(wTile+(kW-1)*2, 8) × 4 预算
TEST_F(MaxPool3DGradNdhwcTiling, ncdhw_big_kv_big_kernel_buffer)
{
    // N=1, C=16, D=H=W=8, k=s=(4,4,8)：kv=128；shape 顺序 [N,C,D,H,W]
    gert::StorageShape xShape = {{1, 16, 8, 8, 8}, {1, 16, 8, 8, 8}};
    gert::StorageShape yInShape = {{1, 16, 2, 2, 1}, {1, 16, 2, 2, 1}};
    gert::StorageShape gradShape = {{1, 16, 2, 2, 1}, {1, 16, 2, 2, 1}};
    NcdhwTilingResult r;
    RunNcdhwTiling(xShape, yInShape, gradShape, ge::DT_FLOAT, {1, 1, 4, 4, 8}, {1, 1, 4, 4, 8}, "VALID",
                   {0, 0, 0, 0, 0, 0}, r);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.td.isBigKernel, 1);

    // NCDHW 大 kernel 对齐装载口径: input = dK × CeilAlign(hK×wK, 32B/元素宽) × 4
    // (NoSplit 判据同口径; 超前向预算时按预算封顶, 由 kernel 侧分块处理)
    const int64_t hwAligned = (4 * 8 + 7) / 8 * 8;
    EXPECT_EQ(r.td.inputBufferSize, 4 * hwAligned * 4);
}

// ==================== 多核优化验证: 效果回归 + 划分语义 ====================

// 多核调整效果回归(64 核平台): usedCoreNum 不低于优化前基线, normalCoreProcessNum 不劣化,
// tiling key 不变, 块数守恒
TEST_F(MaxPool3DGradNdhwcTiling, ndhwc_multicore_effect_sweep)
{
    struct EffectCase {
        const char* tag;
        gert::StorageShape xShape;
        gert::StorageShape yShape;
        std::vector<int64_t> ksize;
        std::vector<int64_t> strides;
        std::string padding;
        std::vector<int64_t> pads;
        ge::DataType dtype;
        uint64_t baseKey;
        int64_t baseUsed;
        int64_t baseNormal;
    };
    const std::vector<EffectCase> cases = {
        {"big_kernel_h_4_to_3",
         {{1, 12, 12, 12, 32}, {1, 12, 12, 12, 32}},
         {{1, 2, 2, 2, 32}, {1, 2, 2, 2, 32}},
         {1, 6, 6, 6, 1},
         {1, 6, 6, 6, 1},
         "CALCULATED",
         {0, 0, 0, 0, 0, 0},
         ge::DT_FLOAT16,
         GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0),
         36,
         2},
        {"valid_already_full",
         {{2, 4, 8, 8, 16}, {2, 4, 8, 8, 16}},
         {{2, 2, 4, 4, 16}, {2, 2, 4, 4, 16}},
         {1, 2, 2, 2, 1},
         {1, 2, 2, 2, 1},
         "VALID",
         {0, 0, 0, 0, 0, 0},
         ge::DT_FLOAT,
         GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 0, 0),
         64,
         2},
        {"pad_already_full",
         {{2, 8, 8, 8, 16}, {2, 8, 8, 8, 16}},
         {{2, 5, 5, 5, 16}, {2, 5, 5, 5, 16}},
         {1, 2, 2, 2, 1},
         {1, 2, 2, 2, 1},
         "CALCULATED",
         {1, 1, 1, 1, 1, 1},
         ge::DT_FLOAT,
         GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0),
         64,
         1},
    };
    for (const auto& ec : cases) {
        NdhwcTilingResult r;
        RunNdhwcTiling(ec.xShape, ec.yShape, ec.yShape, ec.dtype, ec.ksize, ec.strides, ec.padding, ec.pads, r);
        ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(r.tilingKey, ec.baseKey) << ec.tag;
        EXPECT_GE(r.td.base.usedCoreNum, ec.baseUsed) << ec.tag;
        EXPECT_LE(r.td.base.normalCoreProcessNum, ec.baseNormal) << ec.tag;
        const int64_t totalBlocks = r.td.base.highAxisOuter * r.td.cOutputOuter * r.td.base.dOutputOuter *
                                    r.td.base.hOutputOuter * r.td.base.wOutputOuter;
        EXPECT_EQ(r.td.base.normalCoreProcessNum * (r.td.base.usedCoreNum - 1) + r.td.base.tailCoreProcessNum,
                  totalBlocks)
            << ec.tag;
        std::cout << "[multicore-effect][ndhwc][" << ec.tag << "] usedCoreNum " << ec.baseUsed << " -> "
                  << r.td.base.usedCoreNum << "/64" << std::endl;
    }
}

// 划分语义验证(纯 host): 复刻 kernel 侧 PStart/PEnd/窗口公式, 对 d/h/w 轴穷举
// (grad 位置 p, 窗口内位置 a), 断言恰好一个 tile 同时覆盖; isCheckRange=0 时窗口完整落在 tile 内
TEST_F(MaxPool3DGradNdhwcTiling, ndhwc_partition_semantics)
{
    struct SemCase {
        const char* tag;
        gert::StorageShape xShape;
        gert::StorageShape yShape;
        std::vector<int64_t> ksize;
        std::vector<int64_t> strides;
        std::string padding;
        std::vector<int64_t> pads;
        ge::DataType dtype;
    };
    const std::vector<SemCase> cases = {
        {"big_kernel_h_4_to_3",
         {{1, 12, 12, 12, 32}, {1, 12, 12, 12, 32}},
         {{1, 2, 2, 2, 32}, {1, 2, 2, 2, 32}},
         {1, 6, 6, 6, 1},
         {1, 6, 6, 6, 1},
         "CALCULATED",
         {0, 0, 0, 0, 0, 0},
         ge::DT_FLOAT16},
        {"valid_align_dhw",
         {{2, 4, 8, 8, 16}, {2, 4, 8, 8, 16}},
         {{2, 2, 4, 4, 16}, {2, 2, 4, 4, 16}},
         {1, 2, 2, 2, 1},
         {1, 2, 2, 2, 1},
         "VALID",
         {0, 0, 0, 0, 0, 0},
         ge::DT_FLOAT},
        {"pad_unalign_dhw",
         {{2, 8, 8, 8, 16}, {2, 8, 8, 8, 16}},
         {{2, 5, 5, 5, 16}, {2, 5, 5, 5, 16}},
         {1, 2, 2, 2, 1},
         {1, 2, 2, 2, 1},
         "CALCULATED",
         {1, 1, 1, 1, 1, 1},
         ge::DT_FLOAT},
        {"nonoverlap_big_kv",
         {{1, 8, 8, 8, 16}, {1, 8, 8, 8, 16}},
         {{1, 2, 2, 1, 16}, {1, 2, 2, 1, 16}},
         {1, 4, 4, 8, 1},
         {1, 4, 4, 8, 1},
         "VALID",
         {0, 0, 0, 0, 0, 0},
         ge::DT_FLOAT},
    };
    const uint64_t keyCk0 = GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 0, 0);
    for (const auto& sc : cases) {
        NdhwcTilingResult r;
        RunNdhwcTiling(sc.xShape, sc.yShape, sc.yShape, sc.dtype, sc.ksize, sc.strides, sc.padding, sc.pads, r);
        ASSERT_EQ(r.status, ge::GRAPH_SUCCESS) << sc.tag;
        const bool checkRange = (r.tilingKey != keyCk0);
        const auto& b = r.td.base;
        EXPECT_EQ(b.dilationD, 1);
        EXPECT_EQ(b.dilationH, 1);
        EXPECT_EQ(b.dilationW, 1);
        mp3dg_partition_verify::VerifyAxisPartition(b.dOutput, b.dArgmax, b.dKernel, b.dStride, b.padD, b.dOutputInner,
                                                    b.dOutputTail, b.dOutputOuter, !checkRange);
        mp3dg_partition_verify::VerifyAxisPartition(b.hOutput, b.hArgmax, b.hKernel, b.hStride, b.padH, b.hOutputInner,
                                                    b.hOutputTail, b.hOutputOuter, !checkRange);
        mp3dg_partition_verify::VerifyAxisPartition(b.wOutput, b.wArgmax, b.wKernel, b.wStride, b.padW, b.wOutputInner,
                                                    b.wOutputTail, b.wOutputOuter, !checkRange);
        std::cout << "[partition-semantics][ndhwc][" << sc.tag << "] inner(d,h,w)=(" << b.dOutputInner << ","
                  << b.hOutputInner << "," << b.wOutputInner << ") checkRange=" << checkRange << " PASS" << std::endl;
    }
}
