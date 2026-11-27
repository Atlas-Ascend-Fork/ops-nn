/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_case_runner.h
 * \brief LpNormReduceV2 TilingFunc UT 用例执行器（主 tiling 与 crosscore tiling 用例共用，
 *        经 ops-nn 共享 Host UT runner 编入同一测试二进制）
 *
 * TilingCaseParam + SetUpPlatformInfo + RunTilingCase：
 *   - fake 平台信息（SoCInfo/AICoreSpec/version 三组 map，950PR/DT 仿真口径）
 *   - 断言：tilingFunc 返回值 / workspace 字节数 / blockDim / TilingKey / 全部 TilingData 字段
 *   - 边界反向校验：单 tile UB 预算不溢出（§5.3 BLK-2 闭式反向推导，含 fixedExtra）
 *
 * TilingKey 位布局（GET_TPL_TILING_KEY 编码，template_argument.h）：
 *   key = dtype | CROSS_CORE<<8 | LAYOUT<<16 | BUFFER_MODE<<24
 *   dtype 值 = DT_FLOAT(0)/DT_FLOAT16(1)/DT_BF16(27)
 */

#ifndef LP_NORM_REDUCE_V2_TILING_CASE_RUNNER_H
#define LP_NORM_REDUCE_V2_TILING_CASE_RUNNER_H

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "tiling_context_faker.h"
#include "op_impl_registry.h"
#include "platform/platform_infos_def.h"
// 相对路径回溯至算子根目录的 op_kernel/arch35（共享 UT runner 的固定 include 路径不含本算子目录）
#include "../../../../op_kernel/arch35/lp_norm_reduce_v2_tiling_data.h"

// TilingParse<LpNormReduceV2CompileInfo> 编译信息占位（与 op_host/arch35 tiling.cpp 定义一致，空结构）
namespace optiling {
struct LpNormReduceV2CompileInfo {};
} // namespace optiling

namespace lp_ut {

// constexpr 内链接：本头文件被主 tiling 与 crosscore 两个用例 TU 共同包含，避免跨 TU 重定义
constexpr const char* OP_TYPE = "LpNormReduceV2";
// 950PR/DT 仿真平台参数（DAV_3510 典型 UB 248KB，AIV 几十核量级取 64）
constexpr uint64_t UT_DEFAULT_UB_SIZE = 253952UL; // 248KB
constexpr uint64_t UT_DEFAULT_CORE_NUM = 64UL;

struct TilingCaseParam {
    std::string caseName;
    std::vector<int64_t> xShape;
    ge::DataType dtype;
    float p;
    std::vector<int64_t> axes;
    bool keepdim;
    uint64_t coreNum = UT_DEFAULT_CORE_NUM;
    uint64_t ubSize = UT_DEFAULT_UB_SIZE;
    ge::graphStatus expectStatus;
    // 期望 TilingData（expectStatus == GRAPH_SUCCESS 时校验）
    int64_t bLength = 0;
    int64_t rLength = 0;
    int64_t cLength = 1;
    int64_t coreNumUsed = 1;
    int64_t outBlockNum = 1;
    int64_t rGroupNum = 1;
    int64_t bFactor = 0;
    int64_t rowsTile = 1;
    int64_t bTile = 1;
    int64_t ubFactorR = 1;
    int64_t ubFactorC = 1;
    int64_t rAlignedCols = 1;
    int64_t rFactorPerGroup = 0;
    int32_t pBranch = LPNORM_P_BRANCH_INT_POW;
    int32_t powTimes = 1;
    int32_t zeroOutputFlag = 0;
    int32_t segNum = 1;  // 期望 R 段数（P1=1 默认；0=全 A 退化；≥2=P2 段表，迭代三）
    int32_t aSegNum = 0; // P2：期望 A 段数
    uint64_t expectTilingKey = 0;
    uint32_t expectBlockDim = 1;
    uint64_t expectWsBytes = 0; // workspace 期望字节数（CROSS_CORE=0 → 0；=1 → outBlockNum×rGroupNum×slot）
    // 病态平台（UB 小于单块）用例关闭"单 tile ≤ ubAvail"预算断言（防御钳位用例专用，默认开）
    bool checkUbBudget = true;
    // P2 段表逐字段期望（迭代三 A2）：非空时按 segNum/aSegNum 逐字段断言 segSize/segStride/aSize/aStride，
    // 并校验尾部槽位清零（memset 完整性）；空 = 不校验（遗留用例兼容）
    std::vector<int64_t> expectSegSize;
    std::vector<int64_t> expectSegStride;
    std::vector<int64_t> expectASize;
    std::vector<int64_t> expectAStride;
    // workspace 注入开关（false：不注入 → GetWorkspaceSizes 返回空 → DeclareWorkspace 防御分支，
    // 覆盖 O==0/R==0/P2 三条快速路径的 workspace 声明失败传播）
    bool withWorkspace = true;
};

// fake 平台信息（对齐 add_example/ops-nn UT executor：SoCInfo/AICoreSpec/version 三组 map）
static void SetUpPlatformInfo(gert::TilingContext* context, uint64_t coreNum, uint64_t ubSize)
{
    std::map<std::string, std::string> socInfos = {
        {"ai_core_cnt", std::to_string(coreNum)},
        {"vector_core_cnt", std::to_string(coreNum)},
        {"cube_core_cnt", "1"},
        {"core_type_list", "AICore"},
        {"l2_size", "33554432"},
    };
    std::map<std::string, std::string> aicoreSpec = {
        {"ub_size", std::to_string(ubSize)},
        {"l1_size", "524288"},
        {"l0_a_size", "65536"},
        {"l0_b_size", "65536"},
        {"l0_c_size", "131072"},
        {"bt_size", "0"},
        {"cube_freq", "0"},
        {"load3d_constraints", "1"},
    };
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersionInfos = {{"NpuArch", "3510"}, {"Short_SoC_version", "Ascend950"}};

    auto* pf = context->GetPlatformInfo();
    ASSERT_NE(pf, nullptr);
    pf->SetPlatformRes("SoCInfo", socInfos);
    pf->SetPlatformRes("AICoreSpec", aicoreSpec);
    pf->SetCoreNumByCoreType("AICore");
    pf->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    pf->SetPlatformRes("version", socVersionInfos);
}

static void RunTilingCase(const TilingCaseParam& c)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
    ASSERT_NE(opImpl, nullptr) << "LpNormReduceV2 未注册到 OpImplRegistry";
    auto tilingFunc = opImpl->tiling;
    ASSERT_NE(tilingFunc, nullptr) << "LpNormReduceV2 未注册 tiling";

    gert::StorageShape shape;
    auto fillShape = [](gert::Shape& s, const std::vector<int64_t>& dims) {
        s.SetDimNum(dims.size());
        for (size_t i = 0; i < dims.size(); i++) {
            s.SetDim(i, dims[i]);
        }
    };
    fillShape(shape.MutableOriginShape(), c.xShape);
    fillShape(shape.MutableStorageShape(), c.xShape);
    gert::StorageShape yShape = {{1}, {1}};

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    // CompileInfo：TilingParse<LpNormReduceV2CompileInfo> 的编译信息占位（空结构）
    optiling::LpNormReduceV2CompileInfo tilingCompileInfo;

    auto tilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(tilingData, nullptr);
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);
    ASSERT_NE(workspace, nullptr);

    gert::TilingContextFaker faker;
    faker.SetOpType(OP_TYPE)
        .NodeIoNum(1, 1)
        .InputShapes({&shape})
        .OutputShapes({&yShape})
        .NodeInputTd(0, c.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, c.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"p", Ops::NN::AnyValue::CreateFrom<float>(c.p)},
                    {"axes", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(c.axes)},
                    {"keepdim", Ops::NN::AnyValue::CreateFrom<bool>(c.keepdim)},
                    {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-12f)}})
        .CompileInfo(&tilingCompileInfo)
        .PlatformInfo(reinterpret_cast<void*>(&platformInfo))
        .TilingData(tilingData.get());
    if (c.withWorkspace) {
        faker.Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
    }
    auto holder = faker.Build();

    gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    SetUpPlatformInfo(context, c.coreNum, c.ubSize);

    std::cout << "[TILING_CASE] " << c.caseName << std::endl;
    EXPECT_EQ(tilingFunc(context), c.expectStatus);
    if (c.expectStatus != ge::GRAPH_SUCCESS) {
        return;
    }

    // workspace：CROSS_CORE=0 → 0（仍声明 1 个 slot）；=1 → 分档实值（outBlockNum×rGroupNum×slot）
    EXPECT_EQ(context->GetWorkspaceNum(), 1U);
    auto wsSizes = context->GetWorkspaceSizes(1);
    EXPECT_EQ(wsSizes[0], c.expectWsBytes);

    // blockDim / TilingKey
    EXPECT_EQ(context->GetBlockDim(), c.expectBlockDim);
    EXPECT_EQ(context->GetTilingKey(), c.expectTilingKey);

    // TilingData 字段
    const auto* td = reinterpret_cast<const LpNormReduceV2TilingData*>(context->GetRawTilingData()->GetData());
    EXPECT_EQ(td->bLength, c.bLength);
    EXPECT_EQ(td->rLength, c.rLength);
    EXPECT_EQ(td->cLength, c.cLength);
    EXPECT_EQ(td->coreNum, c.coreNumUsed);
    EXPECT_EQ(td->outBlockNum, c.outBlockNum);
    EXPECT_EQ(td->rGroupNum, c.rGroupNum);
    EXPECT_EQ(td->bFactor, c.bFactor);
    EXPECT_EQ(td->rowsTile, c.rowsTile);
    EXPECT_EQ(td->bTile, c.bTile);
    EXPECT_EQ(td->ubFactorR, c.ubFactorR);
    EXPECT_EQ(td->ubFactorC, c.ubFactorC);
    EXPECT_EQ(td->rAlignedCols, c.rAlignedCols);
    EXPECT_EQ(td->rFactorPerGroup, c.rFactorPerGroup);
    EXPECT_EQ(td->pBranch, c.pBranch);
    EXPECT_FLOAT_EQ(td->pVal, c.p);
    EXPECT_EQ(td->powTimes, c.powTimes);
    EXPECT_EQ(td->zeroOutputFlag, c.zeroOutputFlag);
    EXPECT_EQ(td->segNum, c.segNum);
    const uint64_t key = context->GetTilingKey();
    const int64_t bufferModeIdx = static_cast<int64_t>((key >> 24U) & 0xFFU);
    const int64_t crossIdx = static_cast<int64_t>((key >> 8U) & 0xFFU);
    const int64_t layoutIdx = static_cast<int64_t>((key >> 16U) & 0xFFU);
    if (c.segNum >= 2) {
        // P2 段表：段数 ≥2 时 TilingKey 恒 (dtype,0,0,0)，走 P2 缓冲几何（P1 单 tile 预算公式不适用）
        EXPECT_EQ(td->aSegNum, c.aSegNum);
        EXPECT_EQ(crossIdx, 0);
        EXPECT_EQ(layoutIdx, 0);
        EXPECT_EQ(bufferModeIdx, 0);
        EXPECT_GE(td->ubFactorR, 1);
        EXPECT_LE(td->ubFactorR, 8192);
        // R 段表逐字段断言（迭代三 A2）：segSize/segStride 后缀积口径 + 尾部槽位清零（memset 完整性）
        if (!c.expectSegSize.empty()) {
            ASSERT_EQ(c.expectSegSize.size(), static_cast<size_t>(c.segNum));
            ASSERT_EQ(c.expectSegStride.size(), static_cast<size_t>(c.segNum));
            for (int32_t j = 0; j < c.segNum; j++) {
                EXPECT_EQ(td->segSize[j], c.expectSegSize[j]) << "segSize[" << j << "]";
                EXPECT_EQ(td->segStride[j], c.expectSegStride[j]) << "segStride[" << j << "]";
            }
            for (int32_t j = c.segNum; j < 8; j++) {
                EXPECT_EQ(td->segSize[j], 0) << "segSize[" << j << "] 尾部应清零";
                EXPECT_EQ(td->segStride[j], 0) << "segStride[" << j << "] 尾部应清零";
            }
        }
        // A 段表逐字段断言（输出混合基数 → GM 基地址分解，DESIGN §4.3/PT-13 §3.2.1）
        if (!c.expectASize.empty()) {
            ASSERT_EQ(c.expectASize.size(), static_cast<size_t>(c.aSegNum));
            ASSERT_EQ(c.expectAStride.size(), static_cast<size_t>(c.aSegNum));
            for (int32_t a = 0; a < c.aSegNum; a++) {
                EXPECT_EQ(td->aSize[a], c.expectASize[a]) << "aSize[" << a << "]";
                EXPECT_EQ(td->aStride[a], c.expectAStride[a]) << "aStride[" << a << "]";
            }
            for (int32_t a = c.aSegNum; a < 4; a++) {
                EXPECT_EQ(td->aSize[a], 0) << "aSize[" << a << "] 尾部应清零";
                EXPECT_EQ(td->aStride[a], 0) << "aStride[" << a << "] 尾部应清零";
            }
        }
        return;
    }

    // 字段合法性（边界）：tile 数为正、单 tile UB 预算不溢出（求解公式反向校验，§5.3 BLK-2）
    EXPECT_GE(td->bFactor, 0);
    EXPECT_GE(td->rowsTile, 1);
    EXPECT_GE(td->ubFactorR, 1);
    EXPECT_GE(td->bTile, 1);
    const int64_t typeSize = (c.dtype == ge::DT_FLOAT) ? 4 : 2;
    const int64_t bufNum = (bufferModeIdx == 1) ? 2 : 1;
    const int64_t perElemBytes = bufNum * typeSize + 8 + ((c.pBranch == LPNORM_P_BRANCH_ZERO) ? 5 : 0);
    const int64_t ubAvail = static_cast<int64_t>(c.ubSize) - 8192 - (256 * bufNum + 1024);
    const int64_t ubBlockElems = (typeSize == 4) ? 8 : 16;
    int64_t singleTileBytes = 0;
    if (layoutIdx == 0) {
        const int64_t ubRAlloc = ((td->ubFactorR + ubBlockElems - 1) / ubBlockElems) * ubBlockElems;
        const int64_t outPerRowBytes = 8 + typeSize + ((crossIdx == 1) ? 8 : 0);
        singleTileBytes = td->rowsTile * (ubRAlloc * perElemBytes + outPerRowBytes);
    } else {
        const int64_t outPerBatchBytes = td->rAlignedCols * (8 + typeSize + ((crossIdx == 1) ? 8 : 0));
        singleTileBytes = td->ubFactorR * td->rAlignedCols * perElemBytes + td->bTile * outPerBatchBytes;
    }
    if (!c.checkUbBudget) {
        return; // 病态平台防御钳位用例：ubAvail < 单块时仅验证不崩溃与钳位正值，不做预算断言
    }
    EXPECT_LE(singleTileBytes, ubAvail) << "single-tile UB budget overflow: " << singleTileBytes << " > " << ubAvail;
}

} // namespace lp_ut

#endif // LP_NORM_REDUCE_V2_TILING_CASE_RUNNER_H
