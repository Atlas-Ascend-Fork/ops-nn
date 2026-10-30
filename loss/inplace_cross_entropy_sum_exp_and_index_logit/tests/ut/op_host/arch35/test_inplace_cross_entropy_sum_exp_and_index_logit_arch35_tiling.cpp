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
 * \file test_inplace_cross_entropy_sum_exp_and_index_logit_arch35_tiling.cpp
 * \brief A5 (ascend950) inplace tiling UT — 复用 base tiling 类，isInplace=true
 */
#include <iostream>
#include <vector>
#include <map>
#include <cstring>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../../cross_entropy_sum_exp_and_index_logit/op_kernel/arch35/cross_entropy_sum_exp_and_index_logit_struct.h"

using namespace std;
using namespace ge;

namespace {
const uint32_t CE_SCH_MODE_DEFAULT = 0;
constexpr int64_t ROW_BLOCK_MIN = 4;
constexpr int64_t ROW_BLOCK_MAX = 40;

struct InplaceCrossEntropySumExpAndIndexLogitCompileInfo {
    int32_t totalCoreNum = 64;
    uint64_t ubSizePlatForm = 0;
};

struct CeTilingRunResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    CrossEntropySumExpAndIndexLogitRegBaseTilingData tiling;
};

struct Shapes3 {
    gert::StorageShape logits;
    gert::StorageShape target;
    gert::StorageShape gmax;
};

Shapes3 MakeShapes3(const vector<int64_t>& logitsShape, int64_t targetElems, int64_t gmaxElems)
{
    Shapes3 s;
    for (const int64_t dim : logitsShape) {
        s.logits.MutableOriginShape().AppendDim(dim);
        s.logits.MutableStorageShape().AppendDim(dim);
    }
    s.target = gert::StorageShape({targetElems}, {targetElems});
    s.gmax = gert::StorageShape({gmaxElems}, {gmaxElems});
    return s;
}
} // namespace

class InplaceCrossEntropySumExpAndIndexLogitTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InplaceCrossEntropySumExpAndIndexLogitTilingArch35 SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "InplaceCrossEntropySumExpAndIndexLogitTilingArch35 TearDown" << std::endl;
    }
};

// 完整跑一遍 tilingParse + tiling，返回状态、TilingKey、BlockDim 与 TilingData
static CeTilingRunResult RunTiling(gert::StorageShape* logitsShape, gert::StorageShape* targetShape,
                                   gert::StorageShape* gmaxShape, ge::DataType logitsDtype, ge::DataType targetDtype,
                                   ge::DataType gmaxDtype, ge::DataType predictedDtype, ge::DataType sumexpDtype,
                                   ge::DataType expDtype, ge::DataType offsetDtype, ge::DataType maskDtype,
                                   int64_t vocabStart, int64_t vocabEnd)
{
    CeTilingRunResult result;
    const std::string opType("InplaceCrossEntropySumExpAndIndexLogit");
    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr) {
        return result;
    }
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 64}
                                    })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    InplaceCrossEntropySumExpAndIndexLogitCompileInfo compileInfo;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    if (kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo() == nullptr ||
        !kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init()) {
        return result;
    }
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    if (tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return result;
    }

    auto param = gert::TilingData::CreateCap(4096);
    if (param == nullptr) {
        return result;
    }
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    std::vector<gert::StorageShape> outShapes(5, gert::StorageShape({1}, {1}));
    std::vector<gert::StorageShape*> outPtrs;
    for (auto& s : outShapes) {
        outPtrs.push_back(&s);
    }
    std::vector<gert::StorageShape*> inShapes = {logitsShape, targetShape, gmaxShape};

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 5)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes(inShapes)
                      .OutputShapes(outPtrs)
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, logitsDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, targetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, gmaxDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, predictedDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, sumexpDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, expDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, offsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, maskDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"vocab_start_index", Ops::NN::AnyValue::CreateFrom<int64_t>(vocabStart)},
                                  {"vocab_end_index", Ops::NN::AnyValue::CreateFrom<int64_t>(vocabEnd)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext->GetPlatformInfo() == nullptr) {
        return result;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    result.status = tilingFunc(tilingContext);
    result.tilingKey = tilingContext->GetTilingKey();
    result.blockDim = tilingContext->GetBlockDim();
    if (result.status == ge::GRAPH_SUCCESS && tilingContext->GetRawTilingData() != nullptr &&
        tilingContext->GetRawTilingData()->GetData() != nullptr) {
        memcpy(&result.tiling, tilingContext->GetRawTilingData()->GetData(),
               sizeof(CrossEntropySumExpAndIndexLogitRegBaseTilingData));
    }
    return result;
}

// inplace: exp output dtype 跟随 input dtype
static CeTilingRunResult RunTilingValid(gert::StorageShape* logitsShape, gert::StorageShape* targetShape,
                                        gert::StorageShape* gmaxShape, ge::DataType logitsDtype,
                                        ge::DataType targetDtype, ge::DataType gmaxDtype, int64_t vocabStart,
                                        int64_t vocabEnd)
{
    return RunTiling(logitsShape, targetShape, gmaxShape, logitsDtype, targetDtype, gmaxDtype, ge::DT_FLOAT,
                     ge::DT_FLOAT, logitsDtype, ge::DT_INT32, ge::DT_INT32, vocabStart, vocabEnd);
}

// ==================== 正常用例 ====================
// FP32 2D [4, 32]
TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_fp32_2d)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    auto result = RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
                                 0, 32);

    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, CE_SCH_MODE_DEFAULT);
    EXPECT_EQ(result.blockDim, 4u);
    EXPECT_EQ(result.tiling.N, 4u);
    EXPECT_EQ(result.tiling.vLocal, 32u);
    EXPECT_EQ(result.tiling.usedCores, 4u);
    EXPECT_EQ(result.tiling.vTile, 32u);
    EXPECT_EQ(result.tiling.vocabStart, 0);
    EXPECT_EQ(result.tiling.vocabEnd, 32);
    EXPECT_GE(result.tiling.rowBlockMax, ROW_BLOCK_MIN);
    EXPECT_LE(result.tiling.rowBlockMax, ROW_BLOCK_MAX);
    EXPECT_GT(result.tiling.reduceTmpBytes, 0u);
}

// BF16 2D [4, 32]
TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_bf16_2d)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    auto result = RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_BF16, ge::DT_INT32, ge::DT_BF16,
                                 0, 32);

    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, CE_SCH_MODE_DEFAULT);
    EXPECT_EQ(result.blockDim, 4u);
    EXPECT_EQ(result.tiling.N, 4u);
    EXPECT_EQ(result.tiling.vLocal, 32u);
    EXPECT_EQ(result.tiling.vTile, 32u);
}

// FP32 大 N=881 满核
TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_fp32_multi_core)
{
    auto shapes = MakeShapes3({881, 64}, 881, 881);
    auto result = RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
                                 0, 64);

    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, CE_SCH_MODE_DEFAULT);
    EXPECT_EQ(result.blockDim, 64u);
    EXPECT_EQ(result.tiling.N, 881u);
    EXPECT_EQ(result.tiling.usedCores, 64u);
    EXPECT_EQ(result.tiling.headCoreNum, 49u);
    EXPECT_EQ(result.tiling.tokensPerCore, 14u);
    EXPECT_EQ(result.tiling.tokensPerCoreTail, 13u);
}

// ==================== 异常用例 ====================
TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_attr_vocab_end_le_start)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 20, 10)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_logits_wrong_dtype)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_logits_wrong_dims_1d)
{
    auto shapes = MakeShapes3({32}, 32, 32);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_logits_N_out_of_upper)
{
    auto shapes = MakeShapes3({32769, 16}, 32769, 32769);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 16)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_logits_V_out_of_lower)
{
    auto shapes = MakeShapes3({1, 8}, 1, 1);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 8)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_logits_V_not_aligned)
{
    auto shapes = MakeShapes3({1, 17}, 1, 1);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 17)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_logits_vocab_mismatch)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 1, 32)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_target_wrong_dtype)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_target_wrong_elems)
{
    auto shapes = MakeShapes3({4, 32}, 3, 4);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_globalmax_wrong_dtype)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_globalmax_dtype_mismatch)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_globalmax_wrong_elems)
{
    auto shapes = MakeShapes3({4, 32}, 4, 3);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}

// inplace: exp output dtype 必须跟随 input dtype
TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_exp_dtype_must_match_logits)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    // BF16 logits but exp output declared as DT_FLOAT — should fail
    EXPECT_EQ(RunTiling(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_BF16, ge::DT_INT32, ge::DT_BF16,
                        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, 0, 32)
                  .status,
              ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_out_predicted_dtype)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    EXPECT_EQ(RunTiling(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
                        ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, 0, 32)
                  .status,
              ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_out_offset_dtype)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    EXPECT_EQ(RunTiling(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
                        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, 0, 32)
                  .status,
              ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_out_mask_dtype)
{
    auto shapes = MakeShapes3({4, 32}, 4, 4);
    EXPECT_EQ(RunTiling(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT,
                        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 32)
                  .status,
              ge::GRAPH_FAILED);
}

// ==================== 空 tensor 校验 ====================
TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_logits_empty_tensor)
{
    auto shapes = MakeShapes3({0, 32}, 0, 0);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_logits_V_empty_tensor)
{
    auto shapes = MakeShapes3({4, 0}, 4, 4);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 0)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_target_empty_tensor)
{
    auto shapes = MakeShapes3({4, 32}, 0, 4);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}

TEST_F(InplaceCrossEntropySumExpAndIndexLogitTilingArch35, inplace_globalmax_empty_tensor)
{
    auto shapes = MakeShapes3({4, 32}, 4, 0);
    EXPECT_EQ(
        RunTilingValid(&shapes.logits, &shapes.target, &shapes.gmax, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, 0, 32)
            .status,
        ge::GRAPH_FAILED);
}
