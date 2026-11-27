/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../../foreach_utils/op_kernel/arch35/foreach_flat_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/foreach_mul_list_tiling_data.h"
#include "../../../../op_kernel/arch35/foreach_mul_list_tiling_key.h"

using namespace ut_util;

class ForeachMulListTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachMulListTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachMulListTilingTest TearDown" << std::endl; }
};

static void RunCase(std::initializer_list<int64_t> x1Shape, std::initializer_list<int64_t> x2Shape,
                    std::initializer_list<int64_t> yShape, ge::DataType x1Dt, ge::DataType x2Dt, ge::DataType yDt,
                    ge::graphStatus expectStatus, uint64_t expectKey)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc, ai, intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachMulListCompileInfo {
    } ci;
    auto opType = std::string("ForeachMulList");
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFn, nullptr);
    ASSERT_NE(tilingParseFn, nullptr);

    std::string ciStr = R"({"device_id":null})";
    auto kh = gert::KernelRunContextFaker()
                  .KernelIONum(2, 1)
                  .Inputs({const_cast<char*>(ciStr.c_str()), reinterpret_cast<void*>(&pf)})
                  .Outputs({&ci})
                  .Build();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    ASSERT_EQ(tilingParseFn(kh.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xS1 = {x1Shape, x1Shape};
    gert::StorageShape xS2 = {x2Shape, x2Shape};
    gert::StorageShape yS = {yShape, yShape};
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .NodeIoNum(2, 1)
                  .IrInstanceNum({1, 1}, {1})
                  .InputShapes({&xS1, &xS2})
                  .OutputShapes({&yS})
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, x1Dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeInputTd(1, x2Dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, yDt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    EXPECT_EQ(tilingFn(ctx), expectStatus);
    if (expectStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(ctx->GetTilingKey(), expectKey);
    }
}

static void DoCase(std::initializer_list<int64_t> xShape, ge::DataType dt, uint64_t expectKey)
{
    RunCase(xShape, xShape, xShape, dt, dt, dt, ge::GRAPH_SUCCESS, expectKey);
}

TEST_F(ForeachMulListTilingTest, foreach_mul_list_float32)
{
    DoCase({4, 4}, ge::DT_FLOAT, GET_TPL_TILING_KEY(FOREACH_MUL_LIST_TPL_SCH_MODE_FLOAT));
}

TEST_F(ForeachMulListTilingTest, foreach_mul_list_float16)
{
    DoCase({4, 4}, ge::DT_FLOAT16, GET_TPL_TILING_KEY(FOREACH_MUL_LIST_TPL_SCH_MODE_FLOAT16));
}

TEST_F(ForeachMulListTilingTest, foreach_mul_list_bf16)
{
    DoCase({4, 4}, ge::DT_BF16, GET_TPL_TILING_KEY(FOREACH_MUL_LIST_TPL_SCH_MODE_BF16));
}

TEST_F(ForeachMulListTilingTest, foreach_mul_list_int32)
{
    DoCase({4, 4}, ge::DT_INT32, GET_TPL_TILING_KEY(FOREACH_MUL_LIST_TPL_SCH_MODE_INT32));
}

TEST_F(ForeachMulListTilingTest, foreach_mul_list_x1_x2_shape_mismatch)
{
    // x2 的 shape 与 x1 不一致，应在 ValidateForeachMulList 的 shape 严格一致校验处被拒绝。
    RunCase({4, 4}, {4, 8}, {4, 4}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::GRAPH_FAILED, 0);
}

TEST_F(ForeachMulListTilingTest, foreach_mul_list_dtype_mismatch)
{
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
    // x2 的 dtype 与 x1 不一致，应在 ValidateForeachMulList 的 x2-dtype 校验处被拒绝。
    RunCase({4, 4}, {4, 4}, {4, 4}, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::GRAPH_FAILED, 0);
    const auto reportedErrors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_FALSE(reportedErrors.empty());
    EXPECT_EQ(reportedErrors.front().error_id, "EZ0020");
}

namespace {
// Check that the launch ranges own every element exactly once, including ragged lists.
void ExpectFlatList(const std::vector<int64_t>& lengths, ge::DataType dtype, uint64_t key, uint64_t ubSize = 262144,
                    bool expectSuccess = true)
{
    std::vector<gert::TilingContextPara::TensorDescription> tensors;
    std::vector<int64_t> prefix{0};
    for (int64_t length : lengths) {
        tensors.emplace_back(gert::StorageShape{{length}, {length}}, dtype, ge::FORMAT_ND);
        prefix.push_back(prefix.back() + length);
    }
    auto inputs = tensors;
    std::vector<uint32_t> instances{static_cast<uint32_t>(lengths.size())};
    inputs.insert(inputs.end(), tensors.begin(), tensors.end());
    instances.push_back(static_cast<uint32_t>(lengths.size()));
    int compileInfo = 0;
    gert::TilingContextPara context("ForeachMulList", inputs, tensors, {}, instances,
                                    {static_cast<uint32_t>(lengths.size())}, &compileInfo, 64, ubSize, 4096);
    TilingInfo info;
    if (!expectSuccess) {
        EXPECT_FALSE(ExecuteTiling(context, info));
        return;
    }
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, key);
    EXPECT_EQ(info.workspaceSizes, std::vector<int64_t>({32}));
    ASSERT_EQ(info.tilingDataSize, sizeof(ForeachFlatTilingData));
    const auto& data = *reinterpret_cast<const ForeachFlatTilingData*>(info.tilingData.get());
    EXPECT_GT(data.tileElems, 0U);
    for (size_t i = 0; i < lengths.size(); ++i) {
        EXPECT_EQ(data.tensorDataCountList[i], lengths[i]);
    }
    ASSERT_GE(info.blockNum, 1U);
    ASSERT_LE(info.blockNum, FOREACH_FLAT_MAX_CORE_COUNT);
    if (prefix.back() == 0) {
        EXPECT_EQ(info.blockNum, 1U);
        EXPECT_EQ(data.tensorEndOffsetList[0], -1);
        return;
    }
    int64_t next = 0;
    for (uint32_t core = 0; core < info.blockNum; ++core) {
        const auto first = data.tensorStartList[core];
        const auto last = data.tensorEndList[core];
        ASSERT_LT(first, lengths.size());
        ASSERT_LT(last, lengths.size());
        EXPECT_EQ(prefix[first] + data.tensorStartOffsetList[core], next);
        next = prefix[last] + data.tensorEndOffsetList[core] + 1;
        EXPECT_LE(next, prefix.back());
    }
    EXPECT_EQ(next, prefix.back());
}
} // namespace

TEST_F(ForeachMulListTilingTest, flat_supported_dtypes_and_tensor_partition)
{
    const std::vector<ge::DataType> dtypes{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32,
                                           ge::DT_INT16, ge::DT_INT8,    ge::DT_UINT8};
    for (size_t key = 0; key < dtypes.size(); ++key) {
        SCOPED_TRACE(key);
        ExpectFlatList({16, 31, 64}, dtypes[key], key);
    }
}

TEST_F(ForeachMulListTilingTest, flat_empty_list_noop) { ExpectFlatList({}, ge::DT_FLOAT, 0); }

TEST_F(ForeachMulListTilingTest, flat_ragged_large_list_complete_coverage)
{
    ExpectFlatList({65536, 0, 32769}, ge::DT_FLOAT, 0);
}

TEST_F(ForeachMulListTilingTest, flat_insufficient_ub_rejected) { ExpectFlatList({16}, ge::DT_FLOAT, 0, 1, false); }
