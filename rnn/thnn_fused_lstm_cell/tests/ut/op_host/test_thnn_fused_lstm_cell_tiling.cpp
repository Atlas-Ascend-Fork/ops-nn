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
 * \file test_thnn_fused_lstm_cell_tiling.cpp
 * \brief
 */

#include <atomic>
#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ut_util;
using namespace std;
using namespace ge;

struct ThnnFusedLstmCellCompileInfo {};

class ThnnFusedLstmCellTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ThnnFusedLstmCellTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ThnnFusedLstmCellTiling TearDown" << std::endl; }
};

std::vector<uint8_t> RunThnnFusedLstmCellTiling(int64_t batchSize, int64_t hiddenSize, ge::DataType dataType)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape commonShape = {{batchSize, hiddenSize}, {batchSize, hiddenSize}};
    gert::StorageShape gatesShape = {{batchSize, 4 * hiddenSize}, {batchSize, 4 * hiddenSize}};
    gert::StorageShape biasShape = {{4 * hiddenSize}, {4 * hiddenSize}};

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string COMPILE_INFO_STRING_910B = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 40}})";
    GetPlatFormInfos(COMPILE_INFO_STRING_910B.c_str(), socInfos, aicoreSpec, intrinsics);

    // Platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    // Compile info
    ThnnFusedLstmCellCompileInfo compileInfo;

    std::string op_type("ThnnFusedLstmCell");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (opImpl == nullptr) {
        ADD_FAILURE() << "ThnnFusedLstmCell implementation is not registered.";
        return {};
    }
    auto tilingFunc = opImpl->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    if (param == nullptr || workspaceSizeHoler == nullptr) {
        ADD_FAILURE() << "Failed to allocate tiling context data.";
        return {};
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType("ThnnFusedLstmCell")
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&gatesShape, &gatesShape, &commonShape, &biasShape, &biasShape})
                      .OutputShapes({&commonShape, &commonShape, &gatesShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        ADD_FAILURE() << "Failed to create tiling context.";
        return {};
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    if (tilingFunc(tilingContext) != ge::GRAPH_SUCCESS) {
        ADD_FAILURE() << "ThnnFusedLstmCell tiling failed.";
        return {};
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr) {
        ADD_FAILURE() << "ThnnFusedLstmCell tiling data is empty.";
        return {};
    }
    auto data = static_cast<uint8_t*>(rawTilingData->GetData());
    return {data, data + rawTilingData->GetDataSize()};
    // dlog_setlevel(0, 3, 0);
}

TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_float_succ)
{
    std::cout << "run case: "
              << "thnn_fused_lstm_cell_float_succ" << std::endl;
    auto tilingData = RunThnnFusedLstmCellTiling(40, 8, ge::DT_FLOAT);
    EXPECT_FALSE(tilingData.empty());
}

static void RunConcurrentTiling(int64_t batchSize, int64_t hiddenSize, std::atomic<int>& ready,
                                std::atomic<bool>& start, const std::vector<uint8_t>& expectedTilingData,
                                std::atomic<bool>& matched)
{
    ready.fetch_add(1);
    while (!start.load()) {
        std::this_thread::yield();
    }
    for (size_t idx = 0; idx < 200; ++idx) {
        auto tilingData = RunThnnFusedLstmCellTiling(batchSize, hiddenSize, ge::DT_FLOAT);
        if (tilingData != expectedTilingData) {
            matched.store(false);
            return;
        }
    }
}

TEST_F(ThnnFusedLstmCellTiling, thnn_fused_lstm_cell_concurrent_tiling)
{
    constexpr size_t thread_num = 4;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> matched{true};
    std::array<std::thread, thread_num> threads;
    std::array<std::vector<uint8_t>, thread_num> expectedTilingData;

    expectedTilingData[0] = RunThnnFusedLstmCellTiling(40, 8, ge::DT_FLOAT);
    expectedTilingData[1] = RunThnnFusedLstmCellTiling(32, 16, ge::DT_FLOAT);
    expectedTilingData[2] = RunThnnFusedLstmCellTiling(24, 24, ge::DT_FLOAT);
    expectedTilingData[3] = RunThnnFusedLstmCellTiling(16, 32, ge::DT_FLOAT);
    for (const auto& data : expectedTilingData) {
        EXPECT_FALSE(data.empty());
    }

    threads[0] = std::thread(RunConcurrentTiling, 40, 8, std::ref(ready), std::ref(start),
                             std::cref(expectedTilingData[0]), std::ref(matched));
    threads[1] = std::thread(RunConcurrentTiling, 32, 16, std::ref(ready), std::ref(start),
                             std::cref(expectedTilingData[1]), std::ref(matched));
    threads[2] = std::thread(RunConcurrentTiling, 24, 24, std::ref(ready), std::ref(start),
                             std::cref(expectedTilingData[2]), std::ref(matched));
    threads[3] = std::thread(RunConcurrentTiling, 16, 32, std::ref(ready), std::ref(start),
                             std::cref(expectedTilingData[3]), std::ref(matched));

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (ready.load() != static_cast<int>(thread_num) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    bool allThreadsReady = ready.load() == static_cast<int>(thread_num);
    start.store(true);
    for (auto& thread : threads) {
        thread.join();
    }
    ASSERT_TRUE(allThreadsReady);
    EXPECT_TRUE(matched.load());
}
