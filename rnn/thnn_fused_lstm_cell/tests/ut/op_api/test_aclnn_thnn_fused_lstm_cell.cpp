/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_thnn_fused_lstm_cell.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_thnn_fused_lstm_cell_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_thnn_fused_lstm_cell_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_thnn_fused_lstm_cell_test TearDown" << std::endl; }
};

struct WorkspaceSizeResult {
    aclnnStatus status = ACLNN_SUCCESS;
    bool workspaceMatched = true;
};

static aclnnStatus GetSerialWorkspaceSize(int64_t B, int64_t H, uint64_t* workspaceSize)
{
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));
    return ut.TestGetWorkspaceSize(workspaceSize);
}

static WorkspaceSizeResult RunConcurrentGetWorkspaceSize(int64_t B, int64_t H, uint64_t expectedWorkspaceSize,
                                                         std::atomic<int>& ready, std::atomic<bool>& start)
{
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    ready.fetch_add(1);
    while (!start.load()) {
        std::this_thread::yield();
    }

    WorkspaceSizeResult result;
    constexpr size_t call_num = 100;
    for (size_t idx = 0; idx < call_num; ++idx) {
        uint64_t workspace_size = 0;
        result.status = ut.TestGetWorkspaceSize(&workspace_size);
        if (result.status != ACL_SUCCESS) {
            return result;
        }
        result.workspaceMatched = result.workspaceMatched && workspace_size == expectedWorkspaceSize;
    }
    return result;
}

// 正常input场景
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_normal_float)
{
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedLstmCell, INPUT(inputGates, hiddenGates, cx, inputBias, hiddenBias),
                        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_concurrent_get_workspace_size)
{
    constexpr size_t thread_num = 8;
    array<WorkspaceSizeResult, thread_num> results{};
    array<uint64_t, thread_num> expectedWorkspaceSizes{};
    atomic<int> ready{0};
    atomic<bool> start{false};
    array<thread, thread_num> threads;

    for (size_t idx = 0; idx < thread_num; ++idx) {
        EXPECT_EQ(GetSerialWorkspaceSize(3 + idx, 5 + 2 * idx, &expectedWorkspaceSizes[idx]), ACL_SUCCESS);
    }
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = thread([&results, &expectedWorkspaceSizes, &ready, &start, idx]() {
            results[idx] = RunConcurrentGetWorkspaceSize(3 + idx, 5 + 2 * idx, expectedWorkspaceSizes[idx], ready,
                                                         start);
        });
    }

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
    for (const auto& result : results) {
        EXPECT_EQ(result.status, ACL_SUCCESS);
        EXPECT_TRUE(result.workspaceMatched);
    }
}
