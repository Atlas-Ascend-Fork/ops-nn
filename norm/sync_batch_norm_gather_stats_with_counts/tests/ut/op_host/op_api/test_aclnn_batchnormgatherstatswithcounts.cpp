/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "../../../op_host/op_api/aclnn_batch_norm_gather_stats_with_counts.h"

#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "base/err_mgr.h"

using namespace std;

class l2BatchNormGatherStatsWithCountsTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2BatchNormGatherStatsWithCountsTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2BatchNormGatherStatsWithCountsTest TearDown" << endl; }
};

// ErrorManager raw message helpers (replaces weak-symbol capture)
static std::vector<std::string> g_ezCodes;
static bool g_ezCodesLoaded = false;

static void InitErrMgr()
{
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrErrorMessage();
    g_ezCodes.clear();
    g_ezCodesLoaded = false;
}

static void LoadEZCodes()
{
    if (g_ezCodesLoaded) {
        return;
    }
    g_ezCodes.clear();
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    for (const auto& e : errors) {
        g_ezCodes.push_back(e.error_id == nullptr ? std::string() : std::string(e.error_id.get()));
    }
    g_ezCodesLoaded = true;
}

static bool HasEZCode(const std::string& code)
{
    LoadEZCodes();
    for (const auto& id : g_ezCodes) {
        if (id == code) {
            return true;
        }
    }
    return false;
}

static int GetEZCodeCount()
{
    LoadEZCodes();
    return static_cast<int>(g_ezCodes.size());
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_float32_3d)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_float16_3d)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_float32_2d)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto meanDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 5, 6,
                                                                                     7, 8, 5, 6, 7, 8, 5, 6, 7, 8});
    auto invstdDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 5, 6,
                                                                                       7, 8, 5, 6, 7, 8, 5, 6, 7, 8});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_float16_2d)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto meanDesc = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 5, 6,
                                                                                       7, 8, 5, 6, 7, 8, 5, 6, 7, 8});
    auto invstdDesc = TensorDesc({5, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 5, 6,
                                                                                         7, 8, 5, 6, 7, 8, 5, 6, 7, 8});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_format_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_NC1HWC0)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_dtype_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_shape_invalid)
{
    auto selfDesc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_float32_2d_runningmean_nullptr)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto meanDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 5, 6,
                                                                                     7, 8, 5, 6, 7, 8, 5, 6, 7, 8});
    auto invstdDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 5, 6,
                                                                                       7, 8, 5, 6, 7, 8, 5, 6, 7, 8});
    auto countsDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, nullptr, nullptr, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_null_tensor)
{
    auto selfDesc = TensorDesc({0, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_mean_dims_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_invstd_dims_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_runningvar_dims_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_runningvar_shape_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_runningmean_shape_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_invalid_counts_shape)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2, 1});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_invalid_mean_nullptr)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({0, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2, 1});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_mean_invstd_first_dim_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2, 1});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_mean_second_dim_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2, 1});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_runing_mean_dim_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2, 1});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_counts_dims_invalid)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2, 1});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormGatherStatsWithCountsTest, l2_batch_norm_gather_stats_with_counts_nullptr)
{
    auto selfDesc = nullptr;
    auto meanDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto invstdDesc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                          .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto runningMeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto runningVarDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto countsDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{4, 5, 3, 2, 1});

    auto meanAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto invstdAllDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormGatherStatsWithCounts,
                        INPUT(selfDesc, meanDesc, invstdDesc, runningMeanDesc, runningVarDesc, 1e-2, 1e-4, countsDesc),
                        OUTPUT(meanAllDesc, invstdAllDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}
// namespace error_message

// After each test, verify no EZ9999 internal code was captured
class SbngwcNoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        InitErrMgr();
        (void)error_message::GetErrMgrErrorMessage(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
    }
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        // 只守护本文件的测试套件, 框架层或其他算子用例的EZ9999不跨套件追责
        static const char* const guardedSuites[] = {"l2BatchNormGatherStatsWithCountsTest"};
        bool guarded = false;
        for (const char* s : guardedSuites) {
            if (std::string(test_info.test_suite_name()) == s) {
                guarded = true;
                break;
            }
        }
        if (!guarded) {
            return;
        }
        if (HasEZCode("EZ9999")) {
            ADD_FAILURE() << "EZ9999 captured during test [" << test_info.name() << "] in " << test_info.file();
        }
    }
};
static const auto _inst_sbngwc = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new SbngwcNoEz9999Check());
    return 0;
}();
