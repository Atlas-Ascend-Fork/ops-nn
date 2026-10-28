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

using namespace std;

class l2BatchNormGatherStatsWithCountsTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2BatchNormGatherStatsWithCountsTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2BatchNormGatherStatsWithCountsTest TearDown" << endl; }
};

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
// ============================================================================
// Error code verification UT for E*9*** remediation (2026-10-22)
// Captures ErrorManager output via weak-symbol override; asserts no EZ9999.
// (The remediated L0 launcher path is not fakable here; this net verifies
//  the op's full UT suite generates no EZ9999.)
// ============================================================================
namespace error_message {
struct CapturedError {
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};
static thread_local CapturedError g_last_error;
static thread_local int g_report_count = 0;
static thread_local std::vector<std::string> g_all_codes; // 记录用例全程上报过的全部错误码
__attribute__((weak)) void ResetCapture()
{
    g_last_error.code.clear();
    g_last_error.keys.clear();
    g_last_error.values.clear();
    g_report_count = 0;
    g_all_codes.clear();
}
__attribute__((weak)) std::string GetLastError() { return g_last_error.code; }
__attribute__((weak)) int GetReportCount() { return g_report_count; }
__attribute__((weak)) bool HasCode(const std::string& c)
{
    for (const auto& code : g_all_codes) {
        if (code == c) {
            return true;
        }
    }
    return false;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code, const std::vector<const char*>& key,
                                                     const std::vector<const char*>& value)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_last_error.keys.clear();
    g_last_error.values.clear();
    for (const auto& k : key)
        g_last_error.keys.push_back(k ? k : "");
    for (const auto& v : value)
        g_last_error.values.push_back(v ? v : "");
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportInnerErrMsg(const char* f, const char* fn, uint32_t l, const char* ec,
                                                const char* fmt, ...)
{
    g_last_error.code = ec ? ec : "";
    if (ec != nullptr) {
        g_all_codes.push_back(ec);
    }
    g_report_count++;
    return 0;
}
} // namespace error_message

// 验证弱符号截获通道在本二进制内生效: 若链接格局变化导致强定义压制弱定义, 该用例显式失败,
// 避免EZ9999守护用例静默失效
TEST_F(l2BatchNormGatherStatsWithCountsTest, error_capture_channel_self_check)
{
    error_message::ResetCapture();
    const std::vector<const char*> selfKey = {"self_check"};
    const std::vector<const char*> selfValue = {"capture channel liveness"};
    error_message::ReportPredefinedErrMsg("EZ0000", selfKey, selfValue);
    EXPECT_TRUE(error_message::HasCode("EZ0000")) << "capture channel is dead: weak override suppressed";
    EXPECT_EQ(error_message::GetReportCount(), 1);
    error_message::ResetCapture();
}
// After each test, verify no EZ9999 internal code was captured
class SbngwcNoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        error_message::ResetCapture(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
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
        if (error_message::HasCode("EZ9999")) {
            ADD_FAILURE() << "EZ9999 captured during test [" << test_info.name() << "] in " << test_info.file();
        }
    }
};
static const auto _inst_sbngwc = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new SbngwcNoEz9999Check());
    return 0;
}();
