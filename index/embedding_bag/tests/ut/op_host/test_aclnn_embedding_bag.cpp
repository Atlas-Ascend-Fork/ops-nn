/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_embedding_bag.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_embedding_bag.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_embedding_bag_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "embedding_bag_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "embedding_bag_test TearDown" << endl; }
};

TEST_F(l2_embedding_bag_test, case_0)
{
    auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1, 2, 3, 4});
    auto tensorOffsetsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{0, 4});
    bool scaleGradByFreq = false;
    int64_t mode = 1;
    bool sparse = false;
    bool includeLastOffset = false;
    int64_t paddingIdx = -1;
    auto tensorOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOffset2BagDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorBagSizeDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorMaxIndicesDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto ut = OP_API_UT(aclnnEmbeddingBag,
                        INPUT(tensorWeightDesc, tensorIndicesDesc, tensorOffsetsDesc, scaleGradByFreq, mode, sparse,
                              (aclTensor*)nullptr, includeLastOffset, paddingIdx),
                        OUTPUT(tensorOutputDesc, tensorOffset2BagDesc, tensorBagSizeDesc, tensorMaxIndicesDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_embedding_bag_test, case_1)
{
    auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1, 2, 3, 4});
    auto tensorOffsetsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{0, 4});
    bool scaleGradByFreq = false;
    int64_t mode = 2;
    bool sparse = false;
    bool includeLastOffset = false;
    int64_t paddingIdx = -1;
    auto tensorOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOffset2BagDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorBagSizeDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorMaxIndicesDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto ut = OP_API_UT(aclnnEmbeddingBag,
                        INPUT(tensorWeightDesc, tensorIndicesDesc, tensorOffsetsDesc, scaleGradByFreq, mode, sparse,
                              (aclTensor*)nullptr, includeLastOffset, paddingIdx),
                        OUTPUT(tensorOutputDesc, tensorOffset2BagDesc, tensorBagSizeDesc, tensorMaxIndicesDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_embedding_bag_test, case_2)
{
    auto tensorWeightDesc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{1, 2, 0, 1});
    auto tensorOffsetsDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5).Value(vector<int>{0, 2});
    bool scaleGradByFreq = false;
    int64_t mode = 0;
    bool sparse = false;
    auto tensorperSampleWeightsDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    bool includeLastOffset = false;
    int64_t paddingIdx = -1;
    auto tensorOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorOffset2BagDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto tensorBagSizeDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto tensorMaxIndicesDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 5);
    auto ut = OP_API_UT(aclnnEmbeddingBag,
                        INPUT(tensorWeightDesc, tensorIndicesDesc, tensorOffsetsDesc, scaleGradByFreq, mode, sparse,
                              tensorperSampleWeightsDesc, includeLastOffset, paddingIdx),
                        OUTPUT(tensorOutputDesc, tensorOffset2BagDesc, tensorBagSizeDesc, tensorMaxIndicesDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_embedding_bag_test, empty_indices_supported_on_a2_a3)
{
    for (auto soc : {SocVersion::ASCEND910B, SocVersion::ASCEND910_93}) {
        SocVersionManager manager(soc);
        for (int64_t mode : {0, 1, 2}) {
            for (bool last : {false, true}) {
                auto weight = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
                auto indices = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
                auto offsets = TensorDesc({last ? 3 : 2}, ACL_INT32, ACL_FORMAT_ND);
                auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
                auto offset2bag = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
                auto bagSize = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
                auto maxIndices = TensorDesc(mode == 2 ? vector<int64_t>{2, 3} : vector<int64_t>{2}, ACL_INT32,
                                             ACL_FORMAT_ND);
                auto ut = OP_API_UT(
                    aclnnEmbeddingBag,
                    INPUT(weight, indices, offsets, false, mode, false, (aclTensor*)nullptr, last, int64_t{-1}),
                    OUTPUT(output, offset2bag, bagSize, maxIndices));
                uint64_t workspaceSize = 0;
                EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
            }
        }
    }
}

TEST_F(l2_embedding_bag_test, empty_optional_weights_supported_for_sum)
{
    for (auto soc : {SocVersion::ASCEND910B, SocVersion::ASCEND910_93}) {
        SocVersionManager manager(soc);
        auto weight = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
        auto indices = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
        auto offsets = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
        auto sampleWeights = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
        auto output = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
        auto offset2bag = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
        auto bagSize = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
        auto maxIndices = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
        auto ut = OP_API_UT(
            aclnnEmbeddingBag,
            INPUT(weight, indices, offsets, false, int64_t{0}, false, sampleWeights, false, int64_t{-1}),
            OUTPUT(output, offset2bag, bagSize, maxIndices));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    }
}

TEST_F(l2_embedding_bag_test, a5_empty_inputs_keep_legacy_fast_return)
{
    SocVersionManager manager(SocVersion::ASCEND950);
    for (int emptyInput = 0; emptyInput < 4; ++emptyInput) {
        for (int64_t mode : {0, 1, 2}) {
            for (bool last : {false, true}) {
                if (emptyInput == 3 && last) {
                    continue;
                }
                SCOPED_TRACE(emptyInput);
                const int64_t dim = emptyInput == 1 ? 0 : 3;
                const int64_t indicesCount = emptyInput == 2 ? 0 : 4;
                const int64_t offsetsCount = emptyInput == 3 ? 0 : last ? 3 : 2;
                const int64_t bags = offsetsCount - static_cast<int64_t>(last);
                auto weight = TensorDesc({emptyInput == 0 ? 0 : 4, dim}, ACL_FLOAT, ACL_FORMAT_ND);
                auto indices = TensorDesc({indicesCount}, ACL_INT32, ACL_FORMAT_ND);
                auto offsets = TensorDesc({offsetsCount}, ACL_INT32, ACL_FORMAT_ND);
                auto output = TensorDesc({bags, dim}, ACL_FLOAT, ACL_FORMAT_ND);
                auto offset2bag = TensorDesc({indicesCount}, ACL_INT32, ACL_FORMAT_ND);
                auto bagSize = TensorDesc({offsetsCount}, ACL_INT32, ACL_FORMAT_ND);
                auto maxIndices = TensorDesc(mode == 2 ? vector<int64_t>{bags, dim} : vector<int64_t>{0}, ACL_INT32,
                                             ACL_FORMAT_ND);
                auto ut = OP_API_UT(
                    aclnnEmbeddingBag,
                    INPUT(weight, indices, offsets, false, mode, false, (aclTensor*)nullptr, last, int64_t{-1}),
                    OUTPUT(output, offset2bag, bagSize, maxIndices));
                uint64_t workspaceSize = 123;
                EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
                EXPECT_EQ(workspaceSize, 0);
            }
        }
    }
}
