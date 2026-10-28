/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_confidence_head_fused.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

class ConfidenceHeadFusedApiTest : public testing::Test {};

TEST_F(ConfidenceHeadFusedApiTest, FP16WithoutOptionalInputs)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(ConfidenceHeadFusedApiTest, BF16AllOptionalInputsAscend350)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND350);
    auto hidden = TensorDesc({2, 17, 33}, ACL_BF16, ACL_FORMAT_ND);
    auto weight = TensorDesc({40}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto markov = TensorDesc({2, 17, 7}, ACL_BF16, ACL_FORMAT_ND);
    auto temperature = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused,
                        INPUT(hidden, weight, markov, temperature, bias, ScalarDesc(-0.25F), ScalarDesc(0.25F)),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(ConfidenceHeadFusedApiTest, TemperatureWithoutMarkov)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto temperature = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused,
                        INPUT(hidden, weight, nullptr, temperature, nullptr, nullptr, nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(ConfidenceHeadFusedApiTest, ZeroReductionWithBias)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, bias, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(ConfidenceHeadFusedApiTest, EmptyOutput)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({0, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

TEST_F(ConfidenceHeadFusedApiTest, NullHidden)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto weight = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(nullptr, weight, nullptr, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(ConfidenceHeadFusedApiTest, UnsupportedFeatureType)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ConfidenceHeadFusedApiTest, MismatchedWeightType)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_BF16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ConfidenceHeadFusedApiTest, InvalidWeightLength)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({34}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ConfidenceHeadFusedApiTest, InvalidOutputShape)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({18}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ConfidenceHeadFusedApiTest, InvalidBiasLength)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, bias, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ConfidenceHeadFusedApiTest, InvalidBiasRank)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({1, 17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, bias, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ConfidenceHeadFusedApiTest, InvalidTemperatureLength)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto temperature = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused,
                        INPUT(hidden, weight, nullptr, temperature, nullptr, nullptr, nullptr), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ConfidenceHeadFusedApiTest, InvalidMarkovLeadingDimension)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({40}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto markov = TensorDesc({16, 7}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, markov, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ConfidenceHeadFusedApiTest, InvalidHiddenRank)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto hidden = TensorDesc({17, 1, 33, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17, 1, 33}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(ConfidenceHeadFusedApiTest, UnsupportedSoc)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910);
    auto hidden = TensorDesc({17, 33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({33}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnConfidenceHeadFused, INPUT(hidden, weight, nullptr, nullptr, nullptr, nullptr, nullptr),
                        OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
