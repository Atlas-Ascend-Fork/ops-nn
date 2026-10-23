/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_turbo_quant.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_turbo_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "turbo_quant_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "turbo_quant_test TearDown" << endl; }
};

// legal baseline: latent [2, 512] fp32, centroids [16] fp32, y [2, 256] uint8, scale [2] fp16
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_float32_nd_uint8_nd)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// legal empty input: numTokens is 0
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_empty_tokens)
{
    auto latentDesc = TensorDesc({0, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({0, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// issue #7101: latent rank 1
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_latent_rank1)
{
    auto latentDesc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7101: latent rank 3
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_latent_rank3)
{
    auto latentDesc = TensorDesc({2, 1, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7101: headDim 511
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_head_dim_511)
{
    auto latentDesc = TensorDesc({2, 511}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7101: headDim 256
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_head_dim_256)
{
    auto latentDesc = TensorDesc({2, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 128}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// review: latent numTokens -1 (unknown dim) passes all shape comparisons, must be rejected in eager mode
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_latent_num_tokens_neg1)
{
    auto latentDesc = TensorDesc({-1, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({-1, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({-1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7101: centroids length 15
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_centroids_len15)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({15}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7101: centroids length 17
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_centroids_len17)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({17}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7101: centroids rank 2
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_centroids_rank2)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({1, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7100: y last dim halved (declared smaller than inferred)
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_y_last_dim_half)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 128}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7100: y last dim plus one
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_y_last_dim_plus1)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 257}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7100: y numTokens mismatch
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_y_num_tokens_mismatch)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7100: y rank 3
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_y_rank3)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7100: scale length halved
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_scale_len_half)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7100: scale length plus one
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_scale_len_plus1)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue #7100: scale rank 2
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_scale_rank2)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error input dtype: latent int8
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_latent_dtype)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error output dtype: y float16
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_y_dtype)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error output dtype: scale float32
TEST_F(l2_turbo_quant_test, ascend910B2_aclnnTurboQuant_error_scale_dtype)
{
    auto latentDesc = TensorDesc({2, 512}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto centroidsDesc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 256}, ACL_UINT8, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTurboQuant, INPUT(latentDesc, centroidsDesc), OUTPUT(yDesc, scaleDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
