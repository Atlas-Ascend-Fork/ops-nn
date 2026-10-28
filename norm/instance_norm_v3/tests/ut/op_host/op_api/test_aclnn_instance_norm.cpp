/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_instance_norm.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_instance_norm_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "instance_norm_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "instance_norm_test TearDown" << endl; }
};

TEST_F(l2_instance_norm_test, case_invalid_dtype_abnormal)
{
    vector<aclDataType> invalidList = {ACL_DOUBLE, ACL_UINT8, ACL_INT8,   ACL_INT16,     ACL_INT32,
                                       ACL_INT64,  ACL_BOOL,  ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128};
    const char* dataFormat = "NCHW";
    double eps = 1e-5;
    int length = invalidList.size();

    for (int i = 0; i < length; i++) {
        auto xDesc = TensorDesc({2, 8, 8, 16}, invalidList[i], ACL_FORMAT_ND);
        auto gammaDesc = TensorDesc({8}, invalidList[i], ACL_FORMAT_ND);
        auto betaDesc = TensorDesc({8}, invalidList[i], ACL_FORMAT_ND);
        auto yDesc = TensorDesc({2, 8, 8, 16}, invalidList[i], ACL_FORMAT_ND);
        auto meanDesc = TensorDesc({2, 8, 1, 1}, invalidList[i], ACL_FORMAT_ND);
        auto varDesc = TensorDesc({2, 8, 1, 1}, invalidList[i], ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                            OUTPUT(yDesc, meanDesc, varDesc));

        uint64_t workspaceSize = 0;
        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_instance_norm_test, case_x_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT((aclTensor*)nullptr, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_gamma_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, (aclTensor*)nullptr, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_beta_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, (aclTensor*)nullptr, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_y_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT((aclTensor*)nullptr, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_mean_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, (aclTensor*)nullptr, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_var_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, (aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_dataformat_nullptr_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, (const char*)nullptr, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_gamma_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_beta_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_y_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_mean_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_diff_dtype_x_var_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_x_not_4d_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_x_5d_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_gamma_not_1d_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_beta_not_1d_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_gamma_shape_ne_c_nchw_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_gamma_shape_ne_c_nhwc_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_beta_shape_ne_c_nchw_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_beta_shape_ne_c_nhwc_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_dataformat_invalid_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "INVALID";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_dataformat_empty_abnormal)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_nchw_float_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_nchw_float16_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_nhwc_float_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_nhwc_float16_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_nchw_float_small_shape_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_nchw_float_eps_zero_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NCHW";
    double eps = 0.0;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_nchw_float_large_shape_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({4, 32, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({4, 32, 16, 32}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({4, 32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({4, 32, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_empty_x_nchw_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 0, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 0, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_empty_y_nchw_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 0, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_empty_gamma_nchw_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 0, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 0, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 0, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 0, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_empty_mean_nchw_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({0, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_empty_variance_nchw_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({0, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_empty_x_nhwc_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 0, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 0, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_non_contiguous_x_nchw_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND, {512, 64, 16, 1}).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_non_contiguous_x_nhwc_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 4, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND, {512, 128, 8, 1}).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 4, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_platform_not_310p_abnormal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);

    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_instance_norm_test, case_nchw_float16_small_shape_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({1, 2, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({1, 2, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_nhwc_float16_small_shape_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({1, 1, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({1, 1, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_empty_beta_nchw_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_nhwc_float_eps_zero_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    const char* dataFormat = "NHWC";
    double eps = 0.0;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_empty_x_float16_nchw_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 8, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 8, 0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_instance_norm_test, case_empty_x_float16_nhwc_normal)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);

    auto xDesc = TensorDesc({2, 0, 16, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 0, 16, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT16, ACL_FORMAT_ND);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps),
                        OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
// ============================================================================
// Error code verification UT for E*9*** remediation (2026-10-22)
// Captures ErrorManager output via weak-symbol override; asserts no EZ9999.
// (The remediated L0 alloc-failure paths are OOM paths, not fakable here;
//  this net verifies the op's full UT suite generates no EZ9999.)
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
TEST_F(l2_instance_norm_test, error_capture_channel_self_check)
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
class Inln3NoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        error_message::ResetCapture(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
    }
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        // 只守护本文件的测试套件, 框架层或其他算子用例的EZ9999不跨套件追责
        static const char* const guardedSuites[] = {"l2_instance_norm_test"};
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
static const auto _inst_inln3 = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new Inln3NoEz9999Check());
    return 0;
}();
