/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>

#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_add_rms_norm_dynamic_quant_v2.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "opdev/platform.h"

using namespace std;

class l2_add_rms_norm_dynamic_quant_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "add_rms_norm_dynamic_quant_v2_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "add_rms_norm_dynamic_quant_v2_test TearDown" << endl; }
};

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910B_case_001)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910B_case_002)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910B_case_003)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910B_case_004)
{
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    double eps = 1e-5;

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend950_case_001)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend950_case_002)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend950_case_003)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend950_case_004)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend950_case_005)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend950_case_006)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);

    auto tensor_desc_y1 = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, error_case_001)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    double eps = 1e-5;
    auto output_mask = BoolArrayDesc(vector<bool>{false, true});

    auto tensor_desc_y1 = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, output_mask),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, error_case_002)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    double eps = 1e-5;
    auto output_mask = BoolArrayDesc(vector<bool>{false, false});

    auto tensor_desc_y1 = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, output_mask),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, error_case_003)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    double eps = 1e-5;

    auto tensor_desc_y1 = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, nullptr, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, error_case_004)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    double eps = 1e-5;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true});

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, output_mask),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, error_case_005)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s1 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_s2 = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    double eps = 1e-5;
    auto output_mask = BoolArrayDesc(vector<bool>{true, true});

    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, tensor_desc_s1, tensor_desc_s2,
                              tensor_desc_beta, eps, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, dtype_relation_x2_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, dtype_relation_beta_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_beta = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, tensor_desc_beta, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, dtype_relation_xout_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, inactive_output_null_rejected)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{false, true});
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicQuantV2,
        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr, (aclTensor*)nullptr,
              (aclTensor*)nullptr, 1e-5, output_mask),
        OUTPUT((aclTensor*)nullptr, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910b_inactive_output_null_rejected)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{false, true});
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnAddRmsNormDynamicQuantV2,
        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr, (aclTensor*)nullptr,
              (aclTensor*)nullptr, 1e-5, output_mask),
        OUTPUT((aclTensor*)nullptr, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend950_rank_zero_x_rejected)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_INNER_TILING_ERROR);
}

// A2 上 x2 dtype 与 x1 不一致应在首段返回 161002（修复前一致性检查被 IsRegbase 门控，
// 混配首段放行、执行段以资料外内部码 561000 失败，issue #8730）
TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910b_dtype_relation_x2_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// A2 上 gamma dtype 与 x1 不一致应在首段返回 161002（同 issue #8730 机制）
TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910b_dtype_relation_gamma_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// A2 上 INT4 原生输出且 x1 尾轴为奇数（513）应在首段返回 161002：资料 :143 约束
// 「输出 INT4 时 x1 尾轴必须能被 2 整除」修复前在 host 侧无对应实现、违约形态完整放行（issue #8732）
TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910b_int4_odd_tail_rejected)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_desc_x1 = TensorDesc({8, 513}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 513}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({513}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({8, 513}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 513}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 513}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// 对照：INT4 原生输出且 x1 尾轴为偶数（514）不被新增奇偶校验拦截。
// 偶数尾轴的最终结果取决于平台对尾轴对齐的既有要求（SUCCESS 或 INNER_TILING_ERROR），
// 故断言失败码不等于奇数尾轴触发的 PARAM_INVALID，即可证明新增校验仅按奇偶生效
TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910b_int4_even_tail_passes_parity_check)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_desc_x1 = TensorDesc({8, 514}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 514}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({514}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({8, 514}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({8, 514}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 514}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_NE(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_INVALID);
}

// A2 上无效输出占位（mask 缺省且 smoothScale 缺席，processOut2=false）的 dtype 不再受
// y 侧白名单/一致性约束：y2Out 占位异 dtype（INT4）应放行（修复前误拒 161002，issue #8728）
TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910b_inactive_y2_int4_placeholder_success)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({1}, ACL_INT4, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

// 同 issue #8728：占位 INT32 同样放行
TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910b_inactive_y2_int32_placeholder_success)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

// A2 上 y2Out 为有效输出（mask=[0,1]）却传空 tensor：不再触发静默早退，
// 放行到 shape 校验按 561002 拒绝，与 y1Out 空的行为对称（issue #8729）
TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910b_active_y2_empty_rejected)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{false, true});
    auto tensor_desc_y1 = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({0, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, output_mask),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_INNER_TILING_ERROR);
}

// A2 上 y2Out 为无效输出占位传空（mask=[1,0]）：不再静默早退，继续计算其余有效输出（issue #8729）
TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend910b_inactive_y2_empty_proceeds)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND910B);
    auto tensor_desc_x1 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto output_mask = BoolArrayDesc(vector<bool>{true, false});
    auto tensor_desc_y1 = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({0, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, output_mask),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACL_SUCCESS);
}

TEST_F(l2_add_rms_norm_dynamic_quant_v2_test, ascend950_rank_zero_active_int32_output_rejected)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    auto tensor_desc_x1 = TensorDesc({2, 8}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_x2 = TensorDesc({2, 8}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_gamma = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_y1 = TensorDesc({}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_y2 = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_desc_x = TensorDesc({2, 8}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_desc_scale1 = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_desc_scale2 = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuantV2,
                        INPUT(tensor_desc_x1, tensor_desc_x2, tensor_desc_gamma, (aclTensor*)nullptr,
                              (aclTensor*)nullptr, (aclTensor*)nullptr, 1e-5, nullptr),
                        OUTPUT(tensor_desc_y1, tensor_desc_y2, tensor_desc_x, tensor_desc_scale1, tensor_desc_scale2));

    uint64_t workspace_size = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_INNER_TILING_ERROR);
}
