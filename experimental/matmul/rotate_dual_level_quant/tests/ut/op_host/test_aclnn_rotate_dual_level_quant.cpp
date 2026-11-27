/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include "../../../op_api/aclnn_rotate_dual_level_quant.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {
using namespace op;
struct ApiCase {
    const char* name;
    std::vector<int64_t> x{4, 544}, rot{32, 32}, y{4, 544}, s0{4, 2}, s1{4, 9, 2};
    aclDataType input = ACL_BF16, rotation = ACL_BF16, target = ACL_FLOAT4_E2M1;
    aclDataType scale0 = ACL_FLOAT, scale1 = ACL_FLOAT8_E8M0;
    int64_t axis = -1, alg = 0, groupType = 0, block0 = 512, block1 = 32;
    double dstMax = 0;
    bool trans = false;
    const char* round = nullptr;
    aclnnStatus expected = ACLNN_SUCCESS;
};

class RotateDualLevelQuantAclnn : public testing::TestWithParam<ApiCase> {};
TEST_P(RotateDualLevelQuantAclnn, WorkspaceValidation)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    const auto c = GetParam();
    SCOPED_TRACE(c.name);
    auto ut = OP_API_UT(aclnnRotateDualLevelQuant,
                        INPUT(TensorDesc(c.x, c.input, ACL_FORMAT_ND), TensorDesc(c.rot, c.rotation, ACL_FORMAT_ND),
                              (aclTensor*)nullptr, (aclTensor*)nullptr, c.groupType, c.axis, const_cast<char*>(c.round),
                              c.alg, c.dstMax, c.trans, c.block0, c.block1),
                        OUTPUT(TensorDesc(c.y, c.target, ACL_FORMAT_ND), TensorDesc(c.s0, c.scale0, ACL_FORMAT_ND),
                               TensorDesc(c.s1, c.scale1, ACL_FORMAT_ND)));
    uint64_t workspace = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), c.expected);
    if (c.expected == ACLNN_SUCCESS && c.x.front() == 0) {
        EXPECT_EQ(workspace, 0);
    }
}

std::vector<ApiCase> ApiCases()
{
    std::vector<ApiCase> cases;
    for (auto in : {ACL_FLOAT16, ACL_BF16}) {
        for (auto out : {ACL_FLOAT4_E2M1, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2}) {
            ApiCase c{"valid_dtype"};
            c.input = c.rotation = in;
            c.target = out;
            cases.push_back(c);
        }
    }
    for (int64_t k : {32, 64, 128, 256}) {
        ApiCase c{"k_and_tail"};
        c.x = c.y = {3, 3 * k};
        c.rot = {3, k, k};
        c.s0 = {3, (3 * k + 511) / 512};
        c.s1 = {3, (3 * k + 63) / 64, 2};
        cases.push_back(c);
    }
    ApiCase c{"rank1"};
    c.x = c.y = {512};
    c.rot = {128, 128};
    c.s0 = {1};
    c.s1 = {8, 2};
    cases.push_back(c);
    c = ApiCase{"empty"};
    c.x = c.y = {0, 544};
    c.s0 = {0, 2};
    c.s1 = {0, 9, 2};
    cases.push_back(c);
    c = ApiCase{"fp4_round"};
    c.round = "round";
    cases.push_back(c);
    c = ApiCase{"fp4_floor"};
    c.round = "floor";
    cases.push_back(c);
    c = ApiCase{"fp4_alg2"};
    c.alg = 2;
    c.dstMax = 6;
    cases.push_back(c);
    c = ApiCase{"fp8_alg1"};
    c.target = ACL_FLOAT8_E5M2;
    c.alg = 1;
    cases.push_back(c);
    c = ApiCase{"dtype_mismatch"};
    c.rotation = ACL_FLOAT16;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"level0_type"};
    c.scale0 = ACL_BF16;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"level1_type"};
    c.scale1 = ACL_UINT8;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"level0_shape"};
    c.s0 = {4, 1};
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"level1_shape"};
    c.s1 = {4, 17};
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"unsupported_y"};
    c.target = ACL_INT8;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"axis"};
    c.axis = 0;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"trans"};
    c.trans = true;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"fp8_floor"};
    c.target = ACL_FLOAT8_E4M3FN;
    c.round = "floor";
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"bad_round"};
    c.round = "trunc";
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"bad_alg"};
    c.alg = 1;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"bad_dst_max"};
    c.alg = 2;
    c.dstMax = 5;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"nan_dst_max"};
    c.dstMax = std::numeric_limits<double>::quiet_NaN();
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"block0"};
    c.block0 = 256;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"block1"};
    c.block1 = 64;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"group_type"};
    c.groupType = 2;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    c = ApiCase{"empty_still_validated"};
    c.x = c.y = {0, 544};
    c.s0 = {0, 2};
    c.s1 = {0, 9, 2};
    c.block0 = 256;
    c.expected = ACLNN_ERR_PARAM_INVALID;
    cases.push_back(c);
    return cases;
}
INSTANTIATE_TEST_SUITE_P(A5, RotateDualLevelQuantAclnn, testing::ValuesIn(ApiCases()));

TEST(RotateDualLevelQuantApi, GroupedMetadataAndAlpha)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    for (int64_t type : {0, 1}) {
        for (bool perBlock : {false, true}) {
            auto rot = perBlock ? TensorDesc({3, 17, 32, 32}, ACL_BF16, ACL_FORMAT_ND) :
                                  TensorDesc({3, 32, 32}, ACL_BF16, ACL_FORMAT_ND);
            auto ut = OP_API_UT(
                aclnnRotateDualLevelQuant,
                INPUT(TensorDesc({7, 544}, ACL_BF16, ACL_FORMAT_ND), rot, TensorDesc({1}, ACL_BF16, ACL_FORMAT_ND),
                      TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND), type, 1, (char*)nullptr, 0, 0.0, false, 512, 32),
                OUTPUT(TensorDesc({7, 544}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND),
                       TensorDesc({7, 2}, ACL_FLOAT, ACL_FORMAT_ND),
                       TensorDesc({7, 9, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
            uint64_t bytes = 0;
            EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_SUCCESS);
        }
    }
}

TEST(RotateDualLevelQuantApi, RejectNonA5AndNullExecution)
{
    EXPECT_EQ(aclnnRotateDualLevelQuant(nullptr, 0, nullptr, nullptr), ACLNN_ERR_PARAM_NULLPTR);
    SocVersionManager soc(SocVersion::ASCEND910B);
    auto ut = OP_API_UT(
        aclnnRotateDualLevelQuant,
        INPUT(TensorDesc({4, 512}, ACL_BF16, ACL_FORMAT_ND), TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND),
              (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false, 512, 32),
        OUTPUT(TensorDesc({4, 512}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND), TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND),
               TensorDesc({4, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
    uint64_t bytes = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_ERR_RUNTIME_ERROR);
}

TEST(RotateDualLevelQuantApi, NoncontiguousGroupedOutputs)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    auto ut = OP_API_UT(aclnnRotateDualLevelQuant,
                        INPUT(TensorDesc({3, 544}, ACL_BF16, ACL_FORMAT_ND, {1088, 2}, 0, {3, 1088}),
                              TensorDesc({1, 32, 32}, ACL_BF16, ACL_FORMAT_ND), (aclTensor*)nullptr,
                              TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).Value<int64_t>({1}), 1, -1, (char*)nullptr, 0,
                              0.0, false, 512, 32),
                        OUTPUT(TensorDesc({3, 544}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND, {1088, 2}, 0, {3, 1088}),
                               TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND, {4, 2}, 0, {3, 4}),
                               TensorDesc({3, 9, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, {36, 4, 2}, 0, {3, 9, 4})));
    uint64_t bytes = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_SUCCESS);
}

TEST(RotateDualLevelQuantApi, InvalidOptionalMetadataAndFormat)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    for (bool badAlphaType : {false, true}) {
        auto alpha = badAlphaType ? TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND) :
                                    TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);
        auto ut = OP_API_UT(
            aclnnRotateDualLevelQuant,
            INPUT(TensorDesc({3, 544}, ACL_BF16, ACL_FORMAT_ND), TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND), alpha,
                  (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false, 512, 32),
            OUTPUT(TensorDesc({3, 544}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND), TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND),
                   TensorDesc({3, 9, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
        uint64_t bytes = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_ERR_PARAM_INVALID);
    }
    for (auto groups : {TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND), TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND),
                        TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND), TensorDesc({1, 1}, ACL_INT64, ACL_FORMAT_ND)}) {
        auto ut = OP_API_UT(
            aclnnRotateDualLevelQuant,
            INPUT(TensorDesc({3, 544}, ACL_BF16, ACL_FORMAT_ND), TensorDesc({1, 32, 32}, ACL_BF16, ACL_FORMAT_ND),
                  (aclTensor*)nullptr, groups, 0, -1, (char*)nullptr, 0, 0.0, false, 512, 32),
            OUTPUT(TensorDesc({3, 544}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND), TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND),
                   TensorDesc({3, 9, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
        uint64_t bytes = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_ERR_PARAM_INVALID);
    }
    auto ut = OP_API_UT(
        aclnnRotateDualLevelQuant,
        INPUT(TensorDesc({3, 544}, ACL_BF16, ACL_FORMAT_NCHW), TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND),
              (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false, 512, 32),
        OUTPUT(TensorDesc({3, 544}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND), TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND),
               TensorDesc({3, 9, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
    uint64_t bytes = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_ERR_PARAM_INVALID);
}

TEST(RotateDualLevelQuantApi, NullX)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    auto ut = OP_API_UT(
        aclnnRotateDualLevelQuant,
        INPUT((aclTensor*)nullptr, TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND), (aclTensor*)nullptr,
              (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false, 512, 32),
        OUTPUT(TensorDesc({4, 512}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND), TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND),
               TensorDesc({4, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
    uint64_t bytes = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(RotateDualLevelQuantApi, NullRotation)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    auto ut = OP_API_UT(
        aclnnRotateDualLevelQuant,
        INPUT(TensorDesc({4, 512}, ACL_BF16, ACL_FORMAT_ND), (aclTensor*)nullptr, (aclTensor*)nullptr,
              (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false, 512, 32),
        OUTPUT(TensorDesc({4, 512}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND), TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND),
               TensorDesc({4, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
    uint64_t bytes = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(RotateDualLevelQuantApi, NullY)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    auto ut = OP_API_UT(
        aclnnRotateDualLevelQuant,
        INPUT(TensorDesc({4, 512}, ACL_BF16, ACL_FORMAT_ND), TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND),
              (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false, 512, 32),
        OUTPUT((aclTensor*)nullptr, TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND),
               TensorDesc({4, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
    uint64_t bytes = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(RotateDualLevelQuantApi, NullLevel0Scale)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    auto ut = OP_API_UT(
        aclnnRotateDualLevelQuant,
        INPUT(TensorDesc({4, 512}, ACL_BF16, ACL_FORMAT_ND), TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND),
              (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false, 512, 32),
        OUTPUT(TensorDesc({4, 512}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND), (aclTensor*)nullptr,
               TensorDesc({4, 8, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND)));
    uint64_t bytes = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_ERR_PARAM_NULLPTR);
}

TEST(RotateDualLevelQuantApi, NullLevel1Scale)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    auto ut = OP_API_UT(
        aclnnRotateDualLevelQuant,
        INPUT(TensorDesc({4, 512}, ACL_BF16, ACL_FORMAT_ND), TensorDesc({32, 32}, ACL_BF16, ACL_FORMAT_ND),
              (aclTensor*)nullptr, (aclTensor*)nullptr, 0, -1, (char*)nullptr, 0, 0.0, false, 512, 32),
        OUTPUT(TensorDesc({4, 512}, ACL_FLOAT4_E2M1, ACL_FORMAT_ND), TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND),
               (aclTensor*)nullptr));
    uint64_t bytes = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&bytes), ACLNN_ERR_PARAM_NULLPTR);
}
} // namespace
