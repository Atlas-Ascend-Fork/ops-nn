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
#include "gtest/gtest.h"
#include "../../../../nll_loss_common/op_api/target_value_range.h"
#include "../../../../nll_loss_grad/op_api/aclnn_nll_loss_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "aclnn/acl_meta.h"

using namespace std;
using namespace op;

// ValidateTargetValues 是纯 host 校验函数（target 值域 [0, nClasses)，ignoreIndex 豁免）。
// UT 环境无 device 数据，aclrtMemcpy 回读入口（CheckTargetValueRange）会降级跳过，
// 因此这里直接单测纯 host 部分，覆盖三 dtype 与各分支。
class nll_loss_target_check_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "nll_loss_target_check_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "nll_loss_target_check_test TearDown" << endl; }
};

TEST_F(nll_loss_target_check_test, int64_all_valid)
{
    std::vector<int64_t> targets = {0, 2, 4, 1, 3};
    EXPECT_TRUE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_INT64, 5, -100));
}

TEST_F(nll_loss_target_check_test, int64_negative_out_of_bounds)
{
    std::vector<int64_t> targets = {1, -1, 2};
    EXPECT_FALSE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_INT64, 5, -100));
}

TEST_F(nll_loss_target_check_test, int64_too_large_out_of_bounds)
{
    std::vector<int64_t> targets = {1, 5, 2};
    EXPECT_FALSE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_INT64, 5, -100));
}

TEST_F(nll_loss_target_check_test, int64_ignore_index_exempt)
{
    std::vector<int64_t> targets = {0, -100, 4};
    EXPECT_TRUE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_INT64, 5, -100));
}

TEST_F(nll_loss_target_check_test, int64_custom_ignore_index_exempt)
{
    std::vector<int64_t> targets = {0, 7, 4};
    EXPECT_TRUE(ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_INT64, 5, 7));
}

TEST_F(nll_loss_target_check_test, int32_all_valid)
{
    std::vector<int32_t> targets = {0, 3, 2, 1};
    EXPECT_TRUE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_INT32, 4, -100));
}

TEST_F(nll_loss_target_check_test, int32_out_of_bounds)
{
    std::vector<int32_t> targets = {0, -2, 2};
    EXPECT_FALSE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_INT32, 4, -100));
}

TEST_F(nll_loss_target_check_test, uint8_all_valid)
{
    std::vector<uint8_t> targets = {0, 3, 1, 2};
    EXPECT_TRUE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_UINT8, 4, -100));
}

TEST_F(nll_loss_target_check_test, uint8_out_of_bounds)
{
    std::vector<uint8_t> targets = {0, 4, 2};
    EXPECT_FALSE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_UINT8, 4, -100));
}

TEST_F(nll_loss_target_check_test, empty_target_passes)
{
    std::vector<int64_t> targets = {0};
    EXPECT_TRUE(ValidateTargetValues(targets.data(), 0, DataType::DT_INT64, 5, -100));
}

TEST_F(nll_loss_target_check_test, null_buffer_passes)
{
    EXPECT_TRUE(ValidateTargetValues(nullptr, 3, DataType::DT_INT64, 5, -100));
}

TEST_F(nll_loss_target_check_test, non_positive_classes_passes)
{
    std::vector<int64_t> targets = {0, 1};
    EXPECT_TRUE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_INT64, 0, -100));
}

TEST_F(nll_loss_target_check_test, unsupported_dtype_passes)
{
    std::vector<float> targets = {0.0f, 1.0f};
    EXPECT_TRUE(
        ValidateTargetValues(targets.data(), static_cast<int64_t>(targets.size()), DataType::DT_FLOAT, 5, -100));
}

// CheckTargetValueRange 入口分支直测：覆盖 numClasses<=0 早退、GetData 为空跳过、
// 非连续/带偏移跳过、aclrtMemcpy 回读失败降级（UT 环境无 device 数据，回读必然失败）。
// aclrtMemcpy 在本环境对非法源指针安全返回错误码，不会崩溃（已实测）。
class nll_loss_target_entry_test : public testing::Test {};

TEST_F(nll_loss_target_entry_test, zero_numel_early_return)
{
    auto target = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    EXPECT_TRUE(CheckTargetValueRange(target.get(), 5, -100));
}

TEST_F(nll_loss_target_entry_test, non_positive_classes_early_return)
{
    auto target = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    EXPECT_TRUE(CheckTargetValueRange(target.get(), 0, -100));
    EXPECT_TRUE(CheckTargetValueRange(target.get(), -1, -100));
}

TEST_F(nll_loss_target_entry_test, null_storage_data_skips_check)
{
    // UT 的 aclTensor storage_data 恒为 nullptr，GetData() 为空应跳过校验
    auto target = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    EXPECT_TRUE(CheckTargetValueRange(target.get(), 5, -100));
}

TEST_F(nll_loss_target_entry_test, non_contiguous_skips_check)
{
    // 列主序 strides {1,3} 对 shape {3,2} 非连续（row-major 连续 strides 应为 {2,1}）
    auto target = TensorDesc({3, 2}, ACL_INT64, ACL_FORMAT_ND, {1, 3}, 0, {3, 2}).ToAclType();
    EXPECT_TRUE(CheckTargetValueRange(target.get(), 5, -100));
}

TEST_F(nll_loss_target_entry_test, non_zero_offset_skips_check)
{
    auto target = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND, {1}, 1, {3}).ToAclType();
    EXPECT_TRUE(CheckTargetValueRange(target.get(), 5, -100));
}

// 用 aclCreateTensor 构造 GetData() 非空的连续零偏移 tensor，使 CheckTargetValueRange 走到
// aclrtMemcpy 回读分支。UT 环境无 device 数据，回读失败降级跳过（回读成功路径需真机，不在此覆盖）。
class nll_loss_target_memcpy_test : public testing::Test {};

TEST_F(nll_loss_target_memcpy_test, memcpy_fallback_skips_check)
{
    const int64_t shape = 3;
    const int64_t stride = 1;
    int64_t hostData[3] = {0, 1, 2};
    aclTensor* target = aclCreateTensor(&shape, 1, ACL_INT64, &stride, 0, ACL_FORMAT_ND, &shape, 1, hostData);
    ASSERT_NE(target, nullptr);
    EXPECT_TRUE(CheckTargetValueRange(target, 5, -100));
    aclDestroyTensor(target);
}

// aclnnNLLLossBackward（1D 反向）此前无 op_api UT，PR 新增的 AbandonCache/target 值域校验行
// 未被任何用例执行过，这里补入口用例。数据在 UT 环境回读失败会降级跳过，仅验证主流程畅通。
class nll_loss_backward_entry_test : public testing::Test {};

TEST_F(nll_loss_backward_entry_test, get_workspace_size_success)
{
    auto gradDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnNLLLossBackward,
                        INPUT(gradDesc, selfDesc, targetDesc, weightDesc, reduction, ignoreIndex, totalWeightDesc),
                        OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}
