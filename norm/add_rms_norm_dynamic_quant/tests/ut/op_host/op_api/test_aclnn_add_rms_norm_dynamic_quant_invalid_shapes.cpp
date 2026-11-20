/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_add_rms_norm_dynamic_quant.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {
struct InvalidDynamicQuantShape {
    const char* name;
    std::vector<int64_t> x1;
    std::vector<int64_t> x2;
    std::vector<int64_t> gamma;
    std::vector<int64_t> smooth1;
    std::vector<int64_t> smooth2;
};

class AddRmsNormDynamicQuantInvalidShape : public testing::TestWithParam<InvalidDynamicQuantShape> {};

TEST_P(AddRmsNormDynamicQuantInvalidShape, RejectsInvalidInput)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    const auto& param = GetParam();
    auto x1 = TensorDesc(param.x1, ACL_BF16, ACL_FORMAT_ND);
    auto x2 = TensorDesc(param.x2, ACL_BF16, ACL_FORMAT_ND);
    auto gamma = TensorDesc(param.gamma, ACL_BF16, ACL_FORMAT_ND);
    auto smooth1 = TensorDesc(param.smooth1, ACL_BF16, ACL_FORMAT_ND);
    auto smooth2 = TensorDesc(param.smooth2, ACL_BF16, ACL_FORMAT_ND);
    auto y1 = TensorDesc(param.x1, ACL_INT8, ACL_FORMAT_ND);
    auto y2 = TensorDesc(param.x1, ACL_INT8, ACL_FORMAT_ND);
    auto xOut = TensorDesc(param.x1, ACL_BF16, ACL_FORMAT_ND);
    auto scale1 = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scale2 = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    double epsilon = 1e-5;
    auto ut = OP_API_UT(aclnnAddRmsNormDynamicQuant, INPUT(x1, x2, gamma, smooth1, smooth2, epsilon),
                        OUTPUT(y1, y2, xOut, scale1, scale2));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_INNER_TILING_ERROR);
}

INSTANTIATE_TEST_SUITE_P(
    ErrorReporting, AddRmsNormDynamicQuantInvalidShape,
    testing::Values(InvalidDynamicQuantShape{"X2ShapeMismatch", {2, 8}, {3, 8}, {8}, {8}, {8}},
                    InvalidDynamicQuantShape{"RankBelowMinimum", {8}, {8}, {8}, {8}, {8}},
                    InvalidDynamicQuantShape{
                        "RankAboveMaximum", {1, 1, 1, 1, 1, 1, 1, 2, 8}, {1, 1, 1, 1, 1, 1, 1, 2, 8}, {8}, {8}, {8}},
                    InvalidDynamicQuantShape{"GammaRankMismatch", {2, 8}, {2, 8}, {1, 8}, {8}, {8}},
                    InvalidDynamicQuantShape{"GammaShapeMismatch", {2, 8}, {2, 8}, {9}, {8}, {8}},
                    InvalidDynamicQuantShape{"Smooth1ShapeMismatch", {2, 8}, {2, 8}, {8}, {9}, {8}},
                    InvalidDynamicQuantShape{"Smooth2ShapeMismatch", {2, 8}, {2, 8}, {8}, {8}, {9}}),
    [](const testing::TestParamInfo<InvalidDynamicQuantShape>& info) { return info.param.name; });
} // namespace
