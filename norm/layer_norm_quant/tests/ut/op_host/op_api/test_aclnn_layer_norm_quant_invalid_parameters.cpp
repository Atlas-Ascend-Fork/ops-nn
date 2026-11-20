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
#include "../../../../op_host/op_api/aclnn_layer_norm_quant.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {
struct InvalidQuantParameter {
    const char* name;
    aclDataType xType;
    aclDataType gammaType;
    aclDataType betaType;
    std::vector<int64_t> xShape;
    std::vector<int64_t> gammaShape;
};

class LayerNormQuantInvalidParameter : public testing::TestWithParam<InvalidQuantParameter> {};

TEST_P(LayerNormQuantInvalidParameter, RejectsInvalidInput)
{
    const auto& param = GetParam();
    auto x = TensorDesc(param.xShape, param.xType, ACL_FORMAT_ND);
    auto gamma = TensorDesc(param.gammaShape, param.gammaType, ACL_FORMAT_ND);
    auto beta = TensorDesc(param.gammaShape, param.betaType, ACL_FORMAT_ND);
    auto scale = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto zeroPoint = TensorDesc({1}, ACL_INT8, ACL_FORMAT_ND);
    auto y = TensorDesc(param.xShape, ACL_INT8, ACL_FORMAT_ND);
    auto scaleOut = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    int quantMode = 0;
    double epsilon = 1e-6;
    auto ut = OP_API_UT(aclnnLayerNormQuant, INPUT(x, gamma, beta, scale, zeroPoint, quantMode, epsilon),
                        OUTPUT(y, scaleOut));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

INSTANTIATE_TEST_SUITE_P(
    ErrorReporting, LayerNormQuantInvalidParameter,
    testing::Values(InvalidQuantParameter{"UnsupportedXType", ACL_INT32, ACL_INT32, ACL_INT32, {2, 8}, {1, 8}},
                    InvalidQuantParameter{"GammaTypeMismatch", ACL_FLOAT16, ACL_BF16, ACL_FLOAT16, {2, 8}, {1, 8}},
                    InvalidQuantParameter{"BetaTypeMismatch", ACL_FLOAT16, ACL_FLOAT16, ACL_BF16, {2, 8}, {1, 8}},
                    InvalidQuantParameter{
                        "GammaRankExceeded", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, {2, 8}, {1, 1, 8}},
                    InvalidQuantParameter{
                        "XRankExceeded", ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, {1, 1, 1, 1, 1, 1, 1, 1, 8}, {1, 8}}),
    [](const testing::TestParamInfo<InvalidQuantParameter>& info) { return info.param.name; });
} // namespace
