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
#include "../../../op_api/aclnn_renorm.h"
#include "../../../op_api/aclnn_embedding_renorm.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

namespace {
struct InvalidRenormParameter {
    const char* name;
    std::vector<int64_t> outShape;
    aclFormat outFormat;
    float p;
};

class RenormInvalidParameter : public testing::TestWithParam<InvalidRenormParameter> {};

TEST_P(RenormInvalidParameter, RejectsInvalidInput)
{
    const auto& param = GetParam();
    auto self = TensorDesc({1, 1, 3, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc(param.outShape, ACL_FLOAT16, param.outFormat);
    auto p = ScalarDesc(param.p);
    auto maxNorm = ScalarDesc(5.0f);
    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnRenorm, INPUT(self, p, dim, maxNorm), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

INSTANTIATE_TEST_SUITE_P(
    ErrorReporting, RenormInvalidParameter,
    testing::Values(InvalidRenormParameter{"OutputShapeMismatch", {1, 1, 3, 4}, ACL_FORMAT_ND, 1.0f},
                    InvalidRenormParameter{"OutputFormatMismatch", {1, 1, 3, 5}, ACL_FORMAT_NCHW, 1.0f},
                    InvalidRenormParameter{"NegativeP", {1, 1, 3, 5}, ACL_FORMAT_ND, -1.0f},
                    InvalidRenormParameter{"OutputRankExceeded", {1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FORMAT_ND, 1.0f}),
    [](const testing::TestParamInfo<InvalidRenormParameter>& info) { return info.param.name; });

class EmbeddingRenormInvalidParameter : public testing::Test {};

TEST_F(EmbeddingRenormInvalidParameter, IndicesRankExceeded)
{
    auto self = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_INT64, ACL_FORMAT_ND);
    double maxNorm = 1.0;
    double normType = 1.0;
    auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(self, indices, maxNorm, normType), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(EmbeddingRenormInvalidParameter, FractalNzStorageFormat)
{
    auto self = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    auto indices = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
    double maxNorm = 1.0;
    double normType = 1.0;
    auto ut = OP_API_UT(aclnnEmbeddingRenorm, INPUT(self, indices, maxNorm, normType), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}
} // namespace
