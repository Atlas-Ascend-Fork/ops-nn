/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/platform.h"
#include <memory>
#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_foreach_addcmul_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_foreach_addcmul_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_addcmul_scalar_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_addcmul_scalar_test TearDown" << endl; }
};

// support non-contiguous input and output
TEST_F(l2_foreach_addcmul_scalar_test, ascend910B2_foreach_addcmul_scalar_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    auto scalar = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    vector<int64_t> outDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto x3 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcmulScalar, INPUT(x1List, x2List, x3List, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// Regression coverage: foreach-addcmul-scalar-validation
TEST_F(l2_foreach_addcmul_scalar_test, ascend910B2_scalar_cardinality_and_rank)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND910B);
    struct Case {
        vector<int64_t> shape;
        aclnnStatus expected;
    };
    const vector<Case> cases = {
        {{}, ACLNN_SUCCESS},
        {{1}, ACLNN_SUCCESS},
        {{1, 1}, ACLNN_SUCCESS},
        {{0}, ACLNN_ERR_PARAM_INVALID},
        {{2}, ACLNN_ERR_PARAM_INVALID},
        {{1, 0}, ACLNN_ERR_PARAM_INVALID},
        {{1, 1, 1, 1, 1, 1, 1, 1}, ACLNN_SUCCESS},
        {{1, 1, 1, 1, 1, 1, 1, 1, 1}, ACLNN_ERR_PARAM_INVALID},
    };
    for (const auto& c : cases) {
        auto tensor = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
        auto list = TensorListDesc({tensor});
        auto outList = TensorListDesc({tensor});
        auto scalar = TensorDesc(c.shape, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnForeachAddcmulScalar, INPUT(list, list, list, scalar), OUTPUT(outList));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), c.expected);
    }
}

TEST_F(l2_foreach_addcmul_scalar_test, ascend910B2_empty_lists_keep_valid_scalar_contract)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND910B);
    // TensorListDesc asserts on zero elements; use the public empty-list API.
    unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> list(aclCreateTensorList(nullptr, 0),
                                                                    &aclDestroyTensorList);
    ASSERT_NE(list.get(), nullptr);
    for (const auto& shape : {vector<int64_t>{}, vector<int64_t>{0}, vector<int64_t>{2}}) {
        auto scalar = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
        ASSERT_NE(scalar.get(), nullptr);
        uint64_t workspaceSize = 99;
        aclOpExecutor* executor = nullptr;
        auto status = aclnnForeachAddcmulScalarGetWorkspaceSize(list.get(), list.get(), list.get(), scalar.get(),
                                                                list.get(), &workspaceSize, &executor);
        unique_ptr<aclOpExecutor> executorGuard(executor);
        EXPECT_EQ(status, shape.empty() ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID);
        if (shape.empty()) {
            EXPECT_EQ(workspaceSize, 0U);
        }
    }
}
