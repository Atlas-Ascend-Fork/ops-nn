/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_foreach_maximum_scalar_list.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_foreach_maximum_scalar_list_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_maximum_scalar_list_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_maximum_scalar_list_test TearDown" << endl; }
};

static aclnnStatus CallGetWorkspaceSize(const aclTensorList* x, const std::vector<float>& scalarsData,
                                        const aclTensorList* out, uint64_t* workspaceSize)
{
    std::vector<aclScalar*> scalars;
    for (float value : scalarsData) {
        float* val = new float(value);
        auto scalar = aclCreateScalar(val, ACL_FLOAT);
        if (scalar == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
        scalars.push_back(scalar);
    }
    aclScalarList* scalarsList = aclCreateScalarList(scalars.data(), scalars.size());
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnForeachMaximumScalarListGetWorkspaceSize(x, scalarsList, const_cast<aclTensorList*>(out),
                                                             workspaceSize, &executor);
    if (executor != nullptr) {
        delete executor;
    }
    aclDestroyScalarList(scalarsList);
    return ret;
}

// A contiguous input must also support a transposed output view.
TEST_F(l2_foreach_maximum_scalar_list_test, foreach_maximum_scalar_list_test_transposed_out)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    uint64_t workspaceSize = 0;
    auto getWorkspaceResult = CallGetWorkspaceSize(DescToAclContainer(xList), {1.0f}, DescToAclContainer(outList),
                                                   &workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_foreach_maximum_scalar_list_test, foreach_maximum_scalar_list_test_non_contiguous_input_and_out)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {9, 2}, 1, {2, 9});
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    uint64_t workspaceSize = 0;
    auto getWorkspaceResult = CallGetWorkspaceSize(DescToAclContainer(xList), {1.0f}, DescToAclContainer(outList),
                                                   &workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// Keep list indices aligned across direct outputs and outputs requiring ViewCopy.
TEST_F(l2_foreach_maximum_scalar_list_test, foreach_maximum_scalar_list_test_mixed_out_layouts)
{
    auto x0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto out2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {7, 2}, 1, {2, 7});
    auto xList = TensorListDesc({x0, x1, x2});
    auto outList = TensorListDesc({out0, out1, out2});

    uint64_t workspaceSize = 0;
    auto getWorkspaceResult = CallGetWorkspaceSize(DescToAclContainer(xList), {1.0f, 2.0f, 3.0f},
                                                   DescToAclContainer(outList), &workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}
