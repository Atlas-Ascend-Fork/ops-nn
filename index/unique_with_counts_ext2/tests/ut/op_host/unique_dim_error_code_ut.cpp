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
#include <iterator>
#include <regex>
#include <string>
#include <vector>

#include "../../../op_host/op_api/aclnn_unique_dim.h"
#include "base/err_mgr.h"
#include "opdev/op_log.h"

namespace {
aclTensor* CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dataType)
{
    std::vector<int64_t> stride(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        stride[i - 2] = stride[i - 1] * shape[i - 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), nullptr);
}

void ClearUniqueDimErrors()
{
    const auto initRet = error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    ASSERT_EQ(initRet, 0) << "ErrorManager initialization failed, ret=" << initRet;
    (void)error_message::GetErrMgrErrorMessage();
}

void ExpectUniqueDimError(const std::string& code, const std::vector<std::string>& fields)
{
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    std::string text;
    size_t targetCodeCount = 0;
    size_t formalCodeCount = 0;
    for (const auto& error : errors) {
        const std::string errorId = error.error_id == nullptr ? "" : error.error_id.get();
        text += errorId + " ";
        text += error.error_message == nullptr ? "" : error.error_message.get();
        for (size_t i = 0; i < error.args_key.size(); ++i) {
            text += " " + std::string(error.args_key[i].get()) + "=" + std::string(error.args_value[i].get());
        }
        targetCodeCount += errorId == code ? 1 : 0;
        formalCodeCount += errorId.rfind("EZ", 0) == 0 ? 1 : 0;
    }
    EXPECT_EQ(targetCodeCount, 1) << text;
    EXPECT_EQ(formalCodeCount, 1) << text;
    EXPECT_FALSE(std::regex_search(text, std::regex("E[A-Z0-9]9[0-9]{3}"))) << text;
    for (const auto& field : fields) {
        EXPECT_NE(text.find(field), std::string::npos) << text;
    }
}

aclnnStatus GetWorkspaceSize(const aclTensor* self, int64_t dim, aclTensor* valueOut, aclTensor* inverseOut,
                             aclTensor* countsOut)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return aclnnUniqueDimGetWorkspaceSize(self, true, true, dim, valueOut, inverseOut, countsOut, &workspaceSize,
                                          &executor);
}
} // namespace

TEST(UniqueDimErrorCode, NullSelfReportsRequiredInput)
{
    auto valueOut = CreateAclTensor({2, 5}, ACL_FLOAT);
    auto inverseOut = CreateAclTensor({2}, ACL_INT64);
    auto countsOut = CreateAclTensor({2}, ACL_INT64);
    ASSERT_NE(valueOut, nullptr);
    ASSERT_NE(inverseOut, nullptr);
    ASSERT_NE(countsOut, nullptr);

    ClearUniqueDimErrors();
    EXPECT_EQ(GetWorkspaceSize(nullptr, 0, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_NULLPTR);
    ExpectUniqueDimError("EZ0004", {"aclnnUniqueDim", "self"});

    aclDestroyTensor(valueOut);
    aclDestroyTensor(inverseOut);
    aclDestroyTensor(countsOut);
}

TEST(UniqueDimErrorCode, NullWorkspaceOutputsReportRequiredInput)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ClearUniqueDimErrors();
    EXPECT_EQ(aclnnUniqueDimGetWorkspaceSize(nullptr, true, true, 0, nullptr, nullptr, nullptr, nullptr, &executor),
              ACLNN_ERR_PARAM_NULLPTR);
    ExpectUniqueDimError("EZ0004", {"aclnnUniqueDim", "workspaceSize"});

    ClearUniqueDimErrors();
    EXPECT_EQ(
        aclnnUniqueDimGetWorkspaceSize(nullptr, true, true, 0, nullptr, nullptr, nullptr, &workspaceSize, nullptr),
        ACLNN_ERR_PARAM_NULLPTR);
    ExpectUniqueDimError("EZ0004", {"aclnnUniqueDim", "executor"});
}

TEST(UniqueDimErrorCode, UnsupportedSelfDtypeReportsDtypeError)
{
    auto self = CreateAclTensor({2, 5}, ACL_COMPLEX64);
    auto valueOut = CreateAclTensor({2, 5}, ACL_COMPLEX64);
    auto inverseOut = CreateAclTensor({2}, ACL_INT64);
    auto countsOut = CreateAclTensor({2}, ACL_INT64);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(valueOut, nullptr);
    ASSERT_NE(inverseOut, nullptr);
    ASSERT_NE(countsOut, nullptr);

    ClearUniqueDimErrors();
    EXPECT_EQ(GetWorkspaceSize(self, 0, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_INVALID);
    ExpectUniqueDimError("EZ0020", {"aclnnUniqueDim", "self", "COMPLEX64", "dtype must be one of"});

    aclDestroyTensor(self);
    aclDestroyTensor(valueOut);
    aclDestroyTensor(inverseOut);
    aclDestroyTensor(countsOut);
}

TEST(UniqueDimErrorCode, InvalidInverseDtypeReportsExpectedDtype)
{
    auto self = CreateAclTensor({2, 5}, ACL_FLOAT);
    auto valueOut = CreateAclTensor({2, 5}, ACL_FLOAT);
    auto inverseOut = CreateAclTensor({2}, ACL_INT8);
    auto countsOut = CreateAclTensor({2}, ACL_INT64);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(valueOut, nullptr);
    ASSERT_NE(inverseOut, nullptr);
    ASSERT_NE(countsOut, nullptr);

    ClearUniqueDimErrors();
    EXPECT_EQ(GetWorkspaceSize(self, 0, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_INVALID);
    ExpectUniqueDimError("EZ0019", {"aclnnUniqueDim", "inverseOut", "INT8", "int64"});

    aclDestroyTensor(self);
    aclDestroyTensor(valueOut);
    aclDestroyTensor(inverseOut);
    aclDestroyTensor(countsOut);
}

TEST(UniqueDimErrorCode, MismatchedValueDtypeReportsDtypeRelation)
{
    auto self = CreateAclTensor({2, 5}, ACL_FLOAT);
    auto valueOut = CreateAclTensor({2, 5}, ACL_FLOAT16);
    auto inverseOut = CreateAclTensor({2}, ACL_INT64);
    auto countsOut = CreateAclTensor({2}, ACL_INT64);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(valueOut, nullptr);
    ASSERT_NE(inverseOut, nullptr);
    ASSERT_NE(countsOut, nullptr);

    ClearUniqueDimErrors();
    EXPECT_EQ(GetWorkspaceSize(self, 0, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_INVALID);
    ExpectUniqueDimError("EZ0021", {"aclnnUniqueDim", "self, valueOut", "FLOAT", "FLOAT16"});

    aclDestroyTensor(self);
    aclDestroyTensor(valueOut);
    aclDestroyTensor(inverseOut);
    aclDestroyTensor(countsOut);
}

TEST(UniqueDimErrorCode, OversizedRankReportsShapeDimError)
{
    const std::vector<int64_t> shape(10, 1);
    auto self = CreateAclTensor(shape, ACL_FLOAT);
    auto valueOut = CreateAclTensor(shape, ACL_FLOAT);
    auto inverseOut = CreateAclTensor({1}, ACL_INT64);
    auto countsOut = CreateAclTensor({1}, ACL_INT64);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(valueOut, nullptr);
    ASSERT_NE(inverseOut, nullptr);
    ASSERT_NE(countsOut, nullptr);

    ClearUniqueDimErrors();
    EXPECT_EQ(GetWorkspaceSize(self, 0, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_INVALID);
    ExpectUniqueDimError("EZ0012", {"aclnnUniqueDim", "self", "10", "must not exceed 8"});

    aclDestroyTensor(self);
    aclDestroyTensor(valueOut);
    aclDestroyTensor(inverseOut);
    aclDestroyTensor(countsOut);
}

TEST(UniqueDimErrorCode, InvalidDimReportsValueRange)
{
    auto self = CreateAclTensor({2, 5}, ACL_FLOAT);
    auto valueOut = CreateAclTensor({2, 5}, ACL_FLOAT);
    auto inverseOut = CreateAclTensor({2}, ACL_INT64);
    auto countsOut = CreateAclTensor({2}, ACL_INT64);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(valueOut, nullptr);
    ASSERT_NE(inverseOut, nullptr);
    ASSERT_NE(countsOut, nullptr);

    ClearUniqueDimErrors();
    EXPECT_EQ(GetWorkspaceSize(self, 2, valueOut, inverseOut, countsOut), ACLNN_ERR_PARAM_INVALID);
    ExpectUniqueDimError("EZ0024", {"aclnnUniqueDim", "dim", "2", "[-2, 1]"});

    aclDestroyTensor(self);
    aclDestroyTensor(valueOut);
    aclDestroyTensor(inverseOut);
    aclDestroyTensor(countsOut);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
