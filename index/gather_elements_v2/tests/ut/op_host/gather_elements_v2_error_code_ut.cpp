/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "../../../op_host/op_api/gather_elements_v2.h"
#include "base/err_mgr.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

class GatherElementsV2ErrorCode : public testing::Test {
protected:
    static aclTensor* CreateAclTensor(const std::vector<int64_t>& viewShape, const std::vector<int64_t>& stride,
                                      int64_t offset, const std::vector<int64_t>& storageShape,
                                      aclDataType dataType = ACL_FLOAT)
    {
        return aclCreateTensor(viewShape.data(), viewShape.size(), dataType, stride.data(), offset, ACL_FORMAT_ND,
                               storageShape.data(), storageShape.size(), nullptr);
    }
};

namespace {
void ClearGatherApiErrors()
{
    const auto initRet = error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    ASSERT_EQ(initRet, 0) << "ErrorManager initialization failed, ret=" << initRet;
    (void)error_message::GetErrMgrErrorMessage();
}

void ExpectGatherApiError(const std::string& code, const std::vector<std::string>& fields)
{
    auto message = error_message::GetErrMgrErrorMessage();
    ASSERT_NE(message, nullptr);
    const std::string text(message.get());
    const auto firstCode = text.find(code);
    ASSERT_NE(firstCode, std::string::npos) << "ErrorManager text=[" << text << "]";
    EXPECT_EQ(text.find(code, firstCode + code.size()), std::string::npos) << text;
    EXPECT_FALSE(std::regex_search(text, std::regex("E[A-Z0-9]9[0-9]{3}"))) << text;
    const std::regex codePattern("E[A-Z0-9][0-9]{4}");
    EXPECT_EQ(std::distance(std::sregex_iterator(text.begin(), text.end(), codePattern), std::sregex_iterator()), 1)
        << text;
    for (const auto& field : fields) {
        EXPECT_NE(text.find(field), std::string::npos) << text;
    }
}
} // namespace

TEST_F(GatherElementsV2ErrorCode, gather_elements_v2_null_input_reports_parameter_error)
{
    auto tensor = CreateAclTensor({2, 3}, {3, 1}, 0, {2, 3});
    ASSERT_NE(tensor, nullptr);
    auto executor = CREATE_EXECUTOR();
    ASSERT_NE(executor.get(), nullptr);
    ClearGatherApiErrors();
    EXPECT_EQ(l0op::GatherElementsV2(nullptr, tensor, 1, executor.get()), nullptr);
    ExpectGatherApiError("EZ1011", {"GatherElementsV2", "self"});
    ClearGatherApiErrors();
    EXPECT_EQ(l0op::GatherElementsV2(tensor, nullptr, 1, executor.get()), nullptr);
    ExpectGatherApiError("EZ1011", {"GatherElementsV2", "index"});
    aclDestroyTensor(tensor);
}

TEST_F(GatherElementsV2ErrorCode, gather_elements_v2_unsupported_soc_reports_capability_error)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310P);
    auto self = CreateAclTensor({2, 3}, {3, 1}, 0, {2, 3});
    auto index = CreateAclTensor({2, 3}, {3, 1}, 0, {2, 3}, ACL_INT32);
    ASSERT_NE(self, nullptr);
    ASSERT_NE(index, nullptr);
    auto executor = CREATE_EXECUTOR();
    ASSERT_NE(executor.get(), nullptr);
    ClearGatherApiErrors();
    EXPECT_EQ(l0op::GatherElementsV2(self, index, 1, executor.get()), nullptr);
    ExpectGatherApiError("EZ1009", {"GatherElementsV2", "actual SoC", "ASCEND910B", "ASCEND910_93", "dtype", "format"});
    aclDestroyTensor(self);
    aclDestroyTensor(index);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
