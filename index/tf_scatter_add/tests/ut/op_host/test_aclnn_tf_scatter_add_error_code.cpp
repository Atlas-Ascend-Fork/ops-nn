/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "../../../op_host/op_api/aclnn_tf_scatter_add.h"

#include "base/err_mgr.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

namespace {
// The shared API UT executable links nnopbase, which loads liberror_manager.so at
// runtime. Declare the ErrorManager entry points as weak symbols so this UT can
// read raw messages without adding error_manager to the shared CMake target.
extern "C" {
int32_t ErrMgrInitWeak(error_message::ErrorMessageMode mode) asm(
    "_ZN13error_message10ErrMgrInitENS_16ErrorMessageModeE") __attribute__((weak));
error_message::unique_const_char_array GetErrMgrErrorMessageWeak() asm("_ZN13error_message21GetErrMgrErrorMessageEv")
    __attribute__((weak));
std::vector<error_message::ErrMsgRawItem> GetErrMgrRawErrorMessagesWeak() asm(
    "_ZN13error_message25GetErrMgrRawErrorMessagesEv") __attribute__((weak));
}
} // namespace

class l2_tf_scatter_add_error_code_test : public testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_NE(ErrMgrInitWeak, nullptr) << "liberror_manager.so is not loaded";
        ASSERT_NE(GetErrMgrErrorMessageWeak, nullptr) << "GetErrMgrErrorMessage is unavailable";
        ASSERT_NE(GetErrMgrRawErrorMessagesWeak, nullptr) << "GetErrMgrRawErrorMessages is unavailable";
    }

    static void ClearErrorMessages()
    {
        (void)ErrMgrInitWeak(error_message::ErrorMessageMode::INTERNAL_MODE);
        (void)GetErrMgrErrorMessageWeak();
        (void)GetErrMgrRawErrorMessagesWeak();
    }

    static string RawItemToString(const error_message::ErrMsgRawItem& item)
    {
        string text;
        if (item.error_id != nullptr) {
            text += item.error_id.get();
        }
        if (item.error_message != nullptr) {
            text += " ";
            text += item.error_message.get();
        }
        for (size_t i = 0; i < item.args_key.size(); ++i) {
            if (item.args_key[i] == nullptr) {
                continue;
            }
            text += " ";
            text += item.args_key[i].get();
            text += "=";
            if (i < item.args_value.size() && item.args_value[i] != nullptr) {
                text += item.args_value[i].get();
            }
        }
        return text;
    }

    static string RawItemsToString(const vector<error_message::ErrMsgRawItem>& items)
    {
        string text;
        for (const auto& item : items) {
            text += RawItemToString(item);
            text += "\n";
        }
        return text;
    }

    static bool HasCode(const vector<error_message::ErrMsgRawItem>& items, const string& code)
    {
        for (const auto& item : items) {
            if (item.error_id != nullptr && string(item.error_id.get()) == code) {
                return true;
            }
        }
        return false;
    }

    static void ExpectErrorCode(const string& code, const vector<string>& fields)
    {
        const auto items = GetErrMgrRawErrorMessagesWeak();
        const string dump = RawItemsToString(items);
        bool found = false;
        for (const auto& item : items) {
            if (item.error_id == nullptr || string(item.error_id.get()) != code) {
                continue;
            }
            const string text = RawItemToString(item);
            bool allFieldsFound = true;
            for (const auto& field : fields) {
                if (text.find(field) == string::npos) {
                    allFieldsFound = false;
                    break;
                }
            }
            if (allFieldsFound) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "expect code " << code << ", raw messages:\n" << dump;
        EXPECT_FALSE(HasCode(items, "EZ1001")) << "old generic code still reported, raw messages:\n" << dump;
    }
};

// 1. CheckDtypeValid: updates/varRef dtype mismatch -> EZ0021
TEST_F(l2_tf_scatter_add_error_code_test, dtype_mismatch_reports_ez0021)
{
    auto varRefDesc = TensorDesc({3, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updatesDesc = TensorDesc({2, 3, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRefDesc, indicesDesc, updatesDesc), OUTPUT());

    ClearErrorMessages();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectErrorCode("EZ0021", {"updates, varRef", "FLOAT16", "FLOAT", "must have the same dtype"});
}

// 2. CheckShapeForScatterAdd: varRef.shape[0] == 0 but indices is not empty -> EZ0010
TEST_F(l2_tf_scatter_add_error_code_test, empty_var_first_dim_reports_ez0010)
{
    auto varRefDesc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indicesDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updatesDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRefDesc, indicesDesc, updatesDesc), OUTPUT());

    ClearErrorMessages();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectErrorCode("EZ0010", {"varRef, indices", "[0, 3]", "[2]", "when varRef.shape[0] is 0"});
}

// 3. CheckShapeForScatterAdd: updates rank relation mismatch -> EZ0013
TEST_F(l2_tf_scatter_add_error_code_test, scatter_add_rank_mismatch_reports_ez0013)
{
    auto varRefDesc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updatesDesc = TensorDesc({2, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRefDesc, indicesDesc, updatesDesc), OUTPUT());

    ClearErrorMessages();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectErrorCode("EZ0013", {"updates, indices, varRef", "indices rank + varRef rank - 1"});
}

// 4. CheckShapeForScatterAdd: updates dims do not match indices dims -> EZ0010
TEST_F(l2_tf_scatter_add_error_code_test, scatter_add_indices_dim_mismatch_reports_ez0010)
{
    auto varRefDesc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updatesDesc = TensorDesc({3, 3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRefDesc, indicesDesc, updatesDesc), OUTPUT());

    ClearErrorMessages();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectErrorCode("EZ0010", {"updates, indices", "first indices rank dimensions"});
}

// 5. CheckShapeForScatterAdd: updates dims do not match varRef dims -> EZ0010
TEST_F(l2_tf_scatter_add_error_code_test, scatter_add_var_dim_mismatch_reports_ez0010)
{
    auto varRefDesc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updatesDesc = TensorDesc({2, 3, 3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRefDesc, indicesDesc, updatesDesc), OUTPUT());

    ClearErrorMessages();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectErrorCode("EZ0010", {"updates, varRef", "after the first varRef dimension"});
}

// 6. CheckShapeForScatterNdAdd: indices.shape[-1] > varRef rank -> EZ0012
TEST_F(l2_tf_scatter_add_error_code_test, scatter_nd_indices_last_dim_gt_var_rank_reports_ez0012)
{
    auto varRefDesc = TensorDesc({3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updatesDesc = TensorDesc({2, 4, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRefDesc, indicesDesc, updatesDesc), OUTPUT());

    ClearErrorMessages();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectErrorCode("EZ0012", {"varRef", "indices last dim 3"});
}

// 7. CheckShapeForScatterNdAdd: updates rank relation mismatch -> EZ0013
TEST_F(l2_tf_scatter_add_error_code_test, scatter_nd_rank_mismatch_reports_ez0013)
{
    auto varRefDesc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updatesDesc = TensorDesc({2, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRefDesc, indicesDesc, updatesDesc), OUTPUT());

    ClearErrorMessages();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectErrorCode("EZ0013", {"updates, indices, varRef", "indices rank - 1"});
}

// 8. CheckShapeForScatterNdAdd: updates dims do not match indices dims -> EZ0010
TEST_F(l2_tf_scatter_add_error_code_test, scatter_nd_indices_dim_mismatch_reports_ez0010)
{
    auto varRefDesc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updatesDesc = TensorDesc({3, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRefDesc, indicesDesc, updatesDesc), OUTPUT());

    ClearErrorMessages();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectErrorCode("EZ0010", {"updates, indices", "all but the last indices dimension"});
}

// 9. CheckShapeForScatterNdAdd: updates dims do not match varRef dims -> EZ0010
TEST_F(l2_tf_scatter_add_error_code_test, scatter_nd_var_dim_mismatch_reports_ez0010)
{
    auto varRefDesc = TensorDesc({3, 3, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 3);
    auto updatesDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto ut = OP_API_UT(aclnnTfScatterAdd, INPUT(varRefDesc, indicesDesc, updatesDesc), OUTPUT());

    ClearErrorMessages();
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectErrorCode("EZ0010", {"updates, varRef", "after indices.shape[-1] dimensions"});
}
