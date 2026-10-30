/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_trans_quant_param.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "base/err_mgr.h"

using namespace std;

class l2_trans_quant_param_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_trans_quant_param_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_trans_quant_param_test TearDown" << endl; }
};

// ErrorManager raw message helpers (replaces weak-symbol capture)
static std::vector<std::string> g_ezCodes;
static bool g_ezCodesLoaded = false;

static void InitErrMgr()
{
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrErrorMessage();
    g_ezCodes.clear();
    g_ezCodesLoaded = false;
}

static void LoadEZCodes()
{
    if (g_ezCodesLoaded) {
        return;
    }
    g_ezCodes.clear();
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    for (const auto& e : errors) {
        g_ezCodes.push_back(e.error_id == nullptr ? std::string() : std::string(e.error_id.get()));
    }
    g_ezCodesLoaded = true;
}

static bool HasEZCode(const std::string& code)
{
    LoadEZCodes();
    for (const auto& id : g_ezCodes) {
        if (id == code) {
            return true;
        }
    }
    return false;
}

static int GetEZCodeCount()
{
    LoadEZCodes();
    return static_cast<int>(g_ezCodes.size());
}

TEST_F(l2_trans_quant_param_test, ascend910B2_test_normal_input1)
{
    float scaleArray[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    uint64_t scaleSize = 5;
    float* offsetArray = nullptr;
    uint64_t offsetSize = 0;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    aclnnStatus aclRet = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(resultSize, 16);
    uint64_t golden[16] = {70369810369741,
                           70369818758349,
                           70369823372083,
                           70369827146957,
                           70369829453824,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0,
                           0};
    for (auto i = 0; i < resultSize; ++i) {
        EXPECT_EQ(result[i], golden[i]);
    }
    free(result);
}

TEST_F(l2_trans_quant_param_test, ascend910B2_test_normal_input2)
{
    float scaleArray[3] = {1.0, 1.0, 1.0};
    uint64_t scaleSize = 3;
    float offsetArray[3] = {1.0, 2.0, 3.0};
    uint64_t offsetSize = 3;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    aclnnStatus aclRet = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(resultSize, 16);
    uint64_t golden[16] = {70507248484352, 70644687437824, 70782126391296, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (auto i = 0; i < resultSize; ++i) {
        EXPECT_EQ(result[i], golden[i]);
    }
    free(result);
}

TEST_F(l2_trans_quant_param_test, ascend910B2_test_normal_input3)
{
    float scaleArray[1] = {1.1};
    uint64_t scaleSize = 1;
    float offsetArray[1] = {0.0};
    uint64_t offsetSize = 1;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    aclnnStatus aclRet = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(resultSize, 1);
    uint64_t golden[1] = {70369810369741};
    for (auto i = 0; i < resultSize; ++i) {
        EXPECT_EQ(result[i], golden[i]);
    }
    free(result);
}

TEST_F(l2_trans_quant_param_test, ascend910B2_test_abnormal_input_scaleSize)
{
    float scaleArray[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    uint64_t scaleSize = 0;
    float* offsetArray = nullptr;
    uint64_t offsetSize = 0;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    aclnnStatus aclRet = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_quant_param_test, ascend910B2_test_abnormal_input_offsetSize)
{
    float* scaleArray = nullptr;
    uint64_t scaleSize = 0;
    float offsetArray[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    uint64_t offsetSize = 0;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    aclnnStatus aclRet = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_quant_param_test, ascend910B2_test_abnormal_input_Size)
{
    float* scaleArray = nullptr;
    uint64_t scaleSize = 0;
    float* offsetArray = nullptr;
    uint64_t offsetSize = 0;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    aclnnStatus aclRet = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_quant_param_test, ascend910B2_test_abnormal_input_scaleSize_with_nullptr)
{
    float* scaleArray = nullptr;
    uint64_t scaleSize = 5;
    float offsetArray[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    uint64_t offsetSize = 0;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    aclnnStatus aclRet = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_trans_quant_param_test, ascend910B2_test_abnormal_input_offsetSize_with_nullptr)
{
    float scaleArray[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    uint64_t scaleSize = 0;
    float* offsetArray = nullptr;
    uint64_t offsetSize = 5;
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    aclnnStatus aclRet = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// namespace error_message

// After each test, verify no EZ9999 internal code was captured
class Tqp1NoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        InitErrMgr();
        (void)error_message::GetErrMgrErrorMessage(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
    }
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        // 只守护本文件的测试套件, 框架层或其他算子用例的EZ9999不跨套件追责
        static const char* const guardedSuites[] = {"l2_trans_quant_param_test"};
        bool guarded = false;
        for (const char* s : guardedSuites) {
            if (std::string(test_info.test_suite_name()) == s) {
                guarded = true;
                break;
            }
        }
        if (!guarded) {
            return;
        }
        if (HasEZCode("EZ9999")) {
            ADD_FAILURE() << "EZ9999 captured during test [" << test_info.name() << "] in " << test_info.file();
        }
    }
};
static const auto _inst_tqp1 = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new Tqp1NoEz9999Check());
    return 0;
}();
