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

using namespace std;

class l2_trans_quant_param_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_trans_quant_param_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_trans_quant_param_test TearDown" << endl; }
};

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

// ============================================================================
// Error code verification UT for E*9*** remediation (2026-10-22)
// Captures ErrorManager output via weak-symbol override; asserts no EZ9999.
// (The two remediated INNER-code sites are OOM/defensive paths, not fakable
//  here; this net verifies the op's full UT suite generates no EZ9999.)
// ============================================================================
namespace error_message {
struct CapturedError {
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};
static thread_local CapturedError g_last_error;
static thread_local int g_report_count = 0;
static thread_local std::vector<std::string> g_all_codes; // 记录用例全程上报过的全部错误码
__attribute__((weak)) void ResetCapture()
{
    g_last_error.code.clear();
    g_last_error.keys.clear();
    g_last_error.values.clear();
    g_report_count = 0;
    g_all_codes.clear();
}
__attribute__((weak)) std::string GetLastError() { return g_last_error.code; }
__attribute__((weak)) int GetReportCount() { return g_report_count; }
__attribute__((weak)) bool HasCode(const std::string& c)
{
    for (const auto& code : g_all_codes) {
        if (code == c) {
            return true;
        }
    }
    return false;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code, const std::vector<const char*>& key,
                                                     const std::vector<const char*>& value)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_last_error.keys.clear();
    g_last_error.values.clear();
    for (const auto& k : key)
        g_last_error.keys.push_back(k ? k : "");
    for (const auto& v : value)
        g_last_error.values.push_back(v ? v : "");
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code)
{
    g_last_error.code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    g_report_count++;
    return 0;
}
__attribute__((weak)) int32_t ReportInnerErrMsg(const char* f, const char* fn, uint32_t l, const char* ec,
                                                const char* fmt, ...)
{
    g_last_error.code = ec ? ec : "";
    if (ec != nullptr) {
        g_all_codes.push_back(ec);
    }
    g_report_count++;
    return 0;
}
} // namespace error_message

// 验证弱符号截获通道在本二进制内生效: 若链接格局变化导致强定义压制弱定义, 该用例显式失败,
// 避免EZ9999守护用例静默失效
TEST_F(l2_trans_quant_param_test, error_capture_channel_self_check)
{
    error_message::ResetCapture();
    const std::vector<const char*> selfKey = {"self_check"};
    const std::vector<const char*> selfValue = {"capture channel liveness"};
    error_message::ReportPredefinedErrMsg("EZ0000", selfKey, selfValue);
    EXPECT_TRUE(error_message::HasCode("EZ0000")) << "capture channel is dead: weak override suppressed";
    EXPECT_EQ(error_message::GetReportCount(), 1);
    error_message::ResetCapture();
}
// After each test, verify no EZ9999 internal code was captured
class Tqp1NoEz9999Check : public testing::EmptyTestEventListener {
public:
    void OnTestStart(const testing::TestInfo& test_info) override
    {
        error_message::ResetCapture(); // 清除上一用例残留捕获, 避免多算子联合UT下跨套件级联误判
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
        if (error_message::HasCode("EZ9999")) {
            ADD_FAILURE() << "EZ9999 captured during test [" << test_info.name() << "] in " << test_info.file();
        }
    }
};
static const auto _inst_tqp1 = []() {
    testing::UnitTest::GetInstance()->listeners().Append(new Tqp1NoEz9999Check());
    return 0;
}();
