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
#include <cstdint>
#include <sstream>
#include <vector>
#include "gtest/gtest.h"

#include "aclnn_matmul_compress_dequant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_matmul_compress_dequant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_matmul_compress_dequant_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_matmul_compress_dequant_test TearDown" << endl; }
    static void MatMulCompressDequantCommonTest(TensorDesc a_desc, TensorDesc b_desc, TensorDesc index_desc,
                                                TensorDesc bias_desc, TensorDesc deqScale_desc,
                                                IntArrayDesc compress_info_desc, TensorDesc out_desc,
                                                aclnnStatus expect_status)
    {
        auto ut = OP_API_UT(aclnnMatmulCompressDequant,
                            INPUT(a_desc, b_desc, index_desc, bias_desc, deqScale_desc, nullptr, 1, compress_info_desc),
                            OUTPUT(out_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, expect_status);
    }
};

TEST_F(l2_matmul_compress_dequant_test, ascend310P_test_normal_input)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc deqScale_desc = TensorDesc({16, 32}, ACL_UINT64, ACL_FORMAT_NC1HWC0);
    vector<int64_t> compress_info = {8, 8, 32, 16, 1};
    IntArrayDesc compress_info_desc = IntArrayDesc(compress_info);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCompressDequantCommonTest(a_desc, b_desc, index_desc, bias_desc, deqScale_desc, compress_info_desc, out_desc,
                                    ACLNN_SUCCESS);
}

TEST_F(l2_matmul_compress_dequant_test, ascend310P_test_empty)
{
    // 使用**Desc描述host api输入输出
    TensorDesc a_desc = TensorDesc({16, 0}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({0}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc deqScale_desc = TensorDesc({16, 32}, ACL_UINT64, ACL_FORMAT_NC1HWC0);
    vector<int64_t> compress_info = {8, 8, 0, 16, 1};
    IntArrayDesc compress_info_desc = IntArrayDesc(compress_info);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCompressDequantCommonTest(a_desc, b_desc, index_desc, bias_desc, deqScale_desc, compress_info_desc, out_desc,
                                    ACLNN_SUCCESS);
}

TEST_F(l2_matmul_compress_dequant_test, ascend310P_invalid_x1_dtype)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc deqScale_desc = TensorDesc({16, 32}, ACL_UINT64, ACL_FORMAT_NC1HWC0);
    vector<int64_t> compress_info = {8, 8, 32, 16, 1};
    IntArrayDesc compress_info_desc = IntArrayDesc(compress_info);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCompressDequantCommonTest(a_desc, b_desc, index_desc, bias_desc, deqScale_desc, compress_info_desc, out_desc,
                                    ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_compress_dequant_test, ascend310P_invalid_x1_shape_3d)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    TensorDesc a_desc = TensorDesc({2, 16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc deqScale_desc = TensorDesc({16, 32}, ACL_UINT64, ACL_FORMAT_NC1HWC0);
    vector<int64_t> compress_info = {8, 8, 32, 16, 1};
    IntArrayDesc compress_info_desc = IntArrayDesc(compress_info);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCompressDequantCommonTest(a_desc, b_desc, index_desc, bias_desc, deqScale_desc, compress_info_desc, out_desc,
                                    ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_matmul_compress_dequant_test, ascend310P_invalid_k_mismatch)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    TensorDesc a_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc deqScale_desc = TensorDesc({16, 32}, ACL_UINT64, ACL_FORMAT_NC1HWC0);
    vector<int64_t> compress_info = {8, 8, 32, 16, 1};
    IntArrayDesc compress_info_desc = IntArrayDesc(compress_info);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCompressDequantCommonTest(a_desc, b_desc, index_desc, bias_desc, deqScale_desc, compress_info_desc, out_desc,
                                    ACLNN_ERR_PARAM_INVALID);
}

// deqScale元素数不是16的整数倍时，第一段接口应返回参数非法错误码，而非走填零路径返回成功
TEST_F(l2_matmul_compress_dequant_test, ascend310P_invalid_deq_scale_numel_not_aligned)
{
    SocVersionManager versionManager(SocVersion::ASCEND310P);
    TensorDesc a_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc b_desc = TensorDesc({64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc index_desc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({16, 16}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc deqScale_desc = TensorDesc({4, 10}, ACL_UINT64, ACL_FORMAT_ND);
    vector<int64_t> compress_info = {8, 8, 32, 16, 1};
    IntArrayDesc compress_info_desc = IntArrayDesc(compress_info);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    MatMulCompressDequantCommonTest(a_desc, b_desc, index_desc, bias_desc, deqScale_desc, compress_info_desc, out_desc,
                                    ACLNN_ERR_PARAM_INVALID);
}

// Load dependencies globally before including the implementation in an isolated whitebox namespace.
#include "../../../../op_host/op_api/matmul_compress_dequant.h"
#include "level0/fill.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "base/err_mgr.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

namespace compress_dequant_whitebox {
namespace l0op {
__attribute__((visibility("hidden"))) const aclTensor* Contiguous(const aclTensor*, aclOpExecutor*);
__attribute__((visibility("hidden"))) const aclTensor* TransData(const aclTensor*, op::Format, int64_t, aclOpExecutor*);
__attribute__((visibility("hidden"))) const aclTensor* Fill(const aclTensor*, const aclTensor*, const aclIntArray*,
                                                            aclOpExecutor*);
__attribute__((visibility("hidden"))) const aclTensor* ViewCopy(const aclTensor*, const aclTensor*, aclOpExecutor*);
__attribute__((visibility("hidden"))) const aclTensor* MatMulCompressDequant(const aclTensor*, const aclTensor*,
                                                                             const aclTensor*, const aclTensor*,
                                                                             const aclTensor*, const aclTensor*, bool,
                                                                             bool, const aclIntArray*, int,
                                                                             const std::string&, aclOpExecutor*);
} // namespace l0op

// The implementation has C linkage; give its public entry points unique test-only names.
#define aclnnMatmulCompressDequantGetWorkspaceSize CompressDequantWhiteboxGetWorkspaceSize
#define aclnnMatmulCompressDequant CompressDequantWhiteboxRun
#include "../../../../op_host/op_api/aclnn_matmul_compress_dequant.cpp"
#undef aclnnMatmulCompressDequant
#undef aclnnMatmulCompressDequantGetWorkspaceSize

bool failContiguous = false;
int contiguousCalls = 0;
int transDataCalls = 0;
int matmulCalls = 0;

void ReportLowerLayerFailure(const char* callee)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {callee, "injected graph preparation failure"};
    REPORT_PREDEFINED_ERR_MSG("EZ1009", keys, values);
}

namespace l0op {
const aclTensor* Contiguous(const aclTensor* input, aclOpExecutor*)
{
    ++contiguousCalls;
    if (failContiguous && contiguousCalls == 1) {
        ReportLowerLayerFailure("Contiguous");
        return nullptr;
    }
    return input;
}

const aclTensor* TransData(const aclTensor*, op::Format, int64_t, aclOpExecutor*)
{
    ++transDataCalls;
    ReportLowerLayerFailure("TransData");
    return nullptr;
}

const aclTensor* Fill(const aclTensor*, const aclTensor*, const aclIntArray*, aclOpExecutor*) { return nullptr; }
const aclTensor* ViewCopy(const aclTensor*, const aclTensor*, aclOpExecutor*) { return nullptr; }
const aclTensor* MatMulCompressDequant(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*,
                                       const aclTensor*, const aclTensor*, bool, bool, const aclIntArray*, int,
                                       const std::string&, aclOpExecutor*)
{
    ++matmulCalls;
    return nullptr;
}
} // namespace l0op
} // namespace compress_dequant_whitebox

class MatmulCompressDequantFailureTest : public testing::Test {
protected:
    void SetUp() override
    {
        savedContext_ = error_message::GetErrMgrContext();
        auto context = savedContext_;
        context.work_stream_id = reinterpret_cast<std::uintptr_t>(this);
        error_message::SetErrMgrContext(context);
        (void)error_message::GetErrMgrRawErrorMessages();
        compress_dequant_whitebox::contiguousCalls = 0;
        compress_dequant_whitebox::transDataCalls = 0;
        compress_dequant_whitebox::matmulCalls = 0;
    }

    void TearDown() override
    {
        (void)error_message::GetErrMgrRawErrorMessages();
        error_message::SetErrMgrContext(savedContext_);
    }

    void CheckPreparationFailure(bool contiguousFailure)
    {
        using namespace compress_dequant_whitebox;
        SocVersionManager versionManager(SocVersion::ASCEND310P);
        auto tensorOwner = CREATE_EXECUTOR();
        ASSERT_NE(tensorOwner.get(), nullptr);
        auto x1 = tensorOwner->AllocTensor(op::Shape({16, 32}), op::DataType::DT_INT8, op::Format::FORMAT_ND);
        auto x2 = tensorOwner->AllocTensor(op::Shape({64}), op::DataType::DT_INT8, op::Format::FORMAT_ND);
        auto index = tensorOwner->AllocTensor(op::Shape({8}), op::DataType::DT_INT8, op::Format::FORMAT_ND);
        auto bias = tensorOwner->AllocTensor(op::Shape({16, 16}), op::DataType::DT_INT32, op::Format::FORMAT_ND);
        auto scale = tensorOwner->AllocTensor(op::Shape({16, 32}), op::DataType::DT_UINT64, op::Format::FORMAT_ND);
        auto out = tensorOwner->AllocTensor(op::Shape({16, 16}), op::DataType::DT_FLOAT16, op::Format::FORMAT_ND);
        const int64_t compressInfoValues[] = {8, 8, 32, 16, 1};
        auto compressInfo = tensorOwner->AllocIntArray(compressInfoValues, 5);
        failContiguous = contiguousFailure;
        // Exercise graph construction directly, without the public API's Runtime-dependent DFX setup.
        (void)error_message::GetErrMgrRawErrorMessages();
        MatmulUnzipInput input = {x1, x2, index, bias, scale, nullptr};
        EXPECT_EQ(BuildMatMulUnzipGraph(input, 1, compressInfo, out, tensorOwner.get()), nullptr);
        EXPECT_EQ(contiguousCalls, 1);
        EXPECT_EQ(transDataCalls, contiguousFailure ? 0 : 1);
        EXPECT_EQ(matmulCalls, 0);
        const auto errors = error_message::GetErrMgrRawErrorMessages();
        std::ostringstream errorDetails;
        for (const auto& error : errors) {
            errorDetails << "code=" << (error.error_id == nullptr ? "null" : error.error_id.get())
                         << " message=" << (error.error_message == nullptr ? "null" : error.error_message.get())
                         << '\n';
            for (size_t i = 0; i < error.args_key.size() && i < error.args_value.size(); ++i) {
                errorDetails << (error.args_key[i] == nullptr ? "null" : error.args_key[i].get()) << '='
                             << (error.args_value[i] == nullptr ? "null" : error.args_value[i].get()) << '\n';
            }
        }
        ASSERT_EQ(errors.size(), 1U) << errorDetails.str();
        ASSERT_NE(errors[0].error_id, nullptr);
        EXPECT_STREQ(errors[0].error_id.get(), "EZ1009");
        ASSERT_EQ(errors[0].args_key.size(), 2U);
        ASSERT_EQ(errors[0].args_value.size(), 2U);
        EXPECT_STREQ(errors[0].args_key[0].get(), "opName");
        EXPECT_STREQ(errors[0].args_key[1].get(), "reason");
        EXPECT_STREQ(errors[0].args_value[0].get(), contiguousFailure ? "Contiguous" : "TransData");
        EXPECT_STREQ(errors[0].args_value[1].get(), "injected graph preparation failure");
    }

private:
    error_message::ErrorManagerContext savedContext_;
};

TEST_F(MatmulCompressDequantFailureTest, ContiguousFailureStopsGraphAndPreservesFirstError)
{
    CheckPreparationFailure(true);
}

TEST_F(MatmulCompressDequantFailureTest, TransDataFailureStopsGraphAndPreservesFirstError)
{
    CheckPreparationFailure(false);
}
