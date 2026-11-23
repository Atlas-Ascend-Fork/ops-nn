/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gelu_grad_v2_error_code_tiling.cpp
 * \brief GeluGradV2 tiling 错误码单元测试
 *
 * 校验 DoOpTiling 外层 OP_LOGE_WITHOUT_REPORT 去重后，每条失败路径只保留 1 条正式错误码
 * (EZ0020 / EZ0021 / EZ0026)，不再叠加内部兜底码。
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base/err_mgr.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "atvoss/elewise/elewise_tiling.h"

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;

// 独立 work_stream_id 隔离本文件记录，避免共享 Host UT 进程内串扰。
class ScopedErrorContext {
public:
    ScopedErrorContext() : savedContext_(error_message::GetErrMgrContext())
    {
        auto context = savedContext_;
        context.work_stream_id = reinterpret_cast<std::uintptr_t>(this);
        error_message::SetErrMgrContext(context);
        (void)error_message::GetErrMgrRawErrorMessages();
    }

    ~ScopedErrorContext()
    {
        (void)error_message::GetErrMgrRawErrorMessages();
        error_message::SetErrMgrContext(savedContext_);
    }

private:
    error_message::ErrorManagerContext savedContext_;
};

struct CapturedError {
    size_t count = 0;
    std::string code;
};

CapturedError CaptureError()
{
    CapturedError captured;
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    captured.count = errors.size();
    if (errors.empty()) {
        return captured;
    }
    const auto& error = errors.back();
    captured.code = error.error_id == nullptr ? "" : error.error_id.get();
    return captured;
}

CapturedError RunTilingErrorCase(const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs,
                                 const std::string& approximate)
{
    Ops::Base::ElewiseCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 262144;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"approximate", Ops::NN::AnyValue::CreateFrom<std::string>(approximate)},
    };
    gert::TilingContextPara para("GeluGradV2", inputs, outputs, attrs, &compileInfo);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    return CaptureError();
}
} // namespace

class GeluGradV2ErrorCodeTiling : public testing::Test {};

TEST_F(GeluGradV2ErrorCodeTiling, tiling_input_dtype_invalid_reports_ez0020)
{
    const std::vector<TensorDesc> inputs = {
        {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
    };
    const std::vector<TensorDesc> outputs = {
        {{{2, 3}, {2, 3}}, ge::DT_INT32, ge::FORMAT_ND},
    };

    const CapturedError error = RunTilingErrorCase(inputs, outputs, "none");
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0020");
}

TEST_F(GeluGradV2ErrorCodeTiling, tiling_input_dtype_mismatch_reports_ez0021)
{
    const std::vector<TensorDesc> inputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    };
    const std::vector<TensorDesc> outputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };

    const CapturedError error = RunTilingErrorCase(inputs, outputs, "none");
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0021");
}

TEST_F(GeluGradV2ErrorCodeTiling, tiling_output_dtype_mismatch_reports_ez0021)
{
    const std::vector<TensorDesc> inputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    const std::vector<TensorDesc> outputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    };

    const CapturedError error = RunTilingErrorCase(inputs, outputs, "none");
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0021");
}

TEST_F(GeluGradV2ErrorCodeTiling, tiling_attr_approximate_invalid_reports_ez0026)
{
    const std::vector<TensorDesc> inputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    const std::vector<TensorDesc> outputs = {
        {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };

    const CapturedError error = RunTilingErrorCase(inputs, outputs, "relu");
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0026");
}
