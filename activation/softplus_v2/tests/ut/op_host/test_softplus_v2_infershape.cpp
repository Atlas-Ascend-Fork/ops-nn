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
 * \file test_softplus_v2_infershape.cpp
 * \brief SoftplusV2 op_host UT - InferShape unit tests
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "base/err_mgr.h"

namespace {
// ErrorManager 上下文隔离方式与 rnn/gru_grad/tests/ut/op_host/test_gru_grad_tiling.cpp 一致：
// 用独立 work_stream_id 记录，避免共享 Host UT 进程内的用例串扰。
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

    ScopedErrorContext(const ScopedErrorContext&) = delete;
    ScopedErrorContext& operator=(const ScopedErrorContext&) = delete;

private:
    error_message::ErrorManagerContext savedContext_;
};

struct CapturedError {
    size_t count = 0;
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
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
    for (const auto& key : error.args_key) {
        captured.keys.emplace_back(key == nullptr ? "" : key.get());
    }
    for (const auto& value : error.args_value) {
        captured.values.emplace_back(value == nullptr ? "" : value.get());
    }
    return captured;
}

std::string ErrorValue(const CapturedError& error, const std::string& key)
{
    for (size_t i = 0; i < error.keys.size() && i < error.values.size(); ++i) {
        if (error.keys[i] == key) {
            return error.values[i];
        }
    }
    return "<missing:" + key + ">";
}
} // namespace

class SoftplusV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SoftplusV2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SoftplusV2Infershape TearDown" << std::endl; }
};

TEST_F(SoftplusV2Infershape, infershape_1d)
{
    gert::StorageShape x_shape = {{1024}, {1024}};
    gert::Shape y_shape = {{1024}};
    gert::Shape expected_output_shape = {1024};

    ge::DataType dtype = ge::DT_FLOAT;

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftplusV2")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expected_output_shape));
}

// infershape rank 9 > MAX_DIM_NUM(8) → EZ0012（OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON）
TEST_F(SoftplusV2Infershape, infershape_rank_gt_8_reports_ez0012)
{
    gert::StorageShape x_shape;
    for (int64_t i = 0; i < 9; ++i) {
        x_shape.MutableOriginShape().AppendDim(1);
        x_shape.MutableStorageShape().AppendDim(1);
    }
    gert::Shape y_shape = {};

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftplusV2")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("SoftplusV2")
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ScopedErrorContext errorContext;
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    const auto error = CaptureError();

    EXPECT_EQ(error.count, 1U) << "expected exactly one first error, actual code=" << error.code;
    EXPECT_EQ(error.code, "EZ0012");
    EXPECT_EQ(ErrorValue(error, "param_name"), "x");
    EXPECT_EQ(ErrorValue(error, "op_name"), "SoftplusV2");
    EXPECT_EQ(ErrorValue(error, "incorrect_dim"), "9D");
    EXPECT_EQ(ErrorValue(error, "reason"), "the dim num of x must be less than or equal to 8");
}
