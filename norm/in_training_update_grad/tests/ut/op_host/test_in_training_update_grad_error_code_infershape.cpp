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
 * \file test_in_training_update_grad_error_code_infershape.cpp
 * \brief INTrainingUpdateGrad InferShape/InferDataType 错误码 UT（EZ0009/EZ0012/EZ0020/EZ0021）。
 *
 *        逐条覆盖 Host 图构建门的形状/dtype 校验分支，断言：
 *        - 返回 GRAPH_FAILED；
 *        - 首错唯一（count == 1，外层不再追加框架内部兜底码）；
 *        - 错误码精确等于目标码，且 Arglist 参数名/关键 reason 与实现一致。
 *
 *        不可达项说明：infershape 中"D/H/W 为负"分支（源码 :84）在公开用例中不可构造 ——
 *        任一维为 -1 时 IsUnknownShape 已提前判定为动态形状并透传 variance，故不写负向用例。
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "base/err_mgr.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "graph/types.h"
#include "platform/platform_info.h"

namespace {
constexpr const char* kOpType = "INTrainingUpdateGrad";
constexpr ge::Format kFormat = ge::FORMAT_NDC1HWC0;

// ErrorManager 上下文隔离：用独立 work_stream_id 记录，避免共享 Host UT 进程内的用例串扰。
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

// GetErrMgrRawErrorMessages() 读取即消费：每条用例只调用一次。
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

// 按 Arglist 的 key 取实际错误值；缺失时返回可读标记，便于断言失败时定位。
std::string ErrorValue(const CapturedError& error, const std::string& key)
{
    for (size_t i = 0; i < error.keys.size() && i < error.values.size(); ++i) {
        if (error.keys[i] == key) {
            return error.values[i];
        }
    }
    return "<missing:" + key + ">";
}

bool Contains(const std::string& text, const std::string& sub) { return text.find(sub) != std::string::npos; }

gert::Shape MakeShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return shape;
}

// 四输入(dy/x/variance/mean) 两输出(res_gamma/res_beta)，format 固定 NDC1HWC0。
ge::graphStatus RunInferShape(const std::vector<int64_t>& dyDims, const std::vector<int64_t>& xDims,
                              const std::vector<int64_t>& varDims, const std::vector<int64_t>& meanDims,
                              ge::DataType dyDt = ge::DT_FLOAT, ge::DataType xDt = ge::DT_FLOAT,
                              ge::DataType varDt = ge::DT_FLOAT, ge::DataType meanDt = ge::DT_FLOAT)
{
    gert::Shape dy = MakeShape(dyDims);
    gert::Shape x = MakeShape(xDims);
    gert::Shape variance = MakeShape(varDims);
    gert::Shape mean = MakeShape(meanDims);
    gert::Shape resGamma = MakeShape({});
    gert::Shape resBeta = MakeShape({});

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType(kOpType)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&dy, &x, &variance, &mean})
                      .OutputShapes({&resGamma, &resBeta})
                      .NodeInputTd(0, dyDt, kFormat, kFormat)
                      .NodeInputTd(1, xDt, kFormat, kFormat)
                      .NodeInputTd(2, varDt, kFormat, kFormat)
                      .NodeInputTd(3, meanDt, kFormat, kFormat)
                      .NodeOutputTd(0, ge::DT_FLOAT, kFormat, kFormat)
                      .NodeOutputTd(1, ge::DT_FLOAT, kFormat, kFormat)
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType)->infer_shape;
    if (inferShapeFunc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return inferShapeFunc(context);
}

ge::graphStatus RunInferDataType(ge::DataType dyDt, ge::DataType xDt, ge::DataType varDt, ge::DataType meanDt)
{
    ge::DataType outDt = ge::DT_FLOAT;
    auto holder = gert::InferDataTypeContextFaker()
                      .SetOpType(kOpType)
                      .IrInputNum(4)
                      .NodeIoNum(4, 2)
                      .NodeInputTd(0, dyDt, kFormat, kFormat)
                      .NodeInputTd(1, xDt, kFormat, kFormat)
                      .NodeInputTd(2, varDt, kFormat, kFormat)
                      .NodeInputTd(3, meanDt, kFormat, kFormat)
                      .NodeOutputTd(0, ge::DT_FLOAT, kFormat, kFormat)
                      .NodeOutputTd(1, ge::DT_FLOAT, kFormat, kFormat)
                      .InputDataTypes({&dyDt, &xDt, &varDt, &meanDt})
                      .OutputDataTypes({&outDt, &outDt})
                      .Build();
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType)->infer_datatype;
    if (inferDataTypeFunc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return inferDataTypeFunc(context);
}

// 合法的 dy 形状（6D NDC1HWC0），其统计量形状 (N, 1, C1, 1, 1, C0)。
const std::vector<int64_t> kGoodDy = {2, 4, 2, 8, 8, 16};
const std::vector<int64_t> kGoodStat = {2, 1, 2, 1, 1, 16};
} // namespace

class INTrainingUpdateGradErrorCodeInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "INTrainingUpdateGrad ErrorCode InferShape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "INTrainingUpdateGrad ErrorCode InferShape TearDown" << std::endl; }
};

// 1. dy 非 6 维（5D）：EZ0012，param_name=dy，incorrect_dim=5
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_rank_not_six_ez0012)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({2, 4, 2, 8, 8}, {2, 4, 2, 8, 8}, kGoodStat, kGoodStat), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0012");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_EQ(ErrorValue(error, "incorrect_dim"), "5");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must be 6"));
}

// 2. N=0：EZ0009，param_name=dy（N/C1/C0 必须为正）
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_non_positive_n_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({0, 4, 2, 8, 8, 16}, {0, 4, 2, 8, 8, 16}, {0, 1, 2, 1, 1, 16}, {0, 1, 2, 1, 1, 16}),
              ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[0, 4, 2, 8, 8, 16]");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must be positive"));
}

// 3. C1=0：EZ0009，param_name=dy
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_non_positive_c1_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({2, 4, 0, 8, 8, 16}, {2, 4, 0, 8, 8, 16}, {2, 1, 0, 1, 1, 16}, {2, 1, 0, 1, 1, 16}),
              ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[2, 4, 0, 8, 8, 16]");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must be positive"));
}

// 4. C0=0：先命中 N/C1/C0 非正（与 C0>64 的 lane 上界是两条独立分支）
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_non_positive_c0_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({2, 4, 2, 8, 8, 0}, {2, 4, 2, 8, 8, 0}, {2, 1, 2, 1, 1, 0}, {2, 1, 2, 1, 1, 0}),
              ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[2, 4, 2, 8, 8, 0]");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must be positive"));
}

// 5. C0>64（超出单 fp32 向量寄存器 lane 数）：EZ0009，reason 含 must not exceed 64
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_c0_over_64_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({2, 4, 2, 8, 8, 128}, {2, 4, 2, 8, 8, 128}, {2, 1, 2, 1, 1, 128}, {2, 1, 2, 1, 1, 128}),
              ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[2, 4, 2, 8, 8, 128]");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must not exceed 64"));
}

// 6. H*W 超 uint32（65536*65536 = 2^32）：EZ0009，reason 含 H*W
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_hw_product_overflow_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({1, 1, 1, 65536, 65536, 16}, {1, 1, 1, 65536, 65536, 16}, {1, 1, 1, 1, 1, 16},
                            {1, 1, 1, 1, 1, 16}),
              ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "H*W"));
}

// 7. D*H*W 超 uint32：H*W=65535 先通过，D=70000 使 D*H*W 超界 → EZ0009，reason 含 D*H*W
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_dhw_product_overflow_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({1, 70000, 1, 65535, 1, 16}, {1, 70000, 1, 65535, 1, 16}, {1, 1, 1, 1, 1, 16},
                            {1, 1, 1, 1, 1, 16}),
              ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "D*H*W"));
}

// 8. N*C1 超 uint32（70000*70000）：EZ0009，reason 含 N*C1
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_nc1_product_overflow_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({70000, 1, 70000, 1, 1, 16}, {70000, 1, 70000, 1, 1, 16}, {70000, 1, 70000, 1, 1, 16},
                            {70000, 1, 70000, 1, 1, 16}),
              ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "N*C1"));
}

// 9. H*W*C0 超 uint32：H*W=268435456 通过 H*W / D*H*W / N*C1 三关，*C0=16 超界 → EZ0009
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_hwc0_product_overflow_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({1, 1, 1, 268435456, 1, 16}, {1, 1, 1, 268435456, 1, 16}, {1, 1, 1, 1, 1, 16},
                            {1, 1, 1, 1, 1, 16}),
              ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "H*W*C0"));
}

// 10. N*C1*C0 超 uint32：H*W*C0=16 通过，groupNum=268435456 使 N*C1*C0 超界 → EZ0009
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_nc1c0_product_overflow_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape({268435456, 1, 1, 1, 1, 16}, {268435456, 1, 1, 1, 1, 16}, {268435456, 1, 1, 1, 1, 16},
                            {268435456, 1, 1, 1, 1, 16}),
              ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "N*C1*C0"));
}

// 11. x 与 dy 不同形：EZ0009，param_name=x
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_x_shape_mismatch_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape(kGoodDy, {2, 4, 2, 8, 8, 8}, kGoodStat, kGoodStat), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "x");
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[2, 4, 2, 8, 8, 8]");
    EXPECT_EQ(ErrorValue(error, "reason"), "The shape of x must be the same as dy");
}

// 12. variance 形状不符期望 (N,1,C1,1,1,C0)：EZ0009，param_name=variance
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_variance_shape_mismatch_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape(kGoodDy, kGoodDy, {2, 1, 2, 1, 1, 8}, kGoodStat), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "variance");
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[2, 1, 2, 1, 1, 8]");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "(N, 1, C1, 1, 1, C0)"));
}

// 13. mean 形状不符期望 (N,1,C1,1,1,C0)：EZ0009，param_name=mean
TEST_F(INTrainingUpdateGradErrorCodeInferShape, infershape_reject_mean_shape_mismatch_ez0009)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferShape(kGoodDy, kGoodDy, kGoodStat, {2, 1, 2, 1, 1, 8}), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "mean");
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[2, 1, 2, 1, 1, 8]");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "(N, 1, C1, 1, 1, C0)"));
}

// 14. dy dtype=INT32：EZ0020，param_name=dy
TEST_F(INTrainingUpdateGradErrorCodeInferShape, inferdtype_reject_dy_dtype_ez0020)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferDataType(ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0020");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dy");
    EXPECT_EQ(ErrorValue(error, "incorrect_dtype"), "DT_INT32");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must be FLOAT or FLOAT16"));
}

// 15. dy=FLOAT、x=FLOAT16：EZ0021，param_names 含 dy 与 x
TEST_F(INTrainingUpdateGradErrorCodeInferShape, inferdtype_reject_dy_x_dtype_mismatch_ez0021)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferDataType(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0021");
    EXPECT_EQ(ErrorValue(error, "param_names"), "dy and x");
    EXPECT_EQ(ErrorValue(error, "incorrect_dtypes"), "DT_FLOAT, DT_FLOAT16");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must be the same"));
}

// 16. variance=FLOAT16（dy/x=FLOAT）：EZ0020，param_name=variance
TEST_F(INTrainingUpdateGradErrorCodeInferShape, inferdtype_reject_variance_dtype_ez0020)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferDataType(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0020");
    EXPECT_EQ(ErrorValue(error, "param_name"), "variance");
    EXPECT_EQ(ErrorValue(error, "incorrect_dtype"), "DT_FLOAT16");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must be FLOAT"));
}

// 17. mean=FLOAT16（dy/x=FLOAT）：EZ0020，param_name=mean
TEST_F(INTrainingUpdateGradErrorCodeInferShape, inferdtype_reject_mean_dtype_ez0020)
{
    ScopedErrorContext errorContext;
    EXPECT_EQ(RunInferDataType(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0020");
    EXPECT_EQ(ErrorValue(error, "param_name"), "mean");
    EXPECT_EQ(ErrorValue(error, "incorrect_dtype"), "DT_FLOAT16");
    EXPECT_TRUE(Contains(ErrorValue(error, "reason"), "must be FLOAT"));
}
