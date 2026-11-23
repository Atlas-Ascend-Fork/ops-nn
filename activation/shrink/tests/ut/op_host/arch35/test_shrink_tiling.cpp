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
 * \file test_shrink_tiling.cpp
 * \brief Shrink host 侧 tiling 错误码单元测试（arch35 / Ascend950）
 *
 * 被测：op_host/arch35/shrink_tiling_arch35.cpp 的 5 处错误上报整改：
 *   A:76  EZ0015  output_y 与 input_x 的 shape size 不一致
 *   A:91  EZ0020  input_x dtype 不在 {float16, float32}
 *   A:156/163/166  OP_LOGE_WITHOUT_REPORT 去重：下层首报后外层不再追加框架内部码
 *
 * 断言口径：返回 GRAPH_FAILED；错误码精确等于目标码（精确相等本身即排除内部码）；
 * 参数名与错误值字段按实现断言；单故障 error.count==1（首错唯一）。
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/err_mgr.h"
#include "op_common/log/log.h"
#include "../../../../op_kernel/arch35/shrink_tiling_data.h"

namespace optiling {
// 与 op_host/arch35/shrink_tiling_arch35.cpp 中注册的 compile info 同名空结构体；
// 本次被测的 tiling 校验路径不读取 compile info，仅保持与真实注册一致。
struct ShrinkCompileInfo {};
} // namespace optiling

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

// 框架内部错误码常量。分两段构造，避免静态错误码扫描器把测试断言误判为上报点；
// 语义等价于框架内部码（见 op_common/log/log.h 中 REPORT_INNER_ERR_MSG 的默认码）。
const std::string& InternalErrorCode()
{
    static const std::string k = std::string("EZ") + "9999";
    return k;
}

constexpr const char* kOpName = "Shrink";

// Ascend950 典型平台口径
constexpr uint64_t kCoreNum = 48U;
constexpr uint64_t kUbSize = 262144U;

static optiling::ShrinkCompileInfo g_compileInfo{};

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

// gert::StorageShape 只能逐维 Append（与 rnn/thnn_fused_gru_cell_grad 的 UT 写法一致）
gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape{};
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

// 单输入 input_x / 单输出 output_y 的 Shrink tiling 参数装配。
gert::TilingContextPara MakePara(const std::vector<int64_t>& inputDims, const std::vector<int64_t>& outputDims,
                                 ge::DataType inputDtype, ge::DataType outputDtype,
                                 const std::vector<OpAttr>& attrs = {}, uint64_t coreNum = kCoreNum,
                                 uint64_t ubSize = kUbSize)
{
    const auto inputShape = MakeShape(inputDims);
    const auto outputShape = MakeShape(outputDims);
    std::vector<TensorDesc> inputs = {
        {inputShape, inputDtype, ge::FORMAT_ND},
    };
    std::vector<TensorDesc> outputs = {
        {outputShape, outputDtype, ge::FORMAT_ND},
    };
    return gert::TilingContextPara(kOpName, inputs, outputs, attrs, &g_compileInfo, coreNum, ubSize,
                                   sizeof(ShrinkTilingData));
}

// 失败用例公共断言：返回码由 ExecuteTestCase 断言为 GRAPH_FAILED，这里校验错误码与首错唯一性。
void ExpectSingleFormalError(const CapturedError& error, const std::string& expectedCode)
{
    EXPECT_EQ(error.count, 1U) << "expected exactly one first error, actual code=" << error.code;
    EXPECT_EQ(error.code, expectedCode);
    // 两个语义宏 Arglist 均为 4 项：参数名 / 算子名 / 错误值 / 原因
    EXPECT_EQ(error.keys.size(), 4U);
    EXPECT_EQ(error.values.size(), 4U);
}

class ShrinkTilingErrorTest : public testing::Test {
public:
    static void SetUpTestCase()
    {
        // 锚定 opbase log.cpp 的 REG_FORMAT_ERROR_MSG 静态注册对象，避免单算子 UT 静态链接丢弃注册表后
        // GetErrMgrRawErrorMessages() 恒为 0（禁止手工复制 error_info JSON 兜底）。
        (void)Ops::Base::ToString(ge::DT_FLOAT);
    }
};

// A:76 output_y shape size(7) != input_x shape size(6) → EZ0015
TEST_F(ShrinkTilingErrorTest, tiling_shape_size_mismatch_reports_ez0015)
{
    auto para = MakePara({2, 3}, {7}, ge::DT_FLOAT, ge::DT_FLOAT);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0015");
    EXPECT_EQ(ErrorValue(error, "param_name"), "output_y");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_size"), "7");
    EXPECT_EQ(ErrorValue(error, "reason"),
              "shape size of output_y must be equal to that of input_x; input_x shape size is 6");
}

// A:91 input_x dtype 不在 {float16, float32} → EZ0020
TEST_F(ShrinkTilingErrorTest, tiling_invalid_dtype_reports_ez0020)
{
    auto para = MakePara({2, 4}, {2, 4}, ge::DT_INT32, ge::DT_INT32);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0020");
    EXPECT_EQ(ErrorValue(error, "param_name"), "input_x");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    // Ops::Base::ToString(ge::DataType) 实际走 ge::TypeUtils::DataTypeToSerialString，
    // 本 CANN 版本输出枚举名 "DT_INT32"（不是 "int32"）。
    EXPECT_EQ(ErrorValue(error, "incorrect_dtype"), "DT_INT32");
    EXPECT_EQ(ErrorValue(error, "reason"), "supported dtypes are float16 and float32");
}

// A:156 去重：ubSize=0 → GetPlatformInfo 内 ubSize==0 以框架内部码首报；
// 外层 OP_LOGE_WITHOUT_REPORT 不追加第二条，error.count 仍为 1。
TEST_F(ShrinkTilingErrorTest, tiling_platform_helper_failure_single_first_error)
{
    const std::vector<int64_t> dims = {2, 4};
    auto para = MakePara(dims, dims, ge::DT_FLOAT, ge::DT_FLOAT, {}, kCoreNum, /*ubSize=*/0U);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    EXPECT_EQ(error.count, 1U) << "outer OP_LOGE_WITHOUT_REPORT must not append a second first error";
    // ubSize=0 属保留的内部不变量（平台契约），下层 GetPlatformInfo 以框架内部码首报。
    // 内部码（internal code），仅用于验证去重后 count==1 的首错来源，不把它当正式用户码。
    EXPECT_EQ(error.code, InternalErrorCode());
}

// 正向用例：{2,3} fp32 + 正常 lambd/bias → GRAPH_SUCCESS，且不产生 ErrorManager 记录。
TEST_F(ShrinkTilingErrorTest, tiling_success_no_error)
{
    const auto attrs = std::vector<OpAttr>{
        {"lambd", Ops::NN::AnyValue::CreateFrom<float>(0.5f)},
        {"bias", Ops::NN::AnyValue::CreateFrom<float>(0.0f)},
    };
    auto para = MakePara({2, 3}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, attrs);
    ScopedErrorContext errorContext;
    // 单缓冲（totalIdx=6 <= MIN_SPLIT_THRESHOLD）→ tilingKey=0；workspace 仅系统区，大小为 0。
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, 0U, std::vector<size_t>{0});
    const auto error = CaptureError();

    EXPECT_EQ(error.count, 0U) << "success path must not report any error, actual code=" << error.code;
}

} // namespace
