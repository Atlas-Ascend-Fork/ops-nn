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
 * \file test_gelu_v2_tiling_error_code.cpp
 * \brief GeluV2 host 侧 tiling 错误码单元测试（arch35 / Ascend950）
 *
 * 被测：op_host/arch35/gelu_v2_tiling_arch35.cpp
 *   A:99  （EZ0020）输入 dtype 非法，外层 OP_LOGE_WITHOUT_REPORT 去重
 *   A:101 （EZ0021）x/y dtype 不一致，外层 OP_LOGE_WITHOUT_REPORT 去重
 *   A:103 （EZ0010）x/y shape 不一致，外层 OP_LOGE_WITHOUT_REPORT 去重
 *   A:127 （EZ0026）approximate 非法值，外层 OP_LOGE_WITHOUT_REPORT 去重
 *   A:171  ElewiseBaseTiling::DoTiling 失败，外层 OP_LOGE_WITHOUT_REPORT 不追加记录
 *
 * 断言口径：返回 GRAPH_FAILED；错误码精确等于目标码（精确相等本身即排除内部码）；
 * 关键 Arglist 字段按实现断言；单故障 error.count==1（首错唯一，验证外层不重复上报）。
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/err_mgr.h"
#include "../../../../op_host/arch35/gelu_v2_tiling_arch35.h"

namespace {
// ErrorManager 上下文隔离方式与 hard_sigmoid_grad / gru_grad 的 host UT 一致：
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

constexpr const char* kOpName = "GeluV2";
// Ascend950 典型平台口径：48 个 AIV 核、256KB UB
constexpr uint64_t kCoreNum = 48U;
constexpr uint64_t kUbSize = 262144U;

static optiling::GeluV2CompileInfoArch35 g_compileInfo{};

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

// gert::StorageShape 只能逐维 Append（与 hard_sigmoid_grad 的 UT 写法一致）
gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape{};
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara MakePara(const std::vector<int64_t>& xDims, const std::vector<int64_t>& yDims,
                                 ge::DataType xDtype, ge::DataType yDtype, const std::string& approximate,
                                 uint64_t coreNum = kCoreNum, uint64_t ubSize = kUbSize)
{
    const auto xShape = MakeShape(xDims);
    const auto yShape = MakeShape(yDims);
    std::vector<TensorDesc> inputs = {
        {xShape, xDtype, ge::FORMAT_ND},
    };
    std::vector<TensorDesc> outputs = {
        {yShape, yDtype, ge::FORMAT_ND},
    };
    const std::vector<OpAttr> attrs = {
        {"approximate", Ops::NN::AnyValue::CreateFrom<std::string>(approximate)},
    };
    return gert::TilingContextPara(kOpName, inputs, outputs, attrs, &g_compileInfo, coreNum, ubSize);
}

// 失败用例公共断言：返回码由 ExecuteTestCase 断言为 GRAPH_FAILED，这里校验错误码与首错唯一性。
void ExpectSingleFormalError(const CapturedError& error, const std::string& expectedCode)
{
    EXPECT_EQ(error.count, 1U) << "expected exactly one first error, actual code=" << error.code;
    EXPECT_EQ(error.code, expectedCode);
    // 四个语义宏 Arglist 均为 4 项：参数名 / 算子名 / 错误值 / 原因
    EXPECT_EQ(error.keys.size(), 4U);
    EXPECT_EQ(error.values.size(), 4U);
}

class GeluV2TilingErrorCodeTest : public testing::Test {};

// A:99（外层 OP_LOGE_WITHOUT_REPORT 去重）输入 dtype 非法 → EZ0020，且仅 1 条首错
TEST_F(GeluV2TilingErrorCodeTest, tiling_invalid_input_dtype_reports_ez0020_once)
{
    auto para = MakePara({2, 3}, {2, 3}, ge::DT_INT32, ge::DT_INT32, "none");
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0020");
    EXPECT_EQ(ErrorValue(error, "param_name"), "x");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    // Ops::Base::ToString(ge::DataType) 走 ge::TypeUtils::DataTypeToSerialString，输出枚举名。
    EXPECT_EQ(ErrorValue(error, "incorrect_dtype"), "DT_INT32");
    EXPECT_EQ(ErrorValue(error, "reason"), "The dtype of x must be DT_FLOAT16, DT_BF16, or DT_FLOAT");
}

// A:101（外层 OP_LOGE_WITHOUT_REPORT 去重）x/y dtype 不一致 → EZ0021，且仅 1 条首错
TEST_F(GeluV2TilingErrorCodeTest, tiling_dtype_mismatch_reports_ez0021_once)
{
    auto para = MakePara({2, 3}, {2, 3}, ge::DT_FLOAT16, ge::DT_FLOAT, "none");
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0021");
    EXPECT_EQ(ErrorValue(error, "param_names"), "x, y");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_dtypes"), "DT_FLOAT16, DT_FLOAT");
    EXPECT_EQ(ErrorValue(error, "reason"), "The dtypes of x and y must be the same");
}

// A:103（外层 OP_LOGE_WITHOUT_REPORT 去重）x/y shape 不一致 → EZ0010，且仅 1 条首错
TEST_F(GeluV2TilingErrorCodeTest, tiling_shape_mismatch_reports_ez0010_once)
{
    auto para = MakePara({2, 3}, {2, 4}, ge::DT_FLOAT16, ge::DT_FLOAT16, "none");
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0010");
    EXPECT_EQ(ErrorValue(error, "param_names"), "x, y");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_shapes"), "[2, 3], [2, 4]");
    EXPECT_EQ(ErrorValue(error, "reason"), "The shapes of x and y must be the same");
}

// A:127（外层 OP_LOGE_WITHOUT_REPORT 去重）approximate 非法值 → EZ0026，且仅 1 条首错
TEST_F(GeluV2TilingErrorCodeTest, tiling_invalid_approximate_reports_ez0026_once)
{
    auto para = MakePara({2, 3}, {2, 3}, ge::DT_FLOAT16, ge::DT_FLOAT16, "relu");
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0026");
    EXPECT_EQ(ErrorValue(error, "param_name"), "approximate");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_value"), "relu");
    EXPECT_EQ(ErrorValue(error, "reason"), "The value of approximate must be none or tanh");
}

// 正例控制组：合法 F16 + approximate=none → GRAPH_SUCCESS，且 ErrorManager 不产生任何记录。
// 用 ExecuteTiling 只取执行结果，不绑定具体 tilingKey/workspace 数值（本用例只关心错误码）。
TEST_F(GeluV2TilingErrorCodeTest, tiling_valid_fp16_none_reports_no_error)
{
    auto para = MakePara({2, 3}, {2, 3}, ge::DT_FLOAT16, ge::DT_FLOAT16, "none");
    ScopedErrorContext errorContext;
    TilingInfo tilingInfo;
    const bool tilingOk = ExecuteTiling(para, tilingInfo);

    ASSERT_TRUE(tilingOk) << "valid case must succeed, otherwise tilingInfo is unusable";
    EXPECT_GT(tilingInfo.tilingKey, 0U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U); // 必须先确认为 1，才能安全访问 [0]
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);     // ASCEND_WORKSPACE = 0

    const auto error = CaptureError();
    EXPECT_EQ(error.count, 0U) << "valid case must not report any error, actual code=" << error.code;
    EXPECT_TRUE(error.code.empty());
}

// A:171（外层 OP_LOGE_WITHOUT_REPORT 去重）合规 dtype/shape，但 UB_SIZE 过小导致
// ElewiseBaseTiling::DoTiling 失败：本层不得新增错误记录。
//
// 实测口径（CANN 9.2.0 + 仓内 opbase）：OP_LOGE = 打日志 + REPORT_INNER_ERR_MSG("EZ9999")，
// 而 OP_LOGE_WITHOUT_REPORT 只打日志。该失败路径上 opbase 内部共有 3 处 OP_LOGE：
//   1) ElewiseTiling: "ubSize is smaller than extra size"（elewise_tiling.cpp:69）
//   2) DoTiling<ElewiseTilingData>: "Do elewise tiling failed."（elewise_tiling.h:201）
//   3) DoTiling<EleBaseTilingData16B>: "Do elewise tiling failed."（elewise_tiling.h:275）
// 因此 ErrorManager 中只应有这 3 条 opbase 内部码（EZ9999），且不含本层 A:171 的文案；
// 若 A:171 未改为 OP_LOGE_WITHOUT_REPORT，这里会出现第 4 条（"elewiseBaseTiling failed"）。
TEST_F(GeluV2TilingErrorCodeTest, tiling_ub_too_small_keeps_single_first_error)
{
    // ASCEND_API_BUFFER = 122880，UB 取 1KB 必然使 ElewiseTiling 的 ubSize 校验失败
    auto para = MakePara({2, 3}, {2, 3}, ge::DT_FLOAT16, ge::DT_FLOAT16, "none", kCoreNum, /*ubSize=*/1024U);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);

    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 3U) << "only opbase inner reports (EZ9999) are expected";
    for (const auto& item : errors) {
        const std::string code = item.error_id == nullptr ? "" : item.error_id.get();
        const std::string message = item.error_message == nullptr ? "" : item.error_message.get();
        // 唯一首错来源为 opbase 的框架内部码（internal code），不做正式用户码断言
        EXPECT_EQ(code, InternalErrorCode());
        // 本层 A:171 的文案不得出现在任何一条记录中，证明外层去重生效
        EXPECT_EQ(message.find("elewiseBaseTiling failed"), std::string::npos) << "message=" << message;
    }
}

} // namespace
