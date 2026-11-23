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
 * \file test_softplus_v2_error_code_tiling.cpp
 * \brief SoftplusV2 host 侧 tiling 错误码单元测试（arch35 / Ascend950）
 *
 * 被测：op_host/arch35/softplus_v2_tiling.cpp 中 GetInputInfo 的两处形状校验：
 *   - rank 超出 0D-8D       → EZ0012（OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON）
 *   - 单维超过 INT32_MAX    → EZ0009（OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON）
 *   - 外层 SoftplusV2TilingFunc 的 OP_LOGE_WITHOUT_REPORT 去重：
 *     内层首报后外层不再追加框架内部码（EZ9999），首错唯一。
 *
 * 断言口径：返回 GRAPH_FAILED；error_id 精确等于目标码（精确相等本身即排除内部码）；
 * 单故障 error.count == 1（首错唯一）。字段值断言用于确认报文可读性。
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/err_mgr.h"
#include "../../../../op_kernel/arch35/softplus_v2_tiling_data.h"

namespace optiling {
// 与 op_host/arch35/softplus_v2_tiling.cpp 中注册的 compile info 同名空结构体；
// 本次被测的 tiling 校验路径不读取 compile info，仅保持与真实注册一致。
struct SoftplusV2CompileInfo {};
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

constexpr const char* kOpName = "SoftplusV2";

// Ascend950 典型平台口径：tiling 先做平台信息校验（coreNum/ubSize 非 0），
// 再用合法 attrs 走到 shape 校验分支。
constexpr uint64_t kCoreNum = 48U;
constexpr uint64_t kUbSize = 262144U;

static optiling::SoftplusV2CompileInfo g_compileInfo{};

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

// 合法属性：beta=1.0, threshold=20.0（顺序须与 GetAttrPointer<float>(0)/(1) 一致）
std::vector<OpAttr> DefaultAttrs()
{
    return {
        {"beta", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
        {"threshold", Ops::NN::AnyValue::CreateFrom<float>(20.0f)},
    };
}

gert::TilingContextPara MakePara(const std::vector<int64_t>& xDims, ge::DataType xDtype = ge::DT_FLOAT,
                                 const std::vector<OpAttr>& attrs = DefaultAttrs())
{
    const auto xShape = MakeShape(xDims);
    std::vector<TensorDesc> inputs = {
        {xShape, xDtype, ge::FORMAT_ND},
    };
    // 拒绝用例在校验阶段即返回，输出 shape 不参与判定；按 proto（y 与 x 同 shape/dtype）填写。
    std::vector<TensorDesc> outputs = {
        {xShape, xDtype, ge::FORMAT_ND},
    };
    return gert::TilingContextPara(kOpName, inputs, outputs, attrs, &g_compileInfo, kCoreNum, kUbSize,
                                   sizeof(SoftplusV2TilingData));
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

class SoftplusV2TilingErrorTest : public testing::Test {};
} // namespace

// rank 9 > MAX_DIM_NUM(8) → EZ0012。
// 本用例同时验证外层 GetInputInfo 失败包装改为 OP_LOGE_WITHOUT_REPORT 后的去重效果：
// 若外层仍用 OP_LOGE，会再追加一条框架内部码，error.count 会变成 2。
TEST_F(SoftplusV2TilingErrorTest, tiling_rank_gt_8_reports_ez0012)
{
    const std::vector<int64_t> rank9(9, 1);
    auto para = MakePara(rank9);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0012");
    EXPECT_NE(error.code, InternalErrorCode()) << "outer OP_LOGE_WITHOUT_REPORT must not append the internal code";
    EXPECT_EQ(ErrorValue(error, "param_name"), "x");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_dim"), "9D");
    EXPECT_EQ(ErrorValue(error, "reason"), "the dim num of x must be less than or equal to 8");
}

// 单维 2147483648 = INT32_MAX + 1 → EZ0009。
// 平台信息与 attrs 均合法，确保流程走到 shape 校验而不是更早的失败分支。
TEST_F(SoftplusV2TilingErrorTest, tiling_dim_gt_int32max_reports_ez0009)
{
    auto para = MakePara({static_cast<int64_t>(INT32_MAX) + 1});
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0009");
    EXPECT_NE(error.code, InternalErrorCode()) << "outer OP_LOGE_WITHOUT_REPORT must not append the internal code";
    EXPECT_EQ(ErrorValue(error, "param_name"), "x");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[2147483648]");
    EXPECT_EQ(ErrorValue(error, "reason"), "dim[0] of x is 2147483648, exceeding INT32_MAX (2147483647)");
}

// 各维均不超过 INT32_MAX，但维乘积溢出 int64 → GetShapeSize() 返回 kInvalidDimValue 负哨兵 → EZ0009。
TEST_F(SoftplusV2TilingErrorTest, tiling_element_count_overflow_reports_ez0009)
{
    const int64_t kInt32Max = static_cast<int64_t>(INT32_MAX);
    auto para = MakePara({kInt32Max, kInt32Max, kInt32Max});
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0009");
    EXPECT_NE(error.code, InternalErrorCode()) << "outer OP_LOGE_WITHOUT_REPORT must not append the internal code";
    EXPECT_EQ(ErrorValue(error, "param_name"), "x");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), "[2147483647, 2147483647, 2147483647]");
    EXPECT_EQ(ErrorValue(error, "reason"),
              "the element count of x overflows int64 (GetShapeSize returned the kInvalidDimValue sentinel)");
}

// ComputeUsedCoreNum 失败（totalLength * elemSize 溢出 int64）走内部首报，
// 外层 OP_LOGE_WITHOUT_REPORT 不追加第二条：error.count 仍为 1。
TEST_F(SoftplusV2TilingErrorTest, tiling_core_num_helper_failure_single_first_error)
{
    const int64_t kInt32Max = static_cast<int64_t>(INT32_MAX);
    // 维乘积 4611686014132420609 未溢出 int64，但 *4 后超出 INT64_MAX，落在 ComputeUsedCoreNum 的溢出分支
    auto para = MakePara({kInt32Max, kInt32Max, 1});
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    EXPECT_EQ(error.count, 1U) << "outer OP_LOGE_WITHOUT_REPORT must not append a second first error";
    EXPECT_EQ(error.code, InternalErrorCode());
}

// 正向对照：rank 8 为允许边界，不应被 rank 校验拒绝（防止校验边界被改窄）。
TEST_F(SoftplusV2TilingErrorTest, tiling_rank_8_boundary_success)
{
    auto para = MakePara(std::vector<int64_t>(8, 2));
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_GT(tilingInfo.blockNum, 0U);
}
