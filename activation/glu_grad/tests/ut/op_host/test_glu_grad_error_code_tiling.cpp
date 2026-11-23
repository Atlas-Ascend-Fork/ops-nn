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
 * \file test_glu_grad_error_code_tiling.cpp
 * \brief GLUGrad host 侧 tiling 错误码单元测试（通用，ascend910b / ascend950 双平台收录）
 *
 * 被测：op_host/glu_grad_tiling.cpp -> RunCommonTiling -> CheckInputParams / GetTilingParam
 *   1) glu_grad_tiling_common.h:148 dtype 白名单不通过        → EZ0020
 *   2) glu_grad_tiling_common.h:206 dim 越界                  → EZ0026
 *   3) glu_grad_tiling_common.h:215 切分维非偶                → EZ0015
 *   4) glu_grad_tiling_common.h:250 空输入维                  → EZ0009
 *
 * 断言口径：失败用例返回 GRAPH_FAILED；错误码精确等于目标正式码；单故障 error.count==1
 * （精确码本身即证明外层 OP_LOGE_WITHOUT_REPORT / 两层 OP_LOGE 未追加框架内部码）；
 * 合法用例返回 true 且无 ErrorManager 记录。
 *
 * 本文件放在 tests/ut/op_host/ 顶层，cmake/ut.cmake 的
 * `file(GLOB OPHOST_TILING_SRCS ${MODULE_DIR}/test_*_tiling.cpp ${MODULE_DIR}/${target_dir}/test_*_tiling.cpp)`
 * 保证 ascend910b(arch22) 与 ascend950(arch35) 两个目标都会收录。
 */

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/err_mgr.h"
#include "log/log.h"
#include "../../../op_host/glu_grad_tiling.h"

namespace {
// ErrorManager 上下文隔离方式与 activation/hard_sigmoid_grad/tests/ut/op_host/arch35/
// test_hard_sigmoid_grad_tiling.cpp 一致：用独立 work_stream_id 记录，避免共享 Host UT 进程内用例串扰。
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

// GetErrMgrRawErrorMessages 读取即消费（每条用例只读一次），必须先快照再断言
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

constexpr const char* kOpName = "GLUGrad";

// 与 glu_grad 现有 UT（test_glu_grad_tiling.cpp）一致的 Ascend 平台口径
constexpr uint64_t kCoreNum = 40U;
constexpr uint64_t kUbSize = 196608U;

// GluGradCompileInfo 定义在 op_host/glu_grad_tiling.h；tiling 从中读取 totalCoreNum / ubSizePlatForm
static optiling::GluGradCompileInfo g_compileInfo = [] {
    optiling::GluGradCompileInfo info;
    info.totalCoreNum = static_cast<int32_t>(kCoreNum);
    info.ubSizePlatForm = kUbSize;
    return info;
}();

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

// gert::StorageShape 只能逐维 Append
gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape{};
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

// 输入顺序按 op 原型：gradOut, self；输出 out；dim 为 attr。两个输入同 dtype。
gert::TilingContextPara MakePara(const std::vector<int64_t>& gradOutDims, const std::vector<int64_t>& selfDims,
                                 const std::vector<int64_t>& outDims, ge::DataType dtype, int64_t dim)
{
    const auto gradOutShape = MakeShape(gradOutDims);
    const auto selfShape = MakeShape(selfDims);
    const auto outShape = MakeShape(outDims);
    std::vector<TensorDesc> inputs = {
        {gradOutShape, dtype, ge::FORMAT_ND},
        {selfShape, dtype, ge::FORMAT_ND},
    };
    std::vector<TensorDesc> outputs = {
        {outShape, dtype, ge::FORMAT_ND},
    };
    const std::vector<OpAttr> attrs = {
        {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(dim)},
    };
    return gert::TilingContextPara(kOpName, inputs, outputs, attrs, &g_compileInfo, kCoreNum, kUbSize,
                                   sizeof(optiling::GluGradTilingData));
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

class GluGradTilingErrorCodeTest : public testing::Test {};
} // namespace

// 1) glu_grad_tiling_common.h:148 dtype 白名单不通过 → EZ0020。
// 同时证明外层 RunCommonTiling 的 OP_LOGE_WITHOUT_REPORT（:286-287）不追加第二条（count==1）。
TEST_F(GluGradTilingErrorCodeTest, tiling_unsupported_dtype_reports_ez0020)
{
    auto para = MakePara({2, 4}, {2, 4}, {2, 4}, ge::DT_INT32, -1);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0020");
    EXPECT_EQ(ErrorValue(error, "param_name"), "self");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_dtype"), std::to_string(static_cast<int>(ge::DT_INT32)));
    EXPECT_EQ(ErrorValue(error, "reason"), "input dtype only support fp16, fp32, bf16 currently, please check.");
}

// 2) dim 越界（self 3 维，合法范围 [-3,2]，传入 3）→ EZ0026。
// 同时验证 GetAttrSplitDim → GetTilingParam → RunCommonTiling 两层 OP_LOGE_WITHOUT_REPORT 去重（count==1）。
TEST_F(GluGradTilingErrorCodeTest, tiling_dim_out_of_range_reports_ez0026)
{
    auto para = MakePara({2, 3, 4}, {2, 6, 4}, {2, 6, 4}, ge::DT_FLOAT, 3);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0026");
    EXPECT_EQ(ErrorValue(error, "param_name"), "dim");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_value"), "3");
    EXPECT_NE(ErrorValue(error, "reason").find("The value of attr [dim] must be in the range"), std::string::npos)
        << "actual reason=" << ErrorValue(error, "reason");
}

// 3) 切分维非偶（self 最后一维 5）→ EZ0015。
TEST_F(GluGradTilingErrorCodeTest, tiling_odd_halving_dim_reports_ez0015)
{
    auto para = MakePara({2, 2}, {2, 5}, {2, 5}, ge::DT_FLOAT, -1);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0015");
    EXPECT_EQ(ErrorValue(error, "param_name"), "self");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_size"), std::to_string(5));
    EXPECT_NE(ErrorValue(error, "reason").find("can not be split with factor"), std::string::npos)
        << "actual reason=" << ErrorValue(error, "reason");
}

// 4) 空输入维（self 首维 0）→ EZ0009；incorrect_shape 取 Ops::Base::ToString(存储 shape)。
TEST_F(GluGradTilingErrorCodeTest, tiling_empty_dim_reports_ez0009)
{
    const std::vector<int64_t> selfDims = {0, 4};
    auto para = MakePara({0, 2}, selfDims, {0, 4}, ge::DT_FLOAT, -1);
    const auto selfShape = MakeShape(selfDims);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0009");
    EXPECT_EQ(ErrorValue(error, "param_name"), "self");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    // 与实现同一函数（Ops::Base::ToString(const gert::Shape&)），避免手写格式假设
    EXPECT_EQ(ErrorValue(error, "incorrect_shape"), Ops::Base::ToString(selfShape.GetStorageShape()));
    EXPECT_NE(ErrorValue(error, "reason").find("Empty input dimensions are unsupported"), std::string::npos)
        << "actual reason=" << ErrorValue(error, "reason");
}

// 5) 合法 fp16 小 shape：tiling 返回成功且无错误记录。
TEST_F(GluGradTilingErrorCodeTest, tiling_valid_fp16_reports_no_error)
{
    auto para = MakePara({1, 80, 1280}, {1, 80, 2560}, {1, 80, 2560}, ge::DT_FLOAT16, -1);
    ScopedErrorContext errorContext;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(para, tilingInfo)) << "valid fp16 case should pass tiling";
    EXPECT_GT(tilingInfo.blockNum, 0U);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 0U) << "valid case must not report any error, actual code=" << error.code;
}

// 6) 合法 bf16 大 shape（走 big-shape 分支）：tiling 返回成功且无错误记录。
TEST_F(GluGradTilingErrorCodeTest, tiling_valid_bf16_big_shape_reports_no_error)
{
    auto para = MakePara({1, 4, 10240}, {1, 8, 10240}, {1, 8, 10240}, ge::DT_BF16, 1);
    ScopedErrorContext errorContext;
    TilingInfo tilingInfo;
    EXPECT_TRUE(ExecuteTiling(para, tilingInfo)) << "valid bf16 big-shape case should pass tiling";
    EXPECT_GT(tilingInfo.blockNum, 0U);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 0U) << "valid case must not report any error, actual code=" << error.code;
}
