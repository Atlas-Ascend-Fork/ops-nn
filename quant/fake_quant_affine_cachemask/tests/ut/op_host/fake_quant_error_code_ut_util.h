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
 * \file fake_quant_error_code_ut_util.h
 * \brief FakeQuantAffineCachemask 错误码 UT 公共设施：错误码捕获、首错唯一性断言、内部兜底码守护。
 *
 * 仅被 tests/ut/op_host 下的 test_*_infershape.cpp / test_*_tiling.cpp 引用，
 * 文件名不匹配 UT 源文件 glob（test_*.cpp），不会被单独编译。
 */

#ifndef FAKE_QUANT_AFFINE_CACHEMASK_ERROR_CODE_UT_UTIL_H
#define FAKE_QUANT_AFFINE_CACHEMASK_ERROR_CODE_UT_UTIL_H

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/err_mgr.h"

namespace fake_quant_ut {

// 内部兜底码，按段拼接构造，避免测试代码中出现该错误码的连续字面量
inline const std::string kInternalErrCode = std::string("EZ") + "9999";

// 按用例独占一个 ErrorManager 上下文（work_stream_id 取本对象地址），生命周期内记录
// 只落在该上下文中；析构时先 drain 掉本次记录，再恢复进入前的上下文。
// 与共享默认上下文（work_stream_id = 0）解耦，避免同一 Host UT 进程内跨用例/跨套件串扰。
// 对齐 rnn/gru_grad/tests/ut/op_host/test_gru_grad_tiling.cpp 的同名实现。
class ScopedErrorContext {
public:
    ScopedErrorContext() : savedContext_(error_message::GetErrMgrContext())
    {
        auto context = savedContext_;
        context.work_stream_id = reinterpret_cast<std::uintptr_t>(this);
        error_message::SetErrMgrContext(context);
        (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
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

inline std::vector<std::string> CollectErrorIds()
{
    std::vector<std::string> ids;
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    for (const auto& item : errors) {
        ids.emplace_back(item.error_id == nullptr ? std::string() : std::string(item.error_id.get()));
    }
    return ids;
}

inline int CountErrorId(const std::string& code)
{
    int count = 0;
    for (const auto& id : CollectErrorIds()) {
        if (id == code) {
            ++count;
        }
    }
    return count;
}

// 断言：期望正式码有且仅有 1 条；不得出现内部兜底码，也不得出现其他正式码（首错唯一性）
inline void ExpectSingleFormalCode(const std::string& code, const char* caseName)
{
    const auto ids = CollectErrorIds();
    std::string dump;
    int hit = 0;
    int fallback = 0;
    int otherFormal = 0;
    for (const auto& id : ids) {
        dump += (id.empty() ? std::string("<empty>") : id) + " ";
        if (id == code) {
            ++hit;
        } else if (id == kInternalErrCode) {
            ++fallback;
        } else {
            ++otherFormal;
        }
    }
    EXPECT_EQ(hit, 1) << caseName << " expects exactly one " << code << ", captured: [" << dump << "]";
    EXPECT_EQ(fallback, 0) << caseName << " must not report internal fallback code, captured: [" << dump << "]";
    EXPECT_EQ(otherFormal, 0) << caseName << " must not report other formal codes, captured: [" << dump << "]";
}

// 只守护指定套件：框架层或其他算子用例的内部兜底码不跨套件追责；
// 非守护套件在 OnTestStart/OnTestEnd 首行即返回，不触碰 ErrorManager 状态。
class NoEz9999Listener : public testing::EmptyTestEventListener {
public:
    explicit NoEz9999Listener(std::string suiteName) : suiteName_(std::move(suiteName)) {}

    void OnTestStart(const testing::TestInfo& testInfo) override
    {
        if (suiteName_ != testInfo.test_suite_name()) {
            return;
        }
        scoped_ = std::make_unique<ScopedErrorContext>();
    }

    void OnTestEnd(const testing::TestInfo& testInfo) override
    {
        if (suiteName_ != testInfo.test_suite_name()) {
            return;
        }
        if (CountErrorId(kInternalErrCode) > 0) {
            ADD_FAILURE() << "internal fallback code captured during test [" << testInfo.name() << "] in "
                          << testInfo.file();
        }
        // 析构时 drain 本用例残留并恢复外层上下文，不再写回共享默认上下文
        scoped_.reset();
    }

private:
    std::string suiteName_;
    std::unique_ptr<ScopedErrorContext> scoped_;
};

} // namespace fake_quant_ut

#endif // FAKE_QUANT_AFFINE_CACHEMASK_ERROR_CODE_UT_UTIL_H
