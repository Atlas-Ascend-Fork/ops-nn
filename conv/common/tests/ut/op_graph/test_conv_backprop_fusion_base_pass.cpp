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
 * \file test_conv_backprop_fusion_base_pass.cpp
 * \brief 覆盖 ConvBackpropFusionBasePass::Update2DInputDesc 的错误码上报：
 *        input shape 非4维且非[-2]时上报EZ0015
 */

#include <gtest/gtest.h>

#include <string>

#include "base/err_mgr.h"
#include "graph/types.h"
#include "../../../op_graph/fusion_pass/conv_backprop_fusion_base_pass.h"

namespace {

class TestConvBackpropBasePass : public ops::ConvBackpropFusionBasePass {
public:
    TestConvBackpropBasePass() : ops::ConvBackpropFusionBasePass({ge::AscendString("TestConvBackpropInput")}) {}

    ge::AscendString GetNodeType() const override { return "ConvBackpropFusionBasePass"; }

    std::unique_ptr<ge::Graph> Replacement(const ge::GNode&) override { return nullptr; }

    bool Update2DInputDescPublic(ge::TensorDesc& inputDesc) { return Update2DInputDesc(inputDesc); }
};

} // namespace

TEST(ConvBackpropFusionBasePass, Update2DInputDescInvalidShapeDimFail)
{
    // 锚定 opbase log.cpp.o：预定义码注册表（REG_FORMAT_ERROR_MSG）随该符号链接进本测试二进制。
    (void)Ops::Base::ToString(ge::DT_FLOAT);
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrRawErrorMessages();
    TestConvBackpropBasePass pass;
    ge::TensorDesc desc;
    desc.SetShape(ge::Shape({2, 32, 16}));
    EXPECT_FALSE(pass.Update2DInputDescPublic(desc));
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_NE(errors[0].error_id, nullptr);
    EXPECT_STREQ(errors[0].error_id.get(), "EZ0015");
    ASSERT_NE(errors[0].error_message, nullptr);
    const std::string message(errors[0].error_message.get());
    EXPECT_NE(message.find("Parameter input of ConvBackpropFusionBasePass"), std::string::npos) << message;
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(ConvBackpropFusionBasePass, Update2DInputDescFourDimSuccess)
{
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrRawErrorMessages();
    TestConvBackpropBasePass pass;
    ge::TensorDesc desc;
    desc.SetShape(ge::Shape({2, 32, 16, 16}));
    EXPECT_TRUE(pass.Update2DInputDescPublic(desc));
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(ConvBackpropFusionBasePass, Update2DInputDescUnknownRankNormalizeSuccess)
{
    (void)error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE);
    (void)error_message::GetErrMgrRawErrorMessages();
    TestConvBackpropBasePass pass;
    ge::TensorDesc desc;
    desc.SetShape(ge::Shape({-2}));
    EXPECT_TRUE(pass.Update2DInputDescPublic(desc));
    EXPECT_EQ(desc.GetShape().GetDimNum(), 4U);
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}
