/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "infershape_test_util.h"
#include "platform/platform_info.h"
#include "ut_op_common.h"

using namespace ge;

namespace {
void SetPlatform()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

void ExpectShapeEq(const gert::Shape& actual, const gert::Shape& expected)
{
    ASSERT_EQ(actual.GetDimNum(), expected.GetDimNum());
    for (size_t i = 0; i < expected.GetDimNum(); ++i) {
        EXPECT_EQ(actual.GetDim(i), expected.GetDim(i));
    }
}

void CheckInferShape(const gert::Shape& inputShape)
{
    SetPlatform();
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceSub");
    ASSERT_NE(opImpl, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::Shape xShape = inputShape;
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &xShape, &xShape})
                      .OutputShapes({&outputShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

    auto outputDesc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(outputDesc, nullptr);
    ExpectShapeEq(*outputDesc, inputShape);
}

ge::graphStatus RunInferShape(const gert::Shape& inputShape)
{
    SetPlatform();
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceSub");
    if (opImpl == nullptr || opImpl->infer_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape xShape = inputShape;
    gert::Shape outputShape = {};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&xShape, &xShape, &xShape})
                      .OutputShapes({&outputShape})
                      .Build();
    return opImpl->infer_shape(holder.GetContext<gert::InferShapeContext>());
}
} // namespace

class InplaceSubInferShapeTest : public testing::Test {};

TEST_F(InplaceSubInferShapeTest, staticShape) { CheckInferShape({4, 8}); }

TEST_F(InplaceSubInferShapeTest, dynamicShape) { CheckInferShape({2, -1, 16}); }

TEST_F(InplaceSubInferShapeTest, unknownRank) { CheckInferShape({-2}); }

TEST_F(InplaceSubInferShapeTest, rejectScalarX) { EXPECT_EQ(RunInferShape({}), ge::GRAPH_FAILED); }

TEST_F(InplaceSubInferShapeTest, acceptRankOneX) { EXPECT_EQ(RunInferShape({4}), ge::GRAPH_SUCCESS); }
