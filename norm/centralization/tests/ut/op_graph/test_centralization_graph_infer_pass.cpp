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
#include "../../../op_graph/centralization_proto.h"

namespace {
ge::graphStatus Verify(const ge::TensorDesc& input, const ge::TensorDesc& output)
{
    ge::op::Centralization op("centralization_verify");
    EXPECT_EQ(op.UpdateInputDesc("x", input), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateOutputDesc("y", output), ge::GRAPH_SUCCESS);
    return op.VerifyAllAttr(true);
}
} // namespace

TEST(CentralizationGraphVerify, AcceptsUnspecifiedOutputForFloat16)
{
    ge::op::Centralization op("centralization_inferred_output");
    const ge::TensorDesc x(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_FLOAT16);
    ASSERT_EQ(op.UpdateInputDesc("x", x), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.VerifyAllAttr(true), ge::GRAPH_SUCCESS);
    EXPECT_EQ(Verify(x, ge::TensorDesc(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_UNDEFINED)), ge::GRAPH_SUCCESS);
}

TEST(CentralizationGraphVerify, RejectsUnsupportedInputTypes)
{
    const ge::Shape shape({2, 3});
    const ge::TensorDesc y(shape, ge::FORMAT_ND, ge::DT_UNDEFINED);
    for (auto type : {ge::DT_BF16, ge::DT_BOOL}) {
        EXPECT_NE(Verify(ge::TensorDesc(shape, ge::FORMAT_ND, type), y), ge::GRAPH_SUCCESS);
    }
}

TEST(CentralizationGraphVerify, LeavesDeclaredOutputTypeToInference)
{
    const ge::Shape shape({2, 3});
    const ge::TensorDesc x(shape, ge::FORMAT_ND, ge::DT_FLOAT16);
    for (auto type : {ge::DT_BF16, ge::DT_BOOL, ge::DT_FLOAT}) {
        EXPECT_EQ(Verify(x, ge::TensorDesc(shape, ge::FORMAT_ND, type)), ge::GRAPH_SUCCESS);
    }
}

TEST(CentralizationGraphVerify, RejectsBlockedInputAndOutputFormats)
{
    const ge::Shape shape({2, 3, 4, 6});
    const ge::TensorDesc nd(shape, ge::FORMAT_ND, ge::DT_FLOAT);
    const ge::TensorDesc nz(shape, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT);
    EXPECT_NE(Verify(nz, nd), ge::GRAPH_SUCCESS);
    EXPECT_NE(Verify(nd, nz), ge::GRAPH_SUCCESS);
}

TEST(CentralizationGraphVerify, ChecksKnownDimensionsBeforeInference)
{
    const ge::TensorDesc x(ge::Shape({-1, 3}), ge::FORMAT_ND, ge::DT_FLOAT);
    auto verifyShape = [&x](const ge::Shape& shape) {
        return Verify(x, ge::TensorDesc(shape, ge::FORMAT_ND, ge::DT_UNDEFINED));
    };
    EXPECT_EQ(verifyShape(ge::Shape({0, 3})), ge::GRAPH_SUCCESS);
    EXPECT_EQ(verifyShape(ge::Shape({2, -1})), ge::GRAPH_SUCCESS);
    EXPECT_EQ(verifyShape(ge::Shape(ge::UNKNOWN_RANK)), ge::GRAPH_SUCCESS);
    EXPECT_NE(verifyShape(ge::Shape({999})), ge::GRAPH_SUCCESS);
    EXPECT_NE(verifyShape(ge::Shape({2, 4})), ge::GRAPH_SUCCESS);
    const ge::TensorDesc unknown(ge::Shape(ge::UNKNOWN_RANK), ge::FORMAT_ND, ge::DT_FLOAT);
    EXPECT_EQ(Verify(unknown, x), ge::GRAPH_SUCCESS);
}
