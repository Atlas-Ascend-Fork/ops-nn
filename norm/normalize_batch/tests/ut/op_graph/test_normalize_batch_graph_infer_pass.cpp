/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// NormalizeBatch ge 风格 Verifier UT（op_graph/graph_infer.cpp 的
// VERIFY_FUNC_REG 注册轨，经 ge::op::NormalizeBatch::VerifyAllAttr(true) 触发）。
// 覆盖：input_x rank=3 门禁、seq_len rank/dtype、声明 shape/dtype 兼容性、
// ND format 门禁、UNKNOWN_RANK / 未声明（-1 / UNDEFINED / RESERVED）透传。
//
// gert 风格 InferShape 的 UT 在 tests/ut/op_host/test_normalize_batch_infershape.cpp
// （faker 符号仅在 op_host UT 链接集可用）。

#include <gtest/gtest.h>

#include "graph/operator_reg.h"
#include "../../../op_graph/normalize_batch_proto.h"

namespace {
// VERIFY_FUNC_REG 注册的 VerifyNormalizeBatch
ge::graphStatus RunVerify(const ge::Shape& xShape, const ge::Shape& seqShape, const ge::Shape& yShape,
                          ge::DataType xType = ge::DT_FLOAT, ge::DataType seqType = ge::DT_INT32,
                          ge::DataType yType = ge::DT_FLOAT, ge::Format xFormat = ge::FORMAT_ND,
                          ge::Format yFormat = ge::FORMAT_ND)
{
    ge::op::NormalizeBatch op("normalize_batch_verify_ut");
    EXPECT_EQ(op.UpdateInputDesc("input_x", ge::TensorDesc(xShape, xFormat, xType)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("seq_len", ge::TensorDesc(seqShape, ge::FORMAT_ND, seqType)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateOutputDesc("output_y", ge::TensorDesc(yShape, yFormat, yType)), ge::GRAPH_SUCCESS);
    (void)op.SetAttr("normalize_type", "per_feature");
    return op.VerifyAllAttr(true);
}
} // namespace

// 拒绝路径
TEST(NormalizeBatchGraphVerify, RejectsWrongInputRank)
{
    EXPECT_EQ(RunVerify(ge::Shape({2, 3}), ge::Shape({2}), ge::Shape({2, 3})), ge::GRAPH_FAILED); // rank=2
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4, 5}), ge::Shape({2}), ge::Shape({2, 3, 4, 5})),
              ge::GRAPH_FAILED); // rank=4
}

TEST(NormalizeBatchGraphVerify, RejectsWrongSeqLenRankAndDtype)
{
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2, 3}), ge::Shape({2, 3, 4})), ge::GRAPH_FAILED); // rank=2
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2}), ge::Shape({2, 3, 4}), ge::DT_FLOAT, ge::DT_INT64),
              ge::GRAPH_FAILED); // seq_len 必须 int32
}

TEST(NormalizeBatchGraphVerify, RejectsConflictingDeclaredOutput)
{
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2}), ge::Shape({2, 3, 5})),
              ge::GRAPH_FAILED); // 声明 shape 冲突
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2}), ge::Shape({2, 3, 4}), ge::DT_FLOAT, ge::DT_INT32,
                        ge::DT_FLOAT16),
              ge::GRAPH_FAILED); // 声明 dtype 冲突
}

TEST(NormalizeBatchGraphVerify, RejectsNonNDFormat)
{
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2}), ge::Shape({2, 3, 4}), ge::DT_FLOAT, ge::DT_INT32,
                        ge::DT_FLOAT, ge::FORMAT_NCHW),
              ge::GRAPH_FAILED);
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2}), ge::Shape({2, 3, 4}), ge::DT_FLOAT, ge::DT_INT32,
                        ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ),
              ge::GRAPH_FAILED);
}

// 接受路径
TEST(NormalizeBatchGraphVerify, AcceptsCompatibleDeclarations)
{
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2}), ge::Shape({2, 3, 4})), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2}), ge::Shape({2, 3, -1})),
              ge::GRAPH_SUCCESS); // 声明 -1 维兼容
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2}), ge::Shape({2, 3, 4}), ge::DT_FLOAT, ge::DT_INT32,
                        ge::DT_UNDEFINED),
              ge::GRAPH_SUCCESS); // 输出 dtype 未声明
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape({2}), ge::Shape({2, 3, 4}), ge::DT_FLOAT, ge::DT_INT32,
                        ge::DT_FLOAT, ge::FORMAT_RESERVED, ge::FORMAT_RESERVED),
              ge::GRAPH_SUCCESS); // format 未声明（RESERVED）
    EXPECT_EQ(RunVerify(ge::Shape({2, 0, 4}), ge::Shape({2}), ge::Shape({2, 0, 4})),
              ge::GRAPH_SUCCESS); // 空 tensor（rank 合法、0 维一致）
}

TEST(NormalizeBatchGraphVerify, AcceptsUnknownRank)
{
    EXPECT_EQ(RunVerify(ge::Shape(ge::UNKNOWN_RANK), ge::Shape({2}), ge::Shape(ge::UNKNOWN_RANK)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunVerify(ge::Shape({2, 3, 4}), ge::Shape(ge::UNKNOWN_RANK), ge::Shape({2, 3, 4})), ge::GRAPH_SUCCESS);
}
