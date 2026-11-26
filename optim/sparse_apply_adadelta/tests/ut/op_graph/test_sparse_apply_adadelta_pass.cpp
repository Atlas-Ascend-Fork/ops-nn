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
#include <map>
#include <string>
#include <vector>

#include "base/err_mgr.h"
#include "../../../op_graph/sparse_apply_adadelta_proto.h"

namespace {
ge::graphStatus RunRegisteredVerifyCase(const ge::Shape& varShape, const ge::Shape& gradShape,
                                        const ge::Shape& indicesShape)
{
    ge::op::SparseApplyAdadelta op("sparse_apply_adadelta_verify_ut");
    const ge::Shape accumShape = varShape;
    const ge::Shape accumUpdateShape = varShape;
    const ge::Shape scalarShape({1});
    EXPECT_EQ(op.UpdateInputDesc("var", ge::TensorDesc(varShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("accum", ge::TensorDesc(accumShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("accum_update", ge::TensorDesc(accumUpdateShape, ge::FORMAT_ND, ge::DT_FLOAT)),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("lr", ge::TensorDesc(scalarShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("rho", ge::TensorDesc(scalarShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("epsilon", ge::TensorDesc(scalarShape, ge::FORMAT_ND, ge::DT_FLOAT)),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("grad", ge::TensorDesc(gradShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateInputDesc("indices", ge::TensorDesc(indicesShape, ge::FORMAT_ND, ge::DT_INT64)),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateOutputDesc("var", ge::TensorDesc(varShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateOutputDesc("accum", ge::TensorDesc(accumShape, ge::FORMAT_ND, ge::DT_FLOAT)), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.UpdateOutputDesc("accum_update", ge::TensorDesc(accumUpdateShape, ge::FORMAT_ND, ge::DT_FLOAT)),
              ge::GRAPH_SUCCESS);
    return op.VerifyAllAttr(true);
}

bool DrainFirstErrorFields(std::string& errorId, std::map<std::string, std::string>& fields)
{
    auto errors = error_message::GetErrMgrRawErrorMessages();
    if (errors.size() != 1U) {
        return false;
    }
    errorId = std::string(errors.front().error_id.get());
    for (size_t i = 0; i < errors.front().args_key.size(); ++i) {
        fields.emplace(std::string(errors.front().args_key[i].get()), std::string(errors.front().args_value[i].get()));
    }
    return true;
}
} // namespace

// ---------------------------------------------------------------------------
// 错误码整改负向用例：VerifySparseApplyAdadelta 的四个正式宏分支与未知秩透传契约。
// ---------------------------------------------------------------------------

TEST(SparseApplyAdadeltaGraphVerifyTest, indices_not_one_dim_reports_ez0011)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape({4, 8}), ge::Shape({2, 8}), ge::Shape({2, 2})), ge::GRAPH_FAILED);
    std::string errorId;
    std::map<std::string, std::string> fields;
    ASSERT_TRUE(DrainFirstErrorFields(errorId, fields));
    EXPECT_EQ(errorId, "EZ0011");
    EXPECT_EQ(fields.at("param_name"), "indices");
    EXPECT_EQ(fields.at("incorrect_dim"), "2D");
    EXPECT_EQ(fields.at("correct_dim"), "1D");
    EXPECT_FALSE(fields.at("op_name").empty());
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(SparseApplyAdadeltaGraphVerifyTest, scalar_var_grad_reports_ez0013)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    EXPECT_EQ(
        RunRegisteredVerifyCase(ge::Shape(std::vector<int64_t>{}), ge::Shape(std::vector<int64_t>{}), ge::Shape({2})),
        ge::GRAPH_FAILED);
    std::string errorId;
    std::map<std::string, std::string> fields;
    ASSERT_TRUE(DrainFirstErrorFields(errorId, fields));
    EXPECT_EQ(errorId, "EZ0013");
    EXPECT_EQ(fields.at("param_names"), "var, grad");
    EXPECT_EQ(fields.at("incorrect_dims"), "0D, 0D");
    EXPECT_EQ(fields.at("reason"), "The rank of var and grad must be at least 1");
    EXPECT_FALSE(fields.at("op_name").empty());
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(SparseApplyAdadeltaGraphVerifyTest, var_grad_rank_mismatch_reports_ez0013)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape({4, 8}), ge::Shape({2, 4, 8}), ge::Shape({2})), ge::GRAPH_FAILED);
    std::string errorId;
    std::map<std::string, std::string> fields;
    ASSERT_TRUE(DrainFirstErrorFields(errorId, fields));
    EXPECT_EQ(errorId, "EZ0013");
    EXPECT_EQ(fields.at("param_names"), "var, grad");
    EXPECT_EQ(fields.at("incorrect_dims"), "2D, 3D");
    EXPECT_EQ(fields.at("reason"), "The rank of var and grad must be the same");
    EXPECT_FALSE(fields.at("op_name").empty());
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

TEST(SparseApplyAdadeltaGraphVerifyTest, var_grad_dim_mismatch_reports_ez0010)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape({4, 8}), ge::Shape({4, 9}), ge::Shape({2})), ge::GRAPH_FAILED);
    std::string errorId;
    std::map<std::string, std::string> fields;
    ASSERT_TRUE(DrainFirstErrorFields(errorId, fields));
    EXPECT_EQ(errorId, "EZ0010");
    EXPECT_EQ(fields.at("param_names"), "var, grad");
    EXPECT_EQ(fields.at("incorrect_shapes"), "[4, 8], [4, 9]");
    EXPECT_EQ(fields.at("reason"), "var and grad must match in each dimension; first mismatch at dimension 1");
    EXPECT_FALSE(fields.at("op_name").empty());
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

// 未知秩/动态维透传契约：indices 未知秩、var/grad 未知秩、动态维、匹配一致均应成功
TEST(SparseApplyAdadeltaGraphVerifyTest, unknown_rank_and_dynamic_dims_pass_through)
{
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape({4, 8}), ge::Shape({2, 8}), ge::Shape(ge::UNKNOWN_RANK)),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape(ge::UNKNOWN_RANK), ge::Shape({2, 8}), ge::Shape({2})),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape({4, 8}), ge::Shape(ge::UNKNOWN_RANK), ge::Shape({2})),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape({4, -1}), ge::Shape({4, 9}), ge::Shape({2})), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunRegisteredVerifyCase(ge::Shape({4, 8}), ge::Shape({4, 8}), ge::Shape({2})), ge::GRAPH_SUCCESS);
}
