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
 * \file test_gn_training_update_infershape.cpp
 * \brief GNTrainingUpdate infershape UT：y <- x，batch_mean/batch_variance <- sum
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "base/err_mgr.h"
#include <map>
#include "../../../op_graph/gn_training_update_proto.h"

static void CheckRankPairFirstError(ge::Operator& op, const std::string& actualRanks)
{
    ASSERT_EQ(error_message::ErrMgrInit(error_message::ErrorMessageMode::INTERNAL_MODE), 0);
    (void)error_message::GetErrMgrRawErrorMessages();
    ASSERT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_FAILED);
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_NE(errors.front().error_id, nullptr);
    EXPECT_EQ(std::string(errors.front().error_id.get()), "EZ0013");
    ASSERT_EQ(errors.front().args_key.size(), 4U);
    ASSERT_EQ(errors.front().args_value.size(), 4U);
    std::map<std::string, std::string> actual;
    for (size_t i = 0; i < errors.front().args_key.size(); ++i) {
        ASSERT_NE(errors.front().args_key[i], nullptr);
        ASSERT_NE(errors.front().args_value[i], nullptr);
        EXPECT_TRUE(actual.emplace(errors.front().args_key[i].get(), errors.front().args_value[i].get()).second);
    }
    const std::map<std::string, std::string> expected = {
        {"param_names", "sum, square_sum"},
        {"op_name", "GnTrainingUpdate"},
        {"incorrect_dims", actualRanks},
        {"reason", "sum and square_sum must be 5D when their ranks are known"}};
    EXPECT_EQ(actual, expected);
    EXPECT_TRUE(error_message::GetErrMgrRawErrorMessages().empty());
}

class GnTrainingUpdateInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // Pull the existing opbase log.cpp registration object into this UT; this is not a negative error test.
        EXPECT_EQ(Ops::Base::ToString(ge::DT_INT32), "DT_INT32");
        std::cout << "GnTrainingUpdate Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() { std::cout << "GnTrainingUpdate Proto Test TearDown" << std::endl; }
};

// NCHW：x[2,4,2,3] fp16，统计量 [2,2,1,1,1]，G=2；y 同 x，统计量输出同 sum
TEST_F(GnTrainingUpdateInferShape, nchw_success)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({2, 4, 2, 3}, ge::DT_FLOAT16));
    op.UpdateInputDesc("sum", create_desc({2, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({2, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("scale", create_desc({1, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("offset", create_desc({1, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)2);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({2, 4, 2, 3}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({2, 2, 1, 1, 1}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({2, 2, 1, 1, 1}));
}

// NHWC：x[2,2,3,4]（C=4 末维），统计量 [2,1,1,2,1]
TEST_F(GnTrainingUpdateInferShape, nhwc_success)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({2, 2, 3, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({2, 1, 1, 2, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({2, 1, 1, 2, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)2);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({2, 2, 3, 4}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({2, 1, 1, 2, 1}));
}

// 无仿射（scale/offset/mean/variance 全缺席）
TEST_F(GnTrainingUpdateInferShape, no_affine_success)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({1, 8, 4, 4}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({1, 4, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({1, 4, 1, 1, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)4);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({1, 8, 4, 4}));
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({1, 4, 1, 1, 1}));
}

// GEIR.03：x 未知 rank（{-2}）时，sum 的具体非法形状（rank=4≠5）仍应被拒，
// 不能借动态支持整块放行
TEST_F(GnTrainingUpdateInferShape, unknown_rank_x_rejects_illegal_sum)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({2, 2, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({2, 2, 1, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)2);

    CheckRankPairFirstError(op, "4D, 4D");
}

// GEIR.03：x 未知 rank + sum 合法 → 通过，统计量输出取 sum 形状
TEST_F(GnTrainingUpdateInferShape, unknown_rank_x_valid_sum_success)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({2, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({2, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", (int64_t)2);

    Runtime2TestParam param{{"num_groups"}};
    EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(1).GetShape().GetDims(), std::vector<int64_t>({2, 2, 1, 1, 1}));
    EXPECT_EQ(op.GetOutputDesc(2).GetShape().GetDims(), std::vector<int64_t>({2, 2, 1, 1, 1}));
}

TEST_F(GnTrainingUpdateInferShape, square_sum_rank_three_reports_actual_pair)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({2, 4, 2, 3}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({2, 2, 1, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({2, 2, 1}, ge::DT_FLOAT));
    op.SetAttr("num_groups", static_cast<int64_t>(2));
    CheckRankPairFirstError(op, "5D, 3D");
}

TEST_F(GnTrainingUpdateInferShape, illegal_sum_with_unknown_square_sum_reports_unknown_rank)
{
    ge::op::GNTrainingUpdate op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT));
    op.UpdateInputDesc("sum", create_desc({2, 2, 1, 1}, ge::DT_FLOAT));
    op.UpdateInputDesc("square_sum", create_desc({-2}, ge::DT_FLOAT));
    op.SetAttr("num_groups", static_cast<int64_t>(2));
    CheckRankPairFirstError(op, "4D, unknown rank (-2)");
}
