/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "../../../op_graph/fatrelu_mul_proto.h"

class FatreluMul : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FatreluMul SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FatreluMul TearDown" << std::endl; }
};

TEST_F(FatreluMul, FatreluMul_infershape_case_0)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({4, 1, 1280}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

// rank=1 低于下界 2：末维折半语义不成立，应拒绝
TEST_F(FatreluMul, FatreluMul_infershape_case_1_rank1_rejected)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({8}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

// rank=9 越旧上界 8：上界已放开（rank(x) ≥ 2，行模型 rank 透明），应放行；输出=x[:-1]+末维减半
TEST_F(FatreluMul, FatreluMul_infershape_case_2_rank9_ok)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({1, 1, 1, 1, 1, 1, 1, 1, 8}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 4}));
}

// rank=9 非退化形态（fp16，前导维含非 1 值）：放行，输出=x[:-1]+末维减半
TEST_F(FatreluMul, FatreluMul_infershape_case_9_rank9_fwd_fp16)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({2, 3, 1, 1, 1, 1, 1, 1, 8}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({2, 3, 1, 1, 1, 1, 1, 1, 4}));
}

// rank=12 深秩：上界放开后仍 rank 透明，应放行
TEST_F(FatreluMul, FatreluMul_infershape_case_10_rank12_ok)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 6}, ge::DT_BF16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3}));
}

// rank=0 标量：末维读取下溢（GetDim(xDimNum-1) 即 GetDim(SIZE_MAX)），应拒绝
TEST_F(FatreluMul, FatreluMul_infershape_case_3_rank0_rejected)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_FAILED);
}

// rank=2 下界合法
TEST_F(FatreluMul, FatreluMul_infershape_case_4_rank2_ok)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({2, 4}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

// rank=8 上界合法
TEST_F(FatreluMul, FatreluMul_infershape_case_5_rank8_ok)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({2, 2, 2, 2, 2, 2, 2, 8}, ge::DT_FLOAT));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
}

// 未知 rank(-2)：按原样传播，输出 rank 跟随输入（-2 特判分支，先于 rank 守卫）
TEST_F(FatreluMul, FatreluMul_infershape_case_6_unknown_rank_prop)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({-2}, ge::DT_FLOAT16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({-2}));
}

// 末维未知(-1)：末维原样传播不折半（lastDim<0 三元分支）
TEST_F(FatreluMul, FatreluMul_infershape_case_7_unknown_last_dim_prop)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({4, -1}, ge::DT_BF16));

    EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetShape().GetDims(), std::vector<int64_t>({4, -1}));
}

// InferDataType：输出 dtype 跟随输入 x
TEST_F(FatreluMul, FatreluMul_infershape_case_8_datatype_prop)
{
    ge::op::FatreluMul op;
    op.UpdateInputDesc("x", create_desc({4, 1, 1280}, ge::DT_FLOAT16));

    EXPECT_EQ(InferDataTypeTest(op), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.GetOutputDesc(0).GetDataType(), ge::DT_FLOAT16);
}
