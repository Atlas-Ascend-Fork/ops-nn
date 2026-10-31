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
 * \file test_acts_ulq_input_grad_infershape.cpp
 * \brief ActsULQInputGrad InferShape 单元测试
 *
 * InferShape 规则（见 op_host/acts_ulq_input_grad_infershape.cpp）：
 *   三输入 shape 必须一致，x_grad.shape = y_grad.shape。
 * 覆盖 1D / 多维 / 动态维 / unknown rank，以及两个 mask 的 shape 不一致场景。
 */

#include <gtest/gtest.h>

#include "infershape_case_executor.h"

class ActsUlqInputGradInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ActsUlqInputGradInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ActsUlqInputGradInfershape TearDown" << std::endl; }
};

// 1D：x_grad.shape = y_grad.shape
TEST_F(ActsUlqInputGradInfershape, infershape_1d)
{
    gert::InfershapeContextPara para("ActsULQInputGrad",
                                     {
                                         {{{128}, {128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{128}, {128}}, ge::DT_BOOL, ge::FORMAT_ND},
                                         {{{128}, {128}}, ge::DT_BOOL, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{128}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 4D：x_grad.shape 跟随 y_grad
TEST_F(ActsUlqInputGradInfershape, infershape_4d)
{
    gert::InfershapeContextPara para("ActsULQInputGrad",
                                     {
                                         {{{32, 3, 5, 5}, {32, 3, 5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{32, 3, 5, 5}, {32, 3, 5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{32, 3, 5, 5}, {32, 3, 5, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 3, 5, 5}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态维：-1 透传
TEST_F(ActsUlqInputGradInfershape, infershape_dynamic)
{
    gert::InfershapeContextPara para("ActsULQInputGrad",
                                     {
                                         {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1, -1, -1, 64}, {1, -1, -1, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, -1, -1, 64}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// unknown rank：-2 透传
TEST_F(ActsUlqInputGradInfershape, infershape_unknown_rank)
{
    gert::InfershapeContextPara para("ActsULQInputGrad",
                                     {
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_BOOL, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_BOOL, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// clamp_min_mask shape 与 y_grad 不一致
TEST_F(ActsUlqInputGradInfershape, infershape_clamp_min_mask_mismatch)
{
    gert::InfershapeContextPara para("ActsULQInputGrad",
                                     {
                                         {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{8, 4}, {8, 4}}, ge::DT_BOOL, ge::FORMAT_ND},
                                         {{{8, 8}, {8, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// clamp_max_mask shape 与 y_grad 不一致
TEST_F(ActsUlqInputGradInfershape, infershape_clamp_max_mask_mismatch)
{
    gert::InfershapeContextPara para("ActsULQInputGrad",
                                     {
                                         {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{8, 8}, {8, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                         {{{4, 8}, {4, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// -------------------- A2 对齐：动态/未知 rank 放宽分支 --------------------

// 静态 y_grad + 动态掩码（含 -1）：编译期跳过逐维校验，输出跟随 y_grad（对齐 A2 is_dynamic 放宽）
TEST_F(ActsUlqInputGradInfershape, infershape_static_y_with_dynamic_mask)
{
    gert::InfershapeContextPara para("ActsULQInputGrad",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{-1, -1}, {-1, -1}}, ge::DT_BOOL, ge::FORMAT_ND},
                                         {{{2, 3}, {2, 3}}, ge::DT_BOOL, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 未知 rank（-2）+ 已知 rank 掩码：跳过维数校验，输出透传 -2（对齐 A2 is_unknown_rank）
TEST_F(ActsUlqInputGradInfershape, infershape_unknown_rank_with_known_mask)
{
    gert::InfershapeContextPara para("ActsULQInputGrad",
                                     {
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{3, 4}, {3, 4}}, ge::DT_BOOL, ge::FORMAT_ND},
                                         {{{3, 4}, {3, 4}}, ge::DT_BOOL, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 全静态但维数不同（rank 2 vs rank 1）：维数校验拦截
TEST_F(ActsUlqInputGradInfershape, infershape_rank_mismatch)
{
    gert::InfershapeContextPara para("ActsULQInputGrad",
                                     {
                                         {{{8, 8}, {8, 8}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{64}, {64}}, ge::DT_BOOL, ge::FORMAT_ND},
                                         {{{8, 8}, {8, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// -------------------- rank 范围校验（spec rank_range [0,8]） --------------------

// rank=8（上界，合法）：三输入一致 → 通过，输出跟随
TEST_F(ActsUlqInputGradInfershape, infershape_rank8_upper_bound)
{
    gert::InfershapeContextPara para(
        "ActsULQInputGrad",
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2, 2, 2, 2, 2, 2, 2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// rank=9（越界）：infershape 自身拦截，不依赖编译管线 para_check
TEST_F(ActsUlqInputGradInfershape, infershape_rank9_exceed)
{
    gert::InfershapeContextPara para(
        "ActsULQInputGrad",
        {
            {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{2, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2, 2}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}
