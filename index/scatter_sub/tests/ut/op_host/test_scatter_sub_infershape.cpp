/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_case_executor.h"

using namespace ge;

class ScatterSubInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScatterSubInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScatterSubInfershape TearDown" << std::endl; }
};

// 常规场景：输出shape与输入var的shape一致
TEST_F(ScatterSubInfershape, scatter_sub_infershape_test_1)
{
    gert::InfershapeContextPara infershapeContextPara("ScatterSub",
                                                      {
                                                          {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 常规场景：3维var
TEST_F(ScatterSubInfershape, scatter_sub_infershape_test_2)
{
    gert::InfershapeContextPara infershapeContextPara("ScatterSub",
                                                      {
                                                          {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{4, 2, 1}, {4, 2, 1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{4, 16, 32}, {4, 16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 16, 32},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 部分未知shape（-1）场景：输出shape透传var的shape
TEST_F(ScatterSubInfershape, scatter_sub_infershape_partial_unknown_shape)
{
    gert::InfershapeContextPara infershapeContextPara("ScatterSub",
                                                      {
                                                          {{{-1, 8}, {-1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 8}, {2, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 全未知shape（-1）场景：输出shape透传var的shape
TEST_F(ScatterSubInfershape, scatter_sub_infershape_unknown_shape)
{
    gert::InfershapeContextPara infershapeContextPara("ScatterSub",
                                                      {
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 未知rank（-2）场景：输出shape置为(-2,)
TEST_F(ScatterSubInfershape, scatter_sub_infershape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("ScatterSub",
                                                      {
                                                          {{{-2}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-2}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
