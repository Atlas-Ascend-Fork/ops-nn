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

class GatherNdInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GatherNdInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GatherNdInfershape TearDown" << std::endl; }
};

// 常规场景：indices末维为2，输出为 indices[:-1] + x.shape[2:]
TEST_F(GatherNdInfershape, gather_nd_infershape_test_1)
{
    gert::InfershapeContextPara infershapeContextPara("GatherNd",
                                                      {
                                                          {{{4, 3, 5}, {4, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 常规场景：indices末维为1，输出为 indices[:-1] + x.shape[1:]
TEST_F(GatherNdInfershape, gather_nd_infershape_test_2)
{
    gert::InfershapeContextPara infershapeContextPara("GatherNd",
                                                      {
                                                          {{{4, 3, 5}, {4, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 1}, {2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// indices末维为未知维（-1）场景：输出置为未知rank（-2）
TEST_F(GatherNdInfershape, gather_nd_infershape_unknown_last_dim)
{
    gert::InfershapeContextPara infershapeContextPara("GatherNd",
                                                      {
                                                          {{{4, 3, 5}, {4, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, -1}, {2, -1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// indices为未知rank（-2）场景：输出置为未知rank（-2）
TEST_F(GatherNdInfershape, gather_nd_infershape_unknown_rank_indices)
{
    gert::InfershapeContextPara infershapeContextPara("GatherNd",
                                                      {
                                                          {{{4, 3, 5}, {4, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// x为未知rank（-2）场景：输出置为未知rank（-2），不因indices末维超过x维度数误判失败
TEST_F(GatherNdInfershape, gather_nd_infershape_unknown_rank_x)
{
    gert::InfershapeContextPara infershapeContextPara("GatherNd",
                                                      {
                                                          {{{-2}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 2}, {2, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// indices末维超过x的维度数场景：推导失败
TEST_F(GatherNdInfershape, gather_nd_infershape_last_dim_exceed)
{
    gert::InfershapeContextPara infershapeContextPara("GatherNd",
                                                      {
                                                          {{{4, 3, 5}, {4, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}
