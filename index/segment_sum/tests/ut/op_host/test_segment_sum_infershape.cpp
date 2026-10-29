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

class SegmentSumInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SegmentSumInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SegmentSumInfershape TearDown" << std::endl; }
};

// 常规场景：segment_ids为int32常量，输出首维为segment_ids末元素+1，其余维度沿用x
TEST_F(SegmentSumInfershape, segment_sum_infershape_test_1)
{
    int32_t segmentIdsData[4] = {0, 0, 1, 2};
    gert::InfershapeContextPara infershapeContextPara(
        "SegmentSum",
        {
            {{{4, 3, 5}, {4, 3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND, true, segmentIdsData},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 常规场景：segment_ids为int64常量
TEST_F(SegmentSumInfershape, segment_sum_infershape_test_2)
{
    int64_t segmentIdsData[4] = {0, 1, 1, 3};
    gert::InfershapeContextPara infershapeContextPara(
        "SegmentSum",
        {
            {{{4, 2}, {4, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND, true, segmentIdsData},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 常规场景：x为1维时输出为1维
TEST_F(SegmentSumInfershape, segment_sum_infershape_test_3)
{
    int32_t segmentIdsData[5] = {0, 0, 0, 1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "SegmentSum",
        {
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND, true, segmentIdsData},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 部分未知shape（-1）场景：x首维为-1时，输出首维仍由segment_ids推导，其余维度透传
TEST_F(SegmentSumInfershape, segment_sum_infershape_partial_unknown_shape)
{
    int32_t segmentIdsData[2] = {0, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "SegmentSum",
        {
            {{{-1, 3}, {-1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, segmentIdsData},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 未知rank（-2）场景：x为未知rank时，输出置为(-2,)
TEST_F(SegmentSumInfershape, segment_sum_infershape_x_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("SegmentSum",
                                                      {
                                                          {{{-2}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 未知rank（-2）场景：segment_ids为未知rank时，输出置为(-2,)
TEST_F(SegmentSumInfershape, segment_sum_infershape_segment_ids_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("SegmentSum",
                                                      {
                                                          {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
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

// segment_ids为空的场景：输出首维为0
TEST_F(SegmentSumInfershape, segment_sum_infershape_empty_segment_ids)
{
    gert::InfershapeContextPara infershapeContextPara("SegmentSum",
                                                      {
                                                          {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// segment_ids长度未知（-1）场景：输出首维保持未知维度-1
TEST_F(SegmentSumInfershape, segment_sum_infershape_unknown_len_segment_ids)
{
    gert::InfershapeContextPara infershapeContextPara("SegmentSum",
                                                      {
                                                          {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// segment_ids非常量（数据未就绪）场景：输出首维保持未知维度-1，不返回失败
TEST_F(SegmentSumInfershape, segment_sum_infershape_non_const_segment_ids)
{
    gert::InfershapeContextPara infershapeContextPara("SegmentSum",
                                                      {
                                                          {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// segment_ids为int64非常量（数据未就绪）场景：输出首维保持未知维度-1
TEST_F(SegmentSumInfershape, segment_sum_infershape_non_const_segment_ids_int64)
{
    gert::InfershapeContextPara infershapeContextPara("SegmentSum",
                                                      {
                                                          {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4}, {4}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
