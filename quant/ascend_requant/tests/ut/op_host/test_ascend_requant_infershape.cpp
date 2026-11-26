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
#include <iostream>
#include "infershape_case_executor.h"

class AscendRequantInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AscendRequantInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AscendRequantInfershape TearDown" << std::endl; }
};

// 负向用例：x rank 0 < 1，infershape 应失败（正式 EZ0012 参数首报）
TEST_F(AscendRequantInfershape, ascend_requant_infershape_x_rank0_fail)
{
    gert::InfershapeContextPara infershapeContextPara("AscendRequant",
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}, // x rank 0
                                                          {{{}, {}}, ge::DT_UINT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 负向用例：req_scale rank 3 > x rank 2，infershape 应失败（正式 EZ0013 参数首报）
TEST_F(AscendRequantInfershape, ascend_requant_infershape_scale_rank_gt_x_rank_fail)
{
    gert::InfershapeContextPara infershapeContextPara("AscendRequant",
                                                      {
                                                          {{{4, 5}, {4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{1, 2, 3}, {1, 2, 3}}, ge::DT_UINT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 负向用例：req_scale dim 不可广播（非 1 且不等 x dim），infershape 应失败（正式 EZ0010 参数首报）
TEST_F(AscendRequantInfershape, ascend_requant_infershape_scale_not_broadcastable_fail)
{
    gert::InfershapeContextPara infershapeContextPara("AscendRequant",
                                                      {
                                                          {{{4, 5}, {4, 5}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{2, 5}, {2, 5}}, ge::DT_UINT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 正常用例：req_scale [1] 广播到 x [4]，输出 y = x shape
TEST_F(AscendRequantInfershape, ascend_requant_infershape_scale_broadcast_ok)
{
    gert::InfershapeContextPara infershapeContextPara("AscendRequant",
                                                      {
                                                          {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_UINT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
