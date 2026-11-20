/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ut_op_util.h"
#include "infer_shape_context_faker.h"

namespace {
class AddLayerNormInvalidRank : public testing::Test {};

TEST_F(AddLayerNormInvalidRank, GammaRankExceedsInput)
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::StorageShape x = {{8}, {8}};
    gert::StorageShape gamma = {{1, 8}, {1, 8}};
    gert::StorageShape y, rstd, sum, extra;
    auto faker = gert::InferShapeContextFaker();
    faker.SetOpType("AddLayerNorm")
        .NodeIoNum(4, 4)
        .InputShapes({&x, &x, &gamma, &gamma})
        .OutputShapes({&y, &rstd, &sum, &extra});
    for (size_t i = 0; i < 4; ++i) {
        faker.NodeInputTd(i, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < 4; ++i) {
        faker.NodeOutputTd(i, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();
    EXPECT_EQ(impl->infer_shape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
} // namespace
