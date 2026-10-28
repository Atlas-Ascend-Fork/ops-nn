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
#include "../../../op_host/arch35/softplus_grad_tiling.h"

TEST(SoftplusGradRankTilingTest, RejectsCollapsibleRank9BeforeDimensionCollapse)
{
    const gert::Shape rank8Shape = {2, 1, 2, 1, 2, 1, 2, 1};
    const gert::Shape rank9Shape = {2, 1, 2, 1, 2, 1, 2, 1, 2};
    EXPECT_TRUE(optiling::IsSoftplusGradRankSupported(rank8Shape));
    EXPECT_FALSE(optiling::IsSoftplusGradRankSupported(rank9Shape));
}

TEST(SoftplusGradRankTilingTest, SkipsUnknownDimensionsInElementCountCheck)
{
    const gert::Shape dynamicDimShape = {-1, 3};
    const gert::Shape unknownRankShape = {-2};
    const gert::Shape overflowShape = {std::numeric_limits<int64_t>::max(), 2};
    EXPECT_TRUE(optiling::IsSoftplusGradKnownElementCountSupported(dynamicDimShape));
    EXPECT_TRUE(optiling::IsSoftplusGradKnownElementCountSupported(unknownRankShape));
    EXPECT_FALSE(optiling::IsSoftplusGradKnownElementCountSupported(overflowShape));
}
