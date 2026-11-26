/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"
#include "op_common/op_host/util/op_const_def.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"

class ConvPlatformUtilArchTest : public testing::Test {};

TEST_F(ConvPlatformUtilArchTest, dwArchSetMatchesLegacySemantics)
{
    const std::vector<NpuArch> after35 = {NpuArch::DAV_3510, NpuArch::DAV_9201, NpuArch::DAV_9202};
    for (const auto& arch : after35) {
        EXPECT_TRUE(Ops::NN::Conv::PlatformUtil::IsDwArchAfter35(arch));
        EXPECT_TRUE(Ops::NN::Conv::PlatformUtil::IsDxArchAfter35(arch));
    }
}

TEST_F(ConvPlatformUtilArchTest, dwExcludes5162WhileDxIncludes)
{
    EXPECT_FALSE(Ops::NN::Conv::PlatformUtil::IsDwArchAfter35(Ops::Base::DAV_5162));
    EXPECT_TRUE(Ops::NN::Conv::PlatformUtil::IsDxArchAfter35(Ops::Base::DAV_5162));
}

TEST_F(ConvPlatformUtilArchTest, pre35ArchAlwaysFalse)
{
    const std::vector<NpuArch> pre35 = {NpuArch::DAV_2201, NpuArch::DAV_2002, NpuArch::DAV_3103,
                                        NpuArch::DAV_3502, NpuArch::DAV_3505, NpuArch::DAV_5102};
    for (const auto& arch : pre35) {
        EXPECT_FALSE(Ops::NN::Conv::PlatformUtil::IsDwArchAfter35(arch));
        EXPECT_FALSE(Ops::NN::Conv::PlatformUtil::IsDxArchAfter35(arch));
    }
}
