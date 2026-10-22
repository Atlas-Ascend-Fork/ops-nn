/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <gtest/gtest.h>

#include "../../../src/op_host/bank_mgr/op_hash.h"

namespace {
// Repeated bytes make these aligned input words independent of host byte order.
constexpr std::array<uint32_t, 4> kInput = {0x01010101U, 0x02020202U, 0x03030303U, 0x04040404U};
constexpr std::array<uint32_t, 4> kSeeds = {0U, 1U, 271828U, UINT32_MAX};

// Frozen CommonHash vectors for lengths 0..16, calculated independently of the
// production helpers. Partial words contain only the remaining bytes; lengths
// divisible by four retain the legacy final combine (including length zero).
constexpr std::array<std::array<uint32_t, 17>, 4> kExpected = {{
    {0x9e37e78eU, 0x9e37e78cU, 0x9e37e48fU, 0x9e38e481U, 0x0a1d374bU, 0x0b1c374aU, 0x0b1c3949U, 0x0b1e394aU,
     0xe3134c47U, 0xe115485fU, 0xe1154d5cU, 0xe1104d58U, 0xbff9ab3aU, 0xb8f6a135U, 0xb8f6ad36U, 0xb8faad3bU,
     0xb305d06bU},
    {0x9e37e7cfU, 0x9e37e7cdU, 0x9e37e4ceU, 0x9e38e4c0U, 0x0a1d28daU, 0x0b1c26dbU, 0x0b1c28d8U, 0x0b1e28dbU,
     0xe31735f6U, 0xe11931feU, 0xe11934fdU, 0xe11434f1U, 0xbce144b4U, 0xb9e646bdU, 0xb9e642beU, 0xb9e242bbU,
     0xf7911ec5U},
    {0x9f4542bfU, 0x9f4542bfU, 0x9f4543bcU, 0x9f4643beU, 0x7947d500U, 0x7846cb01U, 0x7846d502U, 0x7844d501U,
     0x35b16f70U, 0x37b76f70U, 0x37b76273U, 0x37b26277U, 0x72cea2a3U, 0x7fcba8b8U, 0x7fcbacbbU, 0x7fcfacbeU,
     0xc0eb4199U},
    {0x21c8a74fU, 0x21c8a74fU, 0x21c8a44cU, 0x21c7a442U, 0xf98cc195U, 0xf88dc192U, 0xf88dc391U, 0xf88bc396U,
     0x7b205133U, 0x8522b133U, 0x8522ac30U, 0x8527ac34U, 0xb129153fU, 0xbe2c1d3aU, 0xbe2c1939U, 0xbe28193cU,
     0xeac7238fU},
}};

void CheckKnownValues(uint32_t len)
{
    for (size_t i = 0; i < kSeeds.size(); ++i) {
        EXPECT_EQ(RuntimeKb::CommonHash(kInput.data(), len, kSeeds[i]), kExpected[i][len])
            << "length=" << len << ", seed=" << kSeeds[i];
    }
}
} // namespace

TEST(CommonHashTest, ShortInputsMatchKnownValues)
{
    for (uint32_t len = 0; len < sizeof(uint32_t); ++len) {
        CheckKnownValues(len);
    }
}

TEST(CommonHashTest, CompleteBlocksKeepLegacyValues)
{
    for (uint32_t len = sizeof(uint32_t); len <= sizeof(kInput); len += sizeof(uint32_t)) {
        CheckKnownValues(len);
    }
}

TEST(CommonHashTest, PartialBlocksMatchKnownValues)
{
    for (uint32_t len = sizeof(uint32_t) + 1; len < sizeof(kInput); ++len) {
        if (len % sizeof(uint32_t) != 0) {
            CheckKnownValues(len);
        }
    }
}

TEST(CommonHashTest, DefaultSeedMatchesKnownValues)
{
    for (uint32_t len = 0; len <= sizeof(kInput); ++len) {
        EXPECT_EQ(RuntimeKb::CommonHash(kInput.data(), len), kExpected[2][len]) << "length=" << len;
    }
}

TEST(CommonHashTest, NullInputReturnsSeed)
{
    for (uint32_t seed : kSeeds) {
        EXPECT_EQ(RuntimeKb::CommonHash(nullptr, 0U, seed), seed);
        EXPECT_EQ(RuntimeKb::CommonHash(nullptr, 8U, seed), seed);
    }
}
