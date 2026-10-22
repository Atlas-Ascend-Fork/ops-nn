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

#include "../../../../common/op_host/op_tiling/hash.h"

namespace {
constexpr std::array<uint32_t, 4> kInput = {0x01010101U, 0x02020202U, 0x03030303U, 0x04040404U};
constexpr std::array<uint32_t, 4> kSeeds = {0U, 1U, 271828U, UINT32_MAX};

// Frozen MurmurHash3_x86_32 results from Austin Appleby's SMHasher:
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
// Rows use kSeeds; columns cover lengths 0..16, including tails after 1..3 blocks.
constexpr std::array<std::array<uint32_t, 17>, 4> kMurmurExpected = {{
    {0x00000000U, 0x0feb9e1dU, 0x14a6c117U, 0x20fe586cU, 0x6bf83385U, 0x448894c7U, 0x4ea44e7aU, 0x1f3a761fU,
     0x3105de56U, 0x94576c9aU, 0xc78123f9U, 0x24062e20U, 0x8f3b1341U, 0x265a5d64U, 0xb70ad617U, 0x16e5482bU,
     0xb4b21f93U},
    {0x514e28b7U, 0x1a9ac5d9U, 0x165d53c5U, 0x239a84bdU, 0xe1adb943U, 0x6063a539U, 0xdbae1ec7U, 0xbd3b5f8bU,
     0x9abb0197U, 0x3acc7d99U, 0x66f1ff74U, 0x110a06f4U, 0x2497bcf8U, 0x81975eebU, 0x227bad90U, 0xe4b2440cU,
     0x0dd1b664U},
    {0x737948b2U, 0x15608a87U, 0xf82d5271U, 0xdb77f6ecU, 0xb8f9f2e5U, 0x908d4ff8U, 0x0e3aa3e9U, 0xf6f652dfU,
     0x134d231eU, 0x4f349975U, 0x4a46eadbU, 0xca0d513cU, 0x32265fd2U, 0xaf14d570U, 0x55ee496bU, 0x07dddd14U,
     0x91bf120fU},
    {0x81f16f39U, 0xbebd35d3U, 0x15987121U, 0xbd7042aaU, 0x427a3375U, 0x0cdd715fU, 0xe2cd0363U, 0x067fbde6U,
     0xd2194acaU, 0x0bd82831U, 0x5369c149U, 0x381dfcd9U, 0xf1dbe1f3U, 0xed446139U, 0x7b2c96c7U, 0x85948740U,
     0xd2ae21a4U},
}};

std::array<uint32_t, 4> MakeMurmurInput(uint32_t len)
{
    // Full words stay aligned and byte-order independent. Distinct tail bytes
    // catch indexing/order mistakes as well as reading from the original src.
    auto input = kInput;
    constexpr std::array<uint8_t, 3> tail = {0x80U, 0x37U, 0xe9U};
    auto* bytes = reinterpret_cast<uint8_t*>(input.data());
    const uint32_t remainder = len % sizeof(uint32_t);
    for (uint32_t i = 0; i < remainder; ++i) {
        bytes[len - remainder + i] = tail[i];
    }
    return input;
}
} // namespace

TEST(MatmulMurmurHashTest, KnownVectors)
{
    for (uint32_t len = 0; len <= sizeof(kInput); ++len) {
        const auto input = MakeMurmurInput(len);
        for (size_t i = 0; i < kSeeds.size(); ++i) {
            EXPECT_EQ(Ops::NN::MurmurHash(input.data(), len, kSeeds[i]), kMurmurExpected[i][len])
                << "length=" << len << ", seed=" << kSeeds[i];
        }
    }
}

TEST(MatmulMurmurHashTest, DefaultSeedMatchesKnownVectors)
{
    for (uint32_t len = 0; len <= sizeof(kInput); ++len) {
        const auto input = MakeMurmurInput(len);
        EXPECT_EQ(Ops::NN::MurmurHash(input.data(), len), kMurmurExpected[2][len]) << "length=" << len;
    }
}
