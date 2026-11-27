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
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>
#include "tikicpulib.h"
#include "../../../../op_kernel/arch35/rotate_dual_level_quant_tiling_data.h"

extern "C" __global__ __aicore__ void rotate_dual_level_quant(GM_ADDR x, GM_ADDR rot, GM_ADDR alpha, GM_ADDR groups,
                                                              GM_ADDR y, GM_ADDR level0, GM_ADDR level1,
                                                              GM_ADDR workspace, GM_ADDR tiling);

namespace {
namespace RDLQ = RotateDualLevelQuantNS;

struct KernelTilingFixture {
    uint32_t rows;
    uint32_t k;
    uint32_t baseK;
    uint32_t receiveBytes;
    uint32_t workspaceBytes;
};

// Fixed Ascend950 configurations for these kernel cases. Host UT verifies the
// resource selection through the registered tiling entry, independently of these fixtures.
constexpr std::array<KernelTilingFixture, 5> KERNEL_TILING_FIXTURES = {{{16, 32, 32, 1024, 64256},
                                                                        {16, 64, 64, 1536, 64256},
                                                                        {16, 128, 128, 2560, 64256},
                                                                        {16, 256, 64, 4608, 64256},
                                                                        {48, 32, 32, 2048, 81152}}};
constexpr uint32_t TEST_UB_BYTES = 253952;

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : data(static_cast<uint8_t*>(AscendC::GmAlloc(bytes))) {}
    ~GmBuffer() { AscendC::GmFree(data); }
    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    uint8_t* data;
};

// This target instantiates BF16 -> FP4. All six input/output dtype combinations
// are covered by Host/API UT and can use the TTK golden adapters on Ascend950.
void RunIdentity(uint32_t m, uint32_t n, uint32_t k, bool perColumn, bool grouped, int32_t groupType,
                 bool varyingColumn = false, uint32_t receiveSlots = 2, uint32_t tileRows = 16)
{
    const auto fixture = std::find_if(
        KERNEL_TILING_FIXTURES.begin(), KERNEL_TILING_FIXTURES.end(),
        [tileRows, k](const auto& value) { return value.rows == tileRows && value.k == k; });
    ASSERT_NE(fixture, KERNEL_TILING_FIXTURES.end());
    ASSERT_TRUE(receiveSlots == 1 || receiveSlots == 2);
    RDLQ::RotateDualLevelQuantTilingData data;
    data.M = m;
    data.N = n;
    data.rotK = k;
    data.rotationBlocksPerRow = n / k;
    data.rotationMatricesPerGroup = perColumn ? n / k : 1;
    data.level0ScalesPerRow = (n + 511) / 512;
    data.level1ScalePairsPerRow = (n + 63) / 64;
    data.level1ScalesPerRow = data.level1ScalePairsPerRow * 2;
    data.lastLevel0BlockSize = (n - 1) % 512 + 1;
    data.groupMode = grouped;
    data.groupNum = grouped ? 3 : 0;
    data.groupListType = groupType;
    data.logicalBlockCount = static_cast<uint64_t>(m) * data.level0ScalesPerRow;
    data.aivNumRatio = 2;
    data.splitM = 1;
    data.mL1 = data.baseM = fixture->rows;
    data.nL1 = data.baseN = data.kL1 = k;
    data.baseK = fixture->baseK;
    data.maxRowsPerAiv = fixture->rows / 2;
    data.ubDB = receiveSlots;
    data.ubPitchGran = RDLQ::FIXPIPE_ROW_PITCH_GRAN;
    data.fixpipeReceiveBytes = fixture->receiveBytes;
    data.quantWorkspaceBytes = fixture->workspaceBytes;
    data.collectUbOffset = data.mxWorkspaceBytes;
    data.collectUbBytes = data.maxRowsPerAiv * 512 * 2 + 512;
    data.level0ScaleUbOffset = data.collectUbOffset + data.collectUbBytes;
    data.level0ScaleUbBytes = data.maxRowsPerAiv * 32;
    const uint32_t privateEnd = data.fixpipeReceiveBytes + data.quantWorkspaceBytes;
    ASSERT_LE(privateEnd, receiveSlots == 2 ? TEST_UB_BYTES / 2 : TEST_UB_BYTES);
    ASSERT_EQ(data.level0ScaleUbOffset + data.level0ScaleUbBytes, data.quantWorkspaceBytes);
    data.totalUbBytes = receiveSlots == 2 ? std::max(privateEnd, TEST_UB_BYTES / 2 + data.fixpipeReceiveBytes) :
                                            privateEnd;
    data.usedCoreNum = 1;
    data.taskCount = grouped ? 0 : ((m + data.mL1 - 1) / data.mL1) * static_cast<uint64_t>((n + 511) / 512);
    const size_t rotations = (grouped ? 3 : 1) * data.rotationMatricesPerGroup;
    GmBuffer x(static_cast<size_t>(m) * n * 2);
    GmBuffer rot(rotations * k * k * 2);
    GmBuffer groups(3 * sizeof(int64_t));
    GmBuffer y(static_cast<size_t>(m) * n / 2);
    GmBuffer s0(static_cast<size_t>(m) * data.level0ScalesPerRow * sizeof(float));
    GmBuffer s1(static_cast<size_t>(m) * data.level1ScalesPerRow);
    GmBuffer tiling(sizeof(data));
    ASSERT_NE(x.data, nullptr);
    ASSERT_NE(rot.data, nullptr);
    ASSERT_NE(groups.data, nullptr);
    ASSERT_NE(y.data, nullptr);
    ASSERT_NE(s0.data, nullptr);
    ASSERT_NE(s1.data, nullptr);
    ASSERT_NE(tiling.data, nullptr);
    auto* xValues = reinterpret_cast<uint16_t*>(x.data);
    // 6 in the first original 512 block, 12 in the tail block.
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            xValues[row * n + col] = col < 512 ? 0x40c0 : 0x4140;
        }
    }
    std::memset(rot.data, 0, rotations * k * k * 2);
    auto* rotValues = reinterpret_cast<uint16_t*>(rot.data);
    for (size_t matrix = 0; matrix < rotations; ++matrix) {
        const bool negative = grouped && matrix / data.rotationMatricesPerGroup == 1;
        const bool twice = varyingColumn && matrix % data.rotationMatricesPerGroup == 512 / k - 1;
        for (uint32_t diagonal = 0; diagonal < k; ++diagonal) {
            rotValues[matrix * k * k + diagonal * k + diagonal] = twice ? (negative ? 0xc000 : 0x4000) :
                                                                          (negative ? 0xbf80 : 0x3f80);
        }
    }
    const std::vector<int64_t> groupValues = groupType == 0 ? std::vector<int64_t>{0, 1, 1} :
                                                              std::vector<int64_t>{0, 1, 0};
    std::memcpy(groups.data, groupValues.data(), 3 * sizeof(int64_t));
    std::memset(y.data, 0x33, static_cast<size_t>(m) * n / 2);
    std::fill_n(reinterpret_cast<float*>(s0.data), static_cast<size_t>(m) * data.level0ScalesPerRow, -7.0f);
    std::memset(s1.data, 0x7e, static_cast<size_t>(m) * data.level1ScalesPerRow);
    std::memcpy(tiling.data, &data, sizeof(data));
    ICPU_SET_TILING_KEY(grouped ? 1 : 0);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(rotate_dual_level_quant, data.usedCoreNum, x.data, rot.data, nullptr, grouped ? groups.data : nullptr,
                y.data, s0.data, s1.data, nullptr, tiling.data);
    ICPU_SET_TILING_KEY(0);
    for (uint32_t row = 0; row < m; ++row) {
        const bool processed = !grouped || row == 0;
        for (uint32_t byte = 0; byte < n / 2; ++byte) {
            EXPECT_EQ(y.data[row * (n / 2) + byte], processed ? (grouped ? 0xff : 0x77) : 0x33);
        }
        for (uint32_t block = 0; block < data.level0ScalesPerRow; ++block) {
            EXPECT_FLOAT_EQ(reinterpret_cast<float*>(s0.data)[row * data.level0ScalesPerRow + block],
                            processed ? (block == 0 && !varyingColumn ? 1.0f : 2.0f) : -7.0f);
        }
        for (uint32_t scale = 0; scale < data.level1ScalesPerRow; ++scale) {
            const uint8_t expected = !processed ?
                                         0x7e :
                                         (varyingColumn && scale < (512 - k) / 32 ? 126 : (scale < n / 32 ? 127 : 0));
            EXPECT_EQ(s1.data[row * data.level1ScalesPerRow + scale], expected);
        }
    }
}

TEST(RotateDualLevelQuantKernel, Complete512AcrossSixteenRotations)
{
    RunIdentity(3, 512, 32, false, false, 0);
    // The final K-wide matrix doubles its values: all sixteen rotations must
    // share s0=2, while the first fifteen MX blocks have s1=0.5 (code 126).
    RunIdentity(3, 512, 32, true, false, 0, true);
}
TEST(RotateDualLevelQuantKernel, SingleRowSingleAndDoubleReceiveSlots)
{
    // AIV1 has no rows but must return every input credit, including slot 1.
    RunIdentity(1, 512, 32, false, false, 0);
    RunIdentity(1, 512, 32, false, false, 0, false, 1);
    RunIdentity(1, 32, 32, false, false, 0, false, 1);
}
TEST(RotateDualLevelQuantKernel, Tail512AndOddMxPadding) { RunIdentity(17, 544, 32, true, false, 0, false, 2, 48); }
TEST(RotateDualLevelQuantKernel, K64K128AndK256)
{
    RunIdentity(1, 576, 64, false, false, 0);
    RunIdentity(3, 640, 128, false, false, 0);
    RunIdentity(3, 768, 256, true, false, 0);
}
TEST(RotateDualLevelQuantKernel, GroupedEmptyGroupsAndPreservedSuffix)
{
    RunIdentity(3, 544, 32, true, true, 0);
    RunIdentity(3, 544, 32, false, true, 1);
}
} // namespace
