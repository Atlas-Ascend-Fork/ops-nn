/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_dual_level_mx_quant.cpp
 * \brief Kernel UT for DynamicDualLevelMxQuant scale algorithm paths.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include <utility>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/dynamic_dual_level_mx_quant.cpp"

using namespace std;

extern "C" __global__ __aicore__ void dynamic_dual_level_mx_quant(GM_ADDR x, GM_ADDR smooth_scale, GM_ADDR y,
                                                                  GM_ADDR level0_scale, GM_ADDR level1_scale,
                                                                  GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int64_t kCols = 512;
constexpr int64_t kGroups = kCols / 32;
constexpr int64_t kYPackedBytes = kCols / 2;
constexpr int64_t kGroupBytes = 16; // 32 个 FP4 元素打包成 16 字节

// 一行 512 个 half 元素，按 32 个一组构造：
//   group0: max=6        -> 各算法 scale=1
//   group1: max=3.5      -> OCP/max7/max8: 0.5；ceil/max6: 1
//   group2: max=1.75     -> OCP/max7/max8: 0.25；ceil/max6: 0.5
//   group3: 全 0         -> scale=0
//   group4: inf/nan      -> scale=0xFF
//   group5: 2^-24        -> OCP/ceil/max6/max7: scale 2^-26；max8: 2^-27
//   group6~15: 全 0
std::vector<uint16_t> MakeInput()
{
    std::vector<uint16_t> x(kCols, 0x0000);
    auto setGroup = [&](int64_t group, std::initializer_list<uint16_t> values) {
        int64_t base = group * 32;
        int64_t i = 0;
        for (auto value : values) {
            x[base + i] = value;
            ++i;
        }
    };
    // 6.0, 4.0, 3.0, 2.0, 1.0, 0.5
    setGroup(0, {0x4600, 0x4400, 0x4200, 0x4000, 0x3C00, 0x3800});
    // 3.5, 3.0, 2.0, 1.5, 1.0, 0.5
    setGroup(1, {0x4300, 0x4200, 0x4000, 0x3E00, 0x3C00, 0x3800});
    // 1.75, 1.5, 1.0, 0.5
    setGroup(2, {0x3F00, 0x3E00, 0x3C00, 0x3800});
    // inf, nan
    setGroup(4, {0x7C00, 0x7E00});
    // 2^-24 (half subnormal)
    setGroup(5, {0x0001});
    return x;
}

struct Expectation {
    std::vector<uint8_t> scale;                      // 16 个 E8M0 字节
    std::vector<std::pair<uint8_t, uint8_t>> nibble; // 256 个字节，每个字节两个 FP4 nibble
    std::vector<bool> valid;                         // 是否校验该字节
};

// FP4_E2M1 编码: 0->0, 0.5->1, 1->2, 1.5->3, 2->4, 3->5, 4->6, 6->7
Expectation MakeExpectation(bool group1HalfScale, bool group2QuarterScale, uint8_t group5Scale, uint8_t group5Nibble)
{
    Expectation exp;
    exp.scale.assign(kGroups, 0);
    exp.scale[0] = 127; // scale 1
    exp.scale[1] = group1HalfScale ? 126 : 127;
    exp.scale[2] = group2QuarterScale ? 125 : 126;
    exp.scale[3] = 0;
    exp.scale[4] = 255; // E8M0 NaN
    exp.scale[5] = group5Scale;

    exp.nibble.assign(kYPackedBytes, std::pair<uint8_t, uint8_t>{0, 0});
    exp.valid.assign(kYPackedBytes, true);
    auto setPair = [&](int64_t group, int64_t index, uint8_t first, uint8_t second) {
        exp.nibble[group * kGroupBytes + index] = {first, second};
    };

    // group0: x/1 -> 6,4,3,2,1,0.5 -> 7,6,5,4,2,1
    setPair(0, 0, 7, 6);
    setPair(0, 1, 5, 4);
    setPair(0, 2, 2, 1);

    // group1: scale=0.5 时 y=7,6,4,3,2,1；scale=1 时第一个元素 3.5 为 tie，跳过该字节
    if (group1HalfScale) {
        setPair(1, 0, 7, 7);
        setPair(1, 1, 6, 5);
        setPair(1, 2, 4, 2);
    } else {
        setPair(1, 1, 4, 3);
        setPair(1, 2, 2, 1);
        exp.valid[1 * kGroupBytes] = false;
    }

    // group2: scale=0.25 时 y=7,6,4,2；scale=0.5 时第一个元素 1.75 为 tie，跳过该字节
    if (group2QuarterScale) {
        setPair(2, 0, 7, 7);
        setPair(2, 1, 6, 4);
    } else {
        setPair(2, 1, 4, 2);
        exp.valid[2 * kGroupBytes] = false;
    }

    // group4 为 inf/nan，y 不校验
    for (int64_t byte = 4 * kGroupBytes; byte < 5 * kGroupBytes; ++byte) {
        exp.valid[byte] = false;
    }

    // group5: 2^-24 / 2^-26 = 4；2^-24 / 2^-27 = 8 -> 饱和到 6
    setPair(5, 0, group5Nibble, 0);
    return exp;
}

void RunScaleAlgCase(const char* name, int64_t scaleAlg, float dstTypeMax, float invDstTypeMax,
                     const Expectation& expectation)
{
    const size_t xSize = kCols * sizeof(uint16_t);
    const size_t smoothScaleSize = xSize;
    const size_t ySize = kYPackedBytes;
    const size_t level0ScaleSize = 32;
    const size_t level1ScaleSize = 32;
    const size_t tilingSize = sizeof(DynamicDualLevelMxQuantTilingData);
    const uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* smoothScale = (uint8_t*)AscendC::GmAlloc(smoothScaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* level0Scale = (uint8_t*)AscendC::GmAlloc(level0ScaleSize);
    uint8_t* level1Scale = (uint8_t*)AscendC::GmAlloc(level1ScaleSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(smoothScale, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(level0Scale, nullptr);
    ASSERT_NE(level1Scale, nullptr);
    ASSERT_NE(workSpace, nullptr);
    ASSERT_NE(tiling, nullptr);

    std::vector<uint16_t> xHost = MakeInput();
    std::memcpy(x, xHost.data(), xSize);
    std::memset(smoothScale, 0, smoothScaleSize);
    std::memset(y, 0, ySize);
    std::memset(level0Scale, 0, level0ScaleSize);
    std::memset(level1Scale, 0, level1ScaleSize);
    std::memset(tiling, 0, tilingSize);

    auto* tilingData = reinterpret_cast<DynamicDualLevelMxQuantTilingData*>(tiling);
    tilingData->tilingKey = 0;
    tilingData->totalCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->roundMode = 4; // rint
    tilingData->level0BlockSize = 512;
    tilingData->level1BlockSize = 32;
    tilingData->rowSize = kCols;
    tilingData->colSize = 1;
    tilingData->blockSizeRow = 512;
    tilingData->blockSizeCol = 1;
    tilingData->rowBlockNum = 1;
    tilingData->colBlockNum = 1;
    tilingData->rowTileNum = 1;
    tilingData->colTileNum = 1;
    tilingData->normalTileRowBlockNum = 1;
    tilingData->normalTileColBlockNum = 1;
    tilingData->tailTileRowBlockNum = 1;
    tilingData->tailTileColBlockNum = 1;
    tilingData->normalTileRowSize = kCols;
    tilingData->tailTileRowSize = kCols;
    tilingData->normalTileRowLoopNum = 1;
    tilingData->normalTileColLoopNum = 1;
    tilingData->tailTileRowLoopNum = 1;
    tilingData->tailTileColLoopNum = 1;
    tilingData->ubFactor = 1;
    tilingData->tailAlignNum = 0;
    tilingData->copyMethod = 0;
    tilingData->needSmoothScale = 0;
    tilingData->scaleAlg = scaleAlg;
    tilingData->dstTypeMax = dstTypeMax;
    tilingData->invDstTypeMax = invDstTypeMax;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(0);

    auto kernel = [](GM_ADDR xAddr, GM_ADDR smoothScaleAddr, GM_ADDR yAddr, GM_ADDR level0ScaleAddr,
                     GM_ADDR level1ScaleAddr, GM_ADDR workspaceAddr, GM_ADDR tilingAddr) {
        ::dynamic_dual_level_mx_quant<4>(xAddr, smoothScaleAddr, yAddr, level0ScaleAddr, level1ScaleAddr, workspaceAddr,
                                         tilingAddr);
    };
    ICPU_RUN_KF(kernel, blockDim, x, smoothScale, y, level0Scale, level1Scale, workSpace, tiling);

    // level0_scale = max(x)/6 = 6/6 = 1
    EXPECT_NEAR(*reinterpret_cast<float*>(level0Scale), 1.0f, 1e-6f) << name;

    // level1_scale: 16 个 32 元素组的 E8M0 字节
    const uint8_t* level1 = reinterpret_cast<const uint8_t*>(level1Scale);
    for (int64_t group = 0; group < kGroups; ++group) {
        EXPECT_EQ(level1[group], expectation.scale[group]) << name << " group " << group;
    }

    // y: 256 个字节，每字节两个 FP4。nibble 顺序不依赖硬件细节，两个顺序都接受。
    const uint8_t* yData = reinterpret_cast<const uint8_t*>(y);
    for (int64_t byte = 0; byte < kYPackedBytes; ++byte) {
        if (!expectation.valid[byte]) {
            continue;
        }
        uint8_t first = expectation.nibble[byte].first;
        uint8_t second = expectation.nibble[byte].second;
        uint8_t expected1 = static_cast<uint8_t>(first | (second << 4));
        uint8_t expected2 = static_cast<uint8_t>(second | (first << 4));
        EXPECT_TRUE(yData[byte] == expected1 || yData[byte] == expected2)
            << name << " y byte " << byte << " got " << static_cast<int>(yData[byte]) << " expected "
            << static_cast<int>(expected1) << " or " << static_cast<int>(expected2);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(smoothScale);
    AscendC::GmFree(y);
    AscendC::GmFree(level0Scale);
    AscendC::GmFree(level1Scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}
} // namespace

class dynamic_dual_level_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "dynamic_dual_level_mx_quant_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "dynamic_dual_level_mx_quant_test TearDown\n" << endl; }
};

TEST_F(dynamic_dual_level_mx_quant_test, test_case_scaleAlg0_ocp)
{
    RunScaleAlgCase("scaleAlg=0 OCP", 0, 0.0f, 0.0f, MakeExpectation(true, true, 101, 6));
}

TEST_F(dynamic_dual_level_mx_quant_test, test_case_scaleAlg1_ceil)
{
    RunScaleAlgCase("scaleAlg=1 ceil", 1, 0.0f, 1.0f / 6.0f, MakeExpectation(false, false, 101, 6));
}

TEST_F(dynamic_dual_level_mx_quant_test, test_case_scaleAlg2_default6)
{
    RunScaleAlgCase("scaleAlg=2 dstTypeMax=6", 2, 6.0f, 1.0f / 6.0f, MakeExpectation(false, false, 101, 6));
}

TEST_F(dynamic_dual_level_mx_quant_test, test_case_scaleAlg2_default7)
{
    RunScaleAlgCase("scaleAlg=2 dstTypeMax=7", 2, 7.0f, 1.0f / 7.0f, MakeExpectation(true, true, 101, 6));
}

TEST_F(dynamic_dual_level_mx_quant_test, test_case_scaleAlg2_custom8)
{
    RunScaleAlgCase("scaleAlg=2 dstTypeMax=8", 2, 8.0f, 1.0f / 8.0f, MakeExpectation(true, true, 100, 7));
}
