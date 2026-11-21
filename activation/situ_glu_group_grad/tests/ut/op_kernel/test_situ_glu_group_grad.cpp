/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

// The kernel entry is templated on schMode; the data type comes from DTYPE_X (half in this UT).
template <uint64_t schMode>
__global__ __aicore__ void situ_glu_group_grad(GM_ADDR gradY, GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex,
                                               GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR workspace, GM_ADDR tiling);

namespace {
// group_list_type: 0 = cumsum (group_index[-1] is the total row count),
// 1 = count (sum(group_index) is the total row count).
constexpr int64_t GROUP_LIST_TYPE_CUMSUM = 0;
constexpr int64_t GROUP_LIST_TYPE_COUNT = 1;

class SituGluGroupGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SituGluGroupGradKernelTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SituGluGroupGradKernelTest TearDown" << std::endl; }
};

void FillSituGluGroupGradTiling(SituGluGroupGradTilingData* tilingData, int64_t bs, int64_t d, uint32_t coreNum,
                                float beta, float linearBeta, int64_t activateLeft)
{
    int64_t splitD = d / 2;
    tilingData->bs = bs;
    tilingData->d = d;
    tilingData->splitD = splitD;
    tilingData->rowOfFormerBlock = 1;
    tilingData->rowOfTailBlock = 1;
    tilingData->rowLoopOfFormerBlock = 1;
    tilingData->rowLoopOfTailBlock = 1;
    tilingData->rowFactor = 1;
    tilingData->tailRowFactorOfFormerBlock = 1;
    tilingData->tailRowFactorOfTailBlock = 1;
    tilingData->dLoop = 1;
    tilingData->dFactor = splitD;
    tilingData->tailDFactor = splitD;
    tilingData->beta = beta;
    tilingData->linearBeta = linearBeta;
    tilingData->activateLeft = activateLeft;
    tilingData->groupIndexElemNum = 0;
    // Match the host tiling: the kernel pool gets total UB (253952) minus the 8K reserve that
    // absorbs the full-VL tail over-read of the VF loader / any framework stack usage.
    tilingData->ubSize = 253952 - 8192;
    tilingData->groupIndexLoop = 0;
    tilingData->groupIndexFactor = 0;
    tilingData->tailGroupIndexFactor = 0;
    tilingData->coreNum = coreNum;
    tilingData->hasWeight = 0;
    tilingData->hasGroupIndex = 0;
    // 0 = cumsum (group_index[-1] is the total row count), 1 = count (sum(group_index) is the total).
    tilingData->groupListType = 0;
    tilingData->hasGradW = 0;
}

// CPU golden of the backward. Returns grad_gate / grad_up and the unweighted grad_weight integrand.
void RefSituGrad(float gate, float up, float gradY, float beta, float linearBeta, float w, float& gradGate,
                 float& gradUp, float& partial)
{
    const float t = std::tanh(gate / beta);
    const float s = 1.0f / (1.0f + std::exp(-gate));
    const float situA = beta * t * s;
    float upPrime = up;
    float upGrad = 1.0f;
    if (linearBeta > 0.0f) {
        const float tU = std::tanh(up / linearBeta);
        upPrime = linearBeta * tU;
        upGrad = 1.0f - tU * tU;
    }
    const float gradSituA = gradY * upPrime;
    const float gradUpPrime = gradY * situA;
    gradGate = gradSituA * s * ((1.0f - t * t) + beta * t * (1.0f - s)) * w;
    gradUp = gradUpPrime * upGrad * w;
    partial = gradY * upPrime * situA; // unweighted grad_weight integrand
}

// Numeric check of grad_x (and grad_weight when weight is present) against a CPU golden.
// fp16 inputs are quantized first so the golden sees exactly what the kernel sees.
void RunKernelNumeric(int64_t bs, int64_t d, bool withWeight, bool hasLinearBeta, int64_t activateLeft)
{
    const int64_t splitD = d / 2;
    const uint32_t blockDim = static_cast<uint32_t>(bs);
    const float beta = 2.0f;
    const float linearBeta = hasLinearBeta ? 3.0f : 0.0f;

    uint8_t* gradY = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* gradX = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* weight = withWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * sizeof(float))) : nullptr;
    uint8_t* gradWeight = withWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * sizeof(float))) : nullptr;
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupGradTilingData)));

    auto* gradYH = reinterpret_cast<half*>(gradY);
    for (int64_t i = 0; i < bs * splitD; i++) {
        gradYH[i] = half(static_cast<float>((i % 29) - 14) * 0.125f);
    }
    auto* xH = reinterpret_cast<half*>(x);
    for (int64_t i = 0; i < bs * d; i++) {
        xH[i] = half(static_cast<float>((i % 37) - 18) * 0.125f);
    }
    auto* w = reinterpret_cast<float*>(weight);
    if (withWeight) {
        for (int64_t r = 0; r < bs; r++) {
            w[r] = 0.5f + 0.25f * static_cast<float>(r);
        }
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupGradTilingData*>(tiling);
    FillSituGluGroupGradTiling(tilingData, bs, d, blockDim, beta, linearBeta, activateLeft);
    tilingData->hasWeight = withWeight ? 1 : 0;
    tilingData->hasGradW = withWeight ? 1 : 0;

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_GRAD_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group_grad<0>), blockDim, gradY, x, weight, nullptr, gradX, gradWeight, workspace, tiling);

    auto* gradXH = reinterpret_cast<half*>(gradX);
    auto* gradWH = reinterpret_cast<float*>(gradWeight);
    for (int64_t r = 0; r < bs; r++) {
        float wSum = 0.0f;
        for (int64_t c = 0; c < splitD; c++) {
            const int64_t gateIdx = (activateLeft == 1) ? r * d + c : r * d + splitD + c;
            const int64_t upIdx = (activateLeft == 1) ? r * d + splitD + c : r * d + c;
            const float gate = static_cast<float>(xH[gateIdx]);
            const float up = static_cast<float>(xH[upIdx]);
            const float gy = static_cast<float>(gradYH[r * splitD + c]);
            float gradGate = 0.0f;
            float gradUp = 0.0f;
            float partial = 0.0f;
            RefSituGrad(gate, up, gy, beta, linearBeta, withWeight ? w[r] : 1.0f, gradGate, gradUp, partial);
            wSum += partial;
            EXPECT_NEAR(static_cast<float>(gradXH[gateIdx]), gradGate, 1e-2) << "row " << r << " col " << c;
            EXPECT_NEAR(static_cast<float>(gradXH[upIdx]), gradUp, 1e-2) << "row " << r << " col " << c;
        }
        if (withWeight) {
            EXPECT_NEAR(gradWH[r], wSum, 1e-2) << "grad_weight row " << r;
        }
    }

    AscendC::GmFree(gradY);
    AscendC::GmFree(x);
    AscendC::GmFree(gradX);
    if (weight != nullptr) {
        AscendC::GmFree(weight);
    }
    if (gradWeight != nullptr) {
        AscendC::GmFree(gradWeight);
    }
    AscendC::GmFree(tiling);
}

// group_index truncation: only the first realBs rows are computed/written. grad_x is pre-filled with
// a sentinel so rows [realBs, bs) can be asserted untouched; only rows [0, realBs) are compared.
// The group_index values are mode specific and would yield a different realBs in the other mode, so
// a group_list_type mix-up is detected.
void RunKernelPartialGroup(int64_t bs, int64_t d, bool withWeight, bool hasLinearBeta, int64_t activateLeft,
                           int64_t realBs, int64_t groupListType)
{
    const int64_t splitD = d / 2;
    const uint32_t blockDim = static_cast<uint32_t>(bs);
    const float beta = 2.0f;
    const float linearBeta = hasLinearBeta ? 3.0f : 0.0f;
    const int64_t gLen = 2;
    const float kSentinel = 12344.0f; // must be exact in fp16 (12345 rounds to 12344)
    const half kSentinelH = half(kSentinel);

    uint8_t* gradY = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* gradX = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* weight = withWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * sizeof(float))) : nullptr;
    uint8_t* gradWeight = withWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * sizeof(float))) : nullptr;
    uint8_t* groupIndex = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gLen * sizeof(int64_t)));
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupGradTilingData)));

    auto* gradYH = reinterpret_cast<half*>(gradY);
    for (int64_t i = 0; i < bs * splitD; i++) {
        gradYH[i] = half(static_cast<float>((i % 29) - 14) * 0.125f);
    }
    auto* xH = reinterpret_cast<half*>(x);
    for (int64_t i = 0; i < bs * d; i++) {
        xH[i] = half(static_cast<float>((i % 37) - 18) * 0.125f);
    }
    auto* gradXH = reinterpret_cast<half*>(gradX);
    for (int64_t i = 0; i < bs * d; i++) {
        gradXH[i] = kSentinelH;
    }
    auto* w = reinterpret_cast<float*>(weight);
    if (withWeight) {
        for (int64_t r = 0; r < bs; r++) {
            w[r] = 0.5f + 0.25f * static_cast<float>(r);
        }
    }
    auto* gi = reinterpret_cast<int64_t*>(groupIndex);
    if (groupListType == GROUP_LIST_TYPE_COUNT) {
        gi[0] = realBs; // count mode: the group sizes sum to realBs, the last element alone is 0
        gi[1] = 0;
    } else {
        gi[0] = realBs > 1 ? realBs - 1 : 0; // cumsum mode: the running totals end at realBs
        gi[1] = realBs;
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupGradTilingData*>(tiling);
    FillSituGluGroupGradTiling(tilingData, bs, d, blockDim, beta, linearBeta, activateLeft);
    tilingData->hasWeight = withWeight ? 1 : 0;
    tilingData->hasGradW = withWeight ? 1 : 0;
    tilingData->groupIndexElemNum = gLen;
    tilingData->groupIndexLoop = 1;
    tilingData->groupIndexFactor = gLen;
    tilingData->tailGroupIndexFactor = gLen;
    tilingData->hasGroupIndex = 1;
    tilingData->groupListType = groupListType;

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_GRAD_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group_grad<0>), blockDim, gradY, x, weight, groupIndex, gradX, gradWeight, workspace, tiling);

    for (int64_t r = 0; r < realBs; r++) {
        for (int64_t c = 0; c < splitD; c++) {
            const int64_t gateIdx = (activateLeft == 1) ? r * d + c : r * d + splitD + c;
            const int64_t upIdx = (activateLeft == 1) ? r * d + splitD + c : r * d + c;
            const float gate = static_cast<float>(xH[gateIdx]);
            const float up = static_cast<float>(xH[upIdx]);
            const float gy = static_cast<float>(gradYH[r * splitD + c]);
            float gradGate = 0.0f;
            float gradUp = 0.0f;
            float partial = 0.0f;
            RefSituGrad(gate, up, gy, beta, linearBeta, withWeight ? w[r] : 1.0f, gradGate, gradUp, partial);
            EXPECT_NEAR(static_cast<float>(gradXH[gateIdx]), gradGate, 1e-2) << "row " << r << " col " << c;
            EXPECT_NEAR(static_cast<float>(gradXH[upIdx]), gradUp, 1e-2) << "row " << r << " col " << c;
        }
    }
    for (int64_t r = realBs; r < bs; r++) {
        for (int64_t c = 0; c < d; c++) {
            EXPECT_FLOAT_EQ(static_cast<float>(gradXH[r * d + c]), kSentinel)
                << "wrote row " << r << " beyond realBs=" << realBs;
        }
    }

    AscendC::GmFree(gradY);
    AscendC::GmFree(x);
    AscendC::GmFree(gradX);
    if (weight != nullptr) {
        AscendC::GmFree(weight);
    }
    if (gradWeight != nullptr) {
        AscendC::GmFree(gradWeight);
    }
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(tiling);
}

// group_index length not a multiple of the int64 VL (32). In count mode the sum loop's tail mask must
// still yield exactly realBs; in cumsum mode only the last element is read, so the length must not
// matter. grad_x is pre-filled with a sentinel so rows [realBs, bs) assert untouched.
void RunKernelGroupTail(int64_t bs, int64_t d, int64_t gLen, int64_t realBs, int64_t groupListType)
{
    const int64_t splitD = d / 2;
    const uint32_t blockDim = static_cast<uint32_t>(bs);
    const float beta = 2.0f;
    const float linearBeta = 3.0f;
    const float kSentinel = 12344.0f; // must be exact in fp16 (12345 rounds to 12344)
    const half kSentinelH = half(kSentinel);

    uint8_t* gradY = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* gradX = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* groupIndex = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gLen * sizeof(int64_t)));
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupGradTilingData)));

    auto* gradYH = reinterpret_cast<half*>(gradY);
    for (int64_t i = 0; i < bs * splitD; i++) {
        gradYH[i] = half(static_cast<float>((i % 29) - 14) * 0.125f);
    }
    auto* xH = reinterpret_cast<half*>(x);
    for (int64_t i = 0; i < bs * d; i++) {
        xH[i] = half(static_cast<float>((i % 37) - 18) * 0.125f);
    }
    auto* gradXH = reinterpret_cast<half*>(gradX);
    for (int64_t i = 0; i < bs * d; i++) {
        gradXH[i] = kSentinelH;
    }
    auto* gi = reinterpret_cast<int64_t*>(groupIndex);
    for (int64_t i = 0; i < gLen; i++) {
        // count mode: only the last element carries a count, so the sum equals realBs.
        // cumsum mode: monotonic running totals ending at realBs, whose sum far exceeds realBs,
        // so reading the sum instead of the last element would clamp to bs and be detected.
        gi[i] = (groupListType == GROUP_LIST_TYPE_COUNT) ? 0 : (realBs * (i + 1)) / gLen;
    }
    gi[gLen - 1] = realBs;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupGradTilingData*>(tiling);
    FillSituGluGroupGradTiling(tilingData, bs, d, blockDim, beta, linearBeta, 1);
    tilingData->hasWeight = 0;
    tilingData->hasGradW = 0;
    tilingData->groupIndexElemNum = gLen;
    tilingData->groupIndexLoop = 1;
    tilingData->groupIndexFactor = gLen;
    tilingData->tailGroupIndexFactor = gLen;
    tilingData->hasGroupIndex = 1;
    tilingData->groupListType = groupListType;

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_GRAD_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group_grad<0>), blockDim, gradY, x, nullptr, groupIndex, gradX, nullptr, workspace, tiling);

    for (int64_t r = 0; r < realBs; r++) {
        for (int64_t c = 0; c < splitD; c++) {
            const float gate = static_cast<float>(xH[r * d + c]);
            const float up = static_cast<float>(xH[r * d + splitD + c]);
            const float gy = static_cast<float>(gradYH[r * splitD + c]);
            float gradGate = 0.0f;
            float gradUp = 0.0f;
            float partial = 0.0f;
            RefSituGrad(gate, up, gy, beta, linearBeta, 1.0f, gradGate, gradUp, partial);
            EXPECT_NEAR(static_cast<float>(gradXH[r * d + c]), gradGate, 1e-2)
                << "gLen=" << gLen << " row " << r << " col " << c;
            EXPECT_NEAR(static_cast<float>(gradXH[r * d + splitD + c]), gradUp, 1e-2)
                << "gLen=" << gLen << " row " << r << " col " << c;
        }
    }
    for (int64_t r = realBs; r < bs; r++) {
        for (int64_t c = 0; c < d; c++) {
            EXPECT_FLOAT_EQ(static_cast<float>(gradXH[r * d + c]), kSentinel)
                << "gLen=" << gLen << " wrote row " << r << " beyond realBs=" << realBs;
        }
    }

    AscendC::GmFree(gradY);
    AscendC::GmFree(x);
    AscendC::GmFree(gradX);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(tiling);
}

// d == 0 (hence splitD == 0): grad_x is empty and grad_weight[n] is a sum over an empty cut dim,
// i.e. 0. The kernel leaves the main loop before any tile, so it must zero-fill grad_weight for the
// valid rows ([0, bs) without group_index, [0, realBs) with it) and keep the sentinel past them.
// blockDim is a parameter because it no longer equals bs here: it must cover both idle blocks
// (blockDim > rows) and multi-chunk fills (rows per block > the zero-fill chunk).
void RunKernelEmptyD(int64_t bs, uint32_t blockDim, bool withWeight, bool hasGroupIndex, int64_t realBs,
                     int64_t groupListType)
{
    const float kSentinel = 12345.0f;
    const int64_t gLen = 2;
    const int64_t expectRows = hasGroupIndex ? realBs : bs;
    const int64_t wElems = bs > 0 ? bs : 1;

    // x / grad_y / grad_x hold no element; allocate one so GmAlloc never gets 0.
    uint8_t* gradY = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(half)));
    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(half)));
    uint8_t* gradX = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(half)));
    uint8_t* weight = withWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wElems * sizeof(float))) : nullptr;
    uint8_t* gradWeight = withWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(wElems * sizeof(float))) : nullptr;
    uint8_t* groupIndex = hasGroupIndex ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gLen * sizeof(int64_t))) :
                                          nullptr;
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupGradTilingData)));

    auto* gradWH = reinterpret_cast<float*>(gradWeight);
    if (withWeight) {
        auto* w = reinterpret_cast<float*>(weight);
        for (int64_t r = 0; r < bs; r++) {
            w[r] = 0.5f + 0.25f * static_cast<float>(r);
            gradWH[r] = kSentinel;
        }
    }
    if (hasGroupIndex) {
        auto* gi = reinterpret_cast<int64_t*>(groupIndex);
        if (groupListType == GROUP_LIST_TYPE_COUNT) {
            gi[0] = realBs; // count mode: the group sizes sum to realBs
            gi[1] = 0;
        } else {
            gi[0] = realBs > 1 ? realBs - 1 : 0; // cumsum mode: the running totals end at realBs
            gi[1] = realBs;
        }
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupGradTilingData*>(tiling);
    FillSituGluGroupGradTiling(tilingData, bs, 0, blockDim, 2.0f, 0.0f, 1);
    // Mirror the host SetEmptyTiling for the zero-workload case: no row loop, positive factors.
    tilingData->rowOfFormerBlock = 0;
    tilingData->rowOfTailBlock = 0;
    tilingData->rowLoopOfFormerBlock = 0;
    tilingData->rowLoopOfTailBlock = 0;
    tilingData->tailRowFactorOfFormerBlock = 0;
    tilingData->tailRowFactorOfTailBlock = 0;
    tilingData->dFactor = 1;
    tilingData->tailDFactor = 1;
    tilingData->hasWeight = withWeight ? 1 : 0;
    tilingData->hasGradW = withWeight ? 1 : 0;
    if (hasGroupIndex) {
        tilingData->groupIndexElemNum = gLen;
        tilingData->groupIndexLoop = 1;
        tilingData->groupIndexFactor = gLen;
        tilingData->tailGroupIndexFactor = gLen;
        tilingData->hasGroupIndex = 1;
        tilingData->groupListType = groupListType;
    }

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_GRAD_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group_grad<0>), blockDim, gradY, x, weight, groupIndex, gradX, gradWeight, workspace, tiling);

    if (withWeight) {
        for (int64_t r = 0; r < bs; r++) {
            if (r < expectRows) {
                EXPECT_FLOAT_EQ(gradWH[r], 0.0f) << "grad_weight row " << r << " not zeroed";
            } else {
                EXPECT_FLOAT_EQ(gradWH[r], kSentinel) << "zeroed row " << r << " beyond realBs=" << expectRows;
            }
        }
    }

    AscendC::GmFree(gradY);
    AscendC::GmFree(x);
    AscendC::GmFree(gradX);
    if (weight != nullptr) {
        AscendC::GmFree(weight);
    }
    if (gradWeight != nullptr) {
        AscendC::GmFree(gradWeight);
    }
    if (groupIndex != nullptr) {
        AscendC::GmFree(groupIndex);
    }
    AscendC::GmFree(tiling);
}

TEST_F(SituGluGroupGradKernelTest, numeric_activate_left) { RunKernelNumeric(2, 256, false, false, 1); }

TEST_F(SituGluGroupGradKernelTest, numeric_weight_linear_beta) { RunKernelNumeric(2, 256, true, true, 1); }

// d = 250 -> splitD = 125 (not 32B aligned), exercises the masked VF tail path.
TEST_F(SituGluGroupGradKernelTest, numeric_activate_right_unaligned) { RunKernelNumeric(2, 250, true, true, 0); }

// group_index total = 3 < rows = 4: only rows 0..2 are written and compared.
TEST_F(SituGluGroupGradKernelTest, numeric_group_truncation)
{
    RunKernelPartialGroup(4, 256, true, true, 1, 3, GROUP_LIST_TYPE_CUMSUM);
}

TEST_F(SituGluGroupGradKernelTest, numeric_group_truncation_count)
{
    RunKernelPartialGroup(4, 256, true, true, 1, 3, GROUP_LIST_TYPE_COUNT);
}

TEST_F(SituGluGroupGradKernelTest, numeric_group_truncation_no_weight)
{
    RunKernelPartialGroup(2, 256, false, false, 0, 1, GROUP_LIST_TYPE_CUMSUM);
}

TEST_F(SituGluGroupGradKernelTest, numeric_group_truncation_no_weight_count)
{
    RunKernelPartialGroup(2, 256, false, false, 0, 1, GROUP_LIST_TYPE_COUNT);
}

// group_index lengths that are not a multiple of the int64 VL (32). int64 VL = 256B / 8B = 32.
// realBs is pinned to 5 while bs = 8: an under-count leaves rows [0,5) as sentinel, an
// over-count writes rows [5,8), so both directions are detected.
TEST_F(SituGluGroupGradKernelTest, group_index_tail_33) { RunKernelGroupTail(8, 256, 33, 5, GROUP_LIST_TYPE_COUNT); }
TEST_F(SituGluGroupGradKernelTest, group_index_tail_40) { RunKernelGroupTail(8, 256, 40, 5, GROUP_LIST_TYPE_COUNT); }
TEST_F(SituGluGroupGradKernelTest, group_index_tail_65) { RunKernelGroupTail(8, 256, 65, 5, GROUP_LIST_TYPE_COUNT); }
TEST_F(SituGluGroupGradKernelTest, group_index_tail_97) { RunKernelGroupTail(8, 256, 97, 5, GROUP_LIST_TYPE_COUNT); }
// 257 -> loopCount 9, fourLoopCount 2, tailLoopNum 1: exercises the four-unfold main loop
// together with a 1-element tail.
TEST_F(SituGluGroupGradKernelTest, group_index_tail_257) { RunKernelGroupTail(8, 256, 257, 5, GROUP_LIST_TYPE_COUNT); }

// cumsum mode reads group_index[-1] only: the length and the preceding values must not matter.
TEST_F(SituGluGroupGradKernelTest, group_index_cumsum_1) { RunKernelGroupTail(8, 256, 1, 5, GROUP_LIST_TYPE_CUMSUM); }
TEST_F(SituGluGroupGradKernelTest, group_index_cumsum_33) { RunKernelGroupTail(8, 256, 33, 5, GROUP_LIST_TYPE_CUMSUM); }
TEST_F(SituGluGroupGradKernelTest, group_index_cumsum_257)
{
    RunKernelGroupTail(8, 256, 257, 5, GROUP_LIST_TYPE_CUMSUM);
}

// d == 0: grad_weight is an empty sum and must come back zeroed for every valid row.
// blockDim 16 > bs 4 leaves most blocks idle.
TEST_F(SituGluGroupGradKernelTest, empty_d_weight_idle_blocks) { RunKernelEmptyD(4, 16, true, false, 0, 0); }

// bs 7 over blockDim 3: rows per block 3/3/1, so the last block owns a partial range.
TEST_F(SituGluGroupGradKernelTest, empty_d_weight_uneven_blocks) { RunKernelEmptyD(7, 3, true, false, 0, 0); }

// 1500 rows per block exceeds the 1024-row zero-fill chunk, so the fill loops more than once.
TEST_F(SituGluGroupGradKernelTest, empty_d_weight_multi_chunk) { RunKernelEmptyD(3000, 2, true, false, 0, 0); }

// d == 0 with group_index: only the first realBs rows are owed zeros, the rest stay sentinel.
TEST_F(SituGluGroupGradKernelTest, empty_d_group_cumsum)
{
    RunKernelEmptyD(8, 8, true, true, 5, GROUP_LIST_TYPE_CUMSUM);
}

TEST_F(SituGluGroupGradKernelTest, empty_d_group_count) { RunKernelEmptyD(8, 8, true, true, 5, GROUP_LIST_TYPE_COUNT); }

// No weight means no grad_weight output at all: the early exit must stay a no-op.
TEST_F(SituGluGroupGradKernelTest, empty_d_no_weight) { RunKernelEmptyD(4, 4, false, false, 0, 0); }

// bs == 0 empties grad_weight as well, so there is nothing to zero.
TEST_F(SituGluGroupGradKernelTest, empty_bs_zero) { RunKernelEmptyD(0, 1, true, false, 0, 0); }
} // namespace
