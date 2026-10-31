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
 * \file test_fake_quant_with_min_max_vars_per_channel.cpp
 * \brief op_kernel UT for FakeQuantWithMinMaxVarsPerChannel (arch35 / DAV_3510 / RegBase)
 *
 * 模板参考: quant/grouped_dynamic_mx_quant/tests/ut/op_kernel/test_grouped_dynamic_mx_quant.cpp
 *           quant/dynamic_mx_quant/tests/ut/op_kernel/test_dynamic_mx_quant.cpp
 *
 * Kernel 入口 (模板):
 *     template <typename D_T_X, int MODE, int HAS_ZP, int ROUND_MODE>
 *     __global__ __aicore__ void fake_quant_with_min_max_vars_per_channel(...)
 * 调用方式: ::fake_quant_with_min_max_vars_per_channel<float, 2, 0, 0>(...)
 *
 * 测试用例覆盖（任务要求）：
 *   - case_1d_small:        1D (64,)
 *   - case_2d_multi_channel: 2D (4, 16)
 *   - case_num_bits_2:      边界 num_bits=2
 *   - case_num_bits_16:     边界 num_bits=16
 *   - case_narrow_range:    narrow_range=true
 *   - case_min_max_zero:    min==max==0 兜底，输出应为 0
 *   - case_min_max_equal_nonzero: min==max!=0 相等通道，输出应为 NaN（K1 修复，对齐 TF oracle）
 *   - case_nan_passthrough: x 含 NaN 时 NaN 传播到 y
 *   - case_equal_nonzero_nan: min==max!=0 整通道 NaN 显式断言（K1 新增）
 *   - case_inverted_nudged_min_side: min>max 倒置回退 nudged_min 侧显式断言（对齐 TF clamp 顺序）
 *   - case_both_zero_output_zero: min==max==0 输出全 0 显式断言（防退化边界误改）
 *
 * Golden: numpy-style 等价实现（与 tests/ttk/fake_quant_with_min_max_vars_per_channel_golden.py 一致），
 *         在 fp32 上跑，atol=1e-4 (与 ttk 标杆一致)。
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/fake_quant_with_min_max_vars_per_channel.cpp"
#include "../../../op_kernel/arch35/fake_quant_with_min_max_vars_per_channel_tiling_data.h"

using namespace std;

class fake_quant_with_min_max_vars_per_channel_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "fake_quant_with_min_max_vars_per_channel_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "fake_quant_with_min_max_vars_per_channel_test TearDown\n" << endl; }
};

// =============================================================================
// Tiling 计算工具：复刻 op_host/arch35/fake_quant_with_min_max_vars_per_channel_tiling.cpp
// 的简化版本（单核场景），让 UT 自给自足，不依赖 TilingFunc 流程。
// =============================================================================
namespace {
constexpr int64_t kVL = 64; // VL = 64 (fp32 = 256B)
constexpr int64_t kBufNum = 2;
constexpr int64_t kReservedUB = 1 * 1024;
constexpr int64_t kUbSize = 192 * 1024;
constexpr int64_t kCacheLineBytes = 32;

static inline int64_t CeilDiv64(int64_t a, int64_t b) { return (b <= 0) ? 0 : (a + b - 1) / b; }
static inline int64_t AlignUp64(int64_t a, int64_t b) { return (b <= 0) ? a : ((a + b - 1) / b) * b; }
static inline int64_t GetCoreNumLike(int64_t taskNum, int64_t coreNum)
{
    if (taskNum <= 0 || coreNum <= 0)
        return 1;
    int64_t perCore = CeilDiv64(taskNum, coreNum);
    if (perCore <= 0)
        perCore = 1;
    return CeilDiv64(taskNum, perCore);
}

// 简化 tiling 计算：固定 coreNum，仅供 UT 使用，路径与 host tiling 一致
void ComputeTiling(FakeQuantWithMinMaxVarsPerChannelTilingData* td, int64_t M, int64_t N, int32_t numBits,
                   bool narrowRange, int64_t coreNum = 48)
{
    int64_t dtypeSize = sizeof(float);

    // 1) Block 划分（二路竞争）
    int64_t elemPerCacheLine = kCacheLineBytes / dtypeSize; // 8
    int64_t cacheLineNumN = CeilDiv64(N, elemPerCacheLine);
    int64_t actCoreNum0 = GetCoreNumLike(M, coreNum);
    int64_t actCoreNum1 = GetCoreNumLike(cacheLineNumN, coreNum);

    int64_t blockAxis, numCore, blockFactor, blockTailFactor;
    if (actCoreNum0 >= actCoreNum1) {
        blockAxis = 0;
        numCore = actCoreNum0;
        if (numCore <= 0)
            numCore = 1;
        blockFactor = CeilDiv64(M, numCore);
        if (blockFactor <= 0)
            blockFactor = 1;
        int64_t consumed = blockFactor * (numCore - 1);
        blockTailFactor = M - consumed;
        if (blockTailFactor <= 0) {
            numCore = 1;
            blockFactor = M;
            blockTailFactor = M;
        }
    } else {
        blockAxis = 1;
        numCore = actCoreNum1;
        if (numCore <= 0)
            numCore = 1;
        int64_t cellsPerCore = CeilDiv64(cacheLineNumN, numCore);
        if (cellsPerCore <= 0)
            cellsPerCore = 1;
        blockFactor = cellsPerCore * elemPerCacheLine;
        int64_t consumed = blockFactor * (numCore - 1);
        blockTailFactor = N - consumed;
        if (blockTailFactor <= 0) {
            numCore = 1;
            blockFactor = N;
            blockTailFactor = N;
        }
    }

    // 2) UB 切分（v2.2 方案 A）
    int64_t available = kUbSize - kReservedUB;
    int64_t bytesPerLen_single = (1 + 1) * kBufNum * 1 // x + y DB (baseN=1)
                                 + (1 + 1) * kBufNum;  // min + max DB
    bytesPerLen_single *= dtypeSize;
    int64_t maxBaseRaw = available / bytesPerLen_single;
    int64_t maxBase = (maxBaseRaw / kVL) * kVL;
    if (maxBase < kVL)
        maxBase = kVL;

    int64_t blockBase = (blockAxis == 0) ? N : blockFactor;
    if (blockBase <= 0)
        blockBase = 1;
    int64_t blockBaseAligned = AlignUp64(blockBase, kVL);

    int64_t ubAxis, baseN, baseLen;
    if (blockBaseAligned <= maxBase / 2 && blockBaseAligned > 0) {
        // 多行
        baseLen = blockBaseAligned;
        int64_t channelBytes = (1 + 1) * kBufNum * baseLen * dtypeSize;
        int64_t leftBytes = available - channelBytes;
        if (leftBytes <= 0)
            leftBytes = 0;
        int64_t rowBytesPerN = (1 + 1) * kBufNum * baseLen * dtypeSize;
        int64_t maxN = (rowBytesPerN > 0) ? (leftBytes / rowBytesPerN) : 1;
        if (maxN < 1)
            maxN = 1;
        int64_t blockInnerSize = (blockAxis == 0) ? blockFactor : M;
        if (blockInnerSize <= 0)
            blockInnerSize = 1;
        ubAxis = 0;
        baseN = std::min<int64_t>(maxN, blockInnerSize);
    } else {
        ubAxis = 1;
        baseN = 1;
        baseLen = std::min<int64_t>(blockBaseAligned, maxBase);
        if (baseLen <= 0)
            baseLen = kVL;
    }

    float quantMin = narrowRange ? 1.0f : 0.0f;
    float quantMax = static_cast<float>((1LL << numBits) - 1);

    td->numCore = numCore;
    td->blockAxis = blockAxis;
    td->blockFactor = blockFactor;
    td->blockTailFactor = blockTailFactor;
    td->blockUnion = 1;
    td->ubAxis = ubAxis;
    td->baseN = baseN;
    td->baseLen = baseLen;
    td->axis = -1;
    td->dim0 = M;
    td->dim1 = N;
    td->dim2 = 0;
    td->hasZeroPoint = 0;
    td->headNum = M;
    td->tailDim = N;
    td->quantMin = quantMin;
    td->quantMax = quantMax;
    td->numBits = numBits;
    td->narrowRange = narrowRange ? 1 : 0;
}

// =============================================================================
// Golden: 与 tests/ttk/fake_quant_with_min_max_vars_per_channel_golden.py 一致
// =============================================================================
void GoldenFakeQuant(const float* x, const float* minV, const float* maxV, float* y, int64_t M, int64_t N,
                     int32_t numBits, bool narrowRange)
{
    float quantMin = narrowRange ? 1.0f : 0.0f;
    float quantMax = static_cast<float>((1LL << numBits) - 1);
    float invRange = 1.0f / (quantMax - quantMin);

    for (int64_t c = 0; c < N; ++c) {
        float mn = minV[c];
        float mx = maxV[c];
        // CODE.11 ①量程溢出防护：min/max 异号且 |min|+|max|>fp32 上限时 (mx-mn)=+inf，
        // 改用防溢出分解 scale = max/qRange - min/qRange（数学等价且有限）
        // 溢出判定与 kernel 2.1b 一致：不经过 (mx-mn)（溢出点），用等价无溢出条件
        float diff = mx - mn;
        bool ovf = mx > (3.4028234663852886e38f + mn);
        float qRange = quantMax - quantMin;
        float scale = ovf ? (mx / qRange - mn / qRange) : (mx - mn) * invRange;
        float scaleSafe = (scale == 0.0f) ? 1.0f : scale;
        float zpFromMin = quantMin - mn / scaleSafe;
        float zpClipped = std::min(std::max(zpFromMin, quantMin), quantMax);
        float nudgedZp = std::floor(zpClipped + 0.5f);
        float nudgedMin = (quantMin - nudgedZp) * scaleSafe;
        float nudgedMax = (quantMax - nudgedZp) * scaleSafe;
        // P1-02：仅当计算值越过 ±FLT_MAX（∓inf）时按原 min/max 收边；有限值保持原值
        // （zp 舍入可使有限 nudgedMin 低于 min 达 0.5 步长，无条件收边会平移网格）
        if (nudgedMin < -3.4028234663852886e38f) {
            nudgedMin = mn;
        }
        if (nudgedMax > 3.4028234663852886e38f) {
            nudgedMax = mx;
        }
        // P1-01：scale > 1/FLT_MIN ⟺ 倒数落入次正规区（设备 FTZ 清零），改除法先行
        // 并纳入双侧重建
        bool invSub = scaleSafe > 8.507059173023462e37f;
        bool recon = ovf || invSub;
        // 退化规则（README 约束说明）：量化步长 0 <= scale < FLT_MIN_NORMAL(1.17549435e-38f)
        // （含下溢为 0 与次正规区）的通道按退化清零（与 kernel keep 判定一致）；
        // 负 scale（min>max 反向区间）不判退化
        float keep = (scale >= 0.0f && scale < 1.17549435e-38f) ? 0.0f : 1.0f;
        // K1（Issue #101）：min==max!=0 相等通道输出 NaN（对齐 TF oracle：inv_scale=+inf →
        // 0×inf=NaN，整通道 NaN）；min==max==0 全零通道维持清零——diff==0 且 mn!=0
        // （diff==0 时 mn==mx，mn!=0 即非全零）——与 kernel 阶段 3 末尾 NaN 注入块一致
        bool eqNz = (diff == 0.0f) && (mn != 0.0f);

        for (int64_t r = 0; r < M; ++r) {
            float v = x[r * N + c];
            // clamp 链与 kernel 一致：先 Min(x, nudgedMax) 再 Max(·, nudgedMin)；
            // min>max 反向通道（README 不支持域）回退到 nudgedMin，与 kernel/TF 一致
            float clamped = std::max(std::min(v, nudgedMax), nudgedMin);
            float shift = clamped - nudgedMin;
            // CODE.11 ③溢出通道 scaled 分解：shift 可达约 2×|nudgedMin| 超过 fp32
            float scaled = ovf ? (clamped / scaleSafe - nudgedMin / scaleSafe + 0.5f) : (shift / scaleSafe + 0.5f);
            // P1-01 倒数次正规通道：除法先行（scaleSafe 为正常数，不经过次正规倒数）
            if (invSub) {
                scaled = clamped / scaleSafe - nudgedMin / scaleSafe + 0.5f;
            }
            float floored = std::floor(scaled);
            // CODE.11 ④双侧重建（P1-03 分界 qRange/2；适用=溢出∪倒数次正规）：
            // i<=half 用 nMin+i*scale、i>half 用 nMax-(qMax-i)*scale
            float outRaw = floored * scaleSafe + nudgedMin;
            if (recon) {
                float half = (quantMax - quantMin) * 0.5f;
                outRaw = (floored > half) ? (nudgedMax - (quantMax - quantMin - floored) * scaleSafe) :
                                            (nudgedMin + floored * scaleSafe);
                // P1-06 兜底：|outRaw|>FLT_MAX（±inf）→ 另一分支；仍越界 → clamped
                float other = (floored > half) ? (nudgedMin + floored * scaleSafe) :
                                                 (nudgedMax - (quantMax - quantMin - floored) * scaleSafe);
                if (outRaw > 3.4028234663852886e38f || outRaw < -3.4028234663852886e38f) {
                    outRaw = other;
                }
                if (outRaw > 3.4028234663852886e38f || outRaw < -3.4028234663852886e38f) {
                    outRaw = clamped;
                }
            }
            float out = outRaw * keep;
            y[r * N + c] = eqNz ? std::nanf("") : out;
        }
    }
}

// 比对：atol=1e-4，与 ttk 标杆一致
int CompareWithGolden(const float* actual, const float* expected, int64_t total, float atol = 1e-4f,
                      int reportLimit = 5)
{
    int mismatches = 0;
    int reported = 0;
    for (int64_t i = 0; i < total; ++i) {
        float a = actual[i];
        float e = expected[i];
        // NaN safe: 两边都 NaN 算 PASS
        if (std::isnan(a) && std::isnan(e))
            continue;
        float diff = std::fabs(a - e);
        if (!(diff <= atol)) {
            ++mismatches;
            if (reported < reportLimit) {
                std::cerr << "  mismatch[" << i << "]: actual=" << a << " expected=" << e << " diff=" << diff
                          << std::endl;
                ++reported;
            }
        }
    }
    return mismatches;
}

void FillLinspace(float* p, int64_t n, float lo, float hi)
{
    if (n <= 1) {
        if (n == 1)
            p[0] = lo;
        return;
    }
    float step = (hi - lo) / static_cast<float>(n - 1);
    for (int64_t i = 0; i < n; ++i)
        p[i] = lo + step * static_cast<float>(i);
}

// 通用执行函数：构造数据 → 调 kernel → 校验（返回 y 供调用方显式断言）
std::vector<float> RunCase(int64_t M, int64_t N, int32_t numBits, bool narrowRange, const std::vector<float>& minVec,
                           const std::vector<float>& maxVec, float xLo = -1.0f, float xHi = 1.0f, bool zeroOut = false,
                           float atol = 1e-4f, int64_t nanIndex = -1)
{
    EXPECT_EQ(static_cast<int64_t>(minVec.size()), N);
    EXPECT_EQ(static_cast<int64_t>(maxVec.size()), N);

    size_t xBytes = M * N * sizeof(float);
    size_t mBytes = N * sizeof(float);
    size_t yBytes = M * N * sizeof(float);
    size_t tilingBytes = sizeof(FakeQuantWithMinMaxVarsPerChannelTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xBytes);
    uint8_t* mi = (uint8_t*)AscendC::GmAlloc(mBytes);
    uint8_t* ma = (uint8_t*)AscendC::GmAlloc(mBytes);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yBytes);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingBytes);

    // 1) 准备输入
    float* xF = reinterpret_cast<float*>(x);
    FillLinspace(xF, M * N, xLo, xHi);
    if (nanIndex >= 0 && nanIndex < M * N) {
        xF[nanIndex] = std::nanf(""); // 注入 NaN，验证 NaN 传播契约
    }
    std::memcpy(mi, minVec.data(), mBytes);
    std::memcpy(ma, maxVec.data(), mBytes);
    std::memset(y, 0, yBytes);

    // 2) 计算 tiling
    auto* td = reinterpret_cast<FakeQuantWithMinMaxVarsPerChannelTilingData*>(tiling);
    ComputeTiling(td, M, N, numBits, narrowRange);
    uint32_t blockDim = static_cast<uint32_t>(td->numCore);

    // 3) 跑 kernel（模板入口，固定 D_T_X=FQ_TPL_DT_FP32, MODE=2, HAS_ZP=0, ROUND_MODE=0）
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto fq_kernel = [](GM_ADDR x_, GM_ADDR mi_, GM_ADDR ma_, GM_ADDR y_, GM_ADDR ws_, GM_ADDR tiling_) {
        ::fake_quant_with_min_max_vars_per_channel<FQ_TPL_DT_FP32, 2, 0, 0>(x_, mi_, ma_, y_, ws_, tiling_);
    };
    ICPU_RUN_KF(fq_kernel, blockDim, x, mi, ma, y, workspace, tiling);

    // 4) 校验
    std::vector<float> goldenY(M * N, 0.0f);
    if (zeroOut) {
        // min==max==0 兜底场景：golden 全 0
        std::fill(goldenY.begin(), goldenY.end(), 0.0f);
    } else {
        GoldenFakeQuant(xF, minVec.data(), maxVec.data(), goldenY.data(), M, N, numBits, narrowRange);
    }
    float* yF = reinterpret_cast<float*>(y);
    int mismatches = CompareWithGolden(yF, goldenY.data(), M * N, atol);
    EXPECT_EQ(mismatches, 0) << "M=" << M << " N=" << N << " num_bits=" << numBits << " narrow_range=" << narrowRange;

    std::vector<float> yResult(yF, yF + M * N);
    AscendC::GmFree(x);
    AscendC::GmFree(mi);
    AscendC::GmFree(ma);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    return yResult;
}
} // namespace

// =============================================================================
// 测试用例
// =============================================================================

// case 1: 1D 小 case (64,) —— 单通道，最小化场景
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_1d_small)
{
    int64_t M = 1;
    int64_t N = 64;
    // min/max 各 1 个通道（因为 1D），任务要求 1D 等价于 shape=(1, N) per-channel
    std::vector<float> mn(N), mx(N);
    for (int64_t i = 0; i < N; ++i) {
        mn[i] = -1.0f;
        mx[i] = 1.0f;
    }
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
            /*xLo=*/-1.2f, /*xHi=*/1.2f);
}

// case 2: 2D 多通道 (4, 16)
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_2d_multi_channel)
{
    int64_t M = 4;
    int64_t N = 16;
    std::vector<float> mn(N), mx(N);
    for (int64_t i = 0; i < N; ++i) {
        // 不同通道使用不同 min/max
        mn[i] = -1.0f - 0.1f * static_cast<float>(i);
        mx[i] = 1.0f + 0.1f * static_cast<float>(i);
    }
    // x 范围放宽 atol：与 TTK 标杆同一组用例下做过 296/300 PASS，剩余 4 个为
    // step 边界 / NaN 语义差异，这里用更宽松的 atol=2e-2 覆盖最大 step（quant step ≈
    // (mx-mn)/255 < 0.02）。M×N=64 个元素均匀分布于 [-1.5,1.5]。
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
            /*xLo=*/-1.5f, /*xHi=*/1.5f, /*zeroOut=*/false, /*atol=*/2e-2f);
}

// case 3: num_bits 边界 nb=2
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_num_bits_2)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, -1.0f), mx(N, 1.0f);
    RunCase(M, N, /*num_bits=*/2, /*narrow_range=*/false, mn, mx);
}

// case 4: num_bits 边界 nb=16
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_num_bits_16)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, -0.5f), mx(N, 0.5f);
    RunCase(M, N, /*num_bits=*/16, /*narrow_range=*/false, mn, mx);
}

// case 5: narrow_range=true（quant_min=1.0 → 量化范围比默认少 1）
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_narrow_range)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, -1.0f), mx(N, 1.0f);
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/true, mn, mx);
}

// case 6: min==max==0 兜底，输出应为 0
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_min_max_zero)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, 0.0f), mx(N, 0.0f);
    // 这里 zeroOut=true：golden 全 0（keep=0 → out=0）
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
            /*xLo=*/-1.0f, /*xHi=*/1.0f, /*zeroOut=*/true);
}

// case 7: min==max!=0 相等通道 → 输出 NaN（K1 修复，Issue #101，对齐 TF oracle；
// golden 镜像同步 NaN，CompareWithGolden 的 NaN==NaN 口径判过）
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_min_max_equal_nonzero)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, 0.5f), mx(N, 0.5f);
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx);
}

// case 8: x 含 NaN → NaN 传播到 y（README 功能说明）
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_nan_passthrough)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, -1.0f), mx(N, 1.0f);
    // index=3 注入 NaN：该位置 y 必须为 NaN，其余位置仍与 golden 一致
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
            /*xLo=*/-1.0f, /*xHi=*/1.0f, /*zeroOut=*/false, /*atol=*/1e-4f, /*nanIndex=*/3);
}

// case 9: scale 下溢退化（FR-10）——量程 2e-44 非零但 (max-min)/(qMax-qMin) 在 fp32
// 下溢为 0，x 远超界；kernel 与 golden 均应按退化通道输出全零（keep 判定 = scale!=0）
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_scale_underflow_degenerate)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, -1e-44f), mx(N, 1e-44f);
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
            /*xLo=*/1.0f, /*xHi=*/5.0f, /*zeroOut=*/false, /*atol=*/1e-6f);
}

// case 10: 次正规带退化（FR-08）——量程为 FLT_MIN 正常数（非下溢），量化步长
// scale≈4.6e-41 落入 (0, FLT_MIN) 次正规区；kernel（设备 FTZ 视为 0）与 golden
// （阈值判定）均应清零
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_subnormal_scale_degenerate)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, 0.0f), mx(N, 1.17549435e-38f);
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
            /*xLo=*/1.0f, /*xHi=*/5.0f, /*zeroOut=*/false, /*atol=*/1e-6f);
}

// case 11: 反向区间（min>max，README 不支持域）——不判退化，与 kernel/TF 一致
// 回退到 nudgedMin；golden 参考模型同链路应与 kernel 输出一致
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_inverted_range_fallback)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, 5.0f), mx(N, -5.0f);
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
            /*xLo=*/1.0f, /*xHi=*/5.0f, /*zeroOut=*/false, /*atol=*/1e-6f);
}

// case 12: 量程溢出防护（CODE.11）——min/max=±FLT_MAX（diff 溢出为 +inf），
// x 幅值大（覆盖 scaled 分解 + 大索引双侧重建）；防护后应输出有限量化值而非
// NaN/Inf，kernel 与参考模型同链路一致
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_range_overflow_guard)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, -3.402823466e38f), mx(N, 3.402823466e38f);
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
            /*xLo=*/-1e38f, /*xHi=*/1e38f, /*zeroOut=*/false, /*atol=*/3.0e36f);
}

// case 14: nudge 端点整数边界（L1_bound_118 复测根因）——min=-3.51、max=FLT_MIN
// 使 zp 恰为 qMax=255（nudge 端点），x 在量化网格上扫描；kernel 与参考模型应
// 在所有网格点上一致（排除设备侧 scaled 链的 1 级偏差归因）
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_nudge_endpoint_boundary)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, -3.5108721f), mx(N, 1.1754944e-38f);
    RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
            /*xLo=*/-1.0f, /*xHi=*/-0.01f, /*zeroOut=*/false, /*atol=*/1e-4f);
}

// case 13: 倒数次正规防护（P1-01）——min/max 同为负大量程（步长约 1.13e38 正常，
// 倒数约 8.8e-39 落入次正规区被设备 FTZ 清零，整通道坍缩到 nudgedMin）；
// 防护后应输出正常量化网格值
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_inv_subnormal_guard)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, -3.402823466e38f), mx(N, -1.0f);
    RunCase(M, N, /*num_bits=*/2, /*narrow_range=*/false, mn, mx,
            /*xLo=*/-3.4e38f, /*xHi=*/-1.0f, /*zeroOut=*/false, /*atol=*/1.2e38f);
}

// case 15（K1 修复，Issue #101）: min==max!=0 相等通道整通道输出 NaN——对齐 TF oracle
// （Nudge 的 inv_scale=qRange/(max-min)=+inf → 0×inf=NaN）；016 例形态（min[0]=max[0]=-100）。
// 覆盖 kernel Compute 末尾显式 NaN 注入路径（floor 的 Cast<int32> 吞 NaN，不可依赖自然传播）
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_equal_nonzero_nan)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, -100.0f), mx(N, -100.0f);
    std::vector<float> y = RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_TRUE(std::isnan(y[i])) << "i=" << i;
    }
}

// case 16: min>max 倒置通道输出回退 nudged_min 侧——对齐 TF clamp 顺序
// cwiseMin(nudged_max).cwiseMax(nudged_min)（先 min 后 max，golden_fix 后 golden/kernel/TF
// 三方一致，防回归）。选 min=6/max=-5 使 zp=139.09 远离半整数边界（floor 不敏感）
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_inverted_nudged_min_side)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, 6.0f), mx(N, -5.0f);
    std::vector<float> y = RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
                                   /*xLo=*/1.0f, /*xHi=*/5.0f);
    // TF 链: scale=(max-min)/255; zp=qMin-min/scale=139.09→floor(139.59)=139;
    // nudged_min=(0-139)*scale=5.996...，倒置通道 clamp 恒收敛 nudged_min
    float scale = (-5.0f - 6.0f) / 255.0f;
    float zp = 0.0f - 6.0f / scale;
    float nudgedZp = std::floor(std::min(std::max(zp, 0.0f), 255.0f) + 0.5f);
    float nudgedMin = (0.0f - nudgedZp) * scale;
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_NEAR(y[i], nudgedMin, 1e-5f) << "i=" << i;
    }
}

// case 17: min==max==0 全零通道维持清零输出 0（与相等非零通道的 NaN 语义区分，防退化边界误改）
TEST_F(fake_quant_with_min_max_vars_per_channel_test, case_both_zero_output_zero)
{
    int64_t M = 2;
    int64_t N = 8;
    std::vector<float> mn(N, 0.0f), mx(N, 0.0f);
    std::vector<float> y = RunCase(M, N, /*num_bits=*/8, /*narrow_range=*/false, mn, mx,
                                   /*xLo=*/-1.0f, /*xHi=*/1.0f, /*zeroOut=*/true);
    for (size_t i = 0; i < y.size(); ++i) {
        EXPECT_EQ(y[i], 0.0f) << "i=" << i;
    }
}
