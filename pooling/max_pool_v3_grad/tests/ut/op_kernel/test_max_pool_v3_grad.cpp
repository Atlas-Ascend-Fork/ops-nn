/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_max_pool_v3_grad.cpp
 * \brief MaxPoolV3Grad NHWC 快速通道 kernel 精度 UT (CPU 仿真)
 *
 * 覆盖 NDHWC(去D维)三分支:
 *   1. 大C        : ChunkCGather 标量通道 (cDim*sizeof(T) > VREG/2)
 *   2. 小C小kernel : 向量化 gather (ForwardComputeTile 多级打包)
 *   3. 小C大kernel : 窗口流式加载 + merge (与 2 共用 BackwardBase 反向)
 * 另覆盖 isCheckRange=1 跨 tile 窗口重叠的 scatter 过滤。
 * tiling data 按算子 tiling 口径手工构造, golden 在 CPU 上以 first-wins 语义计算。
 *
 * 注: 当前 tikicpulib CPU 仿真器无法从测试 TU 执行 Reg:: 向量通路(最小探针亦失败,
 *     与 max_pool3d_grad 未提供 kernel UT 同因), 本用例暂不在 CI 启用, 待仿真器修复后
 *     将 CMakeLists 中的 AddOpTestCase 打开即可.
 */

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>

#ifdef __CCE_KT_TEST__
#include "../../../op_kernel/arch35/max_pool_v3_grad.cpp"
#include "tikicpulib.h"
#include "data_utils.h"
#endif
#include "gtest/gtest.h"

using namespace std;

namespace {
constexpr int64_t UB_BLOCK_SIZE = 32;
constexpr uint64_t KEY_NHWC_NO_CHECK = 275;
constexpr uint64_t KEY_NHWC_CHECK = 787;

int64_t CeilAlignI(int64_t v, int64_t a) { return AscendC::CeilDivision(v, a) * a; }

struct CaseDesc {
    const char* name;
    int64_t n;
    int64_t c;
    int64_t hIn;
    int64_t wIn;
    int64_t hOut;
    int64_t wOut;
    int64_t kh;
    int64_t kw;
    int64_t sh;
    int64_t sw;
    int64_t padH;
    int64_t padHBack;
    int64_t padW;
    int64_t padWBack;
    int64_t hInner; // H 轴 tile 切分(inner), outer/tail 推导
    int64_t wInner;
    int64_t cInner;
    int64_t isBigKernel;
    int64_t inputBufferSizeOverride; // >0: 大 kernel 流式加载缓冲(测试指定)
    int64_t checkRange;
};

// 按算子 tiling 的 DoBufferCalculate 口径构造 tiling data (N 轴整轴单 tile)
MaxPoolV3GradNHWCNameSpace::MaxPoolV3GradNHWCTilingData BuildTiling(const CaseDesc& d, int64_t elemSize)
{
    MaxPoolV3GradNHWCNameSpace::MaxPoolV3GradNHWCTilingData td{};

    td.base.hArgmax = d.hOut;
    td.base.wArgmax = d.wOut;
    td.base.hOutput = d.hIn;
    td.base.wOutput = d.wIn;
    td.base.hKernel = d.kh;
    td.base.wKernel = d.kw;
    td.base.hStride = d.sh;
    td.base.wStride = d.sw;
    td.base.padH = d.padH;
    td.base.padHBack = d.padHBack;
    td.base.padW = d.padW;
    td.base.padWBack = d.padWBack;

    td.base.highAxisInner = d.n;
    td.base.highAxisTail = d.n;
    td.base.highAxisOuter = 1;

    int64_t hOuter = AscendC::CeilDivision(d.hIn, d.hInner);
    td.base.hOutputInner = d.hInner;
    td.base.hOutputTail = (d.hIn % d.hInner == 0) ? d.hInner : d.hIn % d.hInner;
    td.base.hOutputOuter = hOuter;

    int64_t wOuter = AscendC::CeilDivision(d.wIn, d.wInner);
    td.base.wOutputInner = d.wInner;
    td.base.wOutputTail = (d.wIn % d.wInner == 0) ? d.wInner : d.wIn % d.wInner;
    td.base.wOutputOuter = wOuter;

    int64_t cOuter = AscendC::CeilDivision(d.c, d.cInner);
    td.cDim = d.c;
    td.cOutputInner = d.cInner;
    td.cOutputTail = (d.c % d.cInner == 0) ? d.cInner : d.c % d.cInner;
    td.cOutputOuter = cOuter;

    // buffer 口径: 与 MaxPoolV3GradNHWCTilingImpl::DoBufferCalculate 一致
    int64_t hInTile = AscendC::CeilDivision(d.hInner + d.kh - 1, d.sh);
    int64_t wInTile = AscendC::CeilDivision(d.wInner + d.kw - 1, d.sw);
    int64_t cAligned = CeilAlignI(d.cInner, UB_BLOCK_SIZE / elemSize);
    td.base.gradBufferSize = d.n * hInTile * wInTile * cAligned * elemSize;
    td.base.outputBufferSize = d.n * d.hInner * d.wInner * cAligned * 4;
    td.base.argmaxBufferSize = d.n * d.cInner * hInTile * wInTile * 4;
    if (d.isBigKernel == 1 && d.inputBufferSizeOverride > 0) {
        td.base.inputBufferSize = d.inputBufferSizeOverride;
    } else {
        int64_t hExp = d.hInner + (d.kh - 1) * 2;
        int64_t wExp = d.wInner + (d.kw - 1) * 2;
        td.base.inputBufferSize = d.n * hExp * wExp * cAligned * elemSize;
    }

    td.base.hProBatchSize = (d.kh > d.sh) ? AscendC::CeilDivision(d.kh, d.sh) : 1;
    td.base.wProBatchSize = (d.kw > d.sw) ? AscendC::CeilDivision(d.kw, d.sw) : 1;
    td.base.isBigKernel = d.isBigKernel;

    // 每核 1 个 base block
    int64_t totalBlocks = 1 * cOuter * hOuter * wOuter;
    td.base.normalCoreProcessNum = 1;
    td.base.usedCoreNum = totalBlocks;
    td.base.tailCoreProcessNum = 1;
    td.isBigKernel = d.isBigKernel;
    return td;
}

// golden: first-wins argmax + 梯度散射(pad 位置不接收梯度), 累加用 int64 保证精确
template <typename T>
float ToFloatVal(T v)
{
    return static_cast<float>(v);
}

template <typename T>
T FromFloatVal(float v)
{
    return static_cast<T>(v);
}

template <typename T>
void RunGolden(const CaseDesc& d, const vector<T>& x, const vector<T>& grad, vector<int64_t>& dyGolden)
{
    for (int64_t nn = 0; nn < d.n; nn++) {
        for (int64_t c = 0; c < d.c; c++) {
            for (int64_t ho = 0; ho < d.hOut; ho++) {
                for (int64_t wo = 0; wo < d.wOut; wo++) {
                    int64_t bestH = -1;
                    int64_t bestW = -1;
                    float best = 0.0f;
                    for (int64_t kh = 0; kh < d.kh; kh++) {
                        for (int64_t kw = 0; kw < d.kw; kw++) {
                            int64_t h = ho * d.sh - d.padH + kh;
                            int64_t w = wo * d.sw - d.padW + kw;
                            if (h < 0 || h >= d.hIn || w < 0 || w >= d.wIn) {
                                continue;
                            }
                            float v = ToFloatVal(x[((nn * d.c + c) * d.hIn + h) * d.wIn + w]);
                            if (bestH < 0 || v > best) {
                                best = v;
                                bestH = h;
                                bestW = w;
                            }
                        }
                    }
                    if (bestH < 0) {
                        continue;
                    }
                    int64_t dyIdx = ((nn * d.c + c) * d.hIn + bestH) * d.wIn + bestW;
                    dyGolden[dyIdx] += static_cast<int64_t>(
                        ToFloatVal(grad[(nn * d.c + c) * d.hOut * d.wOut + ho * d.wOut + wo]));
                }
            }
        }
    }
}

uint32_t g_nhwcSeed = 20261028u;

template <typename T>
void FillRandomInt(vector<T>& v, int64_t modulo)
{
    for (auto& e : v) {
        e = FromFloatVal<T>(static_cast<float>(static_cast<int64_t>(rand_r(&g_nhwcSeed) % modulo)));
    }
}

template <typename T>
void CompareResult(const vector<T>& dy, const vector<int64_t>& golden, const string& name)
{
    ASSERT_EQ(dy.size(), golden.size());
    for (size_t i = 0; i < dy.size(); i++) {
        int64_t actual = static_cast<int64_t>(ToFloatVal(dy[i]));
        ASSERT_EQ(actual, golden[i]) << name << " mismatch at index " << i;
    }
}

#ifdef __CCE_KT_TEST__
// GM内存RAII守卫: 申请即登记, 析构统一释放, 保证断言失败提前返回时不泄漏(RED-8)
class GmMemGuard {
public:
    uint8_t* Alloc(size_t size)
    {
        uint8_t* p = (uint8_t*)AscendC::GmAlloc(size);
        ptrs_.push_back(p);
        return p;
    }

    ~GmMemGuard()
    {
        for (auto* p : ptrs_) {
            if (p != nullptr) {
                AscendC::GmFree(p);
            }
        }
    }

private:
    vector<uint8_t*> ptrs_;
};

// 入口模板路径: 走完整 dispatch(含 tilingData.isBigKernel 运行时分流), fp32
void RunEntryCase(const CaseDesc& d)
{
    const int64_t elemSize = 4;
    auto td = BuildTiling(d, elemSize);
    size_t xLen = static_cast<size_t>(d.n * d.c * d.hIn * d.wIn);
    size_t gradLen = static_cast<size_t>(d.n * d.c * d.hOut * d.wOut);

    vector<float> x(xLen);
    vector<float> grad(gradLen);
    vector<int64_t> golden(xLen, 0);
    FillRandomInt(x, 97);
    FillRandomInt(grad, 9);
    RunGolden(d, x, grad, golden);

    GmMemGuard gmGuard;
    uint8_t* xGm = gmGuard.Alloc(xLen * sizeof(float));
    uint8_t* yGm = gmGuard.Alloc(xLen * sizeof(float));
    uint8_t* gradGm = gmGuard.Alloc(gradLen * sizeof(float));
    uint8_t* dyGm = gmGuard.Alloc(xLen * sizeof(float));
    uint8_t* ws = gmGuard.Alloc(1024);
    uint8_t* tiling = gmGuard.Alloc(sizeof(td));
    ASSERT_NE(xGm, nullptr);
    ASSERT_NE(yGm, nullptr);
    ASSERT_NE(gradGm, nullptr);
    ASSERT_NE(dyGm, nullptr);
    ASSERT_NE(ws, nullptr);
    ASSERT_NE(tiling, nullptr);
    auto ret = memcpy_s(xGm, xLen * sizeof(float), x.data(), xLen * sizeof(float));
    ASSERT_EQ(ret, EOK);
    ret = memcpy_s(gradGm, gradLen * sizeof(float), grad.data(), gradLen * sizeof(float));
    ASSERT_EQ(ret, EOK);
    ret = memset_s(dyGm, xLen * sizeof(float), 0, xLen * sizeof(float));
    ASSERT_EQ(ret, EOK);
    ret = memcpy_s(tiling, sizeof(td), &td, sizeof(td));
    ASSERT_EQ(ret, EOK);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(d.checkRange == 1 ? KEY_NHWC_CHECK : KEY_NHWC_NO_CHECK);
    if (d.checkRange == 1) {
        auto fn = static_cast<void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)>(
            max_pool_v3_grad<TPL_NHWC_KERNEL, TPL_NHWC_FORMAT, TPL_INT32, TPL_CHECK_RANGE>);
        ICPU_RUN_KF(fn, static_cast<uint32_t>(td.base.usedCoreNum), xGm, yGm, gradGm, dyGm, ws, tiling);
    } else {
        auto fn = static_cast<void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)>(
            max_pool_v3_grad<TPL_NHWC_KERNEL, TPL_NHWC_FORMAT, TPL_INT32, TPL_NO_CHECK_RANGE>);
        ICPU_RUN_KF(fn, static_cast<uint32_t>(td.base.usedCoreNum), xGm, yGm, gradGm, dyGm, ws, tiling);
    }

    vector<float> dy(xLen);
    ret = memcpy_s(dy.data(), dy.size() * sizeof(float), dyGm, dy.size() * sizeof(float));
    ASSERT_EQ(ret, EOK);
    CompareResult(dy, golden, d.name);
}

// 类直调路径: fp16(u16 gather 打包 + fp32 累加 + RINT 回转), 小C小kernel
void RunFp16SmallKernelCase(const CaseDesc& d)
{
    const int64_t elemSize = 2;
    auto td = BuildTiling(d, elemSize);
    size_t xLen = static_cast<size_t>(d.n * d.c * d.hIn * d.wIn);
    size_t gradLen = static_cast<size_t>(d.n * d.c * d.hOut * d.wOut);

    vector<half> x(xLen);
    vector<half> grad(gradLen);
    vector<int64_t> golden(xLen, 0);
    FillRandomInt(x, 97);
    FillRandomInt(grad, 9);
    RunGolden(d, x, grad, golden);

    GmMemGuard gmGuard;
    uint8_t* xGm = gmGuard.Alloc(xLen * sizeof(half));
    uint8_t* yGm = gmGuard.Alloc(xLen * sizeof(half));
    uint8_t* gradGm = gmGuard.Alloc(gradLen * sizeof(half));
    uint8_t* dyGm = gmGuard.Alloc(xLen * sizeof(half));
    uint8_t* ws = gmGuard.Alloc(1024);
    uint8_t* tiling = gmGuard.Alloc(sizeof(td));
    ASSERT_NE(xGm, nullptr);
    ASSERT_NE(yGm, nullptr);
    ASSERT_NE(gradGm, nullptr);
    ASSERT_NE(dyGm, nullptr);
    ASSERT_NE(ws, nullptr);
    ASSERT_NE(tiling, nullptr);
    auto ret = memcpy_s(xGm, xLen * sizeof(half), x.data(), xLen * sizeof(half));
    ASSERT_EQ(ret, EOK);
    ret = memcpy_s(gradGm, gradLen * sizeof(half), grad.data(), gradLen * sizeof(half));
    ASSERT_EQ(ret, EOK);
    ret = memset_s(dyGm, xLen * sizeof(half), 0, xLen * sizeof(half));
    ASSERT_EQ(ret, EOK);
    ret = memcpy_s(tiling, sizeof(td), &td, sizeof(td));
    ASSERT_EQ(ret, EOK);

    auto fn = [](GM_ADDR origX, GM_ADDR origY, GM_ADDR gradAddr, GM_ADDR yAddr, GM_ADDR, GM_ADDR tilingAddr) {
        TPipe pipe;
        MaxPoolV3GradNHWCNameSpace::MaxPoolV3GradNHWCSmallKernel<half, int32_t, true> op;
        op.Init(origX, origY, gradAddr, yAddr,
                *reinterpret_cast<const MaxPoolV3GradNHWCNameSpace::MaxPoolV3GradNHWCTilingData*>(tilingAddr), pipe);
        op.Process();
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(KEY_NHWC_CHECK);
    ICPU_RUN_KF(fn, static_cast<uint32_t>(td.base.usedCoreNum), xGm, yGm, gradGm, dyGm, ws, tiling);

    vector<half> dy(xLen);
    ret = memcpy_s(dy.data(), dy.size() * sizeof(half), dyGm, dy.size() * sizeof(half));
    ASSERT_EQ(ret, EOK);
    CompareResult(dy, golden, d.name);
}
#endif
} // namespace

#ifdef __CCE_KT_TEST__
class MaxPoolV3GradNhwcKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "MaxPoolV3GradNhwcKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "MaxPoolV3GradNhwcKernelTest TearDown" << endl; }
};

TEST_F(MaxPoolV3GradNhwcKernelTest, nhwc_small_kernel_n_split_fp32)
{
    CaseDesc d{"small_kernel_n_split_fp32", 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 4, 4, 2, 0, 0, 0};
    RunEntryCase(d);
}

// 分支2: 小C小kernel + CALCULATED 对称 pad + H 轴多 tile 重叠窗口(isCheckRange=1 跨 tile 过滤)
TEST_F(MaxPoolV3GradNhwcKernelTest, nhwc_small_kernel_pad_overlap_fp32)
{
    CaseDesc d{"small_kernel_pad_overlap_fp32", 1, 2, 5, 5, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 4, 5, 2, 0, 0, 1};
    RunEntryCase(d);
}

// 分支1: 大C(C=48, 48*4 > VREG/2) 标量通道
TEST_F(MaxPoolV3GradNhwcKernelTest, nhwc_big_c_fp32)
{
    CaseDesc d{"big_c_fp32", 1, 48, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 48, 0, 0, 0};
    RunEntryCase(d);
}

// 分支3: 小C大kernel(kv=256), 整窗一次载入(NoSplit)
TEST_F(MaxPoolV3GradNhwcKernelTest, nhwc_big_kernel_nosplit_fp32)
{
    CaseDesc d{"big_kernel_nosplit_fp32", 1, 2, 32, 32, 2, 2, 16, 16, 16, 16, 0, 0, 0, 0, 32, 32, 2, 1, 8192, 0};
    RunEntryCase(d);
}

// 分支3: 小C大kernel, 流式分批载入 + merge(修复 3D 版 Init 阶段 cAligned_ 未初始化的口径)
TEST_F(MaxPoolV3GradNhwcKernelTest, nhwc_big_kernel_split_merge_fp32)
{
    CaseDesc d{"big_kernel_split_merge_fp32", 1, 2, 32, 32, 2, 2, 16, 16, 16, 16, 0, 0, 0, 0, 32, 32, 2, 1, 4096, 0};
    RunEntryCase(d);
}

// 分支2: 小C小kernel + SAME pad + W 轴多 tile(isCheckRange=1)
TEST_F(MaxPoolV3GradNhwcKernelTest, nhwc_small_kernel_w_split_fp32)
{
    CaseDesc d{"small_kernel_w_split_fp32", 1, 3, 6, 6, 3, 3, 3, 3, 2, 2, 0, 1, 0, 1, 6, 4, 3, 0, 0, 1};
    RunEntryCase(d);
}

// fp16: u16 gather 打包 + fp32 累加 + RINT 回转, 小C小kernel + pad 重叠
TEST_F(MaxPoolV3GradNhwcKernelTest, nhwc_small_kernel_pad_overlap_fp16)
{
    CaseDesc d{"small_kernel_pad_overlap_fp16", 1, 2, 5, 5, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 4, 5, 2, 0, 0, 1};
    RunFp16SmallKernelCase(d);
}

// fp16: 大C 标量通道
TEST_F(MaxPoolV3GradNhwcKernelTest, nhwc_big_c_fp16)
{
    CaseDesc d{"big_c_fp16", 1, 48, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 48, 0, 0, 0};
    RunFp16SmallKernelCase(d);
}
#endif
