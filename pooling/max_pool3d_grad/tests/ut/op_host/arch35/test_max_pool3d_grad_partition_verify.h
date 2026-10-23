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
 * \file test_max_pool3d_grad_partition_verify.h
 * \brief 多核调整后 tiling 划分的语义级验证(纯 host, 不执行 kernel):
 *   1. 轴划分自洽: outer/tail 与 inner 匹配;
 *   2. 覆盖恰好一次: 对每个 grad 位置 p 与其窗口内每个输出位置 a, 恰好存在一个 tile
 *      同时覆盖 p(在 tile 的 argmax 范围内) 与 a(在 tile 的输出范围内) —— 保证贡献不丢不重,
 *      与 kernel 侧 IS_CHECK_RANGE=1 过滤语义共同构成正确性;
 *   3. 无过滤窗口包含(isCheckRange=0 路径): tile 覆盖的每个 p 的窗口完整落在 tile 内 ——
 *      保证无越界过滤的 scatter 不产生 UB 越界写。
 *   PStart/PEnd/窗口公式与 kernel 实现保持一致(max_pool3d_grad_ncdhw_kernel.h /
 *   max_pool3d_grad_ndhwc_impl_scatter.h), dilation 固定为 1(算子约束)。
 */

#ifndef TEST_MAX_POOL3D_GRAD_PARTITION_VERIFY_H_
#define TEST_MAX_POOL3D_GRAD_PARTITION_VERIFY_H_

#include <algorithm>
#include <map>
#include <utility>

namespace mp3dg_partition_verify {

inline int64_t KernelPStart(int64_t index, int64_t pad, int64_t kernel, int64_t stride)
{
    if (stride == 0) {
        return 0;
    }
    return (index + pad < kernel) ? 0 : (index + pad - kernel) / stride + 1;
}

inline int64_t KernelPEnd(int64_t index, int64_t pad, int64_t stride, int64_t pooledSize)
{
    if (stride == 0) {
        return 0;
    }
    int64_t end = (index + pad) / stride + 1;
    return end < pooledSize ? end : pooledSize;
}

inline void VerifyAxisPartition(int64_t xSize, int64_t gradSize, int64_t kernel, int64_t stride, int64_t pad,
                                int64_t inner, int64_t tail, int64_t outer, bool checkWindowInTile)
{
    EXPECT_GE(inner, 1);
    EXPECT_EQ(outer, (xSize + inner - 1) / inner);
    EXPECT_EQ(tail, xSize - (outer - 1) * inner);

    // (p, a) -> 覆盖该组合的 tile 数
    std::map<std::pair<int64_t, int64_t>, int64_t> coverCount;
    for (int64_t p = 0; p < gradSize; p++) {
        int64_t winBegin = std::max(int64_t(0), p * stride - pad);
        int64_t winEnd = std::min(xSize, p * stride - pad + kernel);
        EXPECT_LT(winBegin, winEnd); // 窗口非空(与 grads shape 校验一致)
        for (int64_t t = 0; t < outer; t++) {
            int64_t tileStart = t * inner;
            int64_t tileEnd = tileStart + ((t == outer - 1) ? tail : inner);
            int64_t rangeBegin = KernelPStart(tileStart, pad, kernel, stride);
            int64_t rangeEnd = KernelPEnd(tileEnd - 1, pad, stride, gradSize);
            if (p < rangeBegin || p >= rangeEnd) {
                continue;
            }
            if (checkWindowInTile) {
                // isCheckRange=0: 无越界过滤, 窗口必须完整落在 tile 内
                EXPECT_GE(winBegin, tileStart);
                EXPECT_LE(winEnd, tileEnd);
            }
            for (int64_t a = winBegin; a < winEnd; a++) {
                if (a >= tileStart && a < tileEnd) {
                    coverCount[{p, a}]++;
                }
            }
        }
    }
    // 每个 (p, 窗口内 a) 恰好被一个 tile 覆盖
    int64_t expectPairs = 0;
    for (int64_t p = 0; p < gradSize; p++) {
        int64_t winBegin = std::max(int64_t(0), p * stride - pad);
        int64_t winEnd = std::min(xSize, p * stride - pad + kernel);
        expectPairs += winEnd - winBegin;
    }
    EXPECT_EQ(static_cast<int64_t>(coverCount.size()), expectPairs);
    for (auto& kv : coverCount) {
        EXPECT_EQ(kv.second, 1) << "pair(p=" << kv.first.first << ", a=" << kv.first.second << ")";
        if (kv.second != 1) {
            return;
        }
    }
}

} // namespace mp3dg_partition_verify

#endif // TEST_MAX_POOL3D_GRAD_PARTITION_VERIFY_H_
