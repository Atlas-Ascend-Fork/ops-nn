#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

import numpy as np
import torch


def _resolve_ranges(indices, k):
    """按 kernel 语义解析每行的行范围，返回 (start, end)；空范围以 start==end 表示。

    1. start >= end 或 start == -1 或 end == -1 -> 空范围
    2. 负索引 -> max(K + value, 0)
    3. 裁剪到 [0, K]
    4. 裁剪后 start >= end -> 空范围
    """
    st = indices[:, 0].astype(np.int64)
    en = indices[:, 1].astype(np.int64)
    empty = (st >= en) | (st == -1) | (en == -1)
    st = np.where(st < 0, np.maximum(st + k, 0), st)
    en = np.where(en < 0, np.maximum(en + k, 0), en)
    st = np.minimum(st, k)
    en = np.minimum(en, k)
    empty = empty | (st >= en)
    st = np.where(empty, 0, st).astype(np.int64)
    en = np.where(empty, 0, en).astype(np.int64)
    return st, en


def _reference_float32(x, src, indices):
    """与 SIMT kernel 逐位一致的参考实现。

    - 输入 float16/float32 统一提升到 float32 累加，累加完成后再转回原 dtype；
    - 每个输出元素按 start -> end 顺序做 Kahan 补偿求和；
    - 当中间和或累加和变为非有限值（±Inf/NaN）时，补偿项 (t - acc) - y 会出现
      Inf - Inf = NaN 并污染后续项，因此非有限时置零补偿、退化为普通 IEEE 顺序加法，
      保持 Inf/NaN 传播（Inf + Inf = Inf，Inf + (-Inf) = NaN）。

    实现按 j 逐步向量化所有行，避免逐元素 Python 循环。
    """
    x = np.asarray(x)
    src = np.asarray(src)
    indices = np.asarray(indices)
    m, n = x.shape[0], x.shape[1]
    k = src.shape[0]

    src_f = np.ascontiguousarray(src, dtype=np.float32)
    st, en = _resolve_ranges(indices, k)

    acc = np.zeros((m, n), dtype=np.float32)
    comp = np.zeros((m, n), dtype=np.float32)
    max_end = int(en.max()) if m else 0
    zero = np.float32(0.0)
    # Inf/NaN 中间量会产生 Inf-Inf 的 invalid/overflow 警告，属预期路径
    with np.errstate(invalid="ignore", over="ignore"):
        for j in range(max_end):
            active = (st <= j) & (j < en)
            if not active.any():
                continue
            val = src_f[j]
            a = acc[active]
            c = comp[active]
            y = val - c
            t = a + y
            finite = np.isfinite(t) & np.isfinite(a)
            comp[active] = np.where(finite, (t - a) - y, zero)
            acc[active] = t

    work = x.astype(np.float32) + acc
    return work.astype(x.dtype)


class AddRowRangesComposeSpec:
    """Class form — small-op composition"""

    # -- golden — function form --
    def golden(x, src, indices, **kwargs):
        """
        Golden function for add_row_ranges.
        All the parameters (names and order) follow SE doc prototype definition without outputs.
        All the input Tensors are numpy.ndarray.

        Algorithm:
            For each row r and column c of the output matrix x_out:
                x_out(r, c) = x(r, c) + sum(src(j, c)), j in [indices[r, 0], indices[r, 1])

        累加语义与 kernel 一致：float16/float32 输入统一使用 float32 中间量、按 start -> end
        顺序做 Kahan 补偿求和，非有限值按 IEEE 传播。

        Indices clipping:
            1. start >= end || start == -1 || end == -1 -> skip
            2. negative index: start<0 -> max(K+start, 0); end<0 -> max(K+end, 0)
            3. clip to [0, K]: start>K -> K; end>K -> K
            4. post-clip empty range: start >= end -> skip

        Args:
            x: numpy.ndarray, shape=(M, N), dtype=float16/float32
            src: numpy.ndarray, shape=(K, N), dtype=float16/float32 (same as x)
            indices: numpy.ndarray, shape=(M, 2), dtype=int32
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            Output tensor x_out, shape=(M, N), dtype=same as x
        """
        return _reference_float32(x, src, indices)

    # -- third_party — dict multi-vendor --
    class ThirdPartyImpl:
        def __init__(self, **kwargs):
            pass

        def __call__(self, x, src, indices, **kwargs):
            # add_row_ranges has no direct torch competitor; reuse the kernel-consistent
            # float32 + Kahan reference so cross_check shares the operator's summation
            # semantics (sequential start -> end, IEEE Inf/NaN propagation).
            xn = x.detach().cpu().numpy()
            sn = src.detach().cpu().numpy()
            indn = indices.detach().cpu().numpy()
            out = _reference_float32(xn, sn, indn)
            return torch.from_numpy(out)

    third_party = {
        "torch": ThirdPartyImpl,
    }

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "add_row_ranges": "AddRowRangesComposeSpec",
}
