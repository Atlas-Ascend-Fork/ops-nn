#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""NormalizeBatch TTK TestSpec（golden / tolerance）。

数学语义（对齐 canndev TIK 实现 ops/built-in/tbe/impl/normalize_batch.py）：
    L  = seq_len[n]                                统计长度（kernel 侧钳制到 [1, d]）
    N  = L        (per_feature)
       = c * L    (all_features)
    E(x) = sum(x[0:L]) / N                         统计只用前 L 个元素
    var  = |E(x^2) - E(x)^2| * N/(N-1)             无偏修正 + abs 防负（N < 2 时 var = 0）
    y    = (x - E(x)) / (sqrt(var) + eps)          eps 加在 sqrt 之后（不是 var 之下）

四个易错语义细节：
    1. 归一化范围 ≠ 统计范围：只有 d 维前 L 个元素参与统计，但整行 d 个元素都被归一化；
    2. per_feature 每 (n,c) 行独立统计；all_features 同一 n 的所有 c 共享统计量（N = c*L）；
    3. eps 位置在 sqrt 之后：y = (x-mean)/(std+eps)；
    4. 无偏修正 N/(N-1)，N < 2 时 var 取 0（TIK 原实现 N=1 会 NaN，950 实现按 var=0 防御）。

golden：独立 CPU 参考实现（纯 numpy 真值核），不 import / 复用被测 kernel 任何产物；
计算全程 float64，不提前舍入到输出 dtype；TTK Promote 传入的 float64 不降精度。

注册名（与 TTK 消费方式对齐，范式同 norm/l2_normalize golden.py）：
    normalize_batch / NormalizeBatch -> Kernel / GEIR（CSV op_name；golden 收
    numpy.ndarray，normalize_type/epsilon 按算子属性名关键字传入）
本算子未交付 ACLNN 接口，不注册 ACLNN TestSpec。
"""

__spec__ = {
    "normalize_batch": "NormalizeBatchTestSpec",
    "NormalizeBatch": "NormalizeBatchTestSpec",
}

import numpy as np


class NormalizeBatchTestSpec:
    """NormalizeBatch TestSpec：golden 真值核 + tolerance 声明。"""

    @staticmethod
    def golden(input_x, seq_len, normalize_type="per_feature", epsilon=1e-5, **kwargs):
        """TTK golden 入口：输入/属性按算子原型名关键字传入，返回 float64 真值。"""
        return normalize_batch_golden(input_x, seq_len, normalize_type, epsilon)

    # 逐元素归一化输出（非归约标量）→ mix_tolerance：|a−g| ≤ atol + rtol·|g|。
    # 不用 stat_rel_err：其默认 threshold=2⁻¹³（mare 上限 ~1.2e-3），而 fp32 kernel 在
    # |golden|→0 的近均值元素上固有 ~1 ULP(x)/|x−mean| 量级相对噪声（0.3%~2%），
    # 任何 fp32 实现都无法通过；atol=1e-6 天然覆盖该噪声。
    tolerance = {
        "float32": {"standard": "mix_tolerance", "rtol": 1e-3, "atol": 1e-6},
    }


def normalize_batch_golden(x, seq_len, normalize_type="per_feature", epsilon=1e-5):
    """NormalizeBatch CPU 参考实现。

    Args:
        x: np.ndarray, shape (n, c, d), float32/float64
        seq_len: array-like, shape (n,), int
        normalize_type: "per_feature" | "all_features"
        epsilon: float > 0

    Returns:
        np.ndarray, shape (n, c, d), float64 真值
    """
    x = np.asarray(x, dtype=np.float64)
    n, c, d = x.shape
    seq = np.asarray(seq_len, dtype=np.int64)
    assert seq.shape == (n,), f"seq_len shape {seq.shape} != ({n},)"
    assert normalize_type in ("per_feature", "all_features")
    assert epsilon > 0

    y = np.empty_like(x)
    for i in range(n):
        # kernel 侧运行时钳制口径：L ∈ [1, d]
        length = int(seq[i])
        length = 1 if length < 1 else (d if length > d else length)
        if normalize_type == "per_feature":
            for j in range(c):
                seg = x[i, j, :length]
                cnt = length
                y[i, j, :] = _normalize_row(x[i, j, :], seg, cnt, epsilon)
        else:
            seg = x[i, :, :length]
            cnt = c * length
            mean, rstd = _mean_rstd(seg, cnt, epsilon)
            y[i, :, :] = (x[i, :, :] - mean) * rstd
    return y


def _mean_rstd(seg, cnt, epsilon):
    """由统计段算 mean 与 1/(std+eps)；N<2 时 var=0。"""
    mean = seg.sum() / cnt
    var = 0.0
    if cnt >= 2:
        var = abs((seg * seg).sum() / cnt - mean * mean)
        var = var * (cnt / (cnt - 1))  # 无偏修正
    rstd = 1.0 / (np.sqrt(var) + epsilon)
    return mean, rstd


def _normalize_row(row, seg, cnt, epsilon):
    """per_feature 单行：seg 为统计段，row 整行都被归一化。"""
    mean, rstd = _mean_rstd(seg, cnt, epsilon)
    return (row - mean) * rstd
