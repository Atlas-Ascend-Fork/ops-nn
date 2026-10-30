#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""inplace_cross_entropy_sum_exp_and_index_logit golden.

inplace 与 base CE 的计算逻辑一致，差异：
1. aclnn 接口少一个输出占位（无 expLogitsOut），但 golden 仍返回 5 个值
   （第 3 个 exp_logits 用于与 in-place 写回后的 vocab_parallel_logits 比对）
2. E2E torch 接口返回 4 个值（无 exp_logits），golden 返回 5 个值
   （exp_logits 放最后，对应 inplace_input_indexes=(0,) 的原地回读位）
"""

import numpy as np


def _to_numpy(t):
    """兼容 torch.Tensor / numpy 数组输入。"""
    if t is None:
        return t
    if hasattr(t, "detach"):
        import torch

        t = t.detach().cpu()
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.numpy()
    return np.asarray(t)


def _as_cpu_tensor(t):
    """转成 CPU torch.Tensor：兼容 numpy / torch.Tensor（任意设备）。"""
    import torch

    if isinstance(t, torch.Tensor):
        if t.is_floating_point() and t.dtype in (torch.bfloat16, torch.float16):
            t = t.float()
        return t.detach().cpu()
    return torch.as_tensor(np.asarray(t))


def _as_python_int(value):
    """标量（python int / torch 标量 tensor）统一转 int。"""
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _cross_entropy_numpy_golden(logits, tgt, gmax, s, e):
    """numpy fp32 精度的核心计算，返回 5 元素 list。"""
    target_shape = tgt.shape
    logits_shape = logits.shape

    v_local = logits.shape[-1]
    n = logits.size // v_local
    if e == 0:
        e = v_local

    logits_2d = logits.reshape(n, v_local)
    target_1d = tgt.reshape(n)
    gmax_1d = gmax.reshape(n)

    shifted = logits_2d - gmax_1d[:, None]

    target_mask_1d = ((target_1d < s) | (target_1d >= e)).astype(np.int32)

    target_offset_1d = np.where(target_mask_1d.astype(bool), 0, target_1d - s).astype(
        np.int32
    )

    gathered = logits_2d[np.arange(n), target_offset_1d.astype(np.int64)]
    predicted_1d = np.where(
        target_mask_1d.astype(bool), np.float32(0.0), gathered - gmax_1d
    )

    exp_logits = np.exp(shifted)
    sum_exp_1d = exp_logits.sum(axis=-1)
    return [
        predicted_1d.reshape(target_shape).astype(np.float32),
        sum_exp_1d.reshape(target_shape).astype(np.float32),
        exp_logits.reshape(logits_shape).astype(np.float32),
        target_offset_1d.reshape(target_shape).astype(np.int32),
        target_mask_1d.reshape(target_shape).astype(np.int32),
    ]


def _cross_entropy_torch_golden(logits, tgt, gmax, s, e):
    """torch float64 高精度核心计算，返回 5 元素 list。"""
    import torch

    target_shape = tgt.shape
    logits_shape = logits.shape

    v_local = logits.shape[-1]
    n = logits.numel() // v_local
    if e == 0:
        e = v_local

    logits_2d = logits.reshape(n, v_local)
    target_1d = tgt.reshape(n)
    gmax_1d = gmax.reshape(n)

    shifted = logits_2d - gmax_1d.unsqueeze(-1)

    target_mask_1d = ((target_1d < s) | (target_1d >= e)).to(torch.int32)

    target_offset_1d = torch.where(
        target_mask_1d.bool(),
        torch.zeros_like(target_1d, dtype=torch.int32),
        (target_1d - s).to(torch.int32),
    )

    gathered = logits_2d[torch.arange(n), target_offset_1d.long()]
    predicted_1d = torch.where(
        target_mask_1d.bool(), torch.zeros_like(gathered), gathered - gmax_1d
    )

    exp_logits = torch.exp(shifted)
    sum_exp_1d = exp_logits.sum(dim=-1)

    return [
        predicted_1d.reshape(target_shape).float(),
        sum_exp_1d.reshape(target_shape).float(),
        exp_logits.reshape(logits_shape).float(),
        target_offset_1d.reshape(target_shape).int(),
        target_mask_1d.reshape(target_shape).int(),
    ]


class InplaceCrossEntropySumExpAndIndexLogitKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，返回 5 元素 list。"""

    @staticmethod
    def golden(
        vocab_parallel_logits,
        target,
        global_logits_max,
        *,
        vocab_start_index=0,
        vocab_end_index=0,
        **kwargs,
    ):
        logits = vocab_parallel_logits.astype(np.float32)
        tgt = target.astype(np.int64)
        gmax = global_logits_max.astype(np.float32)
        s = int(vocab_start_index)
        e = int(vocab_end_index)
        return _cross_entropy_numpy_golden(logits, tgt, gmax, s, e)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int32": {"standard": "binary_equal"},
    }


class AclnnInplaceCrossEntropySumExpAndIndexLogitSpec:
    """ACLNN 流程 — golden 收到 torch.Tensor。

    参数顺序对齐 aclnnInplaceCrossEntropySumExpAndIndexLogitGetWorkspaceSize
    （不含 workspaceSize/executor）：
        (vocabParallelLogits, target, globalLogitsMax,
         vocabStartIndex, vocabEndIndex,
         predictedLogitsOut, sumExpLogitsOut, targetOffsetOut, targetMaskOut)

    返回值顺序必须与 CSV 的 ``output_tensor_indexes`` 一致。本算子用例声明
    ``output_tensor_indexes = (0, 3, 4, 5, 6)``、``output_inplace_indexes = (0,)``
    （张量 0 即 vocabParallelLogits，exp_logits 原地写回），因此返回
        [exp_logits, predicted_logits, sum_exp_logits, target_offset, target_mask]
    """

    @staticmethod
    def golden(
        vocabParallelLogits,
        target,
        globalLogitsMax,
        vocabStartIndex,
        vocabEndIndex,
        predictedLogitsOut,
        sumExpLogitsOut,
        targetOffsetOut,
        targetMaskOut,
        **kwargs,
    ):
        logits = _to_numpy(vocabParallelLogits).astype(np.float32)
        tgt = _to_numpy(target)
        if tgt.dtype != np.int64:
            tgt = tgt.astype(np.int64)
        gmax = _to_numpy(globalLogitsMax).astype(np.float32)
        s = int(vocabStartIndex)
        e = int(vocabEndIndex)
        predicted, sum_exp, exp_logits, offset, mask = _cross_entropy_numpy_golden(
            logits, tgt, gmax, s, e
        )
        return [exp_logits, predicted, sum_exp, offset, mask]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int32": {"standard": "binary_equal"},
    }


class TorchInplaceCrossEntropySumExpAndIndexLogitSpec:
    """torch（E2E）流程 — golden 收到 torch.Tensor，返回 5 元素 list。

    接口对齐 torch.ops.cann_ops_nn.inplace_cross_entropy_sum_exp_and_index_logit：
        (vocab_parallel_logits, target, global_logits_max,
         vocab_start_index, vocab_end_index)
        -> (predicted_logits, sum_exp_logits, target_offset, target_mask)

    inplace 版 torch 接口只返回 4 个非原地输出（exp_logits 写回
    vocab_parallel_logits）。CSV 声明 ``inplace_input_indexes=(0,)`` 后，TTK 会把
    被改写的输入追加到输出尾部（execute.py 先收返回值、再 append 原地回读），
    因此 golden 的返回顺序必须是
        [predicted_logits, sum_exp_logits, target_offset, target_mask, exp_logits]
    """

    @staticmethod
    def golden(
        vocab_parallel_logits,
        target,
        global_logits_max,
        vocab_start_index,
        vocab_end_index,
        **kwargs,
    ):
        logits = _as_cpu_tensor(vocab_parallel_logits).double()
        tgt = _as_cpu_tensor(target).long()
        gmax = _as_cpu_tensor(global_logits_max).double()
        s = _as_python_int(vocab_start_index)
        e = _as_python_int(vocab_end_index)
        predicted, sum_exp, exp_logits, offset, mask = _cross_entropy_torch_golden(
            logits, tgt, gmax, s, e
        )
        # exp_logits 放最后：对应 inplace_input_indexes=(0,) 的原地回读位
        return [predicted, sum_exp, offset, mask, exp_logits]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int32": {"standard": "binary_equal"},
    }


__spec__ = {
    "inplace_cross_entropy_sum_exp_and_index_logit": "InplaceCrossEntropySumExpAndIndexLogitKernelSpec",
    "aclnnInplaceCrossEntropySumExpAndIndexLogit": "AclnnInplaceCrossEntropySumExpAndIndexLogitSpec",
    "torch.ops.cann_ops_nn.inplace_cross_entropy_sum_exp_and_index_logit": "TorchInplaceCrossEntropySumExpAndIndexLogitSpec",
}
