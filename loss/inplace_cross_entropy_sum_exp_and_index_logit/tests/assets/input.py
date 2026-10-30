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
"""TTK input generator for InplaceCrossEntropySumExpAndIndexLogit.

TTK's default random generator creates the three inputs independently. That is
not valid: ``global_logits_max`` must be >= max_j(logits[i,j]) per row to
prevent exp overflow in the golden and kernel. This generator corrects
``global_logits_max`` per row.
"""

import numpy as np

__input__ = {
    "kernel": {
        "inplace_cross_entropy_sum_exp_and_index_logit": "inplace_cross_entropy_sum_exp_and_index_logit_inputs",
    },
    "aclnn": {
        "aclnnInplaceCrossEntropySumExpAndIndexLogit": "aclnn_inplace_cross_entropy_sum_exp_and_index_logit_inputs",
    },
    "e2e": {
        "torch.ops.cann_ops_nn.inplace_cross_entropy_sum_exp_and_index_logit": "e2e_inplace_cross_entropy_sum_exp_and_index_logit_inputs",
    },
}


def _fix_global_logits_max_numpy(logits, gmax):
    """numpy in-place: gmax[i] = max(gmax[i], max_j(logits[i,j]))"""
    per_row_max = logits.max(axis=-1, keepdims=False)
    np.maximum(gmax, per_row_max, out=gmax)
    return gmax


def _fix_global_logits_max_torch(logits, gmax):
    """torch in-place: gmax[i] = max(gmax[i], max_j(logits[i,j]))"""
    import torch

    local_max = torch.max(logits, dim=-1, keepdim=False).values
    torch.max(gmax, local_max, out=gmax)
    return gmax


def inplace_cross_entropy_sum_exp_and_index_logit_inputs(
    vocab_parallel_logits,
    target,
    global_logits_max,
    *,
    vocab_start_index=0,
    vocab_end_index=0,
    **kwargs,
):
    _fix_global_logits_max_numpy(vocab_parallel_logits, global_logits_max)
    return (vocab_parallel_logits, target, global_logits_max)


def aclnn_inplace_cross_entropy_sum_exp_and_index_logit_inputs(
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
    """aclnn 输入插件：就地修正 global_logits_max。

    TTK 会消费返回值并按"输入张量槽"（纯输出槽除外）回写，因此返回修正后的
    输入元组（与 dump/golden 共享同一 storage）。
    """
    _fix_global_logits_max_torch(vocabParallelLogits, globalLogitsMax)
    return (vocabParallelLogits, target, globalLogitsMax)


def e2e_inplace_cross_entropy_sum_exp_and_index_logit_inputs(
    vocab_parallel_logits,
    target,
    global_logits_max,
    vocab_start_index,
    vocab_end_index,
    **kwargs,
):
    _fix_global_logits_max_torch(vocab_parallel_logits, global_logits_max)
    return (
        vocab_parallel_logits,
        target,
        global_logits_max,
        vocab_start_index,
        vocab_end_index,
    )
