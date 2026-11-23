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
"""TestSpec for CTCLossV2.

Reference source
----------------
``torch.ops.aten._ctc_loss`` on CPU -- the canonical framework implementation,
independent of the NPU operator under test. NumPy is used only for
dtype/container bridging, never for the reference mathematics.

Output protocol (follows ctc_loss_v2_def.cpp)
---------------------------------------------
Two REQUIRED outputs, in order: ``neg_log_likelihood`` (N,) and ``log_alpha``
(N, T, tail). Both are returned so both are compared; previously only the loss
was returned, which left ``log_alpha`` out of the comparison entirely.

``log_alpha`` defined region
----------------------------
The reference only defines ``log_alpha[b, t, s]`` for ``t < input_lengths[b]``
**and** ``s < 2 * target_lengths[b] + 1``. Everything outside that window is
left untouched by the reference implementation, i.e. it is uninitialised
memory: calling ``aten._ctc_loss`` repeatedly on identical inputs yields
different content there from run to run.

Those positions are normalised to ``-inf``, which is the reference's own
encoding of an unreachable CTC state, established from the reference alone:

* at ``t = 0`` the reference fills ``s >= 2`` with ``-inf`` -- including for a
  batch whose ``target_lengths[b] == max(target_lengths)``, i.e. with no padding
  involved at all;
* inside the defined window the count of ``-inf`` entries decreases as ``t``
  grows, so ``-inf`` precisely marks "state not reachable at this time step".

The normalisation therefore extends the reference's own convention to every
unreachable position. It is not derived from, and does not consult, the output
of the operator under test.

Tail width
----------
The reference tail is ``2 * max(target_lengths) + 1`` (driven by
``target_lengths``, not by ``targets.shape[1]``). ``ctc_loss_v2_infershape.cpp``
rounds the tail up to a multiple of 8. When the declared tail is wider than the
reference tail, the extra columns are unreachable by construction and are filled
with ``-inf`` as well, so the returned shape matches the declared output shape.
"""

import numpy as np

__spec__ = {"ctc_loss_v2": "CtcLossV2TestSpec"}

__golden__ = {"kernel": {"ctc_loss_v2": "ctc_loss_v2_golden"}}

_NEG_INF = float("-inf")


def _to_reference_dtype(arr):
    """Low precision floats (fp16/bf16) are upcast to float32 for the reference."""
    a = np.asarray(arr)
    if a.dtype in (np.float32, np.float64):
        return a
    return a.astype(np.float32)


def _int_list(arr):
    return [int(v) for v in np.asarray(arr).reshape(-1).tolist()]


def _declared_log_alpha_tail(kwargs, reference_shape):
    """Validate the declared layout without cropping reachable reference states."""
    fallback = reference_shape[-1]
    shapes = kwargs.get("output_ori_shapes") or kwargs.get("output_shapes")
    if not shapes or len(shapes) < 2:
        return fallback
    declared = shapes[1]
    if declared is None:
        return fallback
    if len(declared) != 3 or any(
        dim >= 0 and dim != expected
        for dim, expected in zip(declared[:2], reference_shape[:2])
    ):
        raise ValueError("log_alpha must have layout (batch, time, tail)")
    tail = int(declared[-1])
    if 0 <= tail < fallback:
        raise ValueError("log_alpha tail cannot omit reachable reference states")
    return tail if tail > 0 else fallback


def _mask_unreachable(log_alpha, input_lengths, target_lengths, tail):
    """Set every unreachable (b, t, s) position to -inf.

    Unreachable means t >= input_lengths[b] or s >= 2 * target_lengths[b] + 1,
    including the columns beyond the reference tail when the declared tail is
    wider (alignment padding).
    """
    batch, time_steps, ref_tail = log_alpha.shape
    if tail > ref_tail:
        widened = np.full((batch, time_steps, tail), _NEG_INF, dtype=log_alpha.dtype)
        widened[:, :, :ref_tail] = log_alpha
        log_alpha = widened
    else:
        log_alpha = log_alpha.copy()

    for b in range(batch):
        valid_t = (
            min(int(input_lengths[b]), time_steps)
            if b < len(input_lengths)
            else time_steps
        )
        valid_s = 2 * int(target_lengths[b]) + 1 if b < len(target_lengths) else tail
        valid_t = max(valid_t, 0)
        valid_s = max(min(valid_s, tail), 0)
        log_alpha[b, valid_t:, :] = _NEG_INF
        log_alpha[b, :valid_t, valid_s:] = _NEG_INF
    return log_alpha


def ctc_loss_v2_golden(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    *,
    blank=0,
    reduction="mean",
    zero_infinity=False,
    **kwargs,
):
    """
    Golden function for ctc_loss_v2.
    All the parameters (names and order) follow @ctc_loss_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        log_probs: (T, N, C) log probabilities.
        targets: (N, S) or (sum(target_lengths),) target sequences.
        input_lengths: (N,) valid input length per batch.
        target_lengths: (N,) valid target length per batch.
        blank: blank label index.
        reduction: not used by this forward kernel -- neg_log_likelihood is the
                   raw per-batch negative log likelihood, matching aten._ctc_loss.
        zero_infinity: zero out infinite losses.
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [neg_log_likelihood, log_alpha] following the IR output order.
        neg_log_likelihood shape (N,); log_alpha shape (N, T, tail) with every
        unreachable position normalised to -inf (see the module docstring).
    """
    import torch

    out_dtype = np.asarray(log_probs).dtype

    input_lengths_l = _int_list(input_lengths)
    target_lengths_l = _int_list(target_lengths)

    neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
        torch.from_numpy(_to_reference_dtype(log_probs)),
        torch.from_numpy(np.asarray(targets).astype(np.int64, copy=False)),
        input_lengths_l,
        target_lengths_l,
        int(blank),
        bool(zero_infinity),
    )

    log_alpha_np = log_alpha.numpy()
    tail = _declared_log_alpha_tail(kwargs, log_alpha_np.shape)
    log_alpha_np = _mask_unreachable(
        log_alpha_np, input_lengths_l, target_lengths_l, tail
    )

    return [
        neg_log_likelihood.numpy().astype(out_dtype, copy=False),
        log_alpha_np.astype(out_dtype, copy=False),
    ]


class CtcLossV2TestSpec:
    """CTCLossV2 test spec (kernel / GEIR flow, numpy in and out)."""

    golden = ctc_loss_v2_golden
