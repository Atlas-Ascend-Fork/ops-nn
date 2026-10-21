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

import numpy as np
import torch

__golden__ = {"kernel": {"unique_consecutive": "unique_consecutive_golden"}}

# axis sentinel: the operator only supports flatten mode (def.cpp pins axis=1000).
_FLATTEN_AXIS = 1000


def _resolve_np_dtype(dtypes, index, default):
    """Robustly resolve output_dtypes[index] to a numpy dtype, falling back to default."""
    try:
        d = dtypes[index]
    except (TypeError, IndexError, KeyError):
        return default
    if d is None:
        return default
    try:
        return np.dtype(d)
    except TypeError:
        try:
            return np.dtype(getattr(d, "name", d))
        except Exception:
            return default


def _group_tensor(flat):
    """Build a torch tensor for grouping that preserves the kernel's per-type NE
    semantics. torch.from_numpy cannot ingest ml_dtypes.bfloat16, so bf16 is widened
    to float32 — an exact widening that preserves IEEE grouping (nan != nan, -0.0 ==
    +0.0), matching the kernel's Reg::Compare<CMPMODE::NE>. All int (incl. uint16/
    uint32/uint64) and float16/float32 dtypes are supported natively by torch 2.7."""
    # ml_dtypes.bfloat16 reports dtype.kind == "V" (not "f") and torch.from_numpy
    # rejects it, so detect by name and widen to float32 (exact, IEEE-faithful proxy).
    if flat.dtype.name == "bfloat16":
        return torch.from_numpy(flat.astype(np.float32))
    return torch.from_numpy(flat)


def unique_consecutive_golden(
    x, *, return_idx=False, return_counts=False, axis=1000, **kwargs
):
    """CPU golden for UniqueConsecutive (kernel-level TTK plugin).

    Reference implementation delegates the consecutive-grouping / adjacent-comparison /
    counting core to ``torch.unique_consecutive`` (torch 2.7.1+cpu). This golden is the
    one TTK's OnGenGolden stage invokes (registered via ``__golden__``); the dumped
    golden tensors compared against the kernel are produced by this torch reference —
    it is the official comparison reference, not a side cross-check.

    Parameter names/order follow @unique_consecutive_def.cpp inputs (x) plus its
    attributes. All input tensors are numpy.ndarray (bfloat16 arrives as ml_dtypes
    bfloat16). Returns [y, idx, count] mapped positionally to output 0/1/2.

    Contract mapping (def.cpp / CheckSupport):
      * axis        : always 1000 (flatten sentinel) -> torch dim=None (flatten).
                      Per-dim mode is never emitted by this operator.
      * return_idx  : always False (CheckSupport rejects True) -> idx value undefined.
      * return_counts: True/False -> real counts / placeholder (see below).

    Semantics — flattened consecutive dedup:
      * y     : consecutive-dedup group representatives in input order (non-adjacent
                repeats stay separate groups). Group boundaries/counts come from
                torch.unique_consecutive; y is then gathered from x by the group start
                indices so its bits/dtype are preserved EXACTLY (never cast large ints
                to float; bf16 NaN payload bits kept), matching the kernel copying x.
      * idx   : inverse indices. This operator never produces idx (return_idx is
                always False; CheckSupport rejects True). The kernel emits a length-1
                shape descriptor (CopyOutShape sets idx DIM0=1) but NEVER writes the
                idxGm_ data buffer. Consequences, split by shape vs value:
                  - SHAPE (1,) is contractual: it is the descriptor the kernel really
                    emits, and the shape-unknown metadata output is compared against
                    it, so golden idx must be a real (1,) ndarray (not None, not (0,)).
                  - VALUE is contractually UNDEFINED: since idxGm_ is never written,
                    the single element equals whatever pre-fills the HBM buffer. This
                    was verified empirically on Ascend950PR: default run reads back 1
                    (TTK's own initial value, output_generation.py "1 is the initial
                    value for output"), while `--clear-atomic` reads back 0. The value
                    is therefore NOT an expected result; ones((1,)) is only a
                    shape-correct placeholder that follows the framework's initial-value
                    convention. Value-level comparison of idx is not meaningful.
      * count : per-group element counts when return_counts=True (kernel writes it,
                SHAPE2 DIM0=G) -> real (G,) counts, a genuine expected result.
                When return_counts=False the kernel emits SHAPE2 DIM0=1 but does not
                write countGm_ -> its value is undefined exactly like idx, so golden
                returns a shape-correct (1,) placeholder (not an expected value).

    NOTE (framework limitation, see tests verification notes / minimal repro): on the
    current ops-test-kit, multi-output shape-unknown comparison cannot reach a value
    verdict — `standards` is resolved before the shape-metadata output is appended, so
    compare() indexes standards[len] out of range and returns COMPARE_FAILURE for every
    such case regardless of golden content. Bit-for-bit agreement of kernel outputs vs
    this golden was instead verified offline from the dumped tensors.

    Adjacent equality is delegated to torch.unique_consecutive, whose grouping mirrors
    the kernel's Reg::Compare<CMPMODE::NE>:
      * integer dtypes  -> exact (no float promotion; uint16/32/64 & int64 boundaries);
      * float dtypes    -> IEEE (nan != nan -> True, so adjacent nan become
                           separate groups; -0.0 == +0.0 -> same group).

    Empty input (size == 0): the kernel takes tiling key 666 (CopyOutEmptyShape),
    which sets all three output dims to 0, so golden returns three empty arrays.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name.
    """
    # axis is pinned to the flatten sentinel by the operator contract; flatten x.
    if axis != _FLATTEN_AXIS:
        raise ValueError(
            f"unique_consecutive golden only supports flatten mode (axis={_FLATTEN_AXIS}); "
            f"got axis={axis}"
        )
    flat = np.asarray(x).reshape(-1)

    # Resolve idx/count dtype from out_idx (takes precedence over output_dtypes).
    # out_idx: 3 -> DT_INT32, 9 -> DT_INT64 (default).
    out_idx = kwargs.get("out_idx", 9)
    if out_idx == 3:
        idx_dtype = np.dtype(np.int32)
        count_dtype = np.dtype(np.int32)
    elif out_idx == 9:
        idx_dtype = np.dtype(np.int64)
        count_dtype = np.dtype(np.int64)
    else:
        # Unsupported out_idx: fallback to int64 (matches default).
        idx_dtype = np.dtype(np.int64)
        count_dtype = np.dtype(np.int64)

    n = flat.shape[0]

    # Empty input: kernel key 666 CopyOutEmptyShape -> all three outputs dim0 = 0.
    if n == 0:
        return [
            np.empty((0,), dtype=flat.dtype),
            np.empty((0,), dtype=idx_dtype),
            np.empty((0,), dtype=count_dtype),
        ]

    # Core: consecutive grouping / adjacent comparison / counting via torch.
    # dim=None -> flatten mode (matches axis=1000). counts is int64 group sizes.
    counts_t = torch.unique_consecutive(
        _group_tensor(flat), dim=None, return_counts=True
    )[1]
    counts = counts_t.numpy()

    # Group start offsets derived from torch counts (adaptation only; the grouping
    # decision is entirely torch's). y is gathered from the ORIGINAL flat array so
    # its bits/dtype stay exact (large ints not promoted, bf16 nan payload preserved).
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    y = flat[starts]

    # idx: contractual shape (1,) (kernel emits DIM0=1); value UNDEFINED (idxGm_ never
    # written) -> shape-correct placeholder only, following the framework initial value.
    # This is NOT a torch result — return_idx is always False and idx is never computed.
    idx = np.ones((1,), dtype=idx_dtype)

    if return_counts:
        count = counts.astype(count_dtype)  # real per-group counts (G,) -> expected
    else:
        # count not produced (countGm_ unwritten) -> undefined value, (1,) placeholder.
        count = np.ones((1,), dtype=count_dtype)

    return [y, idx, count]
