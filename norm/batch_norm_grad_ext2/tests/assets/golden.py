#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Reference implementation for BatchNormGradExt2.

The reference is *semantic*: it evaluates the operator contract in float64 and performs a single
rounding to the declared output dtype.  It deliberately does not mirror any hardware tiling,
core count, fold factor or reduction order -- the previous revision did, and that mirroring
itself contributed up to 7 ULP of error on large reductions.

Derived scalars keep the operator's fp32 contract:
  * reduction products x*dy are evaluated in fp64 where they are exact for fp16/fp32 inputs;
  * inference rstd is defined as the correctly rounded fp32 value fl32(1 / fl32(sqrt(fl32(var+eps))));
  * scale_backprop / offset_backprop are rounded to fp32 exactly once.
"""

import logging
import math
import os

import numpy as np
import torch


__spec__ = {"batch_norm_grad_ext2": "BatchNormGradExt2Spec"}
__golden__ = {"kernel": {"batch_norm_grad_ext2": "batch_norm_grad_ext2_golden"}}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}

_FP32 = np.dtype(np.float32)
_FP16 = np.dtype(np.float16)
_FP64 = np.dtype(np.float64)

# Inverse of ttk.utilities.DTYPE_PROMOTE_MAP.  Under golden_mode=Promote the framework lifts every
# input/output one step (fp16 -> fp32, fp32 -> fp64); the inverse restores the dtype the operator
# declares.  Promotion is lossless upwards, so the inverse mapping loses no information.
_PROMOTE_INVERSE = {_FP32: _FP16, _FP64: _FP32}

_CHANNEL_SUM_BLOCK = 1 << 16
_SELFCHECK_LOG = "golden_selfcheck.log"
_LAYOUT_CHECKED = False


# Channel axis per data format.  The rank is deliberately not part of the key: the axis is
# resolved against the tensor that is actually handed in, so one entry serves both the 4-D
# (NCHW/NHWC) and the 5-D (NCDHW/NDHWC) spellings; a negative axis counts from the end.
_CHANNEL_AXIS_BY_FORMAT = {
    "NC": 1,
    "NCL": 1,
    "NCHW": 1,
    "NCDHW": 1,
    "NLC": -1,
    "NWC": -1,
    "NHWC": -1,
    "NDHWC": -1,
}

_DEFAULT_EPSILON = 1e-4


def _format_key(data_format):
    """Normalise a data-format spelling; anything that is not a string is unknown."""
    if not isinstance(data_format, str):
        return None
    # Tolerate rank-suffixed spellings such as "NDHWC-5D" / "NHWC_4D".
    key = data_format.strip().upper().replace("-", "_").split("_")[0]
    return key or None


def _as_bool(value, default):
    """Attribute values arrive from the case definition and may be strings."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _channel_axis(x, data_format, channels, input_formats=None):
    """Index of the channel axis of ``x``.

    ``data_format`` is the primary source; the per-input formats the framework forwards with
    the attributes act as a cross-check, and a size-based search is the last resort.  Every
    candidate is validated against the parameter length, because a format that disagrees with
    the tensor must never be allowed to pass a spatial axis off as the channel axis -- doing so
    produced ``shape '[1, H, 1, 1]' is invalid for input of size C`` on channel-last cases.
    """
    rank = x.dim()
    declared = []
    for raw in (data_format, *(input_formats or ())):
        axis = _CHANNEL_AXIS_BY_FORMAT.get(_format_key(raw))
        if axis is None or not -rank <= axis < rank:
            continue
        if axis % rank not in declared:
            declared.append(axis % rank)

    for axis in declared:
        if x.shape[axis] == channels:
            return axis
    sized = [axis for axis in range(rank) if x.shape[axis] == channels]
    if len(sized) == 1:
        return sized[0]
    if not declared and not sized:
        raise ValueError(
            f"no axis of the input shape {tuple(x.shape)} holds the {channels} channel value(s)"
        )
    # Ambiguous by size: prefer the declared format, then the channel-last family (what this
    # operator is graded on).  A declared axis that still disagrees is a contract violation
    # worth reporting instead of computing something plausible but wrong.
    axis = declared[0] if declared else sized[-1]
    if x.shape[axis] != channels:
        raise ValueError(
            f"channel axis {axis} of the input shape {tuple(x.shape)} holds {x.shape[axis]} "
            f"value(s) but {channels} channel value(s) were supplied (data_format={data_format!r})"
        )
    return axis


def _to_nchw_like_np(x, data_format):
    if data_format == "NHWC":
        return np.transpose(x, (0, 3, 1, 2))
    if data_format == "NDHWC":
        return np.transpose(x, (0, 4, 1, 2, 3))
    return x


def _from_nchw_like_np(x, data_format):
    if data_format == "NHWC":
        return np.transpose(x, (0, 2, 3, 1))
    if data_format == "NDHWC":
        return np.transpose(x, (0, 2, 3, 4, 1))
    return x


def _channel_axes(rank):
    return tuple(i for i in range(rank) if i != 1)


def _declared_dtype(dtype, golden_mode):
    """Restore the operator-declared dtype from the (possibly promoted) one."""
    dtype = np.dtype(dtype)
    if golden_mode != "Promote":
        return dtype
    return _PROMOTE_INVERSE.get(dtype, dtype)


def _pairwise_sum_last(x):
    """Explicit pairwise summation along the last axis in float64."""
    acc = np.array(x, dtype=np.float64, copy=True)
    if acc.shape[-1] == 0:
        return np.zeros(acc.shape[:-1], dtype=np.float64)
    while acc.shape[-1] > 1:
        n = acc.shape[-1]
        pair = n // 2
        summed = acc[..., : 2 * pair : 2] + acc[..., 1 : 2 * pair : 2]
        acc = np.concatenate([summed, acc[..., -1:]], axis=-1) if n % 2 else summed
    return acc[..., 0]


def _channel_sum_fp64(x_cr, block=_CHANNEL_SUM_BLOCK):
    """Per-channel fp64 sum of a (C, R) array.

    Error bound ~ (log2(block) + log2(nblocks)) * u64 * sum|x|, i.e. about 1e-3 ULP for
    the cancellation ratios of this operator (see design section 5.4).
    """
    x_cr = np.asarray(x_cr, dtype=np.float64)
    channels, r_dim = x_cr.shape
    nblocks = max(1, (r_dim + block - 1) // block)
    partials = np.empty((channels, nblocks), dtype=np.float64)
    for idx in range(nblocks):
        partials[:, idx] = _pairwise_sum_last(x_cr[:, idx * block : (idx + 1) * block])
    return _pairwise_sum_last(partials)


def _channel_sum_product_fp64(x_cr, y_cr, block=_CHANNEL_SUM_BLOCK):
    """Per-channel fp64 sum of x*y without materialising the full element-wise product."""
    x_cr = np.asarray(x_cr, dtype=np.float64)
    y_cr = np.asarray(y_cr, dtype=np.float64)
    channels, r_dim = x_cr.shape
    nblocks = max(1, (r_dim + block - 1) // block)
    partials = np.empty((channels, nblocks), dtype=np.float64)
    for idx in range(nblocks):
        start = idx * block
        end = min(start + block, r_dim)
        partials[:, idx] = _pairwise_sum_last(x_cr[:, start:end] * y_cr[:, start:end])
    return _pairwise_sum_last(partials)


def _rstd_infer_fp32(var32, eps32):
    """Contract definition of the inference rstd.

    Bit-equal to the kernel's Sqrt+Div path only inside the kernel's 0-ULP sqrt window
    (v = fl32(var + eps) finite, v > 0 and |v| <= FP32_MAX_FINITE * 0.25). Outside that window the
    kernel falls back to a 1-ULP sqrt, so the two may differ by at most 1 ULP there. The contract
    also fixes v == 0 -> +Inf and v < 0 -> NaN.
    """
    variance = (np.asarray(var32, dtype=np.float32) + np.float32(eps32)).astype(
        np.float32
    )
    root = np.sqrt(variance).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return (np.float32(1.0) / root).astype(np.float32)


def _relative_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(a), np.abs(b))
    diff = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scale > 0.0, diff / scale, diff)


def _channel_abs_sum_fp64(x_cr, block=_CHANNEL_SUM_BLOCK):
    """Per-channel fp64 sum of |x| (used as the absolute-error scale of the self-check)."""
    x_cr = np.asarray(x_cr, dtype=np.float64)
    out = np.zeros(x_cr.shape[0], dtype=np.float64)
    for start in range(0, x_cr.shape[1], block):
        out += np.abs(x_cr[:, start : start + block]).sum(axis=1)
    return out


def _channel_abs_product_sum_fp64(x_cr, y_cr, block=_CHANNEL_SUM_BLOCK):
    """Per-channel fp64 sum of |x*y| (absolute-error scale for S1)."""
    x_cr = np.asarray(x_cr, dtype=np.float64)
    y_cr = np.asarray(y_cr, dtype=np.float64)
    out = np.zeros(x_cr.shape[0], dtype=np.float64)
    for start in range(0, x_cr.shape[1], block):
        out += np.abs(
            x_cr[:, start : start + block] * y_cr[:, start : start + block]
        ).sum(axis=1)
    return out


def _report_self_check(message):
    """Self-check reporting never blocks a testcase: it only logs."""
    logging.warning(message)
    try:
        with open(_SELFCHECK_LOG, "a", encoding="utf-8") as handle:
            handle.write(message + os.linesep)
    except OSError:
        pass


def _self_check_small(x_cr, dy_cr, s0, s1):
    """math.fsum cross-check for small reductions.

    Uses an ABSOLUTE tolerance scaled by sum|t| (a relative comparison is meaningless for
    strongly cancelling finite sequences), skips non-finite data, and never raises: the
    self-check must not abort a testcase.
    """
    if not (np.isfinite(x_cr).all() and np.isfinite(dy_cr).all()):
        return
    try:
        ref0 = np.array(
            [math.fsum(dy_cr[c].tolist()) for c in range(x_cr.shape[0])],
            dtype=np.float64,
        )
        ref1 = np.array(
            [math.fsum((x_cr[c] * dy_cr[c]).tolist()) for c in range(x_cr.shape[0])],
            dtype=np.float64,
        )
    except (ValueError, OverflowError):
        return
    scale0 = _channel_abs_sum_fp64(dy_cr)
    scale1 = _channel_abs_product_sum_fp64(x_cr, dy_cr)
    if not (np.isfinite(ref0).all() and np.isfinite(ref1).all()):
        return
    tol = 1e-13
    bad = np.any(
        np.abs(ref0 - s0) > tol * np.maximum(scale0, np.finfo(np.float64).tiny)
    ) or np.any(np.abs(ref1 - s1) > tol * np.maximum(scale1, np.finfo(np.float64).tiny))
    if bad:
        _report_self_check(
            "BatchNormGradExt2 golden self-check: math.fsum absolute mismatch"
        )


def _self_check_large(x_cr, dy_cr, s0, s1):
    """Two block sizes must agree for large reductions (absolute tolerance, never raises)."""
    if not (np.isfinite(x_cr).all() and np.isfinite(dy_cr).all()):
        return
    alt_block = _CHANNEL_SUM_BLOCK // 4
    alt_s0 = _channel_sum_fp64(dy_cr, alt_block)
    alt_s1 = _channel_sum_product_fp64(x_cr, dy_cr, alt_block)
    scale0 = _channel_abs_sum_fp64(dy_cr)
    scale1 = _channel_abs_product_sum_fp64(x_cr, dy_cr)
    tol = 1e-12
    tiny = np.finfo(np.float64).tiny
    if np.any(np.abs(s0 - alt_s0) > tol * np.maximum(scale0, tiny)) or np.any(
        np.abs(s1 - alt_s1) > tol * np.maximum(scale1, tiny)
    ):
        _report_self_check(
            "BatchNormGradExt2 golden self-check: block-size absolute disagreement"
        )


def _layout_selfcheck():
    """Verify that the (C, R) <-> NCHW-like reconstruction is the exact inverse, for every
    supported data format and for C>1 / N>1.  Logs (never raises) on failure."""
    rng = np.random.default_rng(0)
    for fmt, shape in (
        ("NCHW", (2, 3, 4, 5)),
        ("NHWC", (2, 4, 5, 3)),
        ("NDHWC", (2, 4, 5, 6, 3)),
    ):
        raw = rng.standard_normal(shape).astype(np.float32)
        nchw = _to_nchw_like_np(raw, fmt)
        channels = nchw.shape[1]
        flat = np.ascontiguousarray(np.moveaxis(nchw, 1, 0)).reshape(channels, -1)
        restored = np.moveaxis(
            flat.reshape((channels, nchw.shape[0]) + tuple(nchw.shape[2:])), 0, 1
        )
        if not np.array_equal(restored, nchw):
            _report_self_check(
                "BatchNormGradExt2 golden self-check: dx layout round-trip failed for "
                + fmt
            )
            return


class _BatchNormGradExt2Compose:
    """Third-party GPU implementation using the same BN backward formula as the
    operator contract. Outputs are cast to the NPU output dtypes.

    Two properties are load-bearing for the framework that drives this class:

    * Parameters are keyword-only.  The dispatcher binds inputs and attributes **by name** and
      passes each declared parameter positionally in signature order; a keyword-only signature
      bypasses that path entirely, so an attribute the case definition leaves at its default
      (typically ``epsilon``) falls back to the default instead of shifting the remaining
      values into the wrong slots.
    * The computation stays in the caller's layout.  The channel axis is resolved once and
      every reduction keeps its dimension, so ``dx`` is returned in the input shape for
      NCHW/NHWC/NCDHW/NDHWC alike and no permute round-trip can swap channels and batch.
    """

    def __init__(
        self, *, epsilon=_DEFAULT_EPSILON, data_format="NHWC", is_training=True, **_
    ):
        try:
            self.epsilon = _DEFAULT_EPSILON if epsilon is None else float(epsilon)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"epsilon must be a real number, got {epsilon!r}") from exc
        self.data_format = data_format
        self.is_training = _as_bool(is_training, True)

    def __call__(
        self,
        *,
        y_backprop,
        x,
        scale,
        reserve_space_1,
        reserve_space_2,
        input_formats=None,
        **_,
    ):
        channels = int(scale.numel())
        if (
            int(reserve_space_1.numel()) != channels
            or int(reserve_space_2.numel()) != channels
        ):
            raise ValueError(
                "scale / reserve_space_1 / reserve_space_2 must hold one value per channel, got "
                f"{channels} / {int(reserve_space_1.numel())} / {int(reserve_space_2.numel())}"
            )

        axis = _channel_axis(x, self.data_format, channels, input_formats)
        rank = x.dim()
        axes = tuple(idx for idx in range(rank) if idx != axis)
        reduce_count = 1
        for idx in axes:
            reduce_count *= int(x.shape[idx])
        broadcast_shape = [1] * rank
        broadcast_shape[axis] = channels

        scale_n = scale.reshape(-1).to(torch.float32).reshape(broadcast_shape)
        mean_n = reserve_space_1.reshape(-1).to(torch.float32).reshape(broadcast_shape)
        rs2 = reserve_space_2.reshape(-1).to(torch.float32)
        if self.is_training:
            # reserve_space_2 is the saved rstd; it must not be recomputed.
            rstd_n = rs2.reshape(broadcast_shape)
        else:
            # reserve_space_2 is the saved variance.
            rstd_n = torch.rsqrt(rs2 + self.epsilon).reshape(broadcast_shape)

        dy_f = y_backprop.to(torch.float32)
        x_hat = (x.to(torch.float32) - mean_n) * rstd_n
        dscale = torch.sum(dy_f * x_hat, dim=axes, keepdim=True).to(torch.float32)
        doffset = torch.sum(dy_f, dim=axes, keepdim=True).to(torch.float32)
        if self.is_training:
            dx_n = (
                (dy_f - doffset / reduce_count - x_hat * dscale / reduce_count)
                * scale_n
                * rstd_n
            )
        else:
            dx_n = dy_f * scale_n * rstd_n

        # Same layout in, same layout out: no transpose, so dx always has x's shape.
        dx = dx_n.to(x.dtype)
        empty_a = torch.empty((0,), device=x.device, dtype=torch.float32)
        empty_b = torch.empty((0,), device=x.device, dtype=torch.float32)
        return [dx, dscale.reshape(-1), doffset.reshape(-1), empty_a, empty_b]


class BatchNormGradExt2Spec:
    @staticmethod
    def golden(y_backprop, x, scale, reserve_space_1, reserve_space_2, **kwargs):
        return batch_norm_grad_ext2_golden(
            y_backprop, x, scale, reserve_space_1, reserve_space_2, **kwargs
        )

    third_party = {"torch": _BatchNormGradExt2Compose}
    tolerance = _TOL


def _output_dtype(kwargs, index, default, golden_mode):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return _declared_dtype(default, golden_mode)
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return _declared_dtype(dtype if dtype is not None else default, golden_mode)


def batch_norm_grad_ext2_golden(
    y_backprop,
    x,
    scale,
    reserve_space_1,
    reserve_space_2,
    *,
    epsilon=0.0001,
    data_format="NHWC",
    is_training=True,
    **kwargs,
):
    """Semantic fp64 reference with a single rounding to the declared output dtype.

    Training: reserve_space_2 is the saved rstd.  Inference: reserve_space_2 is the variance and
    rstd is re-derived with the operator's fp32 contract definition.
    """
    golden_mode = kwargs.get("golden_mode", "Enable")
    global _LAYOUT_CHECKED
    if not _LAYOUT_CHECKED:
        _LAYOUT_CHECKED = True
        try:
            _layout_selfcheck()
        except Exception:  # self-checks must never abort a testcase
            pass
    x_dtype = _declared_dtype(x.dtype, golden_mode)
    dy_dtype = _declared_dtype(y_backprop.dtype, golden_mode)
    scale_dtype = _declared_dtype(scale.dtype, golden_mode)
    mean_dtype = _declared_dtype(reserve_space_1.dtype, golden_mode)
    var_dtype = _declared_dtype(reserve_space_2.dtype, golden_mode)
    dx_dtype = _output_dtype(kwargs, 0, x.dtype, golden_mode)
    dscale_dtype = _output_dtype(kwargs, 1, np.float32, golden_mode)
    doffset_dtype = _output_dtype(kwargs, 2, np.float32, golden_mode)

    # 1) Restore the declared contract dtype (lossless), then evaluate in fp64.
    x_n = _to_nchw_like_np(np.asarray(x, dtype=x_dtype), data_format)
    dy_n = _to_nchw_like_np(np.asarray(y_backprop, dtype=dy_dtype), data_format)
    nchw_shape = x_n.shape
    channels = nchw_shape[1]
    axes = _channel_axes(x_n.ndim)
    r_dim = int(np.prod([nchw_shape[axis] for axis in axes], dtype=np.int64))
    reduce_count = float(r_dim)

    x_cr = np.ascontiguousarray(np.moveaxis(x_n, 1, 0)).reshape(channels, r_dim)
    dy_cr = np.ascontiguousarray(np.moveaxis(dy_n, 1, 0)).reshape(channels, r_dim)
    x_cr = x_cr.astype(np.float64)
    dy_cr = dy_cr.astype(np.float64)

    scale_c = np.asarray(scale, dtype=scale_dtype).reshape(channels).astype(np.float64)
    mean_c = (
        np.asarray(reserve_space_1, dtype=mean_dtype)
        .reshape(channels)
        .astype(np.float64)
    )
    rs2_c = np.asarray(reserve_space_2, dtype=var_dtype).reshape(channels)
    if is_training:
        rstd_c = rs2_c.astype(np.float64)
    else:
        rstd_c = _rstd_infer_fp32(rs2_c.astype(np.float32), np.float32(epsilon)).astype(
            np.float64
        )

    # 2) fp64 channel reductions, algebraically equivalent to sum(dy*(x-mean)*rstd).
    with np.errstate(over="ignore", invalid="ignore"):
        s1 = _channel_sum_product_fp64(x_cr, dy_cr)
        s0 = _channel_sum_fp64(dy_cr)
        dscale64 = rstd_c * (s1 - mean_c * s0)
        doffset64 = s0

    # Self-checks only log; they must never abort a testcase (same rule as _report_self_check).
    try:
        if channels * r_dim <= 2_000_000:
            _self_check_small(x_cr, dy_cr, s0, s1)
        else:
            _self_check_large(x_cr, dy_cr, s0, s1)
    except Exception:  # noqa: BLE001 - a self-check must never change the test outcome
        pass

    # 3) The only rounding of the reduction outputs, to their declared dtype.
    dscale_out = dscale64.astype(dscale_dtype)
    doffset_out = doffset64.astype(doffset_dtype)
    dscale32 = dscale64.astype(np.float32)[:, None]
    doffset32 = doffset64.astype(np.float32)[:, None]
    mean_col = mean_c[:, None]
    rstd_col = rstd_c[:, None]
    scale_col = scale_c[:, None]

    # 4) dx, block-wise in fp64 with a single rounding at the end.
    dx_cr = np.empty((channels, r_dim), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, r_dim, _CHANNEL_SUM_BLOCK):
            end = min(start + _CHANNEL_SUM_BLOCK, r_dim)
            x_chunk = x_cr[:, start:end]
            dy_chunk = dy_cr[:, start:end]
            if is_training:
                x_hat = (x_chunk - mean_col) * rstd_col
                dx_cr[:, start:end] = (
                    (
                        dy_chunk
                        - doffset32 / reduce_count
                        - x_hat * dscale32 / reduce_count
                    )
                    * scale_col
                    * rstd_col
                )
            else:
                dx_cr[:, start:end] = dy_chunk * scale_col * rstd_col
    # (C, R) -> NCHW-like.  R flattens (N, *spatial), so the inverse must keep N as the second
    # axis of the (C, N, *spatial) view; a bare reshape to nchw_shape would swap C and N for N>1
    # (and scramble every C>1 case).
    dx_nchw = np.moveaxis(
        dx_cr.reshape((channels, nchw_shape[0]) + tuple(nchw_shape[2:])), 0, 1
    )
    with np.errstate(over="ignore", invalid="ignore"):
        dx = _from_nchw_like_np(dx_nchw, data_format).astype(dx_dtype, copy=False)

    empty = np.empty((0,), dtype=np.float32)
    return [dx, dscale_out, doffset_out, empty, empty]
