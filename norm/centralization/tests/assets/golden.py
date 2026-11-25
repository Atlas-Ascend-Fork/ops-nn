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

"""CPU golden and Torch comparator for Centralization kernel and GEIR validation."""

import math

import numpy as np
import torch


def _normalized_axes(axes, rank):
    if not 1 <= rank <= 8:
        raise ValueError(f"x rank must be in [1, 8], got {rank}")
    if axes is None:
        return (rank - 1,)
    if not isinstance(axes, (list, tuple, np.ndarray)) or np.ndim(axes) != 1:
        raise TypeError("axes must be a one-dimensional sequence of integers")
    normalized = set()
    for axis in axes:
        if isinstance(axis, (bool, np.bool_)) or not isinstance(
            axis, (int, np.integer)
        ):
            raise TypeError(f"axes must contain integers, got {axis!r}")
        axis = int(axis)
        if not -rank <= axis < rank:
            raise ValueError(f"axis {axis} is outside [-{rank}, {rank})")
        normalized.add(axis % rank)
    return tuple(sorted(normalized))


def _centralize(x, axes):
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"Centralization expects FP16 or FP32 input, got {x.dtype}")
    normalized = _normalized_axes(axes, x.dim())
    compute = x.to(torch.float32)
    if len(normalized) == 0:
        # 空 axes 表示不归约：有限输入得到 0，NaN 和无穷输入得到 NaN。
        return (compute - compute).to(x.dtype)
    result = compute - torch.mean(compute, dim=normalized, keepdim=True)
    return result.to(x.dtype)


def _attr(kwargs, name, default):
    value = kwargs.get(name)
    if value is None and isinstance(kwargs.get("attributes"), dict):
        value = kwargs["attributes"].get(name)
    return default if value is None else value


def _center_wide_range(group):
    if not np.isfinite(group).all():
        return np.subtract(group, group.mean(dtype=np.float64), dtype=np.float64)

    # Finite binary floats are integer multiples of the smallest subnormal.
    # Work in those units so even a tiny remainder in width*x - sum survives.
    info = np.finfo(group.dtype)
    fractional_bits = info.nmant - info.minexp

    def units(value):
        numerator, denominator = float(value).as_integer_ratio()
        return numerator << (fractional_bits - (denominator.bit_length() - 1))

    width = len(group)
    total = sum(map(units, group))
    divisor = width << fractional_bits
    return np.fromiter(
        ((width * units(value) - total) / divisor for value in group),
        dtype=group.dtype,
        count=width,
    )


def _center_rows(rows):
    width = rows.shape[1]
    centered = np.empty_like(rows)
    # Bound temporaries for batches of short reductions. A single long row
    # uses the accurately summed fallback without allocating a scaled copy.
    batch_size = max(1, (1 << 17) // width)
    for start in range(0, len(rows), batch_size):
        block = rows[start : start + batch_size]
        out = centered[start : start + batch_size]
        exact = np.zeros(len(block), dtype=bool)
        if len(block) > 1:
            wide = block.astype(np.float64, copy=False)
            exponent = np.frexp(np.max(np.abs(wide), axis=1, keepdims=True))[1]
            shift = 52 - (width - 1).bit_length() - exponent
            scaled = np.ldexp(wide, shift)
            # On this common binary grid, both width*x and the sum fit in
            # 52 bits. Their difference fits in 53, so only division rounds.
            exact = np.all(
                np.isfinite(scaled)
                & (scaled == np.rint(scaled))
                & ((scaled != 0) | (wide == 0)),
                axis=1,
            )
            integers = scaled[exact]
            numerator = width * integers - integers.sum(axis=1, keepdims=True)
            out[exact] = np.ldexp(numerator / width, -shift[exact])
        for i in np.flatnonzero(~exact):
            out[i] = _center_wide_range(block[i])
    return centered


def centralization_golden(x, **kwargs):
    value = np.asarray(x)
    axes = _normalized_axes(_attr(kwargs, "axes", [-1]), value.ndim)
    native_types = (np.dtype("float16"), np.dtype("float32"))
    declared = kwargs.get("input_dtypes")
    if declared is not None:
        declared = np.asarray(declared).item()
    # Kernel/GEIR can also promote the dtype metadata together with the array.
    promoted = kwargs.get("golden_mode") == "Promote" and value.dtype == np.float64
    if promoted:
        valid = declared in (None, np.dtype("float32"), np.dtype("float64"))
    else:
        valid = value.dtype in native_types and (
            declared is None or declared in native_types
        )
    if not valid:
        raise TypeError(
            f"unsupported Centralization input dtype {value.dtype} (declared {declared})"
        )

    if value.size == 0:
        return [np.empty_like(value)]
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        if all(value.shape[axis] == 1 for axis in axes):
            return [np.subtract(value, value)]
        kept = tuple(i for i in range(value.ndim) if i not in axes)
        order = kept + axes
        groups = value.transpose(order)
        width = math.prod(value.shape[i] for i in axes)
        result = _center_rows(groups.reshape(-1, width)).reshape(groups.shape)
        return [result.transpose(np.argsort(order))]


class _CentralizationCompose:
    """Independent PyTorch implementation used by the remote GPU comparator."""

    def __init__(self, *, axes=(-1,), **_):
        self.axes = axes

    def __call__(self, x, **_):
        return [_centralize(x, self.axes)]


class CentralizationSpec:
    @staticmethod
    def golden(x, **kwargs):
        return centralization_golden(x, **kwargs)

    third_party = {"torch": _CentralizationCompose}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }


__spec__ = {"centralization": "CentralizationSpec"}
__golden__ = {"kernel": {"centralization": "centralization_golden"}}
