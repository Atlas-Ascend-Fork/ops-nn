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
"""CPU reference: rotation -> BF16/clamp -> FP32 level0/BF16 -> MX level1.

MX encoding follows the dual-level operator's scale and reciprocal rules.
No NPU execution is used to generate expected outputs. FP4 is a logical one-byte NumPy
carrier here (the device stores two nibbles in each byte).
"""

from functools import lru_cache
from math import prod

import numpy as np
import ml_dtypes

try:
    import en_dtypes
except ImportError:
    # CPU-only environments can use the equivalent ml_dtypes bit encodings.
    np_fp4_e2m1 = ml_dtypes.float4_e2m1fn
    np_mx_scale = ml_dtypes.float8_e8m0fnu
else:
    np_fp4_e2m1 = en_dtypes.float4_e2m1
    np_mx_scale = en_dtypes.float8_e8m0
np_bfloat16 = ml_dtypes.bfloat16
np_fp8_e4m3 = ml_dtypes.float8_e4m3fn
np_fp8_e5m2 = ml_dtypes.float8_e5m2
MXFP4_DST_DTYPE = 40
FP8_E4M3FN_DST_DTYPE = 36
FP8_E5M2_DST_DTYPE = 35
_MX_DTYPE_MAP = {
    "float4_e2m1": np_fp4_e2m1,
    "float8_e4m3fn": np_fp8_e4m3,
    "float8_e5m2": np_fp8_e5m2,
}
_Y_DTYPE_TO_MX = {40: "float4_e2m1", 36: "float8_e4m3fn", 35: "float8_e5m2"}
_MAX_VALUE = {"float4_e2m1": 6, "float8_e4m3fn": 448, "float8_e5m2": 57344}


def _as_numpy(value):
    if value is None or isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        try:
            return tensor.numpy()
        except TypeError:
            import torch

            dtype = {
                "bfloat16": np_bfloat16,
                "float8_e4m3fn": np_fp8_e4m3,
                "float8_e5m2": np_fp8_e5m2,
                "float8_e8m0fnu": np_mx_scale,
            }[str(tensor.dtype).split(".")[-1]]
            raw = torch.int16 if np.dtype(dtype).itemsize == 2 else torch.uint8
            return tensor.view(raw).numpy().view(dtype)
    return np.asarray(value)


def _unpack_fp4(packed):
    """Return two logical FP4 values per byte, low nibble first."""
    codes = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    return codes.view(np_fp4_e2m1)


def _logical_fp4(value, golden):
    """Decode a packed actual only when the reference uses the logical FP4 dtype."""
    array = _as_numpy(value)
    if array is None or not isinstance(golden, np.ndarray):
        return array
    if array.dtype != np.dtype(np.uint8) or "float4_e2m1" not in golden.dtype.name:
        return array
    if (
        array.ndim == 0
        or array.shape[:-1] != golden.shape[:-1]
        or array.shape[-1] * 2 != golden.shape[-1]
    ):
        return array
    return _unpack_fp4(array)


def _first_differences(mismatch):
    flat = mismatch.reshape(-1)
    indices = []
    for start in range(0, flat.size, 65536):
        positions = np.flatnonzero(flat[start : start + 65536])[: 8 - len(indices)]
        indices.extend((positions + start).tolist())
        if len(indices) == 8:
            break
    return [
        list(map(int, np.unravel_index(index, mismatch.shape))) for index in indices
    ]


def _result(
    mismatch, *, allowed_per_thousand=0, checked_elements=None, reason="", **metrics
):
    count = int(np.count_nonzero(mismatch))
    size = mismatch.size if checked_elements is None else checked_elements
    result = {
        "pass": count * 1000 <= allowed_per_thousand * size,
        "precision": 100.0 * (size - count) / size if size else 100.0,
        "metrics": {
            "checked_elements": int(size),
            "bad_elements": count,
            "allowed_bad_fraction": allowed_per_thousand / 1000,
            **metrics,
        },
    }
    if not result["pass"]:
        result["error_info"] = (
            reason or "RotateDualLevelQuant output exceeds its tolerance"
        )
        result["diff_indices"] = _first_differences(mismatch)
    return result


def _shape_dtype_failure(output, golden):
    if output.shape == golden.shape and output.dtype == golden.dtype:
        return None
    return {
        "pass": False,
        "precision": 0.0,
        "error_info": (
            f"shape/dtype mismatch: actual={output.shape}/{output.dtype}, "
            f"golden={golden.shape}/{golden.dtype}"
        ),
    }


@lru_cache(maxsize=4)
def _finite_values(dtype_name):
    """Sorted representable values; adjacent entries are one target-format ULP apart."""
    if dtype_name in ("float4_e2m1", "float4_e2m1fn"):
        positive = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    elif dtype_name in ("float8_e4m3fn", "float8_e5m2"):
        mantissa_bits, bias, max_code = (
            (3, 7, 126) if dtype_name == "float8_e4m3fn" else (2, 15, 123)
        )
        code = np.arange(max_code + 1, dtype=np.int32)
        exponent = code >> mantissa_bits
        fraction = (code & ((1 << mantissa_bits) - 1)) / float(1 << mantissa_bits)
        positive = np.where(
            exponent == 0,
            fraction * 2.0 ** (1 - bias),
            (1 + fraction) * np.exp2(exponent - bias),
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported RotateDualLevelQuant output dtype: {dtype_name}")
    return np.concatenate((-positive[:0:-1], positive))


def compare_quantized(output, golden, compare_context=None):
    """Allow at most one in 1,000 logical y elements to exceed one ULP."""
    output, golden = _as_numpy(output), _as_numpy(golden)
    output = _logical_fp4(output, golden)
    failure = _shape_dtype_failure(output, golden)
    if failure is not None:
        return failure
    if golden.dtype == np.uint8:
        output, golden = _unpack_fp4(output), _unpack_fp4(golden)
    values = _finite_values(golden.dtype.name)
    actual, expected = output.astype(np.float32), golden.astype(np.float32)
    # Single-sided NaN and differing infinities always fail, regardless of tolerance.
    special_mismatch = (np.isnan(actual) != np.isnan(expected)) | (
        (np.isinf(actual) | np.isinf(expected)) & (actual != expected)
    )
    if np.any(special_mismatch):
        return _result(special_mismatch, reason="Y NaN/Inf positions or signs differ")
    finite = np.isfinite(actual) & np.isfinite(expected)
    actual_rank = np.searchsorted(values, actual[finite])
    expected_rank = np.searchsorted(values, expected[finite])
    mismatch = np.zeros(actual.shape, dtype=bool)
    distances = np.abs(actual_rank - expected_rank)
    mismatch[finite] = distances > 1
    return _result(
        mismatch,
        allowed_per_thousand=1,
        reason="Y exceeds one adjacent representable value",
        max_adjacent_distance=int(distances.max()) if distances.size else 0,
    )


def compare_scale(output, golden, *, has_padding=False):
    """Compare finite E8M0 exponent codes and check NaN and padding separately."""
    output, golden = _as_numpy(output), _as_numpy(golden)
    failure = _shape_dtype_failure(output, golden)
    if failure is not None:
        return failure
    if "float8_e8m0" not in golden.dtype.name or golden.dtype.itemsize != 1:
        raise ValueError("RotateDualLevelQuant scales must retain their E8M0 dtype")
    actual, expected = output.view(np.uint8), golden.view(np.uint8)
    checked_elements = expected.size
    if has_padding:
        # Each row with an odd number of MX groups ends in one fixed zero byte.
        pad = (..., -1, 1)
        bad_padding = (actual[pad] != 0) | (expected[pad] != 0)
        if np.any(bad_padding):
            mismatch = np.zeros(expected.shape, dtype=bool)
            mismatch[pad] = bad_padding
            return _result(mismatch, reason="level1_scale padding must have code 0")
        checked_elements -= bad_padding.size

    nan_mismatch = (actual == 255) != (expected == 255)
    if np.any(nan_mismatch):
        return _result(
            nan_mismatch,
            checked_elements=checked_elements,
            reason="level1_scale NaN positions differ",
        )
    distances = np.abs(np.subtract(actual, expected, dtype=np.int16))
    return _result(
        distances > 1,
        allowed_per_thousand=1,
        checked_elements=checked_elements,
        reason="level1_scale exceeds one adjacent E8M0 code",
        max_adjacent_distance=int(distances.max()) if distances.size else 0,
    )


def grouped_ranges(group_list, group_list_type, row_count, group_count=None):
    """Validate count/cumsum metadata and return each group's [start,end) rows."""
    groups = _as_numpy(group_list)
    if groups.ndim != 1 or groups.dtype != np.dtype("int64"):
        raise ValueError("group_list must be a one-dimensional int64 tensor")
    if not 1 <= groups.size <= 1024 or (
        group_count is not None and groups.size != group_count
    ):
        raise ValueError("group count must be in [1,1024] and match rotation dim0")
    if group_list_type not in (0, 1):
        raise ValueError("group_list_type must be 0 or 1")
    ranges = []
    start = 0
    for value in groups:
        value = int(value)
        end = value if group_list_type == 0 else start + value
        if value < 0 or end < start or end > row_count:
            raise ValueError("Invalid group boundary")
        ranges.append((start, end))
        start = end
    return ranges


def _mx_scale_and_reciprocal(group_max, mx_ele_dtype, scale_alg, dst_type_max):
    """Mirror ScaleOcpVf / ScaleDynVf / ScaleCublasVf in arch35/mx_quant.h.

    The reciprocal is a BF16 *bit pattern*, not generally 1 / decoded E8M0.
    FP4 OCP clears the reciprocal for a zero source exponent. FP8 OCP and
    the 6/7 dynamic rule clear it for a zero shared exponent. OCP/Dyn
    special-case code 254; cuBLAS uses unsigned subtraction at both endpoints.
    """
    max_bits = group_max.astype(np_bfloat16).view(np.uint16)
    max_exp = max_bits & np.uint16(0x7F80)
    finite = max_exp != 0x7F80
    if scale_alg == 0 or (scale_alg == 2 and dst_type_max in (6, 7)):
        if scale_alg == 0:
            emax = {"float4_e2m1": 2, "float8_e4m3fn": 8, "float8_e5m2": 15}[
                mx_ele_dtype
            ]
            shared_bits = np.maximum(max_exp, emax << 7) - (emax << 7)
        else:
            add_bits = 0x1F if dst_type_max == 7 else 0x3F
            rounded = (max_bits.astype(np.uint32) + add_bits) & 0x7F80
            rounded = np.where(max_exp < 0x0100, 0x0100, rounded)
            shared_bits = rounded - 0x0100
        codes = np.where(finite, shared_bits >> 7, 255).astype(np.uint8)
        reciprocal = (0x7F00 - shared_bits).astype(np.uint16)
        reciprocal = np.where(finite, reciprocal, 0x7F81)
        exponent_for_zero = (
            max_exp if scale_alg == 0 and mx_ele_dtype == "float4_e2m1" else shared_bits
        )
        reciprocal = np.where(exponent_for_zero != 0, reciprocal, 0)
        reciprocal = np.where(shared_bits == 0x7F00, 0x0040, reciprocal)
    else:
        # Configurable FP4 division and FP8 reciprocal multiplication have
        # distinct FP32 rounding contracts at exponent boundaries.
        max_norm = {"float8_e4m3fn": 448, "float8_e5m2": 57344}.get(
            mx_ele_dtype, dst_type_max
        )
        divisor = np.float32(max_norm)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            maximum = group_max.astype(np.float32)
            if mx_ele_dtype == "float4_e2m1":
                scaled_max = maximum / divisor
            else:
                scaled_max = maximum * (np.float32(1) / divisor)
        bits = scaled_max.view(np.uint32)
        exponent = bits >> 23
        mantissa = bits & 0x007FFFFF
        round_up = (bits > 0x00400000) & (bits < 0x7F000000) & (mantissa != 0)
        if mx_ele_dtype == "float4_e2m1":
            round_up &= (
                exponent > 0
            )  # RotateDualLevelQuant FP4 does not round subnormal scales up.
        code = exponent + round_up.astype(np.uint32)
        code = np.where(finite, code, 255)
        code = np.where(max_bits != 0, code, 0)
        codes = code.astype(np.uint8)
        reciprocal = (0x7F00 - (code << 7)).astype(np.uint16)
        reciprocal = np.where(finite, reciprocal, 0x7F81)
        reciprocal = np.where(max_bits != 0, reciprocal, 0)
    return codes, reciprocal.astype(np.uint16).view(np_bfloat16).astype(np.float32)


def _quantize_fp4(values, round_mode):
    """BF16 -> FP4 with the kernel's RINT / ROUND / FLOOR conversion."""
    values = np.clip(values, -6, 6)
    finite_abs = np.where(np.isfinite(values) & (values != 0), np.abs(values), 1)
    exponent = np.maximum(np.floor(np.log2(finite_abs)), 0)
    step = np.exp2(exponent - 1)
    units = values / step
    if round_mode == "rint":
        rounded = np.rint(units)
    elif round_mode == "round":
        rounded = np.copysign(np.floor(np.abs(units) + 0.5), units)
    else:
        rounded = np.floor(units)
    # FP4 has no NaN encoding; keep the existing zero convention for NaNs.
    return np.nan_to_num(np.clip(rounded * step, -6, 6), nan=0).astype(np_fp4_e2m1)


def mx_quantize(
    fp_array: np.ndarray,
    mx_ele_dtype="float4_e2m1",
    axis=-1,
    block_size=32,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0,
):
    """Reference the BF16 RotateDualLevelQuant epilogue, including E8M0 code padding.

    Inputs are narrowed to the Fixpipe BF16 format before the vector stages.
    All paths multiply by the BF16 reciprocal and round the product to BF16
    before the final FP4 or saturating FP8 conversion.
    """
    if not isinstance(fp_array, np.ndarray) or fp_array.ndim == 0:
        raise ValueError("MX input must be an ndarray with at least one dimension")
    if mx_ele_dtype not in _MX_DTYPE_MAP:
        raise ValueError(f"Unsupported MX dtype: {mx_ele_dtype}")
    if axis not in (-1, fp_array.ndim - 1) or block_size != 32:
        raise ValueError(
            "RotateDualLevelQuant requires the last axis and block_size=32"
        )
    if round_mode not in ("rint", "round", "floor"):
        raise ValueError(f"Unsupported RotateDualLevelQuant round_mode: {round_mode}")
    allowed = (0, 2) if mx_ele_dtype == "float4_e2m1" else (0, 1)
    if scale_alg not in allowed:
        raise ValueError(f"Unsupported scale_alg={scale_alg} for {mx_ele_dtype}")
    if mx_ele_dtype != "float4_e2m1" and round_mode != "rint":
        raise ValueError("RotateDualLevelQuant FP8 only supports rint")
    # ACLNN validates the caller's double before the L0 FP32 attribute conversion.
    if not np.isfinite(dst_type_max) or not (
        6 <= dst_type_max <= 12 if scale_alg == 2 else dst_type_max == 0
    ):
        raise ValueError("dst_type_max must be in [6,12] for scale_alg=2, otherwise 0")
    dst_type_max = float(np.float32(dst_type_max))

    shape = fp_array.shape
    n = shape[-1]
    groups = (n + 31) // 32
    values = fp_array.astype(np_bfloat16).astype(np.float32)
    if n % 32:
        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, groups * 32 - n)])
    blocks = values.reshape(shape[:-1] + (groups, 32))
    group_max = np.max(np.abs(blocks), axis=-1, keepdims=True)
    codes, reciprocal = _mx_scale_and_reciprocal(
        group_max, mx_ele_dtype, scale_alg, dst_type_max
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = (blocks * reciprocal).astype(np_bfloat16).astype(np.float32)
    if mx_ele_dtype == "float4_e2m1":
        quantized = _quantize_fp4(scaled, round_mode)
    else:
        max_norm = 448 if mx_ele_dtype == "float8_e4m3fn" else 57344
        # The kernel uses SatMode::SAT: infinities clamp and NaNs become zero.
        saturated = np.nan_to_num(scaled, nan=0, posinf=max_norm, neginf=-max_norm)
        quantized = np.clip(saturated, -max_norm, max_norm).astype(
            _MX_DTYPE_MAP[mx_ele_dtype]
        )
    y = quantized.reshape(shape[:-1] + (groups * 32,))[..., :n]
    codes = codes.squeeze(-1)
    if groups % 2:
        codes = np.pad(codes, [(0, 0)] * (codes.ndim - 1) + [(0, 1)], constant_values=0)
    scale = codes.reshape(shape[:-1] + ((groups + 1) // 2, 2)).view(np_mx_scale)
    return y, scale


def _apply_clamp(y_bf16, alpha):
    """alpha 在 (0.0, 1.0) 开区间内时, 按 32 一组 clamp 到 ±alpha*groupMax (bf16 域)."""
    a = float(np.asarray(alpha).reshape(-1)[0])
    if not (0.0 < a < 1.0):
        return y_bf16
    orig_shape = y_bf16.shape
    n = orig_shape[-1]
    rows = int(np.prod(orig_shape[:-1]))
    y = y_bf16.astype(np.float32).reshape(rows, n)
    g = -(-n // 32)
    y_pad = y.reshape(rows * g, 32)
    group_max = np.max(np.abs(y_pad), axis=-1, keepdims=True)
    limit = (group_max * np.float32(a)).astype(np_bfloat16).astype(np.float32)
    y_pad = np.clip(y_pad, -limit, limit)
    return y_pad.reshape(orig_shape).astype(np_bfloat16)


def level0_quantize(values, mx_ele_dtype):
    """One FP32 amax/CMAX scale per original row and 512 columns.

    Ignore nonfinite inputs only in amax. FP4 uses the rounded FP32 reciprocal
    of 6. FP8 divides by CMAX in FP32. Flush subnormal scales, normalize the
    original values, narrow to BF16 (RNE), and restore signed zeros.
    """
    values = _as_numpy(values).astype(np_bfloat16)
    n = values.shape[-1]
    rows = int(np.prod(values.shape[:-1]))
    blocks = (n + 511) // 512
    data = values.astype(np.float32).reshape(rows, n)
    normalized = np.empty_like(values).reshape(rows, n)
    scale = np.empty((rows, blocks), dtype=np.float32)
    for block in range(blocks):
        start, end = block * 512, min(n, (block + 1) * 512)
        original = data[:, start:end]
        maximum = np.max(np.where(np.isfinite(original), np.abs(original), 0), axis=-1)
        with np.errstate(under="ignore"):
            cmax = np.float32(_MAX_VALUE[mx_ele_dtype])
            if mx_ele_dtype == "float4_e2m1":
                s0 = maximum * (np.float32(1) / cmax)
            else:
                s0 = maximum / cmax
        s0 = np.where(s0 < np.finfo(np.float32).tiny, np.float32(0), s0)
        scale[:, block] = s0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = (original / s0[:, None]).astype(np_bfloat16)
        normalized[:, start:end] = np.where(original == 0, original, result).astype(
            np_bfloat16
        )
    return normalized.reshape(values.shape), scale.reshape(
        values.shape[:-1] + (blocks,)
    )


def _validate(
    x,
    rotation,
    alpha,
    group_list,
    group_list_type,
    axis,
    trans,
    level0_block_size,
    level1_block_size,
):
    if not 1 <= x.ndim <= 7 or x.dtype not in (
        np.dtype(np.float16),
        np.dtype(np_bfloat16),
        np.dtype(np.float32),
    ):
        raise ValueError(
            "x must have rank 1..7 and dtype FP16/BF16 (or promoted float32)"
        )
    if rotation.dtype != x.dtype or rotation.ndim not in (
        (2, 3) if group_list is None else (3, 4)
    ):
        raise ValueError("rotation dtype/rank does not match the mode")
    k, n = rotation.shape[-1], x.shape[-1]
    if k not in (32, 64, 128, 256) or rotation.shape[-2] != k or n % k:
        raise ValueError("rotation must be square with supported K dividing N")
    if (
        rotation.ndim == (3 if group_list is None else 4)
        and rotation.shape[-3] != n // k
    ):
        raise ValueError("per-column rotation count must equal N/K")
    if axis not in (-1, x.ndim - 1) or trans or group_list_type not in (0, 1):
        raise ValueError("last axis, trans=False and group_list_type=0/1 required")
    if (level0_block_size, level1_block_size) != (512, 32):
        raise ValueError("level0_block_size=512 and level1_block_size=32 required")
    if alpha is not None and (
        alpha.shape != (1,)
        or alpha.dtype not in (np.dtype(np_bfloat16), np.dtype(np.float32))
    ):
        raise ValueError("alpha must be BF16 [1] (or promoted float32)")
    if group_list is not None and (x.ndim != 2 or n == 0):
        raise ValueError("grouped input requires x[M,N], N>0")


def _rotate(x, rotation, alpha):
    k, n = rotation.shape[-1], x.shape[-1]
    rows = int(np.prod(x.shape[:-1]))
    x64 = x.astype(np.float64, copy=False)
    rotation64 = rotation.astype(np.float64, copy=False)

    if rotation.ndim == 2:
        result = np.matmul(x64.reshape(rows * (n // k), k), rotation64)
    else:
        result = np.matmul(x64.reshape(rows, n // k, 1, k), rotation64[None])

    # Round the reference dot product to FP32 before the Fixpipe BF16 cast.
    result = result.reshape(x.shape).astype(np.float32).astype(np_bfloat16)
    return _apply_clamp(result, alpha) if alpha is not None else result


def compute_rotate_dual_level_quant(
    x,
    rotation,
    alpha=None,
    group_list=None,
    group_list_type=0,
    mx_ele_dtype="float4_e2m1",
    axis=-1,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    trans=False,
    level0_block_size=512,
    level1_block_size=32,
    initial_y=None,
    initial_level0_scale=None,
    initial_level1_scale=None,
):
    # Reject oversized strided views before converting them to contiguous CPU arrays.
    shape = np.shape(x)
    int32_max = np.iinfo(np.int32).max
    if max(shape, default=0) > int32_max or prod(shape[:-1]) > int32_max:
        raise ValueError("x dimensions and flattened M must fit INT32")
    x, rotation, alpha = map(_as_numpy, (x, rotation, alpha))
    _validate(
        x,
        rotation,
        alpha,
        group_list,
        group_list_type,
        axis,
        trans,
        level0_block_size,
        level1_block_size,
    )
    # Validate all quantization attributes on a zero-row tensor before early exits.
    mx_quantize(
        np.empty((0, 32), dtype=np_bfloat16),
        mx_ele_dtype,
        round_mode=round_mode,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )
    n = x.shape[-1]
    shapes = (
        x.shape,
        x.shape[:-1] + ((n + 511) // 512,),
        x.shape[:-1] + ((n + 63) // 64, 2),
    )
    dtypes = (_MX_DTYPE_MAP[mx_ele_dtype], np.float32, np_mx_scale)
    outputs = []
    for initial, shape, dtype in zip(
        (initial_y, initial_level0_scale, initial_level1_scale), shapes, dtypes
    ):
        array = (
            np.ones(shape, dtype=dtype)
            if initial is None
            else _as_numpy(initial).copy()
        )
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError("Initial output shape/dtype mismatch")
        outputs.append(array)
    if group_list is None:
        ranges = [(0, int(np.prod(x.shape[:-1])))]
    else:
        ranges = grouped_ranges(
            group_list, group_list_type, x.shape[0], rotation.shape[0]
        )
    if x.size == 0:
        return outputs
    x_rows = x.reshape(-1, n)
    y, level0, level1 = [
        value.reshape((-1,) + shape[len(x.shape) - 1 :])
        for value, shape in zip(outputs, shapes)
    ]
    for group, (start, end) in enumerate(ranges):
        if start == end:
            continue
        rotated = _rotate(
            x_rows[start:end],
            rotation if group_list is None else rotation[group],
            alpha,
        )
        normalized, level0[start:end] = level0_quantize(rotated, mx_ele_dtype)
        y[start:end], level1[start:end] = mx_quantize(
            normalized,
            mx_ele_dtype,
            round_mode=round_mode,
            scale_alg=scale_alg,
            dst_type_max=dst_type_max,
        )
    return outputs


def compare_level0(output, golden):
    output, golden = _as_numpy(output), _as_numpy(golden)
    failure = _shape_dtype_failure(output, golden)
    if failure is not None:
        return failure
    if golden.dtype != np.dtype(np.float32):
        raise ValueError("level0_scale must use FP32")
    return _result(
        ~np.isclose(output, golden, rtol=1e-3, atol=1e-8, equal_nan=True),
        allowed_per_thousand=1,
        reason="level0_scale exceeds the 0.1% mismatch budget (rtol=1e-3, atol=1e-8)",
    )


def _covered_rows(compare_context, rows):
    inputs = getattr(compare_context, "input_tensors", ())
    if inputs is None or len(inputs) not in (4, 7) or inputs[3] is None:
        return None
    attrs = getattr(compare_context, "attributes", {})
    mode = attrs.get("group_list_type", attrs.get("groupListType", 0))
    return grouped_ranges(inputs[3], mode, rows)[-1][1]


def compare_outputs(*outputs, compare_context=None, preserve_tail=True, **kwargs):
    """TTK order: y,s0,s1,golden_y,golden_s0,golden_s1."""
    if len(outputs) != 6:
        raise ValueError("Expected three actual and three golden outputs")
    actual = list(map(_as_numpy, outputs[:3]))
    golden = list(map(_as_numpy, outputs[3:]))
    actual[0] = _logical_fp4(actual[0], golden[0])
    covered = _covered_rows(compare_context, golden[0].shape[0])
    valid = slice(None) if covered is None else slice(0, covered)
    logical_n = golden[0].shape[-1] * (2 if golden[0].dtype == np.uint8 else 1)
    scale_groups = (logical_n + 31) // 32

    def compare_rows(a, g, compare, **options):
        failure = _shape_dtype_failure(a, g)
        if failure is not None:
            return failure
        result = compare(a[valid], g[valid], **options)
        if covered is not None and preserve_tail:
            tail_a = np.ascontiguousarray(a[covered:]).view(np.uint8)
            tail_g = np.ascontiguousarray(g[covered:]).view(np.uint8)
            tail = _result(
                tail_a != tail_g, reason="Uncovered output rows were modified"
            )
            result.setdefault("metrics", {})["tail_bad_bytes"] = tail["metrics"][
                "bad_elements"
            ]
            if not tail["pass"]:
                result.update(
                    {
                        "pass": False,
                        "precision": min(result["precision"], tail["precision"]),
                        "error_info": tail["error_info"],
                    }
                )
        return result

    return [
        compare_rows(actual[0], golden[0], compare_quantized),
        compare_rows(actual[1], golden[1], compare_level0),
        compare_rows(
            actual[2], golden[2], compare_scale, has_padding=scale_groups % 2 == 1
        ),
    ]


class RotateDualLevelQuantTestSpec:
    compare = staticmethod(compare_outputs)

    @staticmethod
    def golden(
        x,
        rotation,
        alpha=None,
        group_list=None,
        *,
        y_dtype=40,
        axis=-1,
        round_mode="rint",
        scale_alg=0,
        dst_type_max=0.0,
        trans=False,
        group_list_type=0,
        level0_block_size=512,
        level1_block_size=32,
        **kwargs,
    ):
        if int(y_dtype) not in _Y_DTYPE_TO_MX:
            raise ValueError("Unsupported RotateDualLevelQuant y_dtype")
        return compute_rotate_dual_level_quant(
            x,
            rotation,
            alpha,
            group_list,
            group_list_type,
            _Y_DTYPE_TO_MX[int(y_dtype)],
            axis,
            round_mode,
            scale_alg,
            dst_type_max,
            trans,
            level0_block_size,
            level1_block_size,
            kwargs.get("initial_y"),
            kwargs.get("initial_level0_scale"),
            kwargs.get("initial_level1_scale"),
        )


__spec__ = {"rotate_dual_level_quant": "RotateDualLevelQuantTestSpec"}
