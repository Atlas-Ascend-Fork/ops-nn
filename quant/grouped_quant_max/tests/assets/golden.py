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

__golden__ = {"kernel": {"grouped_quant_max": "grouped_quant_max_golden"}}


DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "double",
    12: "bool",
    16: "complex64",
    17: "complex128",
    18: "qint8",
    19: "qint16",
    20: "qint32",
    21: "quint8",
    22: "quint16",
    23: "resource",
    25: "dual",
    26: "variant",
    27: "bfloat16",
    29: "int4",
    30: "uint1",
    31: "int2",
    32: "uint2",
    33: "complex32",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
    42: "hifloat4",
}

DST_TYPE_HIFLOAT8 = 34
DST_TYPE_FLOAT8_E5M2 = 35
DST_TYPE_FLOAT8_E4M3FN = 36


def _dst_dtype(dst_type):
    """Map the dst_type attribute to the numpy dtype object of output y."""
    if dst_type == DST_TYPE_HIFLOAT8:
        from en_dtypes import hifloat8

        return np.dtype(hifloat8)
    if dst_type == DST_TYPE_FLOAT8_E5M2:
        from ml_dtypes import float8_e5m2

        return np.dtype(float8_e5m2)
    if dst_type == DST_TYPE_FLOAT8_E4M3FN:
        from ml_dtypes import float8_e4m3fn

        return np.dtype(float8_e4m3fn)
    raise ValueError(
        f"Unsupported dst_type {dst_type}, only 34(hifloat8)/35(float8_e5m2)/"
        f"36(float8_e4m3fn) are supported."
    )


def _max_norm(dst_type):
    """Max finite magnitude of the target format (saturation bound, SAT mode)."""
    if dst_type == DST_TYPE_HIFLOAT8:
        return float.fromhex("0x1.p15")  # 32768
    if dst_type == DST_TYPE_FLOAT8_E5M2:
        return float.fromhex("0x1.Cp15")  # 57344
    if dst_type == DST_TYPE_FLOAT8_E4M3FN:
        return float.fromhex("0x1.Cp8")  # 448
    raise ValueError(f"Unsupported dst_type {dst_type}.")


def _cast_to_dst(data_fp32, dst_type):
    """Saturating (SAT) cast of fp32 values to the target low-precision format.

    Mirrors the device side Cast with SatMode::SAT: values outside the finite
    representable range are clamped to the format max before the
    round-to-nearest-even dtype conversion (CAST_RINT). NaN maps to 0.

    Note (hifloat8 / dst_type 34):
        This repository has no dedicated hybrid conversion helper (there is no
        `libs.generate_hifloat8_data` / `trans_np_float_tensor_to_hifuint8`).
        Therefore hifloat8 is cast with `astype(hifloat8)` for both "rint" and
        "hybrid" round modes, i.e. hybrid shares the rint path here. Exactly
        reproducing the per-value hybrid conversion requires the upstream
        conversion implementation; this is flagged as an open item.
    """
    max_norm = _max_norm(dst_type)
    data_fp32 = np.clip(data_fp32, -max_norm, max_norm)
    data_fp32 = np.nan_to_num(data_fp32, nan=0.0, copy=False)
    return data_fp32.astype(_dst_dtype(dst_type), copy=False)


def _grouped_quant_max_core(x, scale, group_list, dst_type):
    """CPU reference for GroupedQuantMax.

    group_list is a cumulative (cumsum) offset over dim0: group g covers rows
    [group_list[g-1], group_list[g]) with group_list[-1] == x.shape[0].
    y = Cast(x_g * scale[g], dst_type); amax[g] = max(|x_g|).

    Semantics mirror the aclnn reference implementation:
      * empty x early-exits before any group_list validation / per-group max;
      * amax[g] = max(|x|) over the group, cast back to x.dtype (empty group -> 0);
      * y = SAT-cast(x_fp32 * repeat(scale, group_sizes)), computed in one pass.
    """
    x = np.asarray(x)
    x_dtype = x.dtype
    x_fp32 = x.astype(np.float32)

    scale_f32 = np.asarray(scale, dtype=np.float32).reshape(-1)
    group_list = np.asarray(group_list).reshape(-1).astype(np.int64)
    num_groups = group_list.size

    # Empty tensor: must early-exit before the per-group np.max (otherwise
    # np.max on a zero-size array raises ValueError). amax keeps one zero
    # entry per group, y is the empty SAT-cast result.
    if x_fp32.size == 0:
        y = _cast_to_dst(x_fp32, dst_type)
        amax = np.zeros(num_groups, dtype=x_dtype)
        return y, amax

    # Validation order aligned with the reference:
    # num_groups == 0 -> last element -> monotonic -> scale length.
    if num_groups == 0:
        raise RuntimeError("group_list cannot be empty")
    if group_list[-1] != x.shape[0]:
        raise RuntimeError(
            "The last element of group_list must equal x.shape[0], but got "
            f"{group_list[-1]} vs {x.shape[0]}."
        )
    if np.any(np.diff(group_list) < 0):
        raise RuntimeError("group_list must be monotonically non-decreasing.")
    if scale_f32.shape[0] != num_groups:
        raise RuntimeError(
            f"scale length ({scale_f32.shape[0]}) must equal num_groups ({num_groups})."
        )

    # Per-group amax: max(|x_f32|) rounded back to x.dtype. Empty group -> 0.
    amax = np.zeros((num_groups,), dtype=x_dtype)
    prev = 0
    for g in range(num_groups):
        end = int(group_list[g])
        start = prev
        prev = end
        if end <= start:
            continue
        seg = x_fp32[start:end]
        amax[g] = np.asarray(np.max(np.abs(seg))).astype(x_dtype)

    # Quantisation: broadcast the per-group scale to every row, multiply in
    # fp32, then a single SAT-cast to the target format.
    group_sizes = np.diff(group_list, prepend=np.int64(0))
    broadcast_scale = np.repeat(scale_f32, group_sizes)
    broadcast_scale = broadcast_scale.reshape(-1, *([1] * (x_fp32.ndim - 1)))
    scaled = x_fp32 * broadcast_scale
    y = _cast_to_dst(scaled, dst_type)

    return y, amax


def grouped_quant_max_golden(
    x, scale, group_list, round_mode="rint", dst_type=35, **kwargs
):
    """
    Golden function for grouped_quant_max.
    All the parameters (names and order) follow @grouped_quant_max_def.cpp without
    outputs. All the input Tensors are numpy.ndarray.

    Args:
        x: input tensor, FLOAT/FLOAT16/BFLOAT16, 2-8 dims.
        scale: per-group scaling factor, FLOAT, shape [num_groups].
        group_list: cumulative (cumsum) offsets over dim0, INT64, shape [num_groups].
        round_mode: string attr, "rint" (dst_type 35/36) or "round"/"hybrid" (dst_type 34).
        dst_type: int attr, 34=HIFLOAT8, 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN.
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        (y, amax): y has the same shape as x with dtype determined by dst_type;
                   amax has shape [num_groups] with the same dtype as x.
    """
    # round_mode is intentionally not branched on: 35/36 always use CAST_RINT,
    # and hifloat8 (34) falls back to astype(hifloat8) for both "rint" and
    # "hybrid" (no hybrid-specific conversion exists in this repository).
    del round_mode
    y, amax = _grouped_quant_max_core(x, scale, group_list, dst_type)
    return y, amax
