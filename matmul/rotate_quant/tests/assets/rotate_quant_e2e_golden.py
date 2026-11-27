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
"""Ascend 950 references for ordinary and grouped Torch RotateQuant APIs."""

import os
import sys
from operator import index

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from rotate_quant_kernel_golden import (
    _as_numpy,
    _unpack_fp4,
    compare_grouped_outputs,
    compute_grouped_rotate_mx,
    compute_rotate_mx,
    np_fp4_e2m1,
    np_mx_scale,
)

_DST_TO_MX = {
    torch.float8_e5m2: "float8_e5m2",
    torch.float8_e4m3fn: "float8_e4m3fn",
    296: "float4_e2m1",
    23: "float8_e5m2",
    24: "float8_e4m3fn",
    291: "float8_e5m2",
    292: "float8_e4m3fn",
}


def _schema_int(value):
    # The Torch dispatcher also accepts a scalar tensor for an int argument.
    return int(value) if isinstance(value, torch.Tensor) else index(value)


def _mx_dtype(dst_dtype):
    try:
        key = (
            dst_dtype if isinstance(dst_dtype, torch.dtype) else _schema_int(dst_dtype)
        )
        return _DST_TO_MX[key]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Ascend 950 E2E golden requires an explicit MX dst_dtype"
        ) from error


def _logical_output(value, golden, name):
    """Validate public Torch dtypes before using byte views for NumPy comparison."""
    if isinstance(value, torch.Tensor):
        dtype_names = {
            "uint8": "uint8",
            "float4_e2m1": "uint8",
            "float4_e2m1fn": "uint8",
            "float8_e4m3fn": "float8_e4m3fn",
            "float8_e5m2": "float8_e5m2",
            "float8_e8m0": "float8_e8m0fnu",
            "float8_e8m0fnu": "float8_e8m0fnu",
        }
        expected = getattr(torch, dtype_names[golden.dtype.name])
        if value.dtype != expected:
            return None, {
                "pass": False,
                "precision": 0.0,
                "error_info": f"{name} dtype mismatch: actual={value.dtype}, expected={expected}",
            }
    array = _as_numpy(value)
    if golden.dtype == np.dtype(np_fp4_e2m1):
        # TTK may replay older logical FP4 goldens with an already unpacked 1-D y.
        if (
            array.dtype == golden.dtype
            and array.ndim == 1
            and array.size == golden.size
        ):
            return array.reshape(golden.shape), None
        if array.dtype == np.uint8 and array.ndim > 0:
            return _unpack_fp4(array), None
    if array.dtype != golden.dtype:
        return None, {
            "pass": False,
            "precision": 0.0,
            "error_info": f"{name} dtype mismatch: actual={array.dtype}, expected={golden.dtype}",
        }
    return array, None


def _compare_torch_outputs(
    y,
    scale,
    golden_y,
    golden_scale,
    *,
    compare_context=None,
):
    golden_y, golden_scale = _as_numpy(golden_y), _as_numpy(golden_scale)
    if golden_y.dtype.name not in (
        "uint8",
        "float4_e2m1",
        "float4_e2m1fn",
        "float8_e4m3fn",
        "float8_e5m2",
    ):
        raise ValueError("Torch golden y must use packed FP4 or native FP4/FP8")
    if golden_scale.dtype != np.dtype(np_mx_scale):
        raise ValueError("Torch golden scale must retain its E8M0 dtype")
    y, y_failure = _logical_output(y, golden_y, "y")
    scale, scale_failure = _logical_output(scale, golden_scale, "scale")
    results = compare_grouped_outputs(
        golden_y if y_failure is not None else y,
        golden_scale if scale_failure is not None else scale,
        golden_y,
        golden_scale,
        compare_context,
        preserve_tail=False,
    )
    for result_idx, failure in enumerate((y_failure, scale_failure)):
        if failure is not None:
            results[result_idx] = failure
    return results


class CannOpsNnRotateQuantTestSpec:
    """Match the Torch output carriers: packed uint8 FP4, native FP8 and E8M0."""

    @staticmethod
    def golden(
        x,
        rotation,
        *,
        alpha=None,
        dst_dtype=None,
        axis=-1,
        round_mode="rint",
        scale_alg=0,
        dst_type_max=0.0,
        transpose_y=False,
        group_list=None,
        group_list_type=0,
        **kwargs,
    ):
        axis = -1 if axis is None else _schema_int(axis)
        if axis not in (-1, x.ndim - 1) or (
            transpose_y is not None and transpose_y != 0
        ):
            raise ValueError("RotateQuant requires the last axis and transpose_y=False")
        group_list_type = _schema_int(group_list_type)
        if group_list_type not in (0, 1):
            raise ValueError("group_list_type must be 0 or 1")
        if isinstance(round_mode, bytes):
            round_mode = round_mode.decode()
        if isinstance(dst_type_max, torch.Tensor):
            dst_type_max = dst_type_max.item()
        quant_options = {
            "mx_ele_dtype": _mx_dtype(dst_dtype),
            "round_mode": "rint" if round_mode is None else round_mode,
            "scale_alg": 0 if scale_alg is None else _schema_int(scale_alg),
            "dst_type_max": 0.0 if dst_type_max is None else dst_type_max,
        }
        if group_list is not None:
            y, scale = compute_grouped_rotate_mx(
                x, rotation, alpha, group_list, group_list_type, **quant_options
            )
        else:
            y, scale = compute_rotate_mx(x, rotation, alpha, **quant_options)
        if y.dtype == np.dtype(np_fp4_e2m1):
            codes = y.view(np.uint8)
            y = (codes[..., 0::2] & 0x0F) | (codes[..., 1::2] << 4)
        return [y, scale]

    @staticmethod
    def compare(y, scale, golden_y, golden_scale, *, compare_context=None, **kwargs):
        return _compare_torch_outputs(
            y,
            scale,
            golden_y,
            golden_scale,
            compare_context=compare_context,
        )


__spec__ = {
    "torch.ops.cann_ops_nn.rotate_quant": "CannOpsNnRotateQuantTestSpec",
}
