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
"""Torch TTK reference adapter; it does not register or implement a Torch op."""

import os
import sys
from operator import index

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # NumPy-only replay uses integer dtype IDs.

sys.path.insert(0, os.path.dirname(__file__))
import rotate_dual_level_quant_kernel_golden as _kernel

_DST_TO_MX = {
    296: "float4_e2m1",
    23: "float8_e5m2",
    24: "float8_e4m3fn",
    291: "float8_e5m2",
    292: "float8_e4m3fn",
}


def _schema_int(value):
    # The Torch dispatcher also accepts a scalar tensor for an int argument.
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value)
    return index(value)


def _mx_dtype(dst_dtype):
    if dst_dtype is None:
        return "float4_e2m1"
    if torch is not None and isinstance(dst_dtype, torch.dtype):
        name = str(dst_dtype).rsplit(".", 1)[-1]
        if name in ("float8_e5m2", "float8_e4m3fn"):
            return name
    try:
        return _DST_TO_MX[_schema_int(dst_dtype)]
    except (KeyError, TypeError) as error:
        raise ValueError("Expected an MX FP4/FP8 dst_dtype") from error


class CannOpsNnRotateDualLevelQuantTestSpec:
    """Use the Torch output layout, including two FP4 elements per uint8 byte."""

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
        level0_block_size=512,
        level1_block_size=32,
        **kwargs,
    ):
        if isinstance(round_mode, bytes):
            round_mode = round_mode.decode()
        if torch is not None and isinstance(dst_type_max, torch.Tensor):
            dst_type_max = dst_type_max.item()
        y, level0_scale, level1_scale = _kernel.compute_rotate_dual_level_quant(
            x,
            rotation,
            alpha,
            group_list,
            _schema_int(group_list_type),
            _mx_dtype(dst_dtype),
            -1 if axis is None else _schema_int(axis),
            "rint" if round_mode is None else round_mode,
            0 if scale_alg is None else _schema_int(scale_alg),
            0.0 if dst_type_max is None else dst_type_max,
            transpose_y is not None and transpose_y != 0,
            _schema_int(level0_block_size),
            _schema_int(level1_block_size),
        )
        if y.dtype == np.dtype(_kernel.np_fp4_e2m1):
            codes = y.view(np.uint8)
            y = (codes[..., 0::2] & 0x0F) | (codes[..., 1::2] << 4)
        return [y, level0_scale, level1_scale]

    @staticmethod
    def compare(
        y,
        level0_scale,
        level1_scale,
        golden_y,
        golden_level0_scale,
        golden_level1_scale,
        *,
        compare_context=None,
        **kwargs,
    ):
        y, golden_y = _kernel._as_numpy(y), _kernel._as_numpy(golden_y)
        # Only legacy logical FP4 goldens may receive TTK's flattened unpacked y.
        if (
            golden_y.dtype == np.dtype(_kernel.np_fp4_e2m1)
            and y.dtype == golden_y.dtype
            and y.ndim == 1
            and y.size == golden_y.size
        ):
            y = y.reshape(golden_y.shape)
        # Torch creates empty outputs; uncovered rows have no specified initial values.
        return _kernel.compare_outputs(
            y,
            level0_scale,
            level1_scale,
            golden_y,
            golden_level0_scale,
            golden_level1_scale,
            compare_context=compare_context,
            preserve_tail=False,
        )


__spec__ = {
    "torch.ops.cann_ops_nn.rotate_dual_level_quant": "CannOpsNnRotateDualLevelQuantTestSpec"
}
