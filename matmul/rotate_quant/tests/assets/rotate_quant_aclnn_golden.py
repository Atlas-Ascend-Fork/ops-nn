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
"""
rotate_quant aclnn golden 实现。

aclnn 接口参数名使用 camelCase（与 C API 一致），golden 委托 kernel golden 的 compute 函数。
"""

import os
import sys


sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
sys.path.insert(0, os.path.dirname(__file__))

import rotate_quant_kernel_golden as _kernel  # noqa: E402
from rotate_quant_kernel_golden import compare_outputs  # noqa: E402

compute_rotate_mx = _kernel.compute_rotate_mx

_DTYPE_TO_MX = {
    "float4_e2m1": "float4_e2m1",
    "float4": "float4_e2m1",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
}


class AclnnRotateQuantTestSpec:
    """aclnnRotateQuant golden。参数名与 C API 一致。"""

    @staticmethod
    def compare(*outputs, compare_context=None, **kwargs):
        return compare_outputs(*outputs, compare_context=compare_context, **kwargs)

    @staticmethod
    def golden(
        x,
        rotation,
        alpha=None,
        axis=-1,
        roundMode="rint",
        scaleAlg=0,
        dstTypeMax=0.0,
        trans=False,
        yOut=None,
        scaleOut=None,
        **kwargs,
    ):
        if (axis is not None and axis not in (-1, x.ndim - 1)) or trans:
            raise ValueError("RotateQuant requires the last axis and trans=False")
        out_dtype_str = None
        candidates = [yOut.dtype] if yOut is not None and hasattr(yOut, "dtype") else []
        candidates.extend(kwargs.get("tensor_dtypes", ()) or ())
        for dt in candidates:
            dt_low = str(dt).split(".")[-1].lower()
            if dt_low in ("float4_e2m1", "float4", "float8_e4m3fn", "float8_e5m2"):
                out_dtype_str = dt_low
                break
        if out_dtype_str is None:
            raise ValueError(
                "Cannot determine RotateQuant output dtype from yOut or tensor_dtypes"
            )
        mx_ele_dtype = _DTYPE_TO_MX[out_dtype_str]
        return compute_rotate_mx(
            x,
            rotation,
            alpha=alpha,
            mx_ele_dtype=mx_ele_dtype,
            round_mode=str(roundMode) if roundMode is not None else "rint",
            scale_alg=int(scaleAlg) if scaleAlg is not None else 0,
            dst_type_max=float(dstTypeMax) if dstTypeMax is not None else 0.0,
        )


__spec__ = {
    "aclnnRotateQuant": "AclnnRotateQuantTestSpec",
}
