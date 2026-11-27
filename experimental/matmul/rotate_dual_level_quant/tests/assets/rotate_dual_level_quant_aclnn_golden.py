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
"""ACLNN adapter, with parameter order matching the public two-stage API."""

import numbers
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import rotate_dual_level_quant_kernel_golden as _kernel

_MX_DTYPE_CODES = {40: "float4_e2m1", 36: "float8_e4m3fn", 35: "float8_e5m2"}


def mx_output_dtype(y_out, tensor_dtypes=()):
    candidates = [y_out.dtype] if y_out is not None else []
    candidates.extend(tensor_dtypes or ())
    for dtype in candidates:
        if isinstance(dtype, numbers.Integral):
            name = _MX_DTYPE_CODES.get(int(dtype))
            if name is not None:
                return name
            continue
        name = str(dtype).split(".")[-1].lower()
        name = {"float4": "float4_e2m1", "float4_e2m1fn": "float4_e2m1"}.get(name, name)
        if name in _kernel._MX_DTYPE_MAP:
            return name
    raise ValueError("Cannot determine output dtype from yOut or tensor_dtypes")


class AclnnRotateDualLevelQuantTestSpec:
    compare = staticmethod(_kernel.compare_outputs)

    @staticmethod
    def golden(
        x,
        rotation,
        alpha=None,
        groupList=None,
        groupListType=0,
        axis=-1,
        roundMode="rint",
        scaleAlg=0,
        dstTypeMax=0.0,
        trans=False,
        level0BlockSize=512,
        level1BlockSize=32,
        yOut=None,
        level0ScaleOut=None,
        level1ScaleOut=None,
        **kwargs,
    ):
        if groupList is not None and any(
            value is None for value in (yOut, level0ScaleOut, level1ScaleOut)
        ):
            raise ValueError(
                "Grouped ACLNN golden requires all three initial output buffers"
            )
        return _kernel.compute_rotate_dual_level_quant(
            x,
            rotation,
            alpha,
            groupList,
            groupListType,
            mx_output_dtype(yOut, kwargs.get("tensor_dtypes", ())),
            axis,
            "rint" if roundMode is None else roundMode,
            scaleAlg,
            dstTypeMax,
            trans,
            level0BlockSize,
            level1BlockSize,
            yOut,
            level0ScaleOut,
            level1ScaleOut,
        )


__spec__ = {"aclnnRotateDualLevelQuant": "AclnnRotateDualLevelQuantTestSpec"}
