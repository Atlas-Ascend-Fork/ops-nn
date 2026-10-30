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
"""Explicitly selected grouped ACLNN reference; outside the ordinary suite."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets"))

import rotate_quant_kernel_golden as _kernel
from rotate_quant_kernel_golden import _as_numpy, compare_outputs
from rotate_quant_aclnn_golden import AclnnRotateQuantTestSpec


class AclnnRotateQuantV2TestSpec:
    """M-axis grouping; preserve the supplied initial outputs outside covered rows."""

    compare = staticmethod(compare_outputs)

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
        yOut=None,
        scaleOut=None,
        **kwargs,
    ):
        options = dict(
            axis=axis,
            roundMode=roundMode,
            scaleAlg=scaleAlg,
            dstTypeMax=dstTypeMax,
            trans=trans,
            **kwargs,
        )
        if groupList is None:
            return AclnnRotateQuantTestSpec.golden(
                x, rotation, alpha, yOut=yOut, scaleOut=scaleOut, **options
            )
        if yOut is None or scaleOut is None:
            raise ValueError(
                "Grouped golden requires initial yOut and scaleOut to preserve uncovered rows"
            )
        x = _kernel._to_numpy(x)
        rotation = _kernel._to_numpy(rotation)
        groups = _as_numpy(groupList)
        if x.ndim != 2 or rotation.ndim not in (3, 4) or groups.ndim != 1:
            raise ValueError(
                "Expected x[M,N], rotation[E,K,K] or [E,N/K,K,K], groupList[E]"
            )
        if (
            not 1 <= len(groups) <= 1024
            or len(groups) != rotation.shape[0]
            or groupListType not in (0, 1)
        ):
            raise ValueError("Invalid group metadata")
        k = rotation.shape[-1]
        if (
            k not in (32, 64, 128, 256)
            or rotation.shape[-2] != k
            or x.shape[1] <= 0
            or x.shape[1] % k
        ):
            raise ValueError("Invalid grouped rotation dimensions")
        if rotation.ndim == 4 and rotation.shape[1] != x.shape[1] // k:
            raise ValueError("Grouped 4D rotation dim1 must equal N/K")
        y = _as_numpy(yOut).copy()
        scale = _as_numpy(scaleOut).copy()
        start = 0
        for i, value in enumerate(groups):
            value = int(value)
            end = value if groupListType == 0 else start + value
            if value < 0 or end < start or end > x.shape[0]:
                raise ValueError("Invalid group boundary")
            if end > start:
                group_y, group_scale = AclnnRotateQuantTestSpec.golden(
                    x[start:end],
                    rotation[i],
                    alpha,
                    yOut=y[start:end],
                    scaleOut=scale[start:end],
                    **options,
                )
                y[start:end] = group_y
                scale[start:end] = group_scale
            start = end
        return [y, scale]


__spec__ = {
    "aclnnRotateQuantV2": "AclnnRotateQuantV2TestSpec",
}
