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

from typing import ClassVar

import numpy as np

__spec__ = {"mvnv2": "MvnV2TestSpec"}

_DEFAULT_AXES = (0, 2, 3)


def _normalize_axes(axes, ndim):
    """Resolve axes exactly as the operator contract requires."""
    import torch

    if axes is None:
        return tuple(axis for axis in _DEFAULT_AXES if axis < ndim)
    elif isinstance(axes, torch.Tensor):
        axes = axes.detach().cpu().reshape(-1).tolist()
    elif isinstance(axes, np.ndarray):
        axes = axes.reshape(-1).tolist()
    elif not isinstance(axes, (list, tuple)):
        axes = [int(axes)]

    if len(axes) == 0:
        return tuple(axis for axis in _DEFAULT_AXES if axis < ndim)

    resolved = set()
    for axis in axes:
        axis = int(axis)
        if axis < 0:
            axis += ndim
        if not 0 <= axis < ndim:
            raise ValueError(f"axis {axis} is out of range for rank {ndim}")
        resolved.add(axis)
    return tuple(sorted(resolved))


def _to_torch(x):
    import torch

    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _mvn_v2_torch(x, eps, axes, compute_dtype):
    """Compose MVNV2 from Torch reduction and elementwise operators."""
    import torch

    x_tensor = _to_torch(x)
    if x_tensor.dim() != 4:
        raise ValueError(f"MVNV2 expects rank 4, got {x_tensor.dim()}")
    resolved_axes = _normalize_axes(axes, x_tensor.dim())
    if x_tensor.numel() == 0:
        return torch.empty_like(x_tensor)
    output_dtype = x_tensor.dtype
    x_compute = x_tensor.to(compute_dtype)
    if not resolved_axes:
        return torch.zeros_like(x_tensor)

    mean = torch.mean(x_compute, dim=resolved_axes, keepdim=True)
    residual = torch.sub(x_compute, mean)
    variance = torch.mean(
        torch.mul(residual, residual), dim=resolved_axes, keepdim=True
    )
    denominator = torch.add(torch.sqrt(variance), float(eps))
    return torch.div(residual, denominator).to(output_dtype)


def _third_party_correctness_torch(x, eps, axes):
    """Independent Torch FP32 var_mean reference with small-value scaling."""
    import torch

    x_tensor = _to_torch(x)
    if x_tensor.dim() != 4:
        raise ValueError(f"MVNV2 expects rank 4, got {x_tensor.dim()}")
    dims = _normalize_axes(axes, x_tensor.dim())
    if x_tensor.numel() == 0:
        return torch.empty_like(x_tensor)
    if not dims:
        return torch.zeros_like(x_tensor)

    x_compute = x_tensor.to(torch.float32)
    eps_value = float(eps)
    ieee_eps_mode = (
        not np.isfinite(eps_value)
        or eps_value < 0.0
        or 0.0 < abs(eps_value) < np.finfo(np.float32).tiny
    )
    if ieee_eps_mode:
        variance, mean = torch.var_mean(x_compute, dim=dims, correction=0, keepdim=True)
        return ((x_compute - mean) / (torch.sqrt(variance) + eps_value)).to(
            x_tensor.dtype
        )

    group_min = torch.amin(x_compute, dim=dims, keepdim=True)
    group_max = torch.amax(x_compute, dim=dims, keepdim=True)
    finite_range = torch.isfinite(group_min) & torch.isfinite(group_max)
    origin = torch.where(
        finite_range, 0.5 * group_min + 0.5 * group_max, torch.zeros_like(group_min)
    )
    centered = x_compute - origin
    max_abs = torch.amax(torch.abs(centered), dim=dims, keepdim=True)
    one = torch.ones_like(max_abs)
    safe_abs = torch.maximum(
        max_abs, torch.full_like(max_abs, torch.finfo(torch.float32).tiny)
    )
    scale_up = torch.minimum(
        torch.reciprocal(safe_abs), torch.full_like(max_abs, 2.0**126)
    )
    scale_down = torch.maximum(
        torch.reciprocal(max_abs),
        torch.full_like(max_abs, torch.finfo(torch.float32).tiny),
    )
    scale_cap = torch.full_like(max_abs, 2.0**126)
    if float(eps) > 0.0:
        eps_scale_cap = min(2.0**126, torch.finfo(torch.float32).max / float(eps))
        scale_cap = torch.full_like(max_abs, eps_scale_cap)
    scale_up = torch.minimum(scale_up, scale_cap)
    scale_factor = torch.where((max_abs > 0.0) & (max_abs < 2.0**-40), scale_up, one)
    scale_factor = torch.where(max_abs > 2.0**60, scale_down, scale_factor)
    x_scaled = centered * scale_factor
    variance, mean = torch.var_mean(x_scaled, dim=dims, correction=0, keepdim=True)
    residual = x_scaled - mean
    denominator = torch.sqrt(variance) + float(eps) * scale_factor
    result = residual / denominator
    if float(eps) > 0.0:
        finite_const = torch.isfinite(group_min) & (group_min == group_max)
        result = torch.where(finite_const, torch.zeros_like(result), result)
    return result.to(x_tensor.dtype)


class MvnV2TestSpec:
    @staticmethod
    def golden(x, eps=1.0e-9, axes=None, **kwargs):
        import torch

        del kwargs
        x_array = np.asarray(x)
        result = _mvn_v2_torch(x_array, eps, axes, torch.float64)
        return [result.detach().cpu().numpy()]

    class ThirdPartyImpl:
        """Correctness-only Torch reference; not a performance endpoint.

        TTK recreates this class inside every profiled invocation and provides no
        input-preparation hook. Consequently validation and dtype conversion must
        remain here for cross-check correctness.
        """

        def __init__(self, *, eps=1.0e-9, axes=None, **kwargs):
            del kwargs
            self.eps = eps
            self.axes = axes

        def __call__(self, x, **kwargs):
            del kwargs
            return [_third_party_correctness_torch(x, self.eps, self.axes)]

    third_party: ClassVar = {"torch": ThirdPartyImpl}
    tolerance: ClassVar = {
        "float16": {"standard": "cross_check", "level": "L0"},
        "float32": {"standard": "cross_check", "level": "L0"},
    }
