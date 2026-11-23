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
"""foreach_maximum_scalar_list Golden — TensorList 与标量列表逐元素最大值（竞品 torch._foreach_clamp_min）。"""

__spec__ = {
    "foreach_maximum_scalar_list": "ForeachMaximumScalarListKernelSpec",
    "aclnnForeachMaximumScalarList": "AclnnForeachMaximumScalarListSpec",
    "torch._foreach_maximum": "TorchForeachMaximumScalarListSpec",
    "torch.ops.aten._foreach_maximum.ScalarList": "TorchForeachMaximumScalarListSpec",
    "torch._foreach_clamp_min": "TorchForeachMaximumScalarListSpec",
}
import numpy as np
import ml_dtypes
import torch

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


def _to_scalars(v):
    """kernel/GEIR 的 scalars 槽为张量/ndarray，拆出数值列表；python 列表原样。"""
    if v is None:
        return None
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return v.reshape(-1).tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _is_bare_scalar(v):
    """单个标量（python 数值 / numpy 标量 / 0 维张量）为 True；列表/张量列表为 False。"""
    if isinstance(v, (list, tuple)):
        return False
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return v.ndim == 0
    return True


def _pick_scalars(xs, scalars, scalar, cached_scalars=None, cached_scalar=None):
    """解析标量：调用期优先，构造期兜底。

    - scalars（列表）用于 ScalarList 变体
    - scalar（单值）用于 Scalar 变体，按 tensor 个数广播
    - 框架按位置传单标量时会落到 scalars 槽，此处识别并转为 scalar 广播
    - 两者都缺失时显式报错，避免 None 静默参与计算
    """
    if scalars is not None and scalar is None and _is_bare_scalar(scalars):
        scalars, scalar = None, scalars
    if (
        cached_scalars is not None
        and cached_scalar is None
        and _is_bare_scalar(cached_scalars)
    ):
        cached_scalars, cached_scalar = None, cached_scalars
    if scalars is not None:
        return _to_scalars(scalars)
    if scalar is not None:
        return [scalar] * len(xs)
    if cached_scalars is not None:
        return _to_scalars(cached_scalars)
    if cached_scalar is not None:
        return [cached_scalar] * len(xs)
    raise ValueError("foreach_maximum_scalar_list requires 'scalars' (or 'scalar')")


def _integer_scalar_to_int64(v):
    """Integer scalar -> int64; None when not representable (negative cases go to the operator)."""
    try:
        iv = int(np.int64(v))
    except (OverflowError, ValueError, TypeError):
        return None
    if iv < -(1 << 63) or iv > (1 << 63) - 1:
        return None
    return iv


def _prepare_compute(xs, scalars):
    """Mirror the kernel dtype policy: floating x computes in float32 and integer x in int64,
    both then cast back to the original x dtype. Return None on unrepresentable integer scalars."""
    if len(xs) != len(scalars):
        raise ValueError(
            "foreach_maximum_scalar_list requires one scalar per tensor: "
            f"got {len(scalars)} scalars for {len(xs)} tensors"
        )
    calc_xs, calc_scalars, out_dtypes = [], [], []
    for t, s in zip(xs, scalars):
        if torch.is_floating_point(t):
            calc_xs.append(t.to(torch.float32))
            with np.errstate(over="ignore"):
                calc_scalars.append(float(np.float32(s)))
            out_dtypes.append(t.dtype)
            continue
        iv = _integer_scalar_to_int64(s)
        if iv is None:
            return None
        calc_xs.append(t.to(torch.int64))
        calc_scalars.append(iv)
        out_dtypes.append(t.dtype)
    return calc_xs, calc_scalars, out_dtypes


def _compute(xs, scalars):
    """Third-party compute with kernel dtype policy.

    Direct torch._foreach_clamp_min(x, scalars) narrows the scalar to x dtype; a double scalar
    outside x dtype range (e.g. bfloat16 with -1.7e308) overflows. Pre-convert via
    _prepare_compute (float→float32, int→int64), compute, then cast back to the original dtype.
    """
    prepared = _prepare_compute(list(xs), list(scalars))
    if prepared is None:
        return list(torch._foreach_clamp_min(list(xs), list(scalars)))
    calc_xs, calc_scalars, out_dtypes = prepared
    result = torch._foreach_clamp_min(calc_xs, calc_scalars)
    return [r.to(dt) for r, dt in zip(result, out_dtypes)]


def _copy_to_out(o, r):
    """写回 out 视图。torch copy_ 拒绝 stride=0 广播视图：广播轴钉 0 号切片、
    其余轴 slice，组合索引一次批量写回（物理存储唯一）；空张量直接返回。"""
    if o.numel() == 0:
        return
    if not any(s == 0 for s in o.stride()):
        o.copy_(r)
        return
    idx = tuple(0 if s == 0 else slice(None) for s in o.stride())
    o[idx].copy_(r[idx])


def _to_torch(t):
    if isinstance(t, torch.Tensor):
        return t, t.dtype
    a = np.ascontiguousarray(t)
    if a.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16), torch.bfloat16
    tensor = torch.from_numpy(a)
    return tensor, tensor.dtype


class _TorchForeachClampMinScalarListImpl:
    """Kernel/GEIR/ACLNN 三方参考（服务端 GPU）：对标竞品 torch._foreach_clamp_min。"""

    def __init__(self, scalars=None, scalar=None, **kwargs):
        self._scalars = scalars
        self._scalar = scalar

    def __call__(self, x, scalars=None, scalar=None, **kwargs):
        xs = list(x)
        return _compute(
            xs, _pick_scalars(xs, scalars, scalar, self._scalars, self._scalar)
        )


class _TorchForeachClampMinScalarListE2EImpl:
    """E2E 三方参考：self TensorList 位置入参 + scalars/scalar 关键字。"""

    def __init__(self, scalars=None, scalar=None, **kwargs):
        self._scalars = scalars
        self._scalar = scalar

    def __call__(self, *args, scalars=None, scalar=None, **kwargs):
        xs = list(args[0])
        if scalars is None and scalar is None and len(args) > 1:
            scalars = args[1]
        return _compute(
            xs, _pick_scalars(xs, scalars, scalar, self._scalars, self._scalar)
        )


_THIRD_PARTY = {"torch": _TorchForeachClampMinScalarListImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachClampMinScalarListE2EImpl}


class ForeachMaximumScalarListKernelSpec:
    """Kernel/GEIR 路径 — (x: list[np.ndarray], scalars: np.ndarray) -> list[np.ndarray]。"""

    def golden(x, scalars, **kwargs):
        s_list = _to_scalars(scalars)
        x_list, out_dtypes = [], []
        for t in x:
            tt, dt = _to_torch(t)
            x_list.append(tt)
            out_dtypes.append(dt)
        result = _compute(x_list, s_list)
        out = []
        for r, dt in zip(result, out_dtypes):
            if dt == torch.bfloat16:
                out.append(r.view(torch.uint16).numpy().view(ml_dtypes.bfloat16))
            else:
                out.append(r.to(dt).numpy())
        return out

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachMaximumScalarListSpec:
    """ACLNN 路径 — aclnnForeachMaximumScalarListGetWorkspaceSize(x, scalars, out, ...)。"""

    def golden(x, scalars, out=None, **kwargs):
        result = _compute(list(x), _to_scalars(scalars))
        if out is not None:
            if len(out) != len(result):
                raise ValueError(
                    f"out length {len(out)} != result length {len(result)}"
                )
            for o, r in zip(out, result):
                if o.numel() != r.numel():
                    raise ValueError(
                        f"out numel {o.numel()} != result numel {r.numel()}"
                    )
                _copy_to_out(o, r)
            return list(out)
        return result

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachMaximumScalarListSpec:
    """E2E 路径 — torch._foreach_clamp_min(self: TensorList, scalars: Scalar[])。"""

    def golden(*args, self=None, scalars=None, scalar=None, **kwargs):
        x = self if self is not None else (args[0] if args else None)
        if x is None:
            raise ValueError(
                "foreach_maximum_scalar_list golden requires input tensors"
            )
        if not isinstance(x, (list, tuple)):
            x = [x]
        x = list(x)
        if scalars is None and scalar is None and len(args) > 1:
            scalars = args[1]
        return _compute(x, _pick_scalars(x, scalars, scalar))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
