# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent CPU oracle and native Torch reference for LpNormReduceV2.

Kernel/GEIR only (no aclnn layer). The oracle computes the Lp-norm reduction
segment (no final 1/p root), migrated from the standalone delivery workspace
(tests/st golden): 6 p-branches, fp32 intermediate semantics, fp64 accumulation,
round-to-nearest-even write-back for fp16/bf16.

p is a **float** attribute (IEEE 754): +-inf and the fp32 storage aliases
2147483647.0/2147483648.0 (+inf) and -2147483648.0 (-inf) route to the
max/min branches; NaN p is rejected.
"""

import hashlib
import importlib
import math

import numpy as np

__spec__ = {
    "lp_norm_reduce_v2": "LpNormReduceV2TestSpec",
    "LpNormReduceV2": "LpNormReduceV2TestSpec",
}

# fp32 存储别名：fp32(2147483647.0) == 2^31，与 +inf 同判（tiling JudgePBranch 口径）
P_POS_ALIAS = 2147483648.0
P_POS_ALIAS_INPUT = 2147483647.0
P_NEG_ALIAS = -2147483648.0


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _parameters(shape, p, axes):
    """Validate metadata only; no shared numerical implementation with Torch."""
    p = _number(p, "p")
    if math.isnan(p):
        raise ValueError("p must not be NaN")
    rank = len(shape)
    if rank > 8:
        raise ValueError("rank must be in [0, 8]")
    if axes is None:
        axes = ()
    if not isinstance(axes, (list, tuple)):
        raise TypeError("axes must be a list or tuple of integers")
    dims = []
    for axis in axes:
        axis = int(axis)
        if axis < -rank or axis >= rank:
            raise ValueError(f"axis {axis} outside [-rank, rank)")
        axis %= rank
        if axis in dims:
            # V2 语义：归一化后重复轴显式拒绝（infershape/tiling 同口径）
            raise ValueError(f"duplicate reduce axis {axis} after normalization")
        dims.append(axis)
    dims = tuple(dims) if axes else tuple(range(rank))
    return p, dims


def _classify_branch(p):
    """Mirror of op_host/arch35 JudgePBranch priority: +inf alias > -inf alias
    > 0 > 1 > integer [2,15] > other finite float."""
    if math.isinf(p):
        return "max" if p > 0 else "min"
    if p in (P_POS_ALIAS, P_POS_ALIAS_INPUT):
        return "max"
    if p == P_NEG_ALIAS:
        return "min"
    if p == 0.0:
        return "count"
    if p == 1.0:
        return "sum"
    return "pow"


def _reduce_abs(x64, dims, keepdim, branch, p):
    """Branch dispatch over |x| (float64 view); NaN propagates per IEEE."""
    a = np.abs(x64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if branch == "max":
            return np.max(a, axis=dims, keepdims=keepdim)
        if branch == "min":
            return np.min(a, axis=dims, keepdims=keepdim)
        if branch == "count":
            # NaN 计为非零元素（IEEE：NaN != 0）
            return np.sum(a != 0, axis=dims, keepdims=keepdim, dtype=np.float64)
        if branch == "sum":
            return np.sum(a, axis=dims, keepdims=keepdim)
        return np.sum(np.power(a, p), axis=dims, keepdims=keepdim)


def _cast_bf16_rne(f32):
    """float32 -> bfloat16 round-to-nearest-even（对齐 kernel CAST_RINT / tbe.round）。"""
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    bf16 = (u + np.uint32(0x7FFF) + lsb) >> 16
    return (bf16.astype(np.uint32) << 16).view(np.float32)


def _write_back(y64, dtype):
    """fp64 结果回写输入 dtype；promote dtype 原样保留（对齐 V1 "including Promote dtype"）。

    TTK 在 mix_tolerance / cross_check 模式下会把输入 promote（如 float32 ->
    float64），此时必须 retain 提升后的 dtype 返回，禁止降精度回写——否则与
    三方参考（torch 同样 retain）对比时 dtype/精度失配。
    """
    dt = np.dtype(dtype)
    if dt == np.float16:
        return np.asarray(y64).astype(np.float64).astype(np.float32).astype(np.float16)
    if dt == np.float32:
        return np.asarray(y64).astype(np.float64).astype(np.float32)
    if "bfloat16" in str(dt):
        return _cast_bf16_rne(np.asarray(y64).astype(np.float64).astype(np.float32))
    # promote dtype（float64 等）：retain，不降精度、不误入 bf16 分支
    return np.asarray(y64).astype(dt)


def _special_kind(name):
    """Testcase-name tag -> injection kind.

    Two naming styles share one hook (TTK generates inputs once, so NPU,
    golden and the remote third_party consume the same injected arrays):
    - ``zeros_*`` / ``infnan_*`` prefix (V1 style, e.g. ``zeros_f32_count``)
    - ``..._zeros_...`` / ``..._infnan_...`` inner tag (V2 style, e.g.
      ``lpnorm_k_zeros_f32_count`` / ``lpnorm_geir_infnan_f32_max``)
    """
    if name.startswith(("zeros_", "infnan_")):
        return name.split("_", 1)[0]
    if "_zeros_" in name:
        return "zeros"
    if "_infnan_" in name:
        return "infnan"
    return ""


def _customize_inputs(x, **kwargs):
    """Inject deterministic special values by testcase name.

    - zeros: about n/3 exact zeros (plus a few -0.0) so the p=0 nonzero
      count semantics is actually exercised.
    - infnan: inject +inf/-inf/NaN/±0.0 for IEEE propagation checks.
    - Other cases pass through unchanged.

    Seed is the first 8 hex digits of md5(testcase_name), keeping runs
    reproducible across processes.
    """
    name = str(kwargs.get("testcase_name", ""))
    kind = _special_kind(name)
    if not kind:
        return (x,)

    arr = np.asarray(x)
    if arr.size == 0:
        return (x,)
    seed = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    flat = arr.reshape(-1).copy()
    n = flat.size

    if kind == "zeros":
        idx = rng.choice(n, size=max(1, n // 3), replace=False)
        flat[idx] = 0.0
        # |-0.0| == 0 must not count as nonzero either.
        flat[idx[: max(1, idx.size // 10)]] = -0.0
    else:
        specials = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=flat.dtype)
        k = min(n, max(specials.size, n // 20))
        idx = rng.choice(n, size=k, replace=False)
        flat[idx] = specials[rng.integers(0, specials.size, size=k)]

    return (np.ascontiguousarray(flat).reshape(arr.shape),)


class TorchLpNormReduceV2:
    """Native Torch composition; FP16/BF16 intermediates use FP32 per spec.

    p is a float: +-inf (and the fp32 storage aliases) route to amax/amin.
    """

    def __init__(self, *, p=2.0, axes=None, keepdim=False, epsilon=1e-12, **kwargs):
        self.p = p
        self.axes = axes
        self.keepdim = keepdim

    def __call__(self, x, **kwargs):
        # Dynamic import: keep torch out of TTK worker preload (forkserver
        # segfault); only the remote third_party server imports torch.
        torch = importlib.import_module("torch")

        p, dims = _parameters(x.shape, self.p, self.axes)
        calc = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
        # 三方参考 = torch 官方 Lp 范数接口（torch.linalg.vector_norm，权威口径）。
        # 语义映射：LpNormReduceV2 输出 sum|x|^p（不开根号）——
        #   ord=inf/-inf/0/1 时 vector_norm 语义即 max|x|/min|x|/非零计数/sum|x|（无根号）；
        #   其他 p：vector_norm(ord=p) = (sum|x|^p)^(1/p) → **p 还原 sum|x|^p（一次 pow）。
        # fp16/bf16 先升 fp32 与算子 dtype 路径一致。
        if math.isinf(p) or p in (0.0, 1.0):
            y = torch.linalg.vector_norm(calc, ord=p, dim=dims, keepdim=self.keepdim)
        else:
            vn = torch.linalg.vector_norm(calc, ord=p, dim=dims, keepdim=self.keepdim)
            y = vn.pow(p)
        if y.ndim == 0:
            # 全归约 keepdim=false：标量以 (1,) 承载（与算子 infershape/GE 数据
            # 回拷口径一致；TTK flat 元素比对不受影响）。
            y = y.reshape(1)
        return [y.to(x.dtype)]


class LpNormReduceV2TestSpec:
    """CPU NumPy reference for the mathematical formula in the operator README."""

    @staticmethod
    def golden(x, p=2.0, axes=None, keepdim=False, epsilon=1e-12, **kwargs):
        """FP64 intermediates over FP32-representable inputs; input dtype on return.

        epsilon is intentionally unused. Empty tensor / empty reduce domain
        (R==0, O>0) returns zeros (spec boundary); no clamping. NaN propagates
        per IEEE (counts as nonzero for p=0). Scalar shape is retained.
        """
        x = np.asarray(x)
        p, dims = _parameters(x.shape, p, axes)
        if x.size == 0:
            # 归约域为空（含 O==0 空输出与 R==0 置零路径）：输出全 0（spec boundary）
            shape = _output_shape(x.shape, dims, keepdim)
            return [np.zeros(shape, dtype=x.dtype)]
        branch = _classify_branch(p)
        x64 = x.astype(np.float64)
        y = _reduce_abs(x64, dims, keepdim, branch, p)
        shape = _output_shape(x.shape, dims, keepdim)
        if tuple(np.shape(y)) != shape:
            # 全归约 keepdim=false：np 归约输出 0 维 ()，统一承载为 (1,)（GE 口径）
            y = np.asarray(y).reshape(shape)
        return [_write_back(y, x.dtype)]

    third_party = {"torch": TorchLpNormReduceV2}
    customize_inputs = _customize_inputs
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


def _output_shape(shape, dims, keepdim):
    out = []
    for i, d in enumerate(shape):
        if i in dims:
            if keepdim:
                out.append(1)
        else:
            out.append(d)
    if not out:
        # 全维度归约 + keepdim=false：标量以 (1,) 承载——GE RunGraph 对 0 维输出
        # 张量 GetSize()=0 不回拷数据，0 维口径下调用方读到未初始化值。
        out.append(1)
    return tuple(out)
