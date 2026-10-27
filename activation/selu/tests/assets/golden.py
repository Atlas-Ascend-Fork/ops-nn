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
"""TestSpec definitions for Selu raw op, ACLNN, Torch, and TensorFlow paths.

Selu: y = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
      alpha = 1.6732632423543772
      scale = 1.0507009873554805

Registered names:
  - "selu" / "Selu"            <- CSV op_name (Kernel / GEIR path, numpy input;
                                  "selu" = legacy lowercase dialect,
                                  "Selu" = ascendc-st-design generated dialect)
  - "aclnnSelu"                <- CSV api_name (ACLNN path, torch input; first
                                  tensor arg is named ``self``)
  - "aclnnInplaceSelu"         <- CSV api_name (ACLNN inplace path; first tensor
                                  arg is named ``selfRef``)
  - "tf.nn.selu" / "tf.raw_ops.Selu" <- CSV api_name (TensorFlow path)

Benchmark: every spec golden() funnels into the module-level
``_benchmark_selu`` — one shared benchmark function owned by no spec. The
benchmark is the single oracle pinned by the spec ``reference_oracle``:
``torch.selu``. The carrier is normalized to torch (torch/tf/numpy input,
bit-exact local recipes); float dtypes (f32/f16/bf16) call ``torch.selu``
directly; int32/int8 — unsupported by ``torch.selu`` — evaluate the torch
basic-op composition ``_selu_torch_composed``: computed in float32 then
cast back to the integer dtype (int32 saturates on overflow, int8 wraps —
device Cast semantics).

Import rule: torch / ml_dtypes are imported lazily inside function bodies
only.
"""

__spec__ = {
    "selu": "SeluKernelTestSpec",
    "Selu": "SeluKernelTestSpec",
    "aclnnSelu": "AclnnSeluTestSpec",
    "aclnnInplaceSelu": "AclnnInplaceSeluTestSpec",
    "tf.nn.selu": "SeluTensorFlowSpec",
    "tf.raw_ops.Selu": "SeluTensorFlowSpec",
    # torch.selu / torch.nn.functional.selu 不注册: torch 路径复合分解走 aclnnElu(aclnn built-in),不经本算子
}

ALPHA = 1.6732632423543772
SCALE = 1.0507009873554805

# Tolerance: floats use the cross_check standard (third-party comparison,
# level L2) with no in-file numeric two-way tolerances; int32/int8 use
# binary_equal. The ``--compare`` CLI flag belongs to the execution layer and
# is intentionally not encoded here.
_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L2"},
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "int32": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
}
_TOLERANCE_FLOAT = {k: _TOLERANCE[k] for k in ("float32", "float16", "bfloat16")}

# third_party is declared per execution path: the framework-agnostic kernel/GEIR
# entry lists all competitors; framework-bound specs list only their own path's
# API (aclnn/torch -> torch, tf -> tf).
_TP_ALL = {"torch": "torch.selu", "tf": "tf.raw_ops.Selu"}
_TP_TORCH = {"torch": "torch.selu"}
_TP_TF = {"tf": "tf.raw_ops.Selu"}


def _np_to_torch(x):
    """numpy ndarray -> torch tensor, bit-exact (bfloat16 via int16 bit view)."""
    import torch

    if x.dtype.name == "bfloat16":
        return torch.from_numpy(x.view("int16")).view(torch.bfloat16)
    return torch.from_numpy(x)


def _torch_to_np(t):
    """torch tensor -> numpy ndarray, bit-exact (bfloat16 via int16 bit view)."""
    import torch

    if t.dtype == torch.bfloat16:
        import ml_dtypes

        return t.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return t.numpy()


def _tf_to_np(t):
    """tf tensor -> numpy ndarray (bfloat16 viewed as ml_dtypes)."""
    import numpy as np

    x = np.asarray(t.numpy())
    if x.dtype.name == "bfloat16":  # TF 2.6-era and ml_dtypes bf16 share storage
        import ml_dtypes

        x = x.view(ml_dtypes.bfloat16)
    return x


def _is_tf_tensor(x):
    """Carrier probe via the type's module name — no tensorflow import
    needed (the benchmark oracle is torch.selu; tf appears only as an
    input carrier)."""
    return type(x).__module__.split(".", 1)[0] == "tensorflow"


def _selu_torch_composed(t):
    """SELU (elementwise) via torch basic-op composition — the int32/int8
    leg (``torch.selu`` does not support integer dtypes).

    Integer inputs compute in float32 then convert back to the integer
    dtype (device Cast semantics): int32 saturates on overflow, int8 wraps.
    """
    import torch

    xf = t.to(torch.float32)
    yt = SCALE * (
        torch.clamp(xf, min=0.0) + torch.clamp(ALPHA * torch.expm1(xf), max=0.0)
    )
    if t.dtype is torch.int32:
        yt = _saturate_to_int32_range(yt)
    return yt.to(t.dtype)


def _saturate_to_int32_range(y):
    """int32 overflow conversion guard (device Cast saturates on overflow).

    torch ``.to(int32)`` yields INT_MIN on out-of-range floats, while the
    device saturates. Clamp runs in float64: float32 cannot represent
    INT32_MAX (rounds up to 2147483648.0), which would push the bound past
    the valid range.
    """
    import torch

    info = torch.iinfo(torch.int32)
    return y.to(torch.float64).clamp(float(info.min), float(info.max))


def _benchmark_selu(x):
    """Shared benchmark SELU — module-level, owned by no spec.

    Benchmark = ``torch.selu`` (pinned by the spec reference_oracle).
    Carrier normalization (torch/tf/numpy -> torch, bit-exact local recipes)
    -> float dtypes call ``torch.selu`` directly; int32/int8 evaluate
    ``_selu_torch_composed`` -> cast back to the input carrier (torch in ->
    torch out; tf/numpy in -> numpy out).
    """
    import numpy as np
    import torch

    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        t = x.detach().cpu()
    else:
        if _is_tf_tensor(x):
            x = _tf_to_np(x)
        t = _np_to_torch(np.asarray(x))
    if t.dtype in (torch.int8, torch.int32):
        # torch.selu does not support integer dtypes -> torch basic-op composition
        y = _selu_torch_composed(t)
    else:
        y = torch.selu(t)
    return y if is_torch else _torch_to_np(y)


class _TorchSelu:
    """torch.selu third-party wrapper (spec-mode class form): the ACLNN
    ``self``-named input cannot be bound to the string API on the remote
    server, so the torch leg dispatches through this class."""

    def __call__(instance, *args, **kwargs):
        import torch

        x = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
        if x is None:
            for name in ("x", "self", "selfRef", "features", "input", "input_x"):
                x = kwargs.get(name)
                if x is not None:
                    break
        if x is None:
            raise ValueError(
                "Missing Selu input: x/self/selfRef/features/input/input_x"
            )
        return [torch.selu(x)]


class SeluKernelTestSpec:
    """Shared GEIR/Kernel spec registered by CSV ``op_name`` "selu"/"Selu"
    (numpy input)."""

    def golden(x, **kwargs):
        return [_benchmark_selu(x)]

    third_party = _TP_ALL
    tolerance = _TOLERANCE


class AclnnSeluTestSpec:
    """ACLNN spec registered by CSV ``api_name`` (torch input).

    The first tensor argument follows the ACLNN name ``self``; the golden
    parameter is renamed ``selfT`` (positional fallback). The ``self``-named
    input cannot be bound to a string third-party API on the remote server,
    so the torch leg dispatches through the ``_TorchSelu`` wrapper class.
    """

    def golden(selfT, out=None, **kwargs):
        result = _benchmark_selu(selfT)
        if out is not None:
            out.copy_(result)
            return [out]
        return [result]

    third_party = {"torch": _TorchSelu}
    tolerance = _TOLERANCE


class AclnnInplaceSeluTestSpec:
    """ACLNN inplace spec registered by CSV ``api_name`` "aclnnInplaceSelu"
    (torch input; first tensor arg named ``selfRef`` — bindable as a string)."""

    def golden(selfRef, **kwargs):
        return [_benchmark_selu(selfRef)]

    third_party = _TP_TORCH
    tolerance = _TOLERANCE


class TorchSeluTestSpec:
    """Torch Single/Graph spec (not registered in ``__spec__``; kept for spec
    cross-validation — E2E dispatches F.selu to aclnnElu, not this operator)."""

    def golden(input, inplace=False, **kwargs):
        return [_benchmark_selu(input)]

    third_party = _TP_TORCH
    tolerance = _TOLERANCE_FLOAT


class SeluTensorFlowSpec:
    """TF raw-op spec registered by CSV ``api_name`` "tf.nn.selu" /
    "tf.raw_ops.Selu".

    golden evaluates the shared benchmark (see ``_benchmark_selu``);
    the TF counterpart is referenced declaratively via third_party.
    """

    def golden(features, **kwargs):
        return [_benchmark_selu(features)]

    third_party = _TP_TF
    tolerance = _TOLERANCE
