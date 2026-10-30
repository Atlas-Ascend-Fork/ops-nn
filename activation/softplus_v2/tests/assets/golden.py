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

"""CPU/torch reference for SoftplusV2 Kernel/GEIR and ACLNN suites.

Implements the documented contract:

    out = log1p(exp(beta * x)) / beta   when beta * x <= threshold
    out = x                             when beta * x > threshold

Two-byte dtypes are promoted to float32 for the intermediate computation and
cast back, matching the device kernel. The reference is PyTorch's
``torch.nn.functional.softplus`` (the operator is exposed as ``aten.softplus``),
so boundary inputs keep the public behavior (e.g. ``beta = +/-0`` -> ``+/-inf``).
"""

from typing import ClassVar

import numpy as np

# Register both the REG_OP type name used by the frozen Kernel/GEIR suites and
# the installed opInterface value; ACLNN keeps its ACLNN API name.
__spec__ = {
    "SoftplusV2": "SoftplusV2KernelSpec",
    "softplus_v2": "SoftplusV2KernelSpec",
    "aclnnSoftplus": "SoftplusV2AclnnSpec",
}

_TOL = {
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
}


def _numpy_to_torch_tensor(array):
    import torch

    if "bfloat16" in array.dtype.name:
        # The int16 reinterpretation below requires a contiguous last axis.
        contiguous = np.ascontiguousarray(array)
        return torch.from_numpy(contiguous.view(dtype=np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy_tensor(tensor):
    import torch

    if tensor.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return tensor.view(torch.int16).numpy().view(dtype=bfloat16)
    return tensor.numpy()


def _as_torch_tensor(value):
    import torch

    return (
        value
        if isinstance(value, torch.Tensor)
        else _numpy_to_torch_tensor(np.asarray(value))
    )


def _contract_softplus_v2(x, beta, threshold):
    """Deterministic reference implementing the documented operator contract."""
    import torch

    x_tensor = _as_torch_tensor(x)
    source_dtype = x_tensor.dtype
    compute_dtype = (
        torch.float32
        if source_dtype in (torch.float16, torch.bfloat16)
        else source_dtype
    )
    result = torch.nn.functional.softplus(
        x_tensor.to(compute_dtype), beta=float(beta), threshold=float(threshold)
    )
    return result.to(source_dtype)


class _SoftplusV2Compose:
    def __call__(_instance, x, beta=1.0, threshold=20.0, **kwargs):
        del kwargs
        return [_contract_softplus_v2(x, beta, threshold)]


def _tensorflow_softplus_v2(x, beta=1.0, threshold=20.0, **kwargs):
    """TensorFlow third-party reference for the softplus formula."""
    import tensorflow as tf

    del kwargs
    x_tensor = tf.convert_to_tensor(x)
    source_dtype = x_tensor.dtype
    compute_dtype = (
        tf.float32 if source_dtype in (tf.float16, tf.bfloat16) else source_dtype
    )
    x_compute = tf.cast(x_tensor, compute_dtype)
    beta_value = tf.constant(float(beta), dtype=compute_dtype)
    threshold_value = tf.constant(float(threshold), dtype=compute_dtype)
    scaled = beta_value * x_compute
    # Numerically stable log1p(exp(scaled)): for scaled > 20 the log1p correction
    # is below fp32 resolution, so use the linear form there. This avoids the
    # fp32 exp overflow (scaled > ~88.72) that would otherwise yield inf while
    # scaled is still below a large threshold.
    softplus = (
        tf.where(scaled > 20.0, scaled, tf.math.log1p(tf.exp(scaled))) / beta_value
    )
    out = tf.where(scaled > threshold_value, x_compute, softplus)
    return [tf.cast(out, source_dtype)]


class _TensorFlowSoftplusV2:
    """TensorFlow third-party entry in the instantiable-class form."""

    def __call__(_instance, x, beta=1.0, threshold=20.0, **kwargs):
        return _tensorflow_softplus_v2(x, beta=beta, threshold=threshold, **kwargs)


class SoftplusV2KernelSpec:
    @staticmethod
    def golden(x, beta=1.0, threshold=20.0, **kwargs):
        del kwargs
        out = _contract_softplus_v2(x, beta, threshold)
        return [_torch_to_numpy_tensor(out)]

    third_party: ClassVar[dict] = {
        "torch": _SoftplusV2Compose,
        "tf": _TensorFlowSoftplusV2,
    }
    tolerance = _TOL


class SoftplusV2AclnnSpec:
    @staticmethod
    def golden(x, beta=1.0, threshold=20.0, out=None, **kwargs):
        del out, kwargs
        return [_contract_softplus_v2(x, beta, threshold)]

    third_party: ClassVar[dict] = {
        "torch": _SoftplusV2Compose,
        "tf": _TensorFlowSoftplusV2,
    }
    tolerance = _TOL
