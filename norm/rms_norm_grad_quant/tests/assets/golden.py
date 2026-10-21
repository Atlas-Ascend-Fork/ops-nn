#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU references for RMSNorm backward followed by static quantization.

The reference evaluates the documented derivatives in float64, independently of
the fused NPU implementation. Kernel and GEIR share the NumPy TestSpec; ACLNN
also accepts Torch inputs. No NPU or test-framework import is required.
"""

import numpy as np
import torch

__spec__ = {
    "rms_norm_grad_quant": "RmsNormGradQuantTestSpec",
    "aclnnRmsNormGradQuant": "AclnnRmsNormGradQuantTestSpec",
}


def _as_float64(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64).numpy()
    return np.asarray(value, dtype=np.float64)


def _quantize(value, dst_type):
    if dst_type == 2:
        # INT8 uses round-to-nearest-even and saturates instead of wrapping.
        return np.clip(np.rint(value), -128, 127).astype(np.int8)
    if dst_type == 34:
        from en_dtypes import hifloat8

        # HIFLOAT8 is a floating-point cast, not an integer rounding operation.
        return np.clip(value, -32768.0, 32768.0).astype(hifloat8)
    raise ValueError(f"Unsupported dx dtype code: {dst_type}; expected 2 or 34")


def rms_norm_grad_quant_golden(
    dy,
    x,
    rstd,
    gamma,
    scales_x,
    offset_x=None,
    *,
    quant_mode="static",
    div_mode=True,
    dst_type=2,
    **kwargs,
):
    """Compute [quantized dx, dgamma] from the supplied forward rstd.

    Reference basis: documented RMSNorm derivatives, with independent float64
    reductions. All trailing gamma axes are normalized, not just the last axis.
    rstd is consumed as an input; epsilon belongs only to forward input creation.
    """
    if quant_mode != "static":
        raise ValueError("RmsNormGradQuant only supports static quantization")
    x_shape, gamma_shape = tuple(x.shape), tuple(gamma.shape)
    rows, cols = int(np.prod(rstd.shape)), int(np.prod(gamma_shape))
    if rows == 0 or cols == 0:
        # This fused operator explicitly emits NaN for nonempty dgamma in
        # EmptyDgamma::VFDuplicateRows, rather than evaluating an empty sum.
        # Empty gamma is handled by the ACLNN early-return path.
        return [
            _quantize(np.empty(x_shape, dtype=np.float64), dst_type),
            np.full(gamma_shape, np.nan, dtype=np.float64),
        ]

    x_value = _as_float64(x).reshape(rows, cols)
    dy_value = _as_float64(dy).reshape(rows, cols)
    gamma_value = _as_float64(gamma).reshape(1, cols)
    rstd_value = _as_float64(rstd).reshape(rows, 1)
    normalized_x = x_value * rstd_value
    weighted_dy = dy_value * gamma_value
    dx = (
        weighted_dy
        - normalized_x
        * np.mean(
            weighted_dy * normalized_x,
            axis=1,
            keepdims=True,
        )
    ) * rstd_value
    dgamma = np.sum(dy_value * normalized_x, axis=0).reshape(gamma_shape)

    scale = _as_float64(scales_x).reshape(())
    if div_mode and scale == 0:
        raise ValueError("scales_x must be nonzero when div_mode is True")
    quantized = dx / scale if div_mode else dx * scale
    if offset_x is not None:
        quantized = quantized + _as_float64(offset_x).reshape(())
    return [_quantize(quantized.reshape(x_shape), dst_type), dgamma]


def rms_norm_grad_quant_inputs(dy, x, rstd, gamma, scales_x, offset_x=None, **kwargs):
    """Generate saved forward statistics, keeping all other valid input values.

    Random rstd values do not describe an RMSNorm forward pass. Use epsilon=1e-5
    only to construct these test inputs, and preserve rstd's declared dtype and
    shape. Empty tensors do not require a reduction.
    """
    cols = int(np.prod(gamma.shape))
    rows = int(np.prod(rstd.shape))
    if cols and rows:
        x_value = _as_float64(x).reshape(rows, cols)
        saved_rstd = (1.0 / np.sqrt(np.mean(x_value * x_value, axis=1) + 1e-5)).reshape(
            rstd.shape
        )
        if isinstance(rstd, torch.Tensor):
            rstd.copy_(torch.from_numpy(saved_rstd).to(dtype=rstd.dtype))
        else:
            rstd[...] = saved_rstd
    # Repair only a randomly generated zero scale; do not change valid scales.
    scales_x[scales_x == 0] = 1
    return dy, x, rstd, gamma, scales_x, offset_x


class RmsNormGradQuantTestSpec:
    golden = staticmethod(rms_norm_grad_quant_golden)
    customize_inputs = staticmethod(rms_norm_grad_quant_inputs)
    # Quantized integers allow one LSB from floating-point reduction order.
    # FP32 uses mixed tolerance. HIFLOAT8 retains the framework's default
    # numerical-neighbor comparison (requant), without overriding its limits.
    tolerance = {
        "int8": {"standard": "quant"},
        "float32": {"standard": "mix_tolerance"},
    }


def aclnn_rms_norm_grad_quant_golden(
    dy,
    x,
    rstd,
    gamma,
    scales_x,
    offset_x_optional=None,
    quant_mode="static",
    div_mode=True,
    dx_out=None,
    dgamma_out=None,
    **kwargs,
):
    """Follow the complete ACLNN positional signature, including output slots."""
    declared = kwargs.get("original_tensor_dtypes", kwargs.get("tensor_dtypes"))
    dtype = str(declared[6]) if declared else str(getattr(dx_out, "dtype", "int8"))
    dtype = dtype.removeprefix("torch.")
    if dtype not in ("int8", "hifloat8"):
        raise ValueError(f"Unsupported ACLNN dx dtype: {dtype}")
    results = rms_norm_grad_quant_golden(
        dy,
        x,
        rstd,
        gamma,
        scales_x,
        offset_x_optional,
        quant_mode=quant_mode,
        div_mode=div_mode,
        dst_type=2 if dtype == "int8" else 34,
    )
    # Torch has no native HIFLOAT8 carrier. Keep both results as NumPy in that
    # path; ordinary ACLNN Torch inputs receive Torch results.
    if isinstance(x, torch.Tensor) and dtype == "int8":
        return [torch.from_numpy(result) for result in results]
    return results


def aclnn_rms_norm_grad_quant_inputs(
    dy,
    x,
    rstd,
    gamma,
    scales_x,
    offset_x_optional=None,
    quant_mode="static",
    div_mode=True,
    dx_out=None,
    dgamma_out=None,
    **kwargs,
):
    """Return only input tensors; output placeholders and attributes are omitted."""
    return rms_norm_grad_quant_inputs(dy, x, rstd, gamma, scales_x, offset_x_optional)


class AclnnRmsNormGradQuantTestSpec:
    golden = staticmethod(aclnn_rms_norm_grad_quant_golden)
    customize_inputs = staticmethod(aclnn_rms_norm_grad_quant_inputs)
    tolerance = RmsNormGradQuantTestSpec.tolerance
