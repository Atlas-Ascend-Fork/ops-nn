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

"""Golden reference for FakeQuantWithMinMaxVarsPerChannelGradient (per-channel QAT backward).

The reference math is expressed with torch tensor-operator composition (not bare NumPy), and the
per-channel nudge is computed in float32 so it matches the Kernel and the TensorFlow reference
implementation exactly (`round(qMin - min/scale)` at the fp32 precision used on device). A
TensorFlow `third_party` competitor is provided for cross-checking.

Contract (README):
  - q_min = 1 if narrow_range else 0, q_max = 2^num_bits - 1,
    scale = (max - min) / (q_max - q_min), nudged zero point = round(q_min - min/scale) clipped
    to [q_min, q_max], nudged_min = (q_min - zp) * scale, nudged_max = (q_max - zp) * scale.
  - degenerate channel (max <= min): pass gradients through, bp_min/bp_max = 0 (realised with
    -inf/+inf nudged bounds, matching the forward/TF reference).
  - backprops_wrt_x = gradients * mask, mask = (nudged_min <= x <= nudged_max); a NaN element
    yields mask = 0.
  - backprops_wrt_min = sum_{x < nudged_min} gradients, backprops_wrt_max = sum_{x > nudged_max}
    gradients over all leading axes, accumulated in float64 to stay close to the Kahan-compensated
    device accumulation.
"""

import numpy as np

__golden__ = {
    "kernel": {
        "FakeQuantWithMinMaxVarsPerChannelGradient": (
            "fake_quant_with_min_max_vars_per_channel_gradient_golden"
        ),
        # Compatibility alias: some prepared TTK case CSVs bind op_name to the installed
        # opInterface value (snake_case) instead of the GE op_type (CamelCase).
        "fake_quant_with_min_max_vars_per_channel_gradient": (
            "fake_quant_with_min_max_vars_per_channel_gradient_golden"
        ),
    }
}

# Explicit module-level TestSpec registration (newer TTK spec convention) alongside the legacy
# __golden__ mapping above, so both discovery mechanisms resolve this operator.
__spec__ = {
    "FakeQuantWithMinMaxVarsPerChannelGradient": (
        "FakeQuantWithMinMaxVarsPerChannelGradientTestSpec"
    )
}


def _as_tensor(value):
    import torch

    return torch.as_tensor(np.asarray(value, dtype=np.float32))


def fake_quant_with_min_max_vars_per_channel_gradient_golden(
    gradients, x, min, max, *, num_bits=8, narrow_range=False, **kwargs
):
    import torch

    g = _as_tensor(gradients)
    x = _as_tensor(x)
    mn = _as_tensor(min).reshape(-1)
    mx = _as_tensor(max).reshape(-1)
    if x.dim() == 0:
        x = x.reshape(1)

    q_min = 1 if narrow_range else 0
    q_max = 2 ** int(num_bits) - 1
    channel = mn.shape[-1]
    q_range = float(q_max - q_min)

    # Nudge precision: float32, scale = (max - min) / qRange via a single Sub then a single
    # Div, matching the forward FakeQuantWithMinMaxVarsPerChannel and the TF reference
    # (README「约束说明」). Extreme/special min/max (+/-FLT_MAX, +/-inf) are outside the
    # documented domain; this reference keeps the plain fp32 evaluation for them.
    degenerate = mx <= mn
    safe_min = torch.where(degenerate, torch.zeros_like(mn), mn)
    safe_max = torch.where(degenerate, torch.ones_like(mx), mx)
    scale = (safe_max - safe_min) / q_range
    zp = q_min - safe_min / scale
    nudged_zp = torch.clamp(torch.round(zp), float(q_min), float(q_max))
    nudged_min = (q_min - nudged_zp) * scale
    nudged_max = (q_max - nudged_zp) * scale
    nudged_min = torch.where(
        degenerate, torch.full_like(nudged_min, float("-inf")), nudged_min
    )
    nudged_max = torch.where(
        degenerate, torch.full_like(nudged_max, float("inf")), nudged_max
    )

    shape = [1] * x.dim()
    shape[-1] = channel
    n_min = nudged_min.reshape(shape)
    n_max = nudged_max.reshape(shape)

    mask = ((x >= n_min) & (x <= n_max)).to(x.dtype)
    bpx = g * mask

    min_mask = (x < n_min).to(x.dtype)
    max_mask = (x > n_max).to(x.dtype)
    not_deg = (~degenerate).to(x.dtype)
    if x.dim() > 1:
        dims = tuple(range(x.dim() - 1))
        bpm = (g * min_mask).to(torch.float64).sum(dim=dims).to(x.dtype)
        bpmx = (g * max_mask).to(torch.float64).sum(dim=dims).to(x.dtype)
    else:
        bpm = g * min_mask
        bpmx = g * max_mask

    return [
        (bpx * 1.0).numpy(),
        (bpm * not_deg).numpy(),
        (bpmx * not_deg).numpy(),
    ]


class _TfCompetitor:
    def __init__(self, *, num_bits=8, narrow_range=False, **kwargs):
        self.num_bits = int(num_bits)
        self.narrow_range = bool(narrow_range)

    def __call__(self, gradients, x, min, max, **kwargs):
        import tensorflow as tf
        import torch

        def arr(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value)

        return tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient(
            arr(gradients),
            arr(x),
            arr(min),
            arr(max),
            num_bits=self.num_bits,
            narrow_range=self.narrow_range,
        )


class FakeQuantWithMinMaxVarsPerChannelGradientTestSpec:
    """TestSpec for the newer TTK spec convention (Kernel/GEIR route)."""

    def golden(gradients, x, min, max, *, num_bits=8, narrow_range=False, **kwargs):
        return fake_quant_with_min_max_vars_per_channel_gradient_golden(
            gradients, x, min, max, num_bits=num_bits, narrow_range=narrow_range
        )

    third_party = {"tf": _TfCompetitor}
    tolerance = {"float32": {"standard": "mix_tolerance"}}
