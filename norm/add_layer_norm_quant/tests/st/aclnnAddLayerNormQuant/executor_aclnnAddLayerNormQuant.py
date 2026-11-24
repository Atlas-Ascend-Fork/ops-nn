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

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


def _kernel_dtype(kt):
    if kt == 1:
        return torch.float16
    if kt == 2:
        return torch.bfloat16
    return torch.float32


def _round_i8(value):
    return torch.clamp(torch.round(value), -128, 127).to(torch.int8)


def _normalize_kwargs(input_data: InputDataset):
    """Normalize optional inputs by the optMode bitmask and record which
    outputs the kernel leaves undefined (zeroed on both paths)."""
    kwargs = input_data.kwargs
    opt_raw = kwargs.get("optMode")
    if opt_raw is not None:
        opt = int(opt_raw)
        kwargs["bias"] = kwargs.get("bias") if (opt & 1) else None
        kwargs["scales1"] = kwargs.get("scales1") if (opt & 2) else None
        kwargs["scales2"] = kwargs.get("scales2") if (opt & 4) else None
        kwargs["zeroPoints1"] = kwargs.get("zeroPoints1") if (opt & 8) else None
        kwargs["zeroPoints2"] = kwargs.get("zeroPoints2") if (opt & 16) else None
    quant_mode = str(kwargs.get("quantMode", "static")).strip().strip('"').strip("'")
    kwargs["quantMode"] = quant_mode
    if quant_mode != "static":
        kwargs["zeroPoints1"] = None
        kwargs["zeroPoints2"] = None
    flags = {
        "static": quant_mode == "static",
        "s2_on": kwargs["scales2"] is not None,
        "y2_zero": kwargs["scales2"] is None,
    }
    return flags


def _add_inputs(x1, x2, bias, kt):
    if kt == 2:
        x = x1.float() + x2.float()
        if bias is not None:
            x = x + bias.float()
        x = x.to(torch.bfloat16)
    else:
        x = x1 + x2
        if bias is not None:
            x = x + bias.to(x1.dtype)
    return x


def _layernorm_y(x, gamma, beta, kt, epsilon):
    xf = x.float()
    row = gamma.numel()
    xr = xf.reshape(-1, row)
    mean = xr.mean(-1, keepdim=True)
    var = xr.var(-1, unbiased=False, keepdim=True)
    rstd = torch.rsqrt(var + epsilon)
    norm = (xr - mean) * rstd
    if kt == 1:
        y = (norm.to(torch.float16) * gamma).float()
    else:
        y = norm * gamma.float()
    if beta is not None:
        y = y + beta.float()
    return y


def _quant_static(y, sv, zv, div_mode):
    sv = sv.float()
    base = y / sv if div_mode else y * sv
    if zv is not None:
        base = base + zv.float()
    return _round_i8(base)


def _quant_dynamic(y, sv, row, os_shape):
    yr = y.reshape(-1, row)
    tmp = yr if sv is None else yr * sv.float()
    row_max = tmp.abs().amax(-1)
    out_scale = (row_max / 127.0).float()
    denom = out_scale.unsqueeze(-1).clone()
    denom[denom == 0] = 1.0
    yq = _round_i8(tmp / denom).reshape(y.shape)
    return yq, out_scale.reshape(os_shape)


def add_layer_norm_quant_golden(
    x1,
    x2,
    gamma,
    beta,
    bias,
    scales1,
    scales2,
    zero_points1,
    zero_points2,
    quant_mode,
    epsilon,
    div_mode,
    kt,
):
    """Return (y1, y2, x, out_scales1, out_scales2). Undefined outputs are
    returned as zeros and zeroed on the NPU path as well."""
    x = _add_inputs(x1, x2, bias, kt)
    y = _layernorm_y(x, gamma, beta, kt, epsilon)
    os_shape = tuple(x.shape[:-1]) if x.dim() > 1 else (1,)
    if quant_mode == "static":
        y1 = _quant_static(y, scales1, zero_points1, div_mode)
        if scales2 is not None:
            y2 = _quant_static(y, scales2, zero_points2, div_mode)
        else:
            y2 = torch.zeros(x.shape, dtype=torch.int8)
        os1 = torch.zeros(os_shape, dtype=torch.float32)
        os2 = torch.zeros(os_shape, dtype=torch.float32)
    else:
        row = gamma.numel()
        y1, os1 = _quant_dynamic(y, scales1, row, os_shape)
        if scales2 is not None:
            y2, os2 = _quant_dynamic(y, scales2, row, os_shape)
        else:
            y2 = torch.zeros(x.shape, dtype=torch.int8)
            os2 = torch.zeros(os_shape, dtype=torch.float32)
    return y1, y2, x, os1, os2


@register("aclnn_add_layer_norm_quant")
class FunctionAddLayerNormQuantApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self._flags = _normalize_kwargs(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = input_data.kwargs
        if self.device == "cpu":
            y1, y2, x, os1, os2 = add_layer_norm_quant_golden(
                kwargs["x1"],
                kwargs["x2"],
                kwargs["gamma"],
                kwargs["beta"],
                kwargs["bias"],
                kwargs["scales1"],
                kwargs["scales2"],
                kwargs["zeroPoints1"],
                kwargs["zeroPoints2"],
                kwargs["quantMode"],
                float(kwargs["epsilon"]),
                bool(kwargs["divMode"]),
                int(kwargs.get("kernelType") or 3),
            )
            return y1, y2, x, os1, os2
        return None


@register("sample_aclnn_api")
class SampleAddLayerNormQuantApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self._flags = _normalize_kwargs(input_data)
        del input_data.kwargs["kernelType"]
        del input_data.kwargs["optMode"]
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages

    def after_call(self, output_packages):
        outputs = super().after_call(output_packages)
        flags = getattr(self, "_flags", {})
        if outputs is None:
            return outputs
        if flags.get("y2_zero"):
            outputs[1].zero_()
        if flags.get("static"):
            outputs[3].zero_()
            outputs[4].zero_()
        elif flags.get("y2_zero"):
            outputs[4].zero_()
        return outputs
