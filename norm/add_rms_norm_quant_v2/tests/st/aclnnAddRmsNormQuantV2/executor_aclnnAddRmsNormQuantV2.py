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


def _round_i8(value):
    return torch.clamp(torch.round(value), -128, 127).to(torch.int8)


def _normalize_kwargs(input_data: InputDataset):
    """A2/A3 optionals: optMode bits: 2=scales2, 8=zeroPoints1, 16=zeroPoints2
    (zeroPoints2 requires scales2), 32=beta. scales1 is always required.
    rmsNormOut must stay nullptr on A2/A3 (same-shape combo), so the executor
    only exposes (y1, y2, x)."""
    kwargs = input_data.kwargs
    opt_raw = kwargs.get("optMode")
    if opt_raw is None:
        return {
            "y2_zero": kwargs.get("scales2") is None
            and kwargs.get("smoothScale2") is None
        }
    opt = int(opt_raw)
    kwargs["scales2"] = kwargs.get("scales2") if (opt & 2) else None
    kwargs["zeroPoints1"] = kwargs.get("zeroPoints1") if (opt & 8) else None
    kwargs["zeroPoints2"] = kwargs.get("zeroPoints2") if (opt & 16) else None
    kwargs["beta"] = kwargs.get("beta") if (opt & 32) else None
    return {"y2_zero": kwargs["scales2"] is None}


def _add_inputs(x1, x2, kt):
    if kt == 2:
        return (x1.float() + x2.float()).to(torch.bfloat16)
    return x1 + x2


def _rmsnorm(x, gamma, epsilon):
    xf = x.float()
    row = gamma.numel()
    xr = xf.reshape(-1, row)
    rstd = torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + epsilon)
    return xr * rstd


def _norm_y(norm, gamma, beta, kt):
    if kt == 1:
        y = (norm.to(torch.float16) * gamma).float()
    else:
        y = norm * gamma.float()
    if beta is not None:
        y = y + beta.float()
    return y


def _quant(y, sv, zv, div_mode):
    sv = sv.float()
    base = y / sv if div_mode else y * sv
    if zv is not None:
        base = base + zv.float()
    return _round_i8(base)


def add_rms_norm_quant_v2_golden(
    x1,
    x2,
    gamma,
    scales1,
    scales2,
    zero_points1,
    zero_points2,
    beta,
    axis,
    epsilon,
    div_mode,
    kt,
):
    """Return (y1, y2, x). rmsNormOut stays nullptr on A2/A3."""
    x = _add_inputs(x1, x2, kt)
    norm = _rmsnorm(x, gamma, epsilon)
    y = _norm_y(norm, gamma, beta, kt)
    y1 = _quant(y, scales1, zero_points1, div_mode)
    if scales2 is not None:
        y2 = _quant(y, scales2, zero_points2, div_mode)
    else:
        y2 = torch.zeros(x.shape, dtype=torch.int8)
    return y1, y2, x


@register("aclnn_add_rms_norm_quant_v2")
class FunctionAddRmsNormQuantV2Api(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self._flags = _normalize_kwargs(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = input_data.kwargs
        if self.device == "cpu":
            y1, y2, x = add_rms_norm_quant_v2_golden(
                kwargs["x1"],
                kwargs["x2"],
                kwargs["gamma"],
                kwargs["scales1"],
                kwargs["scales2"],
                kwargs["zeroPoints1"],
                kwargs["zeroPoints2"],
                kwargs["beta"],
                int(kwargs.get("axis") or -1),
                float(kwargs["epsilon"]),
                bool(kwargs["divMode"]),
                int(kwargs.get("kernelType") or 3),
            )
            return y1, y2, x
        return None


@register("sample_aclnn_api")
class SampleAddRmsNormQuantV2Api(AclnnBaseApi):
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
        return outputs
