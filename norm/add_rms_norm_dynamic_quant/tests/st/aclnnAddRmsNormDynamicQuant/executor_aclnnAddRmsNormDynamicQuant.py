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
    kwargs = input_data.kwargs
    opt_raw = kwargs.get("optMode")
    if opt_raw is None:
        return {
            "y2_zero": kwargs.get("scales2") is None
            and kwargs.get("smoothScale2") is None
        }
    opt = int(opt_raw)
    kwargs["smoothScale1"] = kwargs.get("smoothScale1") if (opt & 2) else None
    kwargs["smoothScale2"] = kwargs.get("smoothScale2") if (opt & 4) else None
    return {"s2_on": kwargs["smoothScale2"] is not None}


def _add_inputs(x1, x2, kt):
    if kt == 2:
        return (x1.float() + x2.float()).to(torch.bfloat16)
    return x1 + x2


def _rmsnorm_y(x, gamma, kt, epsilon):
    xf = x.float()
    row = gamma.numel()
    xr = xf.reshape(-1, row)
    rstd = torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + epsilon)
    norm = xr * rstd
    if kt == 1:
        y = (norm.to(torch.float16) * gamma).float()
    else:
        y = norm * gamma.float()
    return y


def _quant_dynamic(y, sv, row, scale_shape):
    yr = y.reshape(-1, row)
    tmp = yr if sv is None else yr * sv.float()
    row_max = tmp.abs().amax(-1)
    out_scale = (row_max / 127.0).float()
    denom = out_scale.unsqueeze(-1).clone()
    denom[denom == 0] = 1.0
    yq = _round_i8(tmp / denom).reshape(y.shape)
    return yq, out_scale.reshape(scale_shape)


def add_rms_norm_dynamic_quant_golden(x1, x2, gamma, smooth1, smooth2, epsilon, kt):
    """Return (y1, y2, x, scale1, scale2). Both quant paths are computed by the
    public API; a missing smooth scale simply quantizes the plain y."""
    x = _add_inputs(x1, x2, kt)
    y = _rmsnorm_y(x, gamma, kt, epsilon)
    row = gamma.numel()
    scale_shape = tuple(x.shape[:-1]) if x.dim() > 1 else (1,)
    y1, scale1 = _quant_dynamic(y, smooth1, row, scale_shape)
    y2, scale2 = _quant_dynamic(y, smooth2, row, scale_shape)
    return y1, y2, x, scale1, scale2


@register("aclnn_add_rms_norm_dynamic_quant")
class FunctionAddRmsNormDynamicQuantApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self._flags = _normalize_kwargs(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = input_data.kwargs
        if self.device == "cpu":
            y1, y2, x, scale1, scale2 = add_rms_norm_dynamic_quant_golden(
                kwargs["x1"],
                kwargs["x2"],
                kwargs["gamma"],
                kwargs["smoothScale1"],
                kwargs["smoothScale2"],
                float(kwargs["epsilon"]),
                int(kwargs.get("kernelType") or 3),
            )
            return y1, y2, x, scale1, scale2
        return None


@register("sample_aclnn_api")
class SampleAddRmsNormDynamicQuantApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self._flags = _normalize_kwargs(input_data)
        del input_data.kwargs["kernelType"]
        del input_data.kwargs["optMode"]
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages

    def after_call(self, output_packages):
        outputs = super().after_call(output_packages)
        return outputs
