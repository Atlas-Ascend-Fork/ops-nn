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


def add_rms_norm_cast_golden(x1, x2, gamma, epsilon, kt):
    """A2/A3 reference path (add_rms_cast_v1): the residual sum is kept in the
    input dtype, then reduced in FP32. Return (y1, y2, rstd, x)."""
    if kt == 2:
        x = (x1.float() + x2.float()).to(torch.bfloat16)
    else:
        x = x1 + x2

    x_fp32 = x.float()
    row = gamma.numel()
    xr = x_fp32.reshape(-1, row)
    rstd = torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + epsilon)
    rstd = rstd.reshape(*x_fp32.shape[:-1], 1)

    tmp_x = x_fp32 * rstd
    if kt == 2:
        tmp_x = tmp_x.to(torch.bfloat16).float()
        y2 = (tmp_x * gamma.float()).to(torch.bfloat16)
    elif kt == 1:
        tmp_x = tmp_x.to(torch.float16)
        y2 = tmp_x * gamma
    else:
        y2 = tmp_x * gamma
    y1 = y2.float().clone()
    return y1, y2, rstd, x


@register("aclnn_add_rms_norm_cast")
class FunctionAddRmsNormCastApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        pass

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kwargs = input_data.kwargs
        if self.device == "cpu":
            y1, y2, rstd, x = add_rms_norm_cast_golden(
                kwargs["x1"],
                kwargs["x2"],
                kwargs["gamma"],
                float(kwargs["epsilon"]),
                int(kwargs.get("kernelType") or 3),
            )
            return y1, y2, rstd, x
        return None


@register("sample_aclnn_api")
class SampleAddRmsNormCastApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        del input_data.kwargs["kernelType"]
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages

    def after_call(self, output_packages):
        output1, output2, output3, output4 = super().after_call(output_packages)
        return output1, output2, output3, output4
