#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


@register("layer_norm_backward")
class AclnnLayerNormBackward(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        origin_dtype = input_data.kwargs["input"].dtype
        gradOut = input_data.kwargs["gradOut"]
        input = input_data.kwargs["input"]
        normalizedShape = input_data.kwargs["normalizedShape"]
        mean = input_data.kwargs["mean"]
        rstd = input_data.kwargs["rstd"]
        weight = input_data.kwargs["weightOptional"]
        bias = input_data.kwargs["biasOptional"]
        outMask = input_data.kwargs["outputMask"]

        if self.device == "cpu" and (
            input.dtype == torch.float16 or input.dtype == torch.bfloat16
        ):
            res = torch.ops.aten.native_layer_norm_backward(
                gradOut.to(torch.float32),
                input.to(torch.float32),
                normalizedShape,
                mean.to(torch.float32),
                rstd.to(torch.float32),
                weight.to(torch.float32),
                bias.to(torch.float32),
                outMask,
            )
            output = tuple(tensor.to(origin_dtype) for tensor in res)
        else:
            output = torch.ops.aten.native_layer_norm_backward(
                gradOut, input, normalizedShape, mean, rstd, weight, bias, outMask
            )
        return output
