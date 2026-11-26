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
#

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_aclnn_dynamic_block_quant")
class aclnnDynamicblockquant(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input = input_data.kwargs["x"]
        min_scale = input_data.kwargs["min_scale"]
        input_shape = input.shape

        if self.device == "cpu":
            x = input.reshape(-1, input_shape[-1])

            m, n = x.shape
            pad_size = (128 - (n % 128)) % 128
            x = input.view((m, n))
            x = (
                torch.nn.functional.pad(x, (0, pad_size), value=0)
                if pad_size > 0
                else x
            )
            x_view = x.view(m, -1, 128)
            x_amax = x_view.abs().float().amax(dim=2).view(m, -1, 1)

            scale = x_amax / 127.0
            x_amax = 127 / x_amax
            if min_scale > 0:
                scale = torch.minimum(scale, torch.tensor(1 / min_scale))
                x_amax = torch.maximum(x_amax, torch.tensor(min_scale))
            y_data = x_view * x_amax

            y_data = y_data.round()
            y_data = torch.clamp(y_data, -128, 127).to(torch.int8)
            y_data = y_data.view(m, n + pad_size)[:, :n].reshape(input_shape)
            scale = scale.reshape(*input_shape[:-1], -1)
            y_data = y_data.contiguous()
            scale = scale.contiguous()

            return y_data, scale
