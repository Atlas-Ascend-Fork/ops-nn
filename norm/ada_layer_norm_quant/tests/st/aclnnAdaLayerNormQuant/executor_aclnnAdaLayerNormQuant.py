#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


def _optional(v):
    # shape 为 [] 的占位张量表示该可选输入未传入
    return None if v is None or v.shape == torch.Size([]) else v


@register("function_aclnnAdaLayerNormQuant")
class FunctionaclnnAdaLayerNormQuant(BaseApi):
    # 精度对比标杆
    def AdaLayerNormQuantGolden(
        self,
        x,
        weightOptional,
        biasOptional,
        scale,
        shift,
        smoothScalesOptional,
        epsilon,
    ):
        # 计算流程与 aclnnAdaLayerNormQuant 文档一致：LayerNorm 归一化 ->
        # 自适应参数 (1 + scale) / shift 调整 -> smoothScales 平滑 ->
        # 按行求最大绝对值计算量化系数 -> 除以系数四舍五入得到 INT8 输出
        x = x.to(torch.float32)
        weight = None if weightOptional is None else weightOptional.to(torch.float32)
        bias = None if biasOptional is None else biasOptional.to(torch.float32)
        smooth = (
            None
            if smoothScalesOptional is None
            else smoothScalesOptional.to(torch.float32)
        )
        scale = scale.to(torch.float32)
        shift = shift.to(torch.float32)

        factor = torch.tensor((1.0 / x.shape[-1]), dtype=torch.float32)
        mean = torch.sum(x * factor, dim=-1)
        x_mean = x - mean.unsqueeze(-1)
        var = torch.sum(x_mean * x_mean * factor, dim=-1)
        rstd = 1.0 / torch.sqrt(var + epsilon)
        y = x_mean * rstd.unsqueeze(-1)
        if weight is not None:
            y = y * weight
        if bias is not None:
            y = y + bias

        # scale/shift 形如 [B..., H] 或 [B..., 1, H]，统一广播到 [B..., 1, H]
        if len(scale.shape) != len(x.shape):
            scale = scale.unsqueeze(-2)
        if len(shift.shape) != len(x.shape):
            shift = shift.unsqueeze(-2)
        y = y * (1 + scale) + shift
        if smooth is not None:
            y = y * smooth

        # dynamic INT8 量化：quantScale = row_max(abs(y)) / 127，out = round(y / quantScale)
        row = y.reshape(-1, y.shape[-1])
        row_max = row.abs().amax(-1)
        quant_scale = (row_max / 127.0).to(torch.float32)
        denom = quant_scale.unsqueeze(-1).clone()
        denom[denom == 0] = 1.0
        out = torch.clamp(torch.round(row / denom), -128, 127).to(torch.int8)
        return out.reshape(x.shape), quant_scale.reshape(x.shape[:-1])

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            input_data.kwargs["epsilon"] = ctypes.c_double(input_data.kwargs["epsilon"])
        quant_mode = str(input_data.kwargs.get("quantMode", "dynamic"))
        input_data.kwargs["quantMode"] = quant_mode.strip().strip('"').strip("'")

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        scale = input_data.kwargs["scale"]
        shift = input_data.kwargs["shift"]
        weightOptional = _optional(input_data.kwargs["weightOptional"])
        biasOptional = _optional(input_data.kwargs["biasOptional"])
        smoothScalesOptional = _optional(input_data.kwargs["smoothScalesOptional"])
        epsilon = float(input_data.kwargs["epsilon"])

        out, quant_scale = self.AdaLayerNormQuantGolden(
            x, weightOptional, biasOptional, scale, shift, smoothScalesOptional, epsilon
        )
        return out, quant_scale


@register("function_aclnnAdaLayerNormQuant_pgacl")
class FunctionaclnnAdaLayerNormQuant_pgacl(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        # 输出依次为 out(INT8)、quantScale(FLOAT32)，dtype 不随入参变化
        for index, output_data in enumerate(self.task_result.output_info_list):
            output_data.dtype = "torch.int8" if index == 0 else "torch.float32"

        input_args, output_packages = super().init_by_input_data(input_data)

        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

        # 入参顺序与 C 接口一致：x/scale/shift/weightOptional/biasOptional/
        # smoothScalesOptional，可选输入未传时置 nullptr
        if _optional(input_data.kwargs["weightOptional"]) is None:
            input_args[3] = TensorPtr()
        if _optional(input_data.kwargs["biasOptional"]) is None:
            input_args[4] = TensorPtr()
        if _optional(input_data.kwargs["smoothScalesOptional"]) is None:
            input_args[5] = TensorPtr()
        # quantOffsetOptional 当前版本不支持，只能传 nullptr
        input_args.append(TensorPtr())
        return input_args, output_packages
