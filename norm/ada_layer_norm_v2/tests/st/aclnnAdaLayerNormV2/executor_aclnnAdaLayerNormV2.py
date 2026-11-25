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


@register("function_aclnnAdaLayerNormV2")
class FunctionaclnnAdaLayerNormV2(BaseApi):
    # 精度对比标杆
    def AdaLayerNormV2Golden(
        self, x, weightOptional, biasOptional, scale, shift, epsilon
    ):
        # 计算流程与 aclnnAdaLayerNormV2 文档一致：LayerNorm 归一化 ->
        # 自适应参数 (1 + scale) / shift 调整；相比 AdaLayerNorm 额外返回
        # 输入的均值 mean 与标准差倒数 rstd（最后一维固定为 1）
        origin_dtype = x.dtype
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(torch.float32)
            weightOptional = (
                None if weightOptional is None else weightOptional.to(torch.float32)
            )
            biasOptional = (
                None if biasOptional is None else biasOptional.to(torch.float32)
            )
            scale = scale.to(torch.float32)
            shift = shift.to(torch.float32)
        factor = torch.tensor((1.0 / x.shape[-1]), dtype=torch.float32)

        mean = torch.sum(x * factor, dim=-1)
        x_mean = x - mean.unsqueeze(-1)
        var = torch.sum(x_mean * x_mean * factor, dim=-1)
        rstd = 1.0 / torch.sqrt(var + epsilon)
        y = x_mean * rstd.unsqueeze(-1)
        if weightOptional is not None:
            y = y * weightOptional
        if biasOptional is not None:
            y = y + biasOptional

        # scale/shift 形如 [B..., H] 或 [B..., 1, H]，统一广播到 [B..., 1, H]
        if len(scale.shape) != len(x.shape):
            scale = scale.unsqueeze(-2)
        if len(shift.shape) != len(x.shape):
            shift = shift.unsqueeze(-2)
        out = y * (1 + scale) + shift

        # out/meanOutOptional/rstdOutOptional 的 dtype 与入参 x 一致
        out = out.to(origin_dtype)
        mean_out = mean.unsqueeze(-1).to(origin_dtype)
        rstd_out = rstd.unsqueeze(-1).to(origin_dtype)
        return out, mean_out, rstd_out

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "pyaclnn":
            input_data.kwargs["epsilon"] = ctypes.c_double(input_data.kwargs["epsilon"])

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        scale = input_data.kwargs["scale"]
        shift = input_data.kwargs["shift"]
        weightOptional = _optional(input_data.kwargs["weightOptional"])
        biasOptional = _optional(input_data.kwargs["biasOptional"])
        epsilon = float(input_data.kwargs["epsilon"])

        out, mean_out, rstd_out = self.AdaLayerNormV2Golden(
            x, weightOptional, biasOptional, scale, shift, epsilon
        )
        return out, mean_out, rstd_out


@register("function_aclnnAdaLayerNormV2_pgacl")
class FunctionaclnnAdaLayerNormV2_pgacl(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        x = input_data.kwargs["x"]
        # 输出依次为 out/meanOutOptional/rstdOutOptional，dtype 均与入参 x 一致
        for index, output_data in enumerate(self.task_result.output_info_list):
            output_data.dtype = str(x.dtype)

        input_args, output_packages = super().init_by_input_data(input_data)

        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

        # 入参顺序与 C 接口一致：x/scale/shift/weightOptional/biasOptional，
        # 可选输入未传时置 nullptr
        weightOptional = _optional(input_data.kwargs["weightOptional"])
        if weightOptional is None:
            input_args[3] = TensorPtr()
        biasOptional = _optional(input_data.kwargs["biasOptional"])
        if biasOptional is None:
            input_args[4] = TensorPtr()
        return input_args, output_packages
