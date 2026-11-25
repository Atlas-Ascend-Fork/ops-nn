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

import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


def _optional(v):
    # shape 为 [] 的占位张量表示该可选输入未传入
    return None if v is None or v.shape == torch.Size([]) else v


_STR_TO_DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _declared_dtype(task_result, name):
    # 读取用例 JSON 中声明的 dtype（兼容 dict 与 InputCaseConfig 对象两种形态）。
    # single_bm 标杆侧的输入会被 ATK 升精度（fp16/bf16 -> fp32），mean/rstd
    # 必须按声明 dtype 舍入，保证标杆计算与 NPU 侧收到的入参取值完全一致
    try:
        for item in task_result.case_config.inputs:
            if isinstance(item, (list, tuple)):
                continue
            if isinstance(item, dict):
                item_name, item_dtype = item.get("name"), item.get("dtype")
            else:
                item_name = getattr(item, "name", None)
                item_dtype = getattr(item, "dtype", None)
            if item_name == name:
                return _STR_TO_DTYPE.get(str(item_dtype), torch.float32)
    except Exception:
        pass
    return torch.float32


def _synth_mean_rstd(input_tensor, out_dtype, epsilon=1e-5):
    # mean/rstd 是反向接口的入参，按正向语义由 input 现算后舍入到声明 dtype
    # （json 中声明为占位 shape），形状 [B..., S, 1]
    xf = input_tensor.to(torch.float32)
    factor = torch.tensor((1.0 / xf.shape[-1]), dtype=torch.float32)
    mean = torch.sum(xf * factor, dim=-1, keepdim=True)
    x_mean = xf - mean
    var = torch.sum(x_mean * x_mean * factor, dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + epsilon)
    return rstd.to(out_dtype), mean.to(out_dtype)


def _fill_affine(input, weightOptional, biasOptional):
    # weight/bias 未传时按 1/0 补齐（与 aclnn 侧填充语义一致）：
    # 仅 bias 传入时 weight 补齐为与 bias 同 dtype 的 1（对应算子侧
    # gradWeightOut 须与 biasOptional 同 dtype 的校验规则）
    if weightOptional is not None:
        weight = weightOptional
    elif biasOptional is not None:
        weight = torch.ones_like(biasOptional)
    else:
        weight = torch.ones(input.shape[-1], dtype=torch.float32)
    if biasOptional is not None:
        bias = biasOptional
    else:
        bias = torch.zeros_like(weight)
    return weight, bias


@register("function_aclnnAdaLayerNormBackward")
class FunctionaclnnAdaLayerNormBackward(BaseApi):
    # 精度对比标杆
    def AdaLayerNormBackwardGolden(
        self, gradOut, input, rstd, mean, scale, shift, weightOptional, biasOptional
    ):
        # 计算流程与 aclnnAdaLayerNormBackward 文档一致（z = (input - mean) * rstd）：
        #   dy_g  = gradOut * weight * (1 + scale)
        #   temp1 = sum(dy_g, H) / N；temp2 = sum(dy_g * z, H) / N
        #   gradInputOut  = (dy_g - temp1 - z * temp2) * rstd
        #   gradScaleOut  = sum(gradOut * (weight * z + bias), S)
        #   gradShiftOut  = sum(gradOut, S)
        #   gradWeightOut = sum(gradOut * (1 + scale) * z, [B..., S])
        #   gradBiasOut   = sum(gradOut * (1 + scale), [B..., S])
        grad_dtype = gradOut.dtype
        weight, bias = _fill_affine(input, weightOptional, biasOptional)
        weight_dtype = weight.dtype

        gradOut = gradOut.to(torch.float32)
        input = input.to(torch.float32)
        rstd = rstd.to(torch.float32)
        mean = mean.to(torch.float32)
        scale = scale.to(torch.float32)
        weight = weight.to(torch.float32)
        bias = bias.to(torch.float32)

        n = input.shape[-1]
        z = (input - mean) * rstd
        # scale 形如 [B..., H] 或 [B..., 1, H]，统一 reshape 成 [B..., 1, H] 广播
        scale_b = scale.reshape(input.shape[:-2] + (1, input.shape[-1]))
        dy_g = gradOut * weight * (1 + scale_b)
        temp1 = dy_g.sum(-1, keepdim=True) / n
        temp2 = (dy_g * z).sum(-1, keepdim=True) / n
        grad_input = (dy_g - temp1 - z * temp2) * rstd

        # 对 S 维（倒数第二维）规约；gradScale/gradShift 的 shape 与 scale/shift 一致
        grad_scale = (gradOut * (weight * z + bias)).sum(-2)
        grad_shift = gradOut.sum(-2)
        # 对 B...、S 维整体规约；gradWeight/gradBias 的 shape 为 [H]
        reduce_axes = tuple(range(input.dim() - 1))
        grad_weight = (gradOut * (1 + scale_b) * z).sum(reduce_axes)
        grad_bias = (gradOut * (1 + scale_b)).sum(reduce_axes)

        return (
            grad_input.reshape(input.shape).to(grad_dtype),
            grad_scale.reshape(scale.shape).to(grad_dtype),
            grad_shift.reshape(shift.shape).to(grad_dtype),
            grad_weight.reshape(weight.shape).to(weight_dtype),
            grad_bias.reshape(bias.shape).to(weight_dtype),
        )

    def init_by_input_data(self, input_data: InputDataset):
        # 按用例声明的 dtype 舍入，规避 ATK 标杆侧输入升精度带来的两侧取值不一致
        out_dtype = _declared_dtype(self.task_result, "input")
        input_data.kwargs["rstd"], input_data.kwargs["mean"] = _synth_mean_rstd(
            input_data.kwargs["input"], out_dtype
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradOut = input_data.kwargs["gradOut"]
        input = input_data.kwargs["input"]
        rstd = input_data.kwargs["rstd"]
        mean = input_data.kwargs["mean"]
        scale = input_data.kwargs["scale"]
        shift = input_data.kwargs["shift"]
        weightOptional = _optional(input_data.kwargs["weightOptional"])
        biasOptional = _optional(input_data.kwargs["biasOptional"])

        return self.AdaLayerNormBackwardGolden(
            gradOut, input, rstd, mean, scale, shift, weightOptional, biasOptional
        )


@register("function_aclnnAdaLayerNormBackward_pgacl")
class FunctionaclnnAdaLayerNormBackward_pgacl(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        # mean/rstd 为占位 shape，先按 input 现算回填（舍入到声明 dtype），
        # 再交给框架转换
        out_dtype = _declared_dtype(self.task_result, "input")
        input_data.kwargs["rstd"], input_data.kwargs["mean"] = _synth_mean_rstd(
            input_data.kwargs["input"], out_dtype
        )

        grad_dtype = input_data.kwargs["gradOut"].dtype
        weightOptional = _optional(input_data.kwargs["weightOptional"])
        biasOptional = _optional(input_data.kwargs["biasOptional"])
        # 输出依次为 gradInputOut/gradScaleOut/gradShiftOut/gradWeightOut/gradBiasOut，
        # 前 3 个与 gradOut 同 dtype，后 2 个与传入的 affine 参数同 dtype：
        # weight 优先，其次 bias（算子校验 gradWeightOut 与 biasOptional 同 dtype），
        # 均未传时与 input 同 dtype
        if weightOptional is not None:
            weight_dtype = weightOptional.dtype
        elif biasOptional is not None:
            weight_dtype = biasOptional.dtype
        else:
            weight_dtype = out_dtype
        for index, output_data in enumerate(self.task_result.output_info_list):
            output_data.dtype = str(grad_dtype) if index < 3 else str(weight_dtype)

        input_args, output_packages = super().init_by_input_data(input_data)

        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

        # 入参顺序与 C 接口一致：gradOut/input/normalizedShape/rstd/mean/scale/
        # shift/weightOptional/biasOptional，可选输入未传时置 nullptr
        if weightOptional is None:
            input_args[7] = TensorPtr()
        if biasOptional is None:
            input_args[8] = TensorPtr()
        return input_args, output_packages
