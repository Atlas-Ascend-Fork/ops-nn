#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""AdamApplyOneAssign 已批准公式与计算精度，最终三输出完整 cast 版本。

接口来源：ops-nn@9bead91b81171d946f93f2daa8988b5181af6a0d 的
optim/adam_apply_one_assign；A2 对照固定 canndev@
a16a8448cc874fca8e1cd7c27670a0353bb0f677 的 910B 注册、TBE 和框架映射。
十个必选 Tensor 输入，无属性；返回顺序为 [input1, input2, input3]。
三个状态输入必须均具有十个输入的完整广播 shape，rank 不超过 8。
目标输入、输出 dtype 一致且仅支持 float16/float32；float64 仅供参考计算。

通用 torch.optim.Adam 接口不等价：目标系数由十个 Tensor 输入直接给出，
不额外引入 step、bias correction 或 weight decay。采用真实 Torch 小算子
mul/add/sqrt/div/sub 组合，由 Golden 和 Torch provider 独立调用共享核心。
保持 input0 平方、系数乘法、相加、开方、加 add2_y、除法、乘 input4、
相减的依赖顺序；不添加 epsilon 裁剪或 NaN/Inf 清洗。Kernel 的 MulAddDst
允许存在融合舍入差异，因此使用 cross_check，而非宣称逐 bit 复现。
按用户指定的参考精度策略，Golden wrapper 将实际 FP16 输入升 FP32 计算；
实际 FP32/FP64 保持，不因 mode/output dtype 元信息改变中间计算精度。
全部计算结束后，三份结果分别完整 cast 到 output_dtypes；直接调用未提供
元信息时，按对应状态 input1/input2/input3 的原 dtype 转换。无 Inf 掩码、
裁剪或 NaN 清洗。直接 FP64 数学诊断仍保留 FP64；它不是设备输出合同。
TTK Promote 的临时输出元信息可能已提升，私有 runtime 在上下文恢复后再按
真实设备 output_dtypes 完整 cast；正式全量沿用 Enable。
共享核心与 provider 保持调用方 dtype/device；不扩展目标的 BF16 支持。
这是参考计算策略，不宣称 A2 指令内部逐步舍入与其逐 bit 相同。
CPU Golden 的 FP32 开方在 FP64 内部求值后立即转回 FP32，其他节点及最终输出转换
保持原策略；Torch provider 仍调用其原生开方，独立于这一参考修正。
Golden/provider 均只返回三个独立结果，不改动输入；TTK 3.0 的 Golden 调用
直接传测试输入，原地更新必须由用例 output_inplace_indexes=[1,2,3] 绑定，
不能在计算参考值时提前改写。provider 入参保持原 dtype/device。
正式 TTK loader 结果及 GEIR 实跑状态见既有检查记录。
"""

import numpy as np
import torch

__spec__ = {"adam_apply_one_assign": "AdamApplyOneAssignTestSpec"}


def _check_inputs(inputs, *, allow_float64=False):
    """核对同 dtype/device、rank 和可原地写回的广播约束。"""
    allowed = (torch.float16, torch.float32)
    if allow_float64:
        allowed += (torch.float64,)
    first = inputs[0]
    if first.dtype not in allowed:
        raise TypeError(
            "expected float16/float32; float64 is only for Golden reference"
        )
    if any(x.dtype != first.dtype or x.device != first.device for x in inputs):
        raise ValueError("all inputs must have the same dtype and device")
    if any(x.ndim > 8 for x in inputs):
        raise ValueError("input rank must not exceed 8")
    broadcast_shape = torch.broadcast_shapes(*(x.shape for x in inputs))
    if any(x.shape != broadcast_shape for x in inputs[1:4]):
        raise ValueError("input1/input2/input3 must have the full broadcast shape")


def _cpu_reference_sqrt(value):
    """Keep the CPU reference's FP32 sqrt result independent of FP32 VML rounding.

    FP64 is used only inside this primitive; the result is immediately rounded
    back to FP32 before the denominator and parameter update. Native FP64
    reference calls retain their precision. The Torch provider uses its native
    sqrt through the compute core's default argument.
    """
    if value.device.type == "cpu" and value.dtype == torch.float32:
        return torch.sqrt(value.to(dtype=torch.float64)).to(dtype=torch.float32)
    return torch.sqrt(value)


def _torch_compute(
    input0,
    input1,
    input2,
    input3,
    input4,
    mul0_x,
    mul1_x,
    mul2_x,
    mul3_x,
    add2_y,
    *,
    sqrt_fn=torch.sqrt,
):
    """仅 Tensor 公式计算；保持调用方 dtype/device，不修改任何输入。"""
    square = torch.mul(input0, input0)
    out_input1 = torch.add(torch.mul(input1, mul2_x), torch.mul(square, mul3_x))
    out_input2 = torch.add(torch.mul(mul0_x, input2), torch.mul(mul1_x, input0))
    denominator = torch.add(sqrt_fn(out_input1), add2_y)
    step = torch.mul(torch.div(out_input2, denominator), input4)
    out_input3 = torch.sub(input3, step)
    return [out_input1, out_input2, out_input3]


class AdamApplyOneAssignTestSpec:
    """Kernel/GEIR 共用唯一注册；保留原参考精度，最终输出完整 cast。"""

    def golden(
        input0,
        input1,
        input2,
        input3,
        input4,
        mul0_x,
        mul1_x,
        mul2_x,
        mul3_x,
        add2_y,
        **kwargs,
    ):
        inputs = (
            input0,
            input1,
            input2,
            input3,
            input4,
            mul0_x,
            mul1_x,
            mul2_x,
            mul3_x,
            add2_y,
        )
        # copy 保留 0D shape、支持非连续/负 stride，不改动 TTK 的测试输入。
        tensors = [torch.from_numpy(np.array(x, copy=True, order="C")) for x in inputs]
        # 参考精度只按实际 dtype 决定，不能由 mode 或输出元信息降精度。
        _check_inputs(tensors, allow_float64=True)
        compute_tensors = [
            x.to(dtype=torch.float32) if x.dtype == torch.float16 else x
            for x in tensors
        ]
        results = _torch_compute(*compute_tensors, sqrt_fn=_cpu_reference_sqrt)
        output_dtypes = kwargs.get("output_dtypes")
        if output_dtypes is None:
            output_dtypes = tuple(np.asarray(inputs[i]).dtype for i in (1, 2, 3))
        if len(output_dtypes) != len(results):
            raise ValueError("expected one output dtype for each of the three states")
        # Cast every element after the entire reference graph has completed.
        # Intermediate v/m remain at reference precision while computing p.
        with np.errstate(over="ignore", invalid="ignore"):
            return [
                result.detach().cpu().numpy().astype(np.dtype(dtype), copy=False)
                for result, dtype in zip(results, output_dtypes)
            ]

    class ThirdPartyImpl:
        def __init__(self, **kwargs):
            pass

        def __call__(
            self,
            input0,
            input1,
            input2,
            input3,
            input4,
            mul0_x,
            mul1_x,
            mul2_x,
            mul3_x,
            add2_y,
            **kwargs,
        ):
            inputs = (
                input0,
                input1,
                input2,
                input3,
                input4,
                mul0_x,
                mul1_x,
                mul2_x,
                mul3_x,
                add2_y,
            )
            _check_inputs(inputs)
            # provider 入参保持原 dtype/device，原地输出绑定由 TTK 执行层处理。
            with torch.no_grad():
                return _torch_compute(*inputs)

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
