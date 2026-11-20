#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Golden for aclnnBatchNormElemt (arch35 TTK cases).

公式（与 docs/aclnnBatchNormElemt.md 一致）：y = (x - mean) * invstd * weight + bias，
invstd 直接作为乘子参与计算（含负值符号）；eps 为兼容保留参数，不参与计算。

TTK 按 aclnn C 头文件签名顺序传参（含输出张量 output，golden 忽略之）；
tensor_formats 等信息经 kwargs 传入（见 ops-test-kit golden_generation._call_custom_golden）。
输入可能是 numpy（bf16 经 ml_dtypes）或 torch 张量，统一转 torch float64 计算。

不声明 third_party 三方标杆：主流框架同名算子无 CPU 实现（dispatcher 报
NotImplementedError）、拒绝 fp16/bf16 统计量（本算子为超集兼容）且在部分版本被移除，
无法覆盖本用例集，采用两方对比（golden vs NPU）。

判据选型为 mix_tolerance（生态算子开源精度标准）：batch_norm_elemt 的输出存在
灾难性抵消导致的近零元素（|y|~1e-6 而中间量 |A|~1e-1），stat_rel_err 的 mare
对任一正确 fp32 实现（含参考实现，已实测同样 FAIL）在该分布下固有误报；
mix_tolerance 的 atol 下限（fp32: 2^-16）正确处理近零输出，同时保持
rtol（2^-10）紧公差与 99% 匹配率要求。cross_check 的 small_value 处理同样适用
（batch_norm3_d 家族惯例），但其为三方对比、依赖 third-party endpoint（fail-closed），
本算子 third-party 不可用，故不选。
"""

import numpy as np
import torch

__spec__ = {
    "aclnnBatchNormElemt": "BatchNormElemtAclnnSpec",
    "torch.ops.aten.batch_norm_elemt.default": "TorchBatchNormElemtSpec",  # E2E
}

_TOLERANCE = {
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
}

_TORCH_OF = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def _to_torch(t):
    """TTK 传入 numpy 或 torch 张量，统一转 torch CPU（bf16 numpy 经 uint16 位视图无损转换）。"""
    if t is None:
        return None
    if hasattr(t, "detach"):
        return t.detach().cpu()
    arr = np.ascontiguousarray(t)
    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.view(np.float16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _to_orig_form(y, input):
    """结果按输入形态返回：torch 进 torch 出，numpy 进 numpy 出（bf16 位视图）。"""
    if hasattr(input, "detach"):
        return [y.to(input.dtype)]
    arr = np.asarray(input)
    t_dt = _TORCH_OF.get(arr.dtype.name, torch.float32)
    if arr.dtype.name == "bfloat16":
        return [y.to(t_dt).view(torch.float16).numpy().view(arr.dtype)]
    return [y.to(t_dt).numpy().astype(arr.dtype, copy=False)]


def _elemt_core(input, weight, bias, mean, invstd, fmt):
    """公共计算核：y = (x - mean) * invstd * weight + bias（float64，含末维通道感知）。"""
    x = _to_torch(input)
    last_is_c = fmt in ("NDHWC", "NHWC")

    def view64(t):
        # 末维通道：一维 (C,) 直接广播到末维；其余：按 dim1 广播
        t = _to_torch(t).to(torch.float64)
        return t.view(-1) if last_is_c else t.view([1, -1] + [1] * (x.dim() - 2))

    m = view64(mean)
    s_ = view64(invstd)
    w = view64(weight) if weight is not None else 1.0
    b = view64(bias) if bias is not None else 0.0
    return (x.to(torch.float64) - m) * s_ * w + b


def batch_norm_elemt_golden(
    input, weight=None, bias=None, mean=None, invstd=None, eps=1e-5, **kwargs
):
    """E2E golden：torch.ops.aten.batch_norm_elemt，torch 入/出，单输出 y。

    参数位置跟随 aten schema 形参序（input/weight/bias/mean/invstd/eps，
    无输出占位）；eps 同样不参与计算。

    通道固定按 dim=1 广播：torch 的 channels_last/channels_last_3d 仅改变
    步长（物理布局），逻辑 shape 恒为 (N,C,...)，不因 tensor_formats 标注
    的存储格式改变逻辑通道位置。这与 aclnn 入口不同——aclnn 通路中
    NDHWC/NHWC 存储格式的 view shape 通道确实在末维。
    """
    y = _elemt_core(input, weight, bias, mean, invstd, None)
    return _to_orig_form(y, input)


class TorchBatchNormElemtSpec:
    """e2e 流程：torch 入/出，单输出 y（对齐 batch_norm_v3 的 Torch*Spec 惯例）。"""

    golden = staticmethod(batch_norm_elemt_golden)
    tolerance = _TOLERANCE


class BatchNormElemtAclnnSpec:
    """按 CSV api_name "aclnnBatchNormElemt" 注册，参数按 aclnn C 头文件签名顺序
    （input, weight, bias, mean, invstd, eps, output），weight/bias/output 由调用方填充，
    golden 忽略 output。"""

    @staticmethod
    def golden(
        input,
        weight=None,
        bias=None,
        mean=None,
        invstd=None,
        eps=1e-5,
        output=None,
        **kwargs,
    ):
        fmts = kwargs.get("tensor_formats") or ()
        y = _elemt_core(input, weight, bias, mean, invstd, fmts[0] if fmts else None)
        return _to_orig_form(y, input)

    tolerance = _TOLERANCE
