#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK golden for sigmoid_cross_entropy_with_logits_v2 (arch35)。

kernel 通路（参数名与顺序遵循 sigmoid_cross_entropy_with_logits_v2_def.cpp，不含输出）：
    predict   : 必选
    target    : 必选，shape 与 predict 一致
    weight    : 可选，可 broadcast 到 predict
    pos_weight: 可选，可 broadcast 到 predict
    reduction : 属性（string），kernel 侧仅实现 "none"（mean/sum 由 aclnn 层
                调 reduce 算子实现），golden 按 none 语义计算，mean/sum 仅作兼容。

aclnn 通路（参数名与顺序遵循 aclnn_binary_cross_entropy_with_logits.h 的
aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize 形参，不含 workspaceSize/executor）：
    self            : 必选（logits）
    target          : 必选，shape 与 self 一致
    weightOptional  : 可选 tensor，可 broadcast 到 target，None 占位
    posWeightOptional: 可选 tensor，可 broadcast 到 target，None 占位
    reduction       : int64 属性，0(none)/1(mean)/2(sum)
    out             : 输出 tensor，redution=0 时 shape 与 self 一致，否则为 [1]
    空张量语义：none → 空输出；
    mean → NaN；sum → 0（torch 对空张量 mean/sum 的行为一致）。

计算公式：
    logsigmoid(x) = min(x, 0) - log(1 + exp(-|x|))
    loss = (1 - target) * x - logsigmoid(x) * ((pos_weight - 1) * target + 1)
    loss = loss * weight

kernel 通路 golden 收发 numpy.ndarray（边界必须用 numpy），aclnn 通路直接收发
torch.Tensor。
"""

import numpy
import torch

__golden__ = {
    "kernel": {
        "sigmoid_cross_entropy_with_logits_v2": "sigmoid_cross_entropy_with_logits_v2_golden"
    },
    "aclnn": {
        "aclnnBinaryCrossEntropyWithLogits": "aclnn_binary_cross_entropy_with_logits_golden"
    },
}

_REDUCTION_INT_TO_STR = {0: "none", 1: "mean", 2: "sum"}

_TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _numpy_bfloat16():
    """ml_dtypes.bfloat16（优先复用 ttk 的 dtype 解析，失败则直接取 ml_dtypes）。"""
    try:
        from ttk.utilities.dtypes import resolve_numpy_dtype

        return resolve_numpy_dtype("bfloat16")
    except Exception:
        from ml_dtypes import bfloat16

        return bfloat16


def _to_torch(array):
    """numpy.ndarray -> torch.Tensor（bfloat16 走 bit-view 转换）。"""
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        return array
    array = numpy.asarray(array)
    if array.dtype == _numpy_bfloat16():
        return torch.from_numpy(array.view(numpy.int16)).view(torch.bfloat16)
    return torch.from_numpy(numpy.ascontiguousarray(array))


def _to_numpy(tensor):
    """torch.Tensor -> numpy.ndarray（bfloat16 走 bit-view 转换）。"""
    if tensor.dtype == torch.bfloat16:
        return tensor.contiguous().view(torch.int16).numpy().view(_numpy_bfloat16())
    return tensor.numpy()


def _bce_with_logits_elementwise(predict_t, target_t, weight_t=None, pos_weight_t=None):
    """BCEWithLogits：
    logsigmoid(x) = min(x, 0) - log(1 + exp(-|x|))
    loss = (1 - target) * x - logsigmoid(x) * ((pos_weight - 1) * target + 1)
    loss = loss * weight
    """
    zeros = torch.zeros_like(predict_t)
    log_sigmoid = torch.minimum(predict_t, zeros) - torch.log(
        1.0 + torch.exp(-torch.abs(predict_t))
    )
    if pos_weight_t is not None:
        log_sigmoid = log_sigmoid * ((pos_weight_t - 1.0) * target_t + 1.0)
    loss = (1.0 - target_t) * predict_t - log_sigmoid
    if weight_t is not None:
        loss = loss * weight_t
    return loss


def sigmoid_cross_entropy_with_logits_v2_golden(
    predict, target, weight=None, pos_weight=None, *, reduction="none", **kwargs
):
    """
    Golden for sigmoid_cross_entropy_with_logits_v2 (kernel path).

    Args:
        predict:   numpy.ndarray, fp16/fp32/bf16
        target:    numpy.ndarray, 与 predict 同 dtype 同 shape
        weight:    numpy.ndarray/None, 可 broadcast 到 predict
        pos_weight: numpy.ndarray/None, 可 broadcast 到 predict
        reduction: 属性，string，"none"/"mean"/"sum"（kernel 侧仅支持 none）
        **kwargs:  {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                   full_soc_version, short_soc_version, testcase_name

    Returns:
        [loss]: 输出 tensor 列表（单输出）
    """
    predict_t = _to_torch(predict).to(torch.float32)
    target_t = _to_torch(target).to(torch.float32)
    weight_t = None if weight is None else _to_torch(weight).to(torch.float32)
    pos_weight_t = (
        None if pos_weight is None else _to_torch(pos_weight).to(torch.float32)
    )

    loss = _bce_with_logits_elementwise(predict_t, target_t, weight_t, pos_weight_t)

    reduction_str = str(reduction)
    if reduction_str == "mean":
        loss = torch.mean(loss)
    elif reduction_str == "sum":
        loss = torch.sum(loss)

    # 按输出 dtype 舍入（与 kernel 的 CAST_MODE_RINT 一致：round-to-nearest-even）
    output_dtypes = kwargs.get("output_dtypes") or []
    if output_dtypes:
        out_dtype = _TORCH_DTYPE_MAP.get(str(output_dtypes[0]).lower(), torch.float32)
        loss = loss.to(out_dtype)

    return [_to_numpy(loss)]


def aclnn_binary_cross_entropy_with_logits_golden(
    self,
    target,
    weightOptional=None,
    posWeightOptional=None,
    reduction=0,
    out=None,
    **kwargs,
):
    """
    Golden for aclnnBinaryCrossEntropyWithLogits (aclnn path).

    Args:
        self:             torch.Tensor, logits, fp16/fp32/bf16
        target:           torch.Tensor, 与 self 同 shape
        weightOptional:   torch.Tensor/None, 可 broadcast 到 target
        posWeightOptional: torch.Tensor/None, 可 broadcast 到 target
        reduction:        int, 0(none)/1(mean)/2(sum)
        out:              框架分配的输出 tensor（可忽略）
        **kwargs:         short_soc_version, testcase_name, tensor_dtypes, ...

    空张量语义：none → 空输出；mean → NaN；
    sum → 0。torch 对空张量 mean/sum 的行为与之一致。

    Returns:
        [loss]: 输出 tensor 列表（单输出）
    """
    self_t = self.to(torch.float32)
    target_t = target.to(torch.float32)
    weight_t = None if weightOptional is None else weightOptional.to(torch.float32)
    pos_weight_t = (
        None if posWeightOptional is None else posWeightOptional.to(torch.float32)
    )

    reduction_str = _REDUCTION_INT_TO_STR.get(int(reduction), "none")
    out_dtype = out.dtype if out is not None and hasattr(out, "dtype") else None

    # aclnn 层 HandleEmptyTensor：空张量（任一维度为 0）直接返回，
    # 不进入计算（weight/pos_weight 不参与）：none → 空输出；mean → NaN；sum → 0
    if self_t.numel() == 0:
        if reduction_str == "mean":
            loss = torch.full((), float("nan"), dtype=torch.float32)
        elif reduction_str == "sum":
            loss = torch.zeros((), dtype=torch.float32)
        else:
            loss = self_t.new_empty(self_t.shape)
        if out_dtype is not None:
            loss = loss.to(out_dtype)
            if loss.dim() == 0 and out.dim() > 0:
                loss = loss.reshape(out.shape)
        return [loss]

    loss = _bce_with_logits_elementwise(self_t, target_t, weight_t, pos_weight_t)

    if reduction_str == "mean":
        loss = torch.mean(loss)
    elif reduction_str == "sum":
        loss = torch.sum(loss)

    # 归约完成后 Cast 到 out dtype
    if out_dtype is not None:
        loss = loss.to(out_dtype)
        # 归约输出对齐 out 布局（0-dim 标量 → [1]，元素数一致）
        if loss.dim() == 0 and out.dim() > 0:
            loss = loss.reshape(out.shape)

    return [loss]
