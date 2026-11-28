#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Golden reference for sigmoid_cross_entropy_with_logits_grad.

gradient = (sigmoid(predict) - target) * dout

Kernel 语义：非 float32 输入先提升到 float32 计算，结果用 CAST_RINT 转回原 dtype。

精度契约（Promote）：cross_check 场景框架把输入抬一档喂进来（fp16/bf16→fp32、
fp32→fp64），此处照单全收、绝不向下砍，并统一提升到 float64 计算后按 float64 输出，
作为与实现无关的高精度基准：若按 fp32 + torch 计算，golden 会与 third_party
（同为 torch 实现）误差强相关（甚至逐位一致），cross_check 的 mare/mere/rmse
比值分母趋零被 small_value 夹底，NPU 任意 1ulp 误差都会放大为超标。

"""

import numpy as np

__spec__ = {
    "sigmoid_cross_entropy_with_logits_grad": "SigmoidCrossEntropyWithLogitsGradKernelSpec",
}

_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_BF16_NUMPY = None
try:
    import ml_dtypes

    _BF16_NUMPY = ml_dtypes.bfloat16
except ImportError:  # pragma: no cover - TTK 环境必然提供
    _BF16_NUMPY = None


def _is_bfloat16(dtype):
    if _BF16_NUMPY is not None and dtype == _BF16_NUMPY:
        return True
    return str(dtype) in ("bfloat16", "<V2")


def _to_float32(arr):
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32)


def _to_float64(arr):
    if arr.dtype == np.float64:
        return arr
    return arr.astype(np.float64)


def _cast_back(arr, orig_dtype):
    if orig_dtype == np.float32:
        return arr
    if orig_dtype == np.float16:
        return arr.astype(np.float16)
    if _is_bfloat16(orig_dtype) and _BF16_NUMPY is not None:
        return arr.astype(_BF16_NUMPY)
    return arr


class SigmoidCrossEntropyWithLogitsGradKernelSpec:
    """Kernel/GEIR 共用 TestSpec（CSV op_name = sigmoid_cross_entropy_with_logits_grad）。

    输入按 def.cpp 顺序位置传入（predict, target, dout），无属性、无可选输入。
    golden 接收 numpy，返回 numpy list。
    """

    @staticmethod
    def golden(predict, target, dout, **kwargs):
        import torch

        orig_dtype = predict.dtype
        # Promote（cross_check 恒走）：输入已被框架抬档（fp16/bf16→fp32、fp32→fp64），
        # 照单全收并统一提升到 float64 计算、按 float64 输出（高精度基准）。
        # 兼容未下发 golden_mode 的旧框架：fp64 到达必为 Promote 抬档的 fp32 用例。
        if kwargs.get("golden_mode") == "Promote" or orig_dtype == np.float64:
            p = torch.from_numpy(_to_float64(predict))
            t = torch.from_numpy(_to_float64(target))
            d = torch.from_numpy(_to_float64(dout))
            # dout 只需元素总数与 predict 相等，shape 可能不同；reshape 到 predict shape 后逐元素计算
            d = d.reshape(p.shape)
            grad = (torch.sigmoid(p) - t) * d
            return [grad.numpy()]
        # 非 Promote（原始 dtype 输入）：对齐 kernel 语义——窄档提升到 fp32 计算，
        # 结果按 CAST_RINT 转回原 dtype
        p = torch.from_numpy(_to_float32(predict))
        t = torch.from_numpy(_to_float32(target))
        d = torch.from_numpy(_to_float32(dout))
        # dout 只需元素总数与 predict 相等，shape 可能不同；reshape 到 predict shape 后逐元素计算
        d = d.reshape(p.shape)
        grad = (torch.sigmoid(p) - t) * d
        return [_cast_back(grad.numpy(), orig_dtype)]

    class ThirdPartyImpl:
        """torch provider：独立 Torch 基础算子组合实现，供 GEIR cross_check。"""

        def __init__(self, **kwargs):
            pass

        def __call__(self, predict, target, dout, **kwargs):
            import torch

            orig_dtype = predict.dtype
            p = predict.float() if predict.dtype != torch.float32 else predict
            t = target.float() if target.dtype != torch.float32 else target
            d = dout.float() if dout.dtype != torch.float32 else dout
            # dout 只需元素总数与 predict 相等，shape 可能不同；reshape 到 predict shape 后逐元素计算
            d = d.reshape(p.shape)
            grad = (torch.sigmoid(p) - t) * d
            if orig_dtype == torch.float16:
                grad = grad.to(torch.float16)
            elif orig_dtype == torch.bfloat16:
                grad = grad.to(torch.bfloat16)
            return [grad]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = _TOLERANCE
