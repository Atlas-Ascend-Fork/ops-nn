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

"""ConfusionSoftmaxGrad golden（仅 kernel/GEIR：无 aclnn 接口，亦无 aten 等价算子）。

公式：y = grad - sum(grad * x, axis=-1)，x 为 softmax 前向输出，无最终乘 x。"""

import numpy as np
import torch

try:
    import ml_dtypes  # noqa: F401
except ImportError:  # pragma: no cover
    ml_dtypes = None

__spec__ = {
    "confusion_softmax_grad": "ConfusionSoftmaxGradSpec",
}


def _np_to_torch(array):
    """numpy → torch CPU 张量（bf16 经 uint16 位视图无损转换）。"""
    array = np.ascontiguousarray(array)
    if ml_dtypes is not None and array.dtype == np.dtype(ml_dtypes.bfloat16):
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _compute_dtype(*tensors):
    """计算精度：任一入参 fp64 则整体 fp64（Promote），否则 fp32。"""
    for tensor in tensors:
        if tensor.dtype == torch.float64:
            return torch.float64
    return torch.float32


def _torch_to_numpy(tensor, dtype_name):
    """结果 → numpy，cast 到输出 dtype。"""
    array = tensor.detach().cpu().numpy()
    return np.ascontiguousarray(array.astype(dtype_name, copy=False))


class ConfusionSoftmaxGradSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    def golden(grad, x, **kwargs):
        """参数跟随 confusion_softmax_grad_def.cpp，无属性。"""
        if tuple(grad.shape) != tuple(x.shape):
            raise ValueError(f"grad/x shape mismatch: {grad.shape} vs {x.shape}")

        g_raw = _np_to_torch(grad)
        x_raw = _np_to_torch(x)
        compute = _compute_dtype(g_raw, x_raw)
        g = g_raw.to(compute)
        p = x_raw.to(compute)
        y = g - (g * p).sum(dim=-1, keepdim=True)

        out_dtypes = [str(d) for d in (kwargs.get("output_dtypes") or ())]
        target = out_dtypes[0] if out_dtypes else str(grad.dtype)
        return _torch_to_numpy(y, target)

    class ThirdPartyImpl:
        """torch 三方标杆：小算子拼接（无直配 API）。"""

        def __init__(self, **kwargs):
            pass

        def __call__(self, grad, x, **kwargs):
            return [grad - (grad * x).sum(dim=-1, keepdim=True)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }
