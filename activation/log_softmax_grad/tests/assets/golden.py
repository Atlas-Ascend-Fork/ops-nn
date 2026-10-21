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

"""LogSoftmaxGrad golden：kernel/GEIR = log_softmax_grad（numpy），aclnn = aclnnLogSoftmaxBackward，
e2e = torch.ops.aten._log_softmax_backward_data（torch）。真值为 CPU aten._log_softmax_backward_data；
x 为 log_softmax 前向输出（exp(x) 即 softmax 概率）。"""

import numpy as np
import torch

try:
    import ml_dtypes  # noqa: F401
except ImportError:  # pragma: no cover
    ml_dtypes = None

__spec__ = {
    "log_softmax_grad": "LogSoftmaxGradSpec",
    "aclnnLogSoftmaxBackward": "AclnnLogSoftmaxBackwardSpec",
    "torch.ops.aten._log_softmax_backward_data": "AtenLogSoftmaxBackwardSpec",
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


def _normalize_axes(axes, ndim):
    """axis/axes 归一化：负轴转正、去重排序，缺省 [-1]。"""
    if axes is None:
        axes = [-1]
    if isinstance(axes, int):
        axes = [axes]
    axes = [int(a) for a in axes]
    if not axes:
        axes = [-1]
    return sorted({a if a >= 0 else a + ndim for a in axes})


def _aten_log_softmax_backward(grad, output, axes, compute_dtype):
    """aten 真值：单轴直调；多轴折叠到末轴整体归约。input_dtype 须与入参一致。"""
    if len(axes) == 1:
        return torch.ops.aten._log_softmax_backward_data(
            grad, output, axes[0], compute_dtype
        )
    keep = [i for i in range(grad.dim()) if i not in axes]
    keep_shape = [grad.shape[i] for i in keep]
    axes_shape = [grad.shape[i] for i in axes]
    perm = keep + axes
    g_folded = grad.permute(perm).reshape(*keep_shape, -1)
    y_folded = output.permute(perm).reshape(*keep_shape, -1)
    out = torch.ops.aten._log_softmax_backward_data(
        g_folded, y_folded, -1, compute_dtype
    )
    return out.reshape(*keep_shape, *axes_shape).permute(
        [perm.index(i) for i in range(grad.dim())]
    )


class LogSoftmaxGradSpec:
    """kernel/GEIR 流程：numpy 入/出。"""

    def golden(grad, x, *, axis=None, axes=None, **kwargs):
        """参数跟随 log_softmax_grad_def.cpp，axis 默认 [-1]；x 为前向输出。"""
        dim = _normalize_axes(axis if axis is not None else axes, grad.ndim)
        g_raw = _np_to_torch(grad)
        x_raw = _np_to_torch(x)
        compute = _compute_dtype(g_raw, x_raw)
        y = _aten_log_softmax_backward(
            g_raw.to(compute), x_raw.to(compute), dim, compute
        )

        out_dtypes = [str(d) for d in (kwargs.get("output_dtypes") or ())]
        target = out_dtypes[0] if out_dtypes else str(grad.dtype)
        return _torch_to_numpy(y, target)

    class ThirdPartyImpl:
        """torch 三方标杆：axis → dim 适配。"""

        def __init__(self, *, axis=None, axes=None, **kwargs):
            axes = axis if axis is not None else axes
            self.dim = int(axes[0]) if axes else -1

        def __call__(self, grad, x, **kwargs):
            return [
                torch.ops.aten._log_softmax_backward_data(grad, x, self.dim, grad.dtype)
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AclnnLogSoftmaxBackwardSpec:
    """aclnn 流程：torch 入/出。"""

    def golden(gradOutput, output, dim, out=None, **kwargs):
        """参数跟随 aclnnLogSoftmaxBackwardGetWorkspaceSize（不含 workspaceSize 与 executor）。"""
        dim = _normalize_axes(dim, gradOutput.dim())[0]
        compute = _compute_dtype(gradOutput, output)
        y = _aten_log_softmax_backward(
            gradOutput.to(compute), output.to(compute), [dim], compute
        )
        target = out.dtype if out is not None else gradOutput.dtype
        return y.to(target)

    class ThirdPartyImpl:
        """torch 三方标杆：dim 直传。"""

        def __init__(self, *, dim=-1, **kwargs):
            self.dim = int(dim)

        def __call__(self, gradOutput, output, **kwargs):
            return [
                torch.ops.aten._log_softmax_backward_data(
                    gradOutput, output, self.dim, gradOutput.dtype
                )
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AtenLogSoftmaxBackwardSpec:
    """e2e 流程：torch 入/出。"""

    def golden(grad_output, output, dim, input_dtype=None, **kwargs):
        """参数跟随 aten schema；输出 dtype 为 input_dtype。"""
        dim = _normalize_axes(dim, grad_output.dim())[0]
        g_raw = grad_output.detach().cpu()
        y_raw = output.detach().cpu()
        compute = _compute_dtype(g_raw, y_raw)
        y = _aten_log_softmax_backward(
            g_raw.to(compute), y_raw.to(compute), [dim], compute
        )
        target = input_dtype if input_dtype is not None else grad_output.dtype
        return y.to(target)

    class ThirdPartyImpl:
        """torch 三方标杆：dim 直传，input_dtype 取输入 dtype。"""

        def __init__(self, *, dim=-1, input_dtype=None, **kwargs):
            self.dim = int(dim)

        def __call__(self, grad_output, output, **kwargs):
            return [
                torch.ops.aten._log_softmax_backward_data(
                    grad_output, output, self.dim, grad_output.dtype
                )
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }
