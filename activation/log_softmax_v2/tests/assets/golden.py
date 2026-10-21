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

"""LogSoftmaxV2 golden：kernel/GEIR = log_softmax_v2（numpy），aclnn = aclnnLogSoftmax，
e2e = torch.ops.aten.log_softmax（torch）。真值为 CPU aten.log_softmax。"""

import numpy as np
import torch

try:
    import ml_dtypes  # noqa: F401
except ImportError:  # pragma: no cover
    ml_dtypes = None

__spec__ = {
    "log_softmax_v2": "LogSoftmaxV2Spec",
    "aclnnLogSoftmax": "AclnnLogSoftmaxSpec",
    "torch.ops.aten.log_softmax": "AtenLogSoftmaxSpec",
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
    """axes/axis 归一化：负轴转正、去重排序，缺省 [-1]。"""
    if axes is None:
        axes = [-1]
    if isinstance(axes, int):
        axes = [axes]
    axes = [int(a) for a in axes]
    if not axes:
        axes = [-1]
    return sorted({a if a >= 0 else a + ndim for a in axes})


def _aten_log_softmax(x, axes):
    """aten 真值：单轴直调；多轴折叠到末轴整体归约。"""
    if len(axes) == 1:
        return torch.ops.aten.log_softmax(x, axes[0])
    keep = [i for i in range(x.dim()) if i not in axes]
    keep_shape = [x.shape[i] for i in keep]
    axes_shape = [x.shape[i] for i in axes]
    perm = keep + axes
    folded = x.permute(perm).reshape(*keep_shape, -1)
    out = torch.ops.aten.log_softmax(folded, -1)
    return out.reshape(*keep_shape, *axes_shape).permute(
        [perm.index(i) for i in range(x.dim())]
    )


class LogSoftmaxV2Spec:
    """kernel/GEIR 流程：numpy 入/出。"""

    def golden(logits, *, axes=None, axis=None, **kwargs):
        """参数跟随 log_softmax_v2_def.cpp，axes 默认 [-1]。"""
        dim = _normalize_axes(axes if axes is not None else axis, logits.ndim)
        x_raw = _np_to_torch(logits)
        y = _aten_log_softmax(x_raw.to(_compute_dtype(x_raw)), dim)

        out_dtypes = [str(d) for d in (kwargs.get("output_dtypes") or ())]
        target = out_dtypes[0] if out_dtypes else str(logits.dtype)
        return _torch_to_numpy(y, target)

    class ThirdPartyImpl:
        """torch 三方标杆：axes → dim 适配。"""

        def __init__(self, *, axes=None, **kwargs):
            self.dim = int(axes[0]) if axes else -1

        def __call__(self, logits, **kwargs):
            return [torch.log_softmax(logits, dim=self.dim)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AclnnLogSoftmaxSpec:
    """aclnn 流程：torch 入/出。"""

    def golden(selfT, dim, out=None, **kwargs):
        """参数跟随 aclnnLogSoftmaxGetWorkspaceSize（不含 workspaceSize 与 executor）。"""
        dim = _normalize_axes(dim, selfT.dim())[0]
        x = selfT.to(_compute_dtype(selfT))
        y = _aten_log_softmax(x, [dim])
        target = out.dtype if out is not None else selfT.dtype
        return y.to(target)

    class ThirdPartyImpl:
        """torch 三方标杆：dim 直传。"""

        def __init__(self, *, dim=-1, **kwargs):
            self.dim = int(dim)

        def __call__(self, *args, **kwargs):
            # 入参名 self 与保留名冲突，按位置取 args[0]
            return [torch.log_softmax(args[0], dim=self.dim)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AtenLogSoftmaxSpec:
    """e2e 流程：torch 入/出。"""

    def golden(selfT, dim, dtype=None, **kwargs):
        """参数跟随 aten schema；dtype 指定时输出为该 dtype，否则同输入。"""
        dim = _normalize_axes(dim, selfT.dim())[0]
        x_raw = selfT.detach().cpu()
        y = _aten_log_softmax(x_raw.to(_compute_dtype(x_raw)), [dim])
        target = dtype if dtype is not None else selfT.dtype
        return y.to(target)

    class ThirdPartyImpl:
        """torch 三方标杆：dim/dtype 透传。"""

        def __init__(self, *, dim=-1, dtype=None, **kwargs):
            self.dim = int(dim)
            self.dtype = dtype

        def __call__(self, *args, **kwargs):
            # 入参名 self 与保留名冲突，按位置取 args[0]
            kwargs = {} if self.dtype is None else {"dtype": self.dtype}
            return [torch.ops.aten.log_softmax(args[0], self.dim, **kwargs)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }
