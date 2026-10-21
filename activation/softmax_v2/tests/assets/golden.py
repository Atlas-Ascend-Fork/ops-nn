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

"""SoftmaxV2 golden：kernel/GEIR = softmax_v2（numpy），aclnn = aclnnSoftmax，
e2e = torch.ops.aten._softmax（torch）。真值为 CPU aten._softmax；
half_to_float=True 且 x 为 fp16 时输出 fp32。"""

import numpy as np
import torch

try:
    import ml_dtypes  # noqa: F401
except ImportError:  # pragma: no cover
    ml_dtypes = None

__spec__ = {
    "softmax_v2": "SoftmaxV2Spec",
    "aclnnSoftmax": "AclnnSoftmaxSpec",
    "torch.ops.aten._softmax": "AtenSoftmaxSpec",
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


def _aten_softmax(x, axes, half_to_float):
    """aten 真值：单轴直调；多轴折叠到末轴整体归约。"""
    if len(axes) == 1:
        return torch.ops.aten._softmax(x, axes[0], bool(half_to_float))
    keep = [i for i in range(x.dim()) if i not in axes]
    keep_shape = [x.shape[i] for i in keep]
    axes_shape = [x.shape[i] for i in axes]
    perm = keep + axes
    folded = x.permute(perm).reshape(*keep_shape, -1)
    out = torch.ops.aten._softmax(folded, -1, bool(half_to_float))
    return out.reshape(*keep_shape, *axes_shape).permute(
        [perm.index(i) for i in range(x.dim())]
    )


class SoftmaxV2Spec:
    """kernel/GEIR 流程：numpy 入/出。"""

    def golden(x, *, axes=None, axis=None, half_to_float=False, **kwargs):
        """参数跟随 softmax_v2_def.cpp，axes 默认 [-1]，half_to_float 为 Bool 属性。"""
        dim = _normalize_axes(axes if axes is not None else axis, x.ndim)
        x_raw = _np_to_torch(x)
        y = _aten_softmax(x_raw.to(_compute_dtype(x_raw)), dim, False)

        out_dtypes = [str(d) for d in (kwargs.get("output_dtypes") or ())]
        if out_dtypes:
            target = out_dtypes[0]
        elif half_to_float and x_raw.dtype == torch.float16:
            target = "float32"
        else:
            target = str(x.dtype)
        return _torch_to_numpy(y, target)

    class ThirdPartyImpl:
        """torch 三方标杆：axes → dim 适配。"""

        def __init__(self, *, axes=None, half_to_float=False, **kwargs):
            self.dim = int(axes[0]) if axes else -1
            self.half_to_float = bool(half_to_float)

        def __call__(self, x, **kwargs):
            if self.half_to_float and not x.is_cuda:
                # half_to_float 仅 CUDA 后端支持，CPU 以等价 fp32 计算实现
                xf = x.float() if x.dtype == torch.float16 else x
                return [torch.softmax(xf, dim=self.dim)]
            return [torch.ops.aten._softmax(x, self.dim, self.half_to_float)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AclnnSoftmaxSpec:
    """aclnn 流程：torch 入/出。"""

    def golden(selfT, dim, out=None, **kwargs):
        """参数跟随 aclnnSoftmaxGetWorkspaceSize（不含 workspaceSize 与 executor）。"""
        dim = _normalize_axes(dim, selfT.dim())[0]
        x = selfT.to(_compute_dtype(selfT))
        y = _aten_softmax(x, [dim], False)
        target = out.dtype if out is not None else selfT.dtype
        return y.to(target)

    class ThirdPartyImpl:
        """torch 三方标杆：dim 直传。"""

        def __init__(self, *, dim=-1, **kwargs):
            self.dim = int(dim)

        def __call__(self, *args, **kwargs):
            # 入参名 self 与保留名冲突，按位置取 args[0]
            return [torch.ops.aten._softmax(args[0], self.dim, False)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AtenSoftmaxSpec:
    """e2e 流程：torch 入/出。"""

    def golden(selfT, dim, half_to_float=False, **kwargs):
        """参数跟随 aten schema；fp16 输入且 half_to_float=True 时输出 fp32。"""
        dim = _normalize_axes(dim, selfT.dim())[0]
        x_raw = selfT.detach().cpu()
        y = _aten_softmax(x_raw.to(_compute_dtype(x_raw)), [dim], False)
        target = (
            torch.float32
            if (half_to_float and x_raw.dtype == torch.float16)
            else selfT.dtype
        )
        return y.to(target)

    class ThirdPartyImpl:
        """torch 三方标杆：dim/half_to_float 透传。"""

        def __init__(self, *, dim=-1, half_to_float=False, **kwargs):
            self.dim = int(dim)
            self.half_to_float = bool(half_to_float)

        def __call__(self, *args, **kwargs):
            # 入参名 self 与保留名冲突，按位置取 args[0]
            x = args[0]
            if self.half_to_float and not x.is_cuda:
                # half_to_float 仅 CUDA 后端支持，CPU 以等价 fp32 计算实现
                xf = x.float() if x.dtype == torch.float16 else x
                return [torch.softmax(xf, dim=self.dim)]
            return [torch.ops.aten._softmax(x, self.dim, self.half_to_float)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }
