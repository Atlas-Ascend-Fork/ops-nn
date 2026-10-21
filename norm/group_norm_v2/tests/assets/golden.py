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

"""GroupNormV2 golden：kernel/GEIR = group_norm_v2（numpy），aclnn = aclnnGroupNorm，
e2e = torch.ops.aten.native_group_norm（torch）。真值为 CPU aten.native_group_norm。

x 通道为 dim[1]（与 data_format 属性取值无关）；gamma/beta 可选（缺省全 1/全 0）；
支持 x fp16/bf16 + gamma/beta fp32 混合精度。mean/rstd 输出形状：
kernel op 为 (N*group,) 一维，aclnn/e2e 为 (N, group)。"""

import numpy as np
import torch

try:
    import ml_dtypes  # noqa: F401
except ImportError:  # pragma: no cover
    ml_dtypes = None

__spec__ = {
    "group_norm_v2": "GroupNormV2Spec",
    "aclnnGroupNorm": "AclnnGroupNormSpec",
    "torch.ops.aten.native_group_norm": "AtenGroupNormSpec",
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


def _nchw(x_tensor):
    """(N, C, ...) → (N, C, HxW)。"""
    n, c = x_tensor.shape[0], x_tensor.shape[1]
    return x_tensor.reshape(n, c, -1)


def _unify_for_aten(*tensors):
    """混合精度时统一升到最宽 dtype 并保证连续（aten 要求连续输入），
    输出由调用方 cast 回原 dtype。"""
    dtypes = [t.dtype for t in tensors if t is not None]
    if not dtypes:
        return tensors
    widest = dtypes[0]
    for d in dtypes[1:]:
        widest = torch.promote_types(widest, d)
    return tuple(t.to(widest).contiguous() if t is not None else None for t in tensors)


def _aten_group_norm(xv, w, b, n, c, hw, group, eps):
    """aten 真值；weight/bias 为 None 时按算子语义取全 1/全 0。"""
    if w is None:
        w = torch.ones([c], dtype=xv.dtype)
    if b is None:
        b = torch.zeros([c], dtype=xv.dtype)
    return torch.ops.aten.native_group_norm(
        xv.contiguous(), w.contiguous(), b.contiguous(), n, c, hw, group, eps
    )


class GroupNormV2Spec:
    """kernel/GEIR 流程：numpy 入/出；mean/rstd 为 (N*group,) 一维。"""

    def golden(
        x,
        gamma=None,
        beta=None,
        *,
        num_groups,
        data_format="NHWC",
        eps=1e-5,
        is_training=True,
        **kwargs,
    ):
        """参数跟随 group_norm_v2_def.cpp，num_groups 必选。"""
        n, c = x.shape[0], x.shape[1]
        if int(num_groups) <= 0 or c % int(num_groups) != 0:
            raise ValueError(
                f"channel {c} must be divisible by num_groups {num_groups}"
            )

        x_raw = _np_to_torch(x)
        gamma_raw = _np_to_torch(gamma) if gamma is not None else None
        beta_raw = _np_to_torch(beta) if beta is not None else None
        raw = [x_raw] + [t for t in (gamma_raw, beta_raw) if t is not None]
        compute = _compute_dtype(*raw)
        xv = _nchw(x_raw.to(compute))
        hw = xv.shape[2]
        w = gamma_raw.to(compute) if gamma_raw is not None else None
        b = beta_raw.to(compute) if beta_raw is not None else None

        y, mean, rstd = _aten_group_norm(
            xv, w, b, n, c, hw, int(num_groups), float(eps)
        )
        y = y.reshape(tuple(x.shape))

        out_dtypes = [str(d) for d in (kwargs.get("output_dtypes") or ())]
        defaults = [str(x.dtype), "float32", "float32"]
        outputs = []
        for i, result in enumerate((y, mean, rstd)):
            target = out_dtypes[i] if i < len(out_dtypes) else defaults[i]
            outputs.append(_torch_to_numpy(result, target))
        outputs[1] = outputs[1].reshape(-1)
        outputs[2] = outputs[2].reshape(-1)
        return outputs

    class ThirdPartyImpl:
        """torch 三方标杆：num_groups → group 适配，mean/rstd 折叠为一维。"""

        def __init__(self, *, num_groups=None, eps=1e-5, **kwargs):
            self.group = int(num_groups) if num_groups is not None else 1
            self.eps = float(eps)

        def __call__(self, x, gamma=None, beta=None, **kwargs):
            orig_dtype = x.dtype
            x, gamma, beta = _unify_for_aten(x, gamma, beta)
            n, c = x.shape[0], x.shape[1]
            hw = max(x.numel() // (n * c), 1)
            y, mean, rstd = torch.ops.aten.native_group_norm(
                x, gamma, beta, n, c, hw, self.group, self.eps
            )
            # 三输出均为 x dtype，mean/rstd 为一维
            return [
                y.to(orig_dtype),
                mean.to(orig_dtype).reshape(-1),
                rstd.to(orig_dtype).reshape(-1),
            ]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AclnnGroupNormSpec:
    """aclnn 流程：torch 入/出；mean/rstd 为 (N, group)。"""

    def golden(
        selfT,
        gamma,
        beta,
        N,
        C,
        HxW,
        group,
        eps,
        out=None,
        meanOut=None,
        rstdOut=None,
        **kwargs,
    ):
        """参数跟随 aclnnGroupNormGetWorkspaceSize（不含 workspaceSize 与 executor）。"""
        raw = [selfT] + [t for t in (gamma, beta) if t is not None]
        compute = _compute_dtype(*raw)
        xv = _nchw(selfT.to(compute))
        w = gamma.to(compute) if gamma is not None else None
        b = beta.to(compute) if beta is not None else None

        y, mean, rstd = _aten_group_norm(
            xv, w, b, int(N), int(C), int(HxW), int(group), float(eps)
        )
        y = y.reshape(tuple(selfT.shape))

        outputs = []
        for result, o in zip((y, mean, rstd), (out, meanOut, rstdOut)):
            if o is not None:
                outputs.append(result.to(o.dtype))
            else:
                outputs.append(result)
        return outputs

    class ThirdPartyImpl:
        """torch 三方标杆：N/C/HxW/group/eps 直传。"""

        def __init__(self, *, N=1, C=1, HxW=1, group=1, eps=1e-5, **kwargs):
            self.n, self.c, self.hw = int(N), int(C), int(HxW)
            self.group, self.eps = int(group), float(eps)

        def __call__(self, selfT, gamma=None, beta=None, **kwargs):
            orig_dtype = selfT.dtype
            selfT, gamma, beta = _unify_for_aten(selfT, gamma, beta)
            y, mean, rstd = torch.ops.aten.native_group_norm(
                selfT, gamma, beta, self.n, self.c, self.hw, self.group, self.eps
            )
            return [y.to(orig_dtype), mean.to(orig_dtype), rstd.to(orig_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }


class AtenGroupNormSpec:
    """e2e 流程：torch 入/出；三输出同输入 dtype。"""

    def golden(input, weight, bias, N, C, HxW, group, eps, **kwargs):
        """参数跟随 aten schema。"""
        x_raw = input.detach().cpu()
        w_raw = weight.detach().cpu() if weight is not None else None
        b_raw = bias.detach().cpu() if bias is not None else None
        raw = [x_raw] + [t for t in (w_raw, b_raw) if t is not None]
        compute = _compute_dtype(*raw)
        xv = _nchw(x_raw.to(compute))

        w = w_raw.to(compute) if w_raw is not None else None
        b = b_raw.to(compute) if b_raw is not None else None
        y, mean, rstd = _aten_group_norm(
            xv, w, b, int(N), int(C), int(HxW), int(group), float(eps)
        )
        y = y.reshape(tuple(input.shape))
        return [y.to(input.dtype), mean.to(input.dtype), rstd.to(input.dtype)]

    class ThirdPartyImpl:
        """torch 三方标杆：N/C/HxW/group/eps 直传。"""

        def __init__(self, *, N=1, C=1, HxW=1, group=1, eps=1e-5, **kwargs):
            self.n, self.c, self.hw = int(N), int(C), int(HxW)
            self.group, self.eps = int(group), float(eps)

        def __call__(self, *args, **kwargs):
            # 入参名 input 与内建同名，按位置取 args[0]=input, [1]=weight, [2]=bias
            x = args[0]
            w = args[1] if len(args) > 1 else kwargs.get("weight")
            b = args[2] if len(args) > 2 else kwargs.get("bias")
            orig_dtype = x.dtype
            x, w, b = _unify_for_aten(x, w, b)
            y, mean, rstd = torch.ops.aten.native_group_norm(
                x, w, b, self.n, self.c, self.hw, self.group, self.eps
            )
            return [y.to(orig_dtype), mean.to(orig_dtype), rstd.to(orig_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }
