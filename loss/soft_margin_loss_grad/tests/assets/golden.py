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
"""SoftMarginLossGrad 独立 Golden：raw、ACLNN、Torch。

两方 Golden 和三方标杆都直接调用
``torch.ops.aten.soft_margin_loss_backward`` 竞品算子。
"""

import numpy
import torch

__spec__ = {
    "soft_margin_loss_grad": "SoftMarginLossGradKernelSpec",
    "aclnnSoftMarginLossBackward": "SoftMarginLossGradAclnnSpec",
    "torch.ops.aten.soft_margin_loss_backward": "SoftMarginLossGradTorchSpec",
}

_L2 = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
}


def _reduction_value(reduction):
    if hasattr(reduction, "item"):
        reduction = reduction.item()
    if isinstance(reduction, str):
        return {"none": 0, "mean": 1, "sum": 2}.get(reduction, 1)
    return int(reduction)


def _np_to_torch(arr):
    """numpy → torch（bf16 经 int16 位桥接，语义等价 ttk.utilities.dtypes.numpy_to_torch_tensor）。"""
    if "bfloat16" in arr.dtype.name:
        return torch.from_numpy(arr.view(numpy.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _torch_to_np(tensor):
    """torch → numpy（bf16 还原 ml_dtypes.bfloat16，语义等价 ttk.utilities.dtypes.torch_to_numpy_tensor）。"""
    if tensor.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return tensor.view(torch.int16).numpy().view(bfloat16)
    return tensor.numpy()


def _compute(grad_output, source, target, reduction=1, *, aclnn_semantics=False):
    # Keep the promoted dtype supplied by TTK for cpu-golden.
    is_torch = isinstance(grad_output, torch.Tensor)
    go = grad_output if is_torch else _np_to_torch(grad_output)
    src = source if isinstance(source, torch.Tensor) else _np_to_torch(source)
    tgt = target if isinstance(target, torch.Tensor) else _np_to_torch(target)
    if aclnn_semantics:
        common = torch.promote_types(
            go.dtype, torch.promote_types(src.dtype, tgt.dtype)
        )
        go, src, tgt = (t.contiguous().to(common) for t in (go, src, tgt))
        shape = torch.broadcast_shapes(go.shape, src.shape, tgt.shape)
        go, src, tgt = (t.expand(shape).contiguous() for t in (go, src, tgt))
    result = torch.ops.aten.soft_margin_loss_backward(
        go, src, tgt, _reduction_value(reduction)
    )
    return result if is_torch else _torch_to_np(result.detach().cpu())


def _kernel_target_output_dtype(predict, kwargs):
    # TTK promotes output_dtypes too; the frozen CSV declaration remains the target.
    if kwargs.get("golden_mode") == "Promote":
        import json
        from pathlib import Path

        declarations = json.loads(
            Path(__file__)
            .with_name("soft_margin_loss_grad_output_dtypes.json")
            .read_text(encoding="utf-8")
        )["cases"]
        case = kwargs.get("testcase_name")
        if case not in declarations:
            raise ValueError(
                f"Missing declared output dtype for {case!r}; refresh golden dtype metadata from CSV"
            )
        return declarations[case]
    dtypes = kwargs.get("output_dtypes")
    if dtypes:
        return dtypes[0]
    return str(predict.dtype).removeprefix("torch.")


def _cast_final_output(result, dtype):
    if isinstance(result, torch.Tensor):
        return result.to(getattr(torch, dtype))
    if dtype == "bfloat16":
        from ml_dtypes import bfloat16

        return result.astype(bfloat16)
    return result.astype(dtype)


def soft_margin_loss_grad_golden(predict, label, dout, reduction="mean", **kwargs):
    dtype = _kernel_target_output_dtype(predict, kwargs)
    result = _compute(dout, predict, label, reduction)
    return [_cast_final_output(result, dtype)]


def aclnn_soft_margin_loss_backward_golden(
    gradOutput, self, target, reduction=1, out=None, **kwargs
):
    del kwargs
    result = _compute(gradOutput, self, target, reduction, aclnn_semantics=True)
    if out is not None:
        # Keep promoted ATen computation; round only the final reference output.
        result = result.reshape(out.shape).to(out.dtype)
    return [result]


def torch_soft_margin_loss_backward_golden(
    grad_output, self, target, reduction, out=None, **kwargs
):
    del kwargs
    result = _compute(grad_output, self, target, reduction)
    if out is not None:
        result = result.reshape(out.shape)
    return [result]


class _TorchSoftMarginKernel:
    def __init__(self, reduction="mean", **kwargs):
        del kwargs
        self.reduction = _reduction_value(reduction)

    @torch.no_grad()
    def __call__(self, predict, label, dout, reduction=None, **kwargs):
        del kwargs
        red = self.reduction if reduction is None else _reduction_value(reduction)
        return [torch.ops.aten.soft_margin_loss_backward(dout, predict, label, red)]


class _TorchSoftMarginAclnn:
    def __init__(self, reduction=1, **kwargs):
        del kwargs
        self.reduction = _reduction_value(reduction)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        gradOutput = kwargs.get("gradOutput")
        if gradOutput is None and len(args) > 0:
            gradOutput = args[0]
        source = kwargs.get("self")
        if source is None and len(args) > 1:
            source = args[1]
        target = kwargs.get("target")
        if target is None and len(args) > 2:
            target = args[2]
        reduction = kwargs.get("reduction")
        red = self.reduction if reduction is None else _reduction_value(reduction)
        common = torch.promote_types(
            gradOutput.dtype, torch.promote_types(source.dtype, target.dtype)
        )
        gradOutput, source, target = (
            t.contiguous().to(common) for t in (gradOutput, source, target)
        )
        shape = torch.broadcast_shapes(gradOutput.shape, source.shape, target.shape)
        gradOutput, source, target = (
            t.expand(shape).contiguous() for t in (gradOutput, source, target)
        )
        return [
            torch.ops.aten.soft_margin_loss_backward(gradOutput, source, target, red)
        ]


class _TorchSoftMarginTorch:
    """Torch 三方标杆（E2E）：形参名匹配 aten schema（grad_output/self/target）。

    ``self`` 输入与 Python 保留槽冲突，服务端按名字绑定时不进 ``**kwargs``，
    只能经 ``*args`` 位置接收；``target``/``reduction`` 经 ``**kwargs`` 提供。
    """

    def __init__(self, reduction=1, **kwargs):
        del kwargs
        self.reduction = _reduction_value(reduction)

    @torch.no_grad()
    def __call__(self, grad_output, *args, **kwargs):
        source = next((a for a in args if isinstance(a, torch.Tensor)), None)
        if source is None:
            raise ValueError(
                "soft_margin_loss_grad third_party: 'self' input not bound to *args"
            )
        target = kwargs.get("target")
        reduction = kwargs.get("reduction")
        red = self.reduction if reduction is None else _reduction_value(reduction)
        return [
            torch.ops.aten.soft_margin_loss_backward(grad_output, source, target, red)
        ]


class _SoftMarginLossGradGraph(torch.nn.Module):
    def forward(self, grad_output, source, target, reduction=1, **kwargs):
        del kwargs
        return torch.ops.aten.soft_margin_loss_backward(
            grad_output, source, target, _reduction_value(reduction)
        )


class SoftMarginLossGradKernelSpec:
    golden = staticmethod(soft_margin_loss_grad_golden)
    third_party = {"torch": _TorchSoftMarginKernel}
    tolerance = _L2


class SoftMarginLossGradAclnnSpec:
    golden = staticmethod(aclnn_soft_margin_loss_backward_golden)
    third_party = {"torch": _TorchSoftMarginAclnn}
    tolerance = _L2


class SoftMarginLossGradTorchSpec:
    golden = staticmethod(torch_soft_margin_loss_backward_golden)
    third_party = {"torch": _TorchSoftMarginTorch}
    torch_graph = _SoftMarginLossGradGraph
    tolerance = _L2
