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
"""TestSpec for NLLLoss.

Reference source
----------------
``torch.nn.functional.nll_loss`` on CPU and a sum of selected class weights,
independent of the NPU operator under test. NumPy is used only for
dtype/container bridging, never for the reference mathematics.

Output protocol (follows nll_loss_def.cpp)
------------------------------------------
Two outputs, in order: ``y``, ``total_weight``.

``total_weight`` and ``reduction="none"``
-----------------------------------------
``docs/aclnnNLLLoss.md`` and ``docs/aclnnNLLLoss2d.md`` both exempt
``totalWeightOut`` for ACLNN reduction=none. Their wrappers explicitly skip
copying that output in this mode, including the 1-D ACLNN path.
That is not a kernel/GEIR exemption: ``op_graph/nll_loss_proto.h`` declares
``total_weight`` without this restriction, but does not define its none value.
This spec retains the previous ATen auxiliary convention as an explicit
reference choice: the selected weight for 1-D none, but zero for 2-D/4-D none. Public
PyTorch NLLLoss returns only loss; its compatibility statement alone cannot
establish that zero is the kernel contract. README also gives a weight-sum
formula without qualifying none. These documents need clarification.
The arch35 2-D/4-D none implementation leaves this output unwritten. Keep that
mismatch visible as a contract discrepancy, not a proven kernel bug or an
automatic SUPPRESSED output based on the ACLNN exemption.
The separately registered Torch E2E API returns only loss by its own contract.
"""

import numpy as np
from torch.nn import NLLLoss

__spec__ = {
    "nll_loss": "NllLossTestSpec",
    "torch.nn.functional.nll_loss": "TorchNllLossTestSpec",
}

__golden__ = {"kernel": {"nll_loss": "nll_loss_golden"}}

_LOW_PRECISION = ("float16", "bfloat16")


def _needs_upcast(dtype):
    name = str(getattr(dtype, "name", dtype))
    return any(low in name for low in _LOW_PRECISION)


def nll_loss_golden(
    x, target, weight=None, *, ignore_index=-100, reduction="mean", **kwargs
):
    """
    Golden function for nll_loss.
    All the parameters (names and order) follow @nll_loss_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Computes loss with torch.nn.functional.nll_loss on CPU for every input rank
    and total_weight from the non-ignored targets and class weights. Low precision
    inputs are upcast to float32 before the reference call if the caller has not
    already promoted them. TTK-promoted float64 inputs return to float32, the
    highest floating dtype accepted by this IR.

    Args:
        x: logits, shape (C,) for 1-D or (N, C) / (N, C, H, W).
        target: class indices.
        weight: per-class weight, same length as C.
        ignore_index: target value to skip.
        reduction: "none" | "mean" | "sum".
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [y, total_weight] following the IR output order; neither is suppressed.
    """
    import torch
    import torch.nn.functional as F

    # Like ops-cv !2491: this IR excludes FP64; undo TTK's FP32 promotion.
    x, weight = (
        value.astype(np.float32)
        if value is not None and value.dtype == np.float64
        else value
        for value in (x, weight)
    )
    out_dtype = x.dtype
    x_f = x.astype(np.float32) if _needs_upcast(x.dtype) else x

    x_t = torch.from_numpy(np.ascontiguousarray(x_f))
    weight_t = None
    if weight is not None:
        weight_f = weight.astype(np.float32) if _needs_upcast(weight.dtype) else weight
        weight_t = torch.from_numpy(np.ascontiguousarray(weight_f))
    target_t = torch.from_numpy(np.ascontiguousarray(target).astype(np.int64))

    is_1d = x_t.dim() == 1
    if is_1d:
        # The public API expects a 0-d target for 1-D input.
        target_t = target_t.reshape(())

    y_t = F.nll_loss(
        x_t, target_t, weight=weight_t, reduction=reduction, ignore_index=ignore_index
    )
    if reduction == "none" and not is_1d:
        # Preserve the prior reference convention, not an inferred kernel
        # exemption. The unresolved none contract is documented above.
        total_weight_t = x_t.new_zeros(())
    else:
        selected = target_t[target_t != ignore_index]
        total_weight_t = (
            x_t.new_tensor(selected.numel())
            if weight_t is None
            else weight_t[selected].sum()
        )

    y = y_t.numpy().astype(out_dtype, copy=False)

    total_weight = np.asarray(total_weight_t.numpy()).reshape(1)
    shapes = kwargs.get("output_ori_shapes") or kwargs.get("output_shapes")
    if shapes:
        y = y.reshape(shapes[0])
        total_weight = total_weight.reshape(shapes[1])
    elif y.ndim == 0:
        y = y.reshape(1)
    return [y, total_weight.astype(out_dtype, copy=False)]


class NllLossTestSpec:
    """NLLLoss test spec (kernel / GEIR flow, numpy in and out)."""

    golden = nll_loss_golden


def _e2e_nll_clamp_target(input, target, *rest, **kwargs):
    """Constrain E2E targets to the legal class range [0, C).

    The framework's default random int targets fall outside [0, C) and the
    reference raises "Target out of bounds".

    The tensor is edited in place because the E2E flow discards the return
    value; the inputs are also returned so the hook satisfies the kernel-flow
    contract, where the returned sequence replaces the generated inputs.
    """
    import torch

    num_classes = input.shape[1] if input.dim() > 1 else input.shape[0]
    # rest follows F.nll_loss: weight, size_average, ignore_index, ...
    ignore_index = rest[2] if len(rest) > 2 else kwargs.get("ignore_index", -100)
    target.copy_(
        torch.where(target == ignore_index, target, target.remainder(num_classes))
    )
    return (input, target, *rest)


def torch_nll_loss_golden(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    **kwargs,
):
    """E2E golden for torch.nn.functional.nll_loss.

    The parameter list mirrors the full torch signature positionally, because the
    E2E flow fills every parameter (inputs and attributes alike) by position.
    Note the framework API returns only the loss, whereas the operator itself has
    two outputs; this spec covers the framework entry's single return value.
    Match the legacy CPU framework reference at the input dtype, including
    intermediate rounding and overflow; do not promote the reference to float64.
    """
    import torch

    weight_cpu = weight.detach().cpu() if weight is not None else None
    result = torch.nn.functional.nll_loss(
        input.detach().cpu(),
        target.detach().to(torch.int64).cpu(),
        weight=weight_cpu,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
    )
    return [result]


class TorchNllLossTestSpec:
    """NLLLoss test spec (e2e flow).

    Registered because an E2E run was verified to dispatch to this operator:
    the eager profile reports kernel ``aclnnNLLLoss_NLLLossAiCore_NLLLoss``.
    """

    golden = torch_nll_loss_golden
    customize_inputs = _e2e_nll_clamp_target
    # NLLLoss calls the same functional API, but keeps ignore_index/reduction as
    # module attributes rather than symbolic scalar inputs to the dynamic graph.
    torch_graph = NLLLoss
