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
"""TestSpec for CrossEntropyLoss.

Reference source
----------------
``torch.nn.functional.cross_entropy`` plus ``torch.nn.functional.log_softmax``
on CPU -- the canonical framework implementation, independent of the NPU
operator under test. NumPy is used only for dtype/container bridging, never for
the reference mathematics.

Output protocol (follows cross_entropy_loss_def.cpp)
----------------------------------------------------
Four outputs, in order: ``loss``, ``log_prob``, ``zloss``, ``lse_for_zloss``.

``zloss`` / ``lse_for_zloss``
----------------------------
The kernel/GEIR contract in ``op_graph/cross_entropy_loss_proto.h`` makes
zloss conditional on ``return_zloss`` and declares its shape (N,), including
an all-zero output when the scale is zero. The documented formula is
``scale * logsumexp(input, dim=1)**2``. ``lse_for_zloss`` is (N,) and is
inactive only when the scale is zero (``docs/aclnnCrossEntropyLoss.md``).
The reference returns None only for these inactive slots, never because a
kernel forgot to write an active output. The arch35 implementation currently
does not write either auxiliary output. README explicitly marks zloss as
unsupported, whereas the proto describes conditional outputs and infershape
reserves (1,) placeholders. Non-default probes expose this contract conflict;
they are not evidence of failed precision within the documented support range.
An active output whose declared shape conflicts with the proto raises an
explicit contract error instead of comparing a per-row reference to a placeholder.

ACLNN separately documents the zloss feature as unsupported. Its wrapper
rejects non-default requests explicitly; that restriction is not applied to
the kernel/GEIR reference, which preserves the proto's conditional outputs
rather than silently suppressing them. The existing 80 kernel cases use the
default False/0 configuration and therefore compare loss/log_prob only.

``loss`` shape
--------------
InferShape makes ``loss`` rank 1 always: ``(N,)`` for ``reduction="none"``,
else ``(1,)`` (``LOSS_SHAPE``). The reference is reshaped to match.
"""

import numpy as np

__spec__ = {
    "cross_entropy_loss": "CrossEntropyLossTestSpec",
    "aclnnCrossEntropyLoss": "AclnnCrossEntropyLossTestSpec",
}

__golden__ = {
    "kernel": {"cross_entropy_loss": "cross_entropy_loss_golden"},
    "aclnn": {"aclnnCrossEntropyLoss": "aclnn_cross_entropy_loss_golden"},
}


def _demote_promoted(value):
    """Like ops-cv !2491: IR/ACLNN exclude FP64 tensors, so undo TTK promotion."""
    if hasattr(value, "detach"):
        return value.float() if str(value.dtype) == "torch.float64" else value
    arr = np.asarray(value)
    return arr.astype(np.float32) if arr.dtype == np.float64 else value


def _to_torch_reference(tensor):
    import torch

    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
        return tensor.detach().cpu().to(dtype)
    arr = np.asarray(tensor)
    if arr.dtype not in (
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.int16,
        np.int8,
        np.uint8,
    ):
        arr = arr.astype(np.float32)
    dtype = torch.float64 if arr.dtype == np.float64 else torch.float32
    return torch.from_numpy(arr).to(dtype)


def _compute_loss_and_logprob(
    logits_t, targets_t, weight_t, reduction, ignore_index, label_smoothing
):
    import torch
    import torch.nn.functional as F

    if weight_t is not None:
        compute = torch.promote_types(logits_t.dtype, weight_t.dtype)
        logits_t = logits_t.to(compute)
        weight_t = weight_t.to(compute)
    N, C = logits_t.shape[0], logits_t.shape[1]

    if N == 0 or C == 0:
        if reduction == "none":
            loss = logits_t.new_zeros((N,))
        elif reduction == "mean":
            loss = logits_t.new_full((1,), float("nan"))
        else:
            loss = logits_t.new_zeros((1,))
        log_prob = F.log_softmax(logits_t, dim=-1)
        return loss, log_prob

    loss = F.cross_entropy(
        logits_t,
        targets_t,
        weight=weight_t,
        reduction=reduction,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )

    log_prob = F.log_softmax(logits_t, dim=-1)

    if reduction != "none" and loss.dim() == 0:
        loss = loss.unsqueeze(0)

    return loss, log_prob


def _compute_zloss(logits, scale, return_zloss):
    import torch

    lse = torch.logsumexp(logits, dim=1) if scale != 0 else None
    zloss = None
    if return_zloss:
        zloss = (
            scale * lse.square()
            if lse is not None
            else logits.new_zeros((logits.shape[0],))
        )
    return zloss, lse


def cross_entropy_loss_golden(
    input,
    target,
    weight=None,
    *,
    reduction="mean",
    ignore_index=-100,
    label_smoothing=0.0,
    lse_square_scale_for_zloss=0.0,
    return_zloss=False,
    **kwargs,
):
    """
    Golden function for cross_entropy_loss.
    All the parameters (names and order) follow @cross_entropy_loss_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Computes the reference with torch.nn.functional.cross_entropy (loss) and
    torch.nn.functional.log_softmax (log_prob) on CPU. Low precision inputs are
    upcast to float32; TTK-promoted float64 inputs return to float32, the highest
    floating dtype accepted by this IR. Low-precision return handling is unchanged.

    Args:
        input: logits, shape (N, C).
        target: class indices, shape (N,), int32 or int64.
        weight: optional per-class weight, shape (C,).
        reduction: "none" | "mean" | "sum".
        ignore_index: target value to skip.
        label_smoothing: smoothing factor in [0, 1).
        lse_square_scale_for_zloss: coefficient of squared logsumexp.
        return_zloss: whether the per-row zloss output is requested.
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [loss, log_prob, zloss, lse_for_zloss] following the IR output order.
        Inactive auxiliary outputs are None; see the module docstring.
    """
    if isinstance(reduction, bytes):
        reduction = reduction.decode()

    input, weight = map(_demote_promoted, (input, weight))
    out_dtype = np.asarray(input).dtype

    logits_t = _to_torch_reference(input)
    weight_t = _to_torch_reference(weight) if weight is not None else None

    import torch

    target_t = torch.from_numpy(np.ascontiguousarray(target).astype(np.int64))

    loss, log_prob = _compute_loss_and_logprob(
        logits_t,
        target_t,
        weight_t,
        reduction,
        int(ignore_index),
        float(label_smoothing),
    )
    zloss, lse = _compute_zloss(logits_t, lse_square_scale_for_zloss, return_zloss)
    shapes = kwargs.get("output_ori_shapes") or kwargs.get("output_shapes")
    for index, name, output in ((2, "zloss", zloss), (3, "lse_for_zloss", lse)):
        if output is None or not shapes or len(shapes) <= index:
            continue
        declared = shapes[index]
        if declared is not None and (
            len(declared) != output.ndim
            or any(
                dim >= 0 and dim != actual
                for dim, actual in zip(declared, output.shape)
            )
        ):
            raise ValueError(
                f"CrossEntropyLoss {name} contract conflict: active reference shape "
                f"{tuple(output.shape)} does not match declared {tuple(declared)}"
            )
    return [
        output.numpy().astype(out_dtype, copy=False) if output is not None else None
        for output in (loss, log_prob, zloss, lse)
    ]


class CrossEntropyLossTestSpec:
    """CrossEntropyLoss test spec (kernel / GEIR flow, numpy in and out)."""

    golden = cross_entropy_loss_golden

    class ThirdParty:
        """Adapt the framework reference to all four kernel output slots."""

        def __init__(
            self,
            reduction="mean",
            ignore_index=-100,
            label_smoothing=0.0,
            lse_square_scale_for_zloss=0.0,
            return_zloss=False,
        ):
            self.reduction = reduction
            self.ignore_index = ignore_index
            self.label_smoothing = label_smoothing
            self.scale = lse_square_scale_for_zloss
            self.return_zloss = return_zloss

        def __call__(self, input, target, weight=None):
            # Keep the caller's device. Only the reference compute dtype changes.
            import torch

            compute = torch.float64 if input.dtype == torch.float64 else torch.float32
            loss, log_prob = _compute_loss_and_logprob(
                input.to(compute),
                target.to(torch.int64),
                weight.to(compute) if weight is not None else None,
                self.reduction,
                self.ignore_index,
                self.label_smoothing,
            )
            zloss, lse = _compute_zloss(
                input.to(compute), self.scale, self.return_zloss
            )
            return [
                output.to(input.dtype) if output is not None else None
                for output in (loss, log_prob, zloss, lse)
            ]

    third_party = {"torch": ThirdParty}


def aclnn_cross_entropy_loss_golden(
    input,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
    lse_square_scale_for_zloss,
    return_zloss,
    loss_out,
    log_prob_out,
    zloss_out,
    lse_for_zloss_out,
    **kwargs,
):
    """
    Aclnn golden for aclnnCrossEntropyLoss.
    All the parameters (name & order) follow \
        function `aclnnCrossEntropyLossGetWorkspaceSize` in @aclnn_cross_entropy_loss.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        (loss, log_prob, zloss, lse_for_zloss) as torch.Tensor or numpy.ndarray.
    """
    import torch

    if return_zloss or lse_square_scale_for_zloss != 0:
        raise NotImplementedError("CrossEntropyLoss zloss outputs are not supported")
    del loss_out, log_prob_out, zloss_out, lse_for_zloss_out

    if isinstance(reduction, bytes):
        reduction = reduction.decode()

    input, weight = map(_demote_promoted, (input, weight))
    is_torch_tensor = isinstance(input, torch.Tensor)
    out_dtype = input.dtype

    logits_t = _to_torch_reference(input)
    if weight is not None:
        weight_t = _to_torch_reference(weight)
    else:
        weight_t = None
    if isinstance(target, torch.Tensor):
        target_t = target.to(torch.int64)
    else:
        target_t = torch.from_numpy(np.asarray(target)).to(torch.int64)

    loss, log_prob = _compute_loss_and_logprob(
        logits_t, target_t, weight_t, reduction, ignore_index, label_smoothing
    )

    if is_torch_tensor:
        return (
            loss.to(out_dtype),
            log_prob.to(out_dtype),
            None,
            None,
        )
    return (
        loss.numpy().astype(out_dtype, copy=False),
        log_prob.numpy().astype(out_dtype, copy=False),
        None,
        None,
    )


class AclnnCrossEntropyLossTestSpec:
    """CrossEntropyLoss test spec (aclnn flow, torch.Tensor in and out)."""

    golden = aclnn_cross_entropy_loss_golden
