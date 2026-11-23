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

import numpy as np

__spec__ = {
    "binary_cross_entropy": "BinaryCrossEntropyTestSpec",
    "aclnnBinaryCrossEntropy": "AclnnBinaryCrossEntropyTestSpec",
    "torch.nn.functional.binary_cross_entropy": "TorchBinaryCrossEntropyTestSpec",
}

__golden__ = {
    "kernel": {"binary_cross_entropy": "binary_cross_entropy_golden"},
    "aclnn": {"aclnnBinaryCrossEntropy": "aclnn_binary_cross_entropy_golden"},
}

_REDUCTION_INT_TO_STR = {0: "none", 1: "mean", 2: "sum"}


def _demote_promoted(value):
    """Like ops-cv !2491: IR/ACLNN exclude FP64 tensors, so undo TTK promotion.

    Do not use this for Torch E2E, which accepts genuine FP64 inputs.
    """
    if hasattr(value, "detach"):
        return value.float() if str(value.dtype) == "torch.float64" else value
    arr = np.asarray(value)
    return arr.astype(np.float32) if arr.dtype == np.float64 else value


def aclnn_binary_cross_entropy_golden(
    self_t, target_t, weight_t, reduction, out_t=None, **kwargs
):
    import torch.nn.functional as F

    self_t, target_t, weight_t = map(_demote_promoted, (self_t, target_t, weight_t))
    red = _REDUCTION_INT_TO_STR.get(int(reduction), "mean")
    # The API requires self/target/weight/out to share a dtype. TTK leaves the
    # pure output slot unpromoted, so its dtype restores the native reference
    # precision without inspecting any device output values.
    dtype = out_t.dtype if out_t is not None else self_t.dtype
    self_f = self_t.detach().to(device="cpu", dtype=dtype)
    target_f = target_t.detach().to(device="cpu", dtype=dtype)
    weight_f = (
        weight_t.detach().to(device="cpu", dtype=dtype)
        if weight_t is not None
        else None
    )
    if weight_f is not None:
        result = F.binary_cross_entropy(
            self_f, target_f, weight=weight_f, reduction=red
        )
    else:
        result = F.binary_cross_entropy(self_f, target_f, reduction=red)
    return result


class Constant:
    CONST_MIN = -100
    CONST_ONE = 1
    CONST_NEG_ONE = -1


def binary_cross_entropy_golden(
    input_data_x1, input_data_x2, input_data_weight=None, *, reduction="mean", **kwargs
):
    """
    Golden function for binary_cross_entropy.
    All the parameters (names and order) follow @binary_cross_entropy_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    input_data_x1, input_data_x2, input_data_weight = map(
        _demote_promoted, (input_data_x1, input_data_x2, input_data_weight)
    )
    ori_dtype = input_data_x1.dtype
    shape = input_data_x1.shape
    if ori_dtype == "float16" or ori_dtype == "bfloat16":
        input_data_x1 = input_data_x1.astype("float32")
        input_data_x2 = input_data_x2.astype("float32")
        if input_data_weight is not None:
            input_data_weight = input_data_weight.astype("float32")

    x_log_tmp = np.log(input_data_x1)
    x_log_tmp = np.maximum(x_log_tmp, Constant.CONST_MIN)
    data_mul1 = np.multiply(x_log_tmp, input_data_x2)
    x_neg_tmp = np.multiply(input_data_x1, Constant.CONST_NEG_ONE)
    x1_tmp = np.add(x_neg_tmp, Constant.CONST_ONE)
    y_neg_tmp = np.multiply(input_data_x2, Constant.CONST_NEG_ONE)
    y1_tmp = np.add(y_neg_tmp, Constant.CONST_ONE)
    x1_log_tmp = np.log(x1_tmp)
    x1_log_tmp = np.maximum(x1_log_tmp, Constant.CONST_MIN)
    data_mul2 = np.multiply(x1_log_tmp, y1_tmp)
    data_sum = np.add(data_mul1, data_mul2)
    result = np.multiply(data_sum, Constant.CONST_NEG_ONE)

    if input_data_weight is not None:
        result = np.multiply(result, input_data_weight)

    reduce_elts = 1.0
    for i in shape:
        reduce_elts *= i
    cof = reduce_elts ** (-1)

    axis_d = []
    for i, _ in enumerate(shape):
        axis_d.append(i)
    axis_d = tuple(axis_d)

    if reduction == "mean":
        result = np.multiply(result, cof)
        result = np.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "sum":
        result = np.sum(result, axis=axis_d, keepdims=False)
    elif reduction == "none":
        pass

    return result.astype(ori_dtype)


class BinaryCrossEntropyTestSpec:
    """BinaryCrossEntropy test spec (kernel / GEIR flow, numpy in and out)."""

    golden = binary_cross_entropy_golden
    third_party = {"torch": "torch.nn.functional.binary_cross_entropy"}


class AclnnBinaryCrossEntropyTestSpec:
    """BinaryCrossEntropy test spec (aclnn flow, torch.Tensor in and out)."""

    golden = aclnn_binary_cross_entropy_golden


def _e2e_bce_clamp_inputs(input, target, *rest, **kwargs):
    """Constrain E2E inputs to the operator's legal domain.

    binary_cross_entropy requires input and target in [0, 1]; the framework's
    default random inputs fall outside that range and the reference refuses to
    run. Preserve all legal probabilities, including 0/1, and leave weights
    unchanged: the API does not restrict them to a positive bounded range.

    The tensors are edited in place because the E2E flow discards the return
    value; the inputs are also returned so the hook satisfies the kernel-flow
    contract, where the returned sequence replaces the generated inputs.
    """
    input.copy_(input.where((input >= 0) & (input <= 1), input.abs().clamp(0.01, 0.99)))
    target.copy_(target.abs().clamp(0.0, 1.0))
    return (input, target, *rest)


def torch_binary_cross_entropy_golden(
    input,
    target,
    weight=None,
    size_average=None,
    reduce=None,
    reduction="mean",
    **kwargs,
):
    """E2E golden for torch.nn.functional.binary_cross_entropy.

    The parameter list mirrors the full torch signature positionally, because the
    E2E flow fills every parameter (inputs and attributes alike) by position.
    Match the legacy CPU framework reference at the input dtype, including
    intermediate rounding and overflow; do not promote the reference to float64.
    """
    import torch

    weight_cpu = weight.detach().cpu() if weight is not None else None
    result = torch.nn.functional.binary_cross_entropy(
        input.detach().cpu(),
        target.detach().cpu(),
        weight=weight_cpu,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )
    return [result]


class TorchBinaryCrossEntropyTestSpec:
    """BinaryCrossEntropy test spec (e2e flow).

    Registered because an E2E run was verified to dispatch to this operator:
    the eager profile reports kernel
    ``aclnnBinaryCrossEntropy_BinaryCrossEntropy_...``.
    """

    golden = torch_binary_cross_entropy_golden
    customize_inputs = _e2e_bce_clamp_inputs
