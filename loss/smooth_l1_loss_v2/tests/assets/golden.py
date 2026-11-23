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


__spec__ = {
    "smooth_l1_loss_v2": "SmoothL1LossV2TestSpec",
    "torch.nn.functional.smooth_l1_loss": "TorchSmoothL1LossTestSpec",
}

__golden__ = {"kernel": {"smooth_l1_loss_v2": "smooth_l1_loss_v2_golden"}}


def smooth_l1_loss_v2_golden(predict, label, *, sigma=1.0, reduction="mean", **kwargs):
    """
    Golden function for smooth_l1_loss_v2.
    All the parameters (names and order) follow @smooth_l1_loss_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    # Like ops-cv !2491: this IR excludes FP64; undo TTK's FP32 promotion.
    predict, label = (
        value.astype("float32") if value.dtype.name == "float64" else value
        for value in (predict, label)
    )
    ori_dtype = predict.dtype
    input_dtype = "float32"
    if "float16" in ori_dtype.name:
        predict = predict.astype(input_dtype)
        label = label.astype(input_dtype)
    predict_t = torch.from_numpy(predict)
    label_t = torch.from_numpy(label)
    res = torch.nn.functional.smooth_l1_loss(
        predict_t, label_t, reduction=reduction, beta=sigma
    ).numpy()
    if "float16" in ori_dtype.name:
        res = res.astype(ori_dtype, copy=False)
    return res


def torch_smooth_l1_loss_golden(
    input, target, size_average=None, reduce=None, reduction="mean", beta=1.0, **kwargs
):
    """E2E golden for torch.nn.functional.smooth_l1_loss.

    The parameter list mirrors the full torch signature positionally, because the
    E2E flow fills every parameter (inputs and attributes alike) by position.
    ``beta`` is the operator's ``sigma`` attribute. Match the legacy CPU framework
    reference at the input dtype, including intermediate rounding and overflow;
    do not promote the reference to float64.
    """
    import torch

    result = torch.nn.functional.smooth_l1_loss(
        input.detach().cpu(),
        target.detach().cpu(),
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        beta=beta,
    )
    return [result]


class SmoothL1LossV2TestSpec:
    """SmoothL1LossV2 test spec (kernel / GEIR flow, numpy in and out)."""

    golden = smooth_l1_loss_v2_golden


class TorchSmoothL1LossTestSpec:
    """SmoothL1LossV2 test spec (e2e flow).

    Registered because an E2E run was verified to dispatch to this operator:
    the eager profile reports kernel
    ``aclnnSmoothL1Loss_SmoothL1Loss_SmoothL1LossV2``.
    """

    golden = torch_smooth_l1_loss_golden
