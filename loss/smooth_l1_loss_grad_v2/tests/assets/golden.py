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
    "smooth_l1_loss_grad_v2": "SmoothL1LossGradV2TestSpec",
}

__golden__ = {"kernel": {"smooth_l1_loss_grad_v2": "smooth_l1_loss_grad_v2_golden"}}


def smooth_l1_loss_grad_v2_golden(
    predict, label, dout, *, sigma=1.0, reduction="mean", **kwargs
):
    """
    Golden function for smooth_l1_loss_grad_v2.
    All the parameters (names and order) follow @smooth_l1_loss_grad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch
    import torch.nn.functional as F

    # Like ops-cv !2491: this IR excludes FP64; undo TTK's FP32 promotion.
    predict, label, dout = (
        value.astype(np.float32) if value.dtype == np.float64 else value
        for value in (predict, label, dout)
    )
    predict_shape = predict.shape
    label_shape = label.shape
    dout_shape = dout.shape

    max_shape = np.broadcast_shapes(predict_shape, label_shape, dout_shape)
    predict = np.broadcast_to(predict, max_shape)
    label = np.broadcast_to(label, max_shape)
    dout = np.broadcast_to(dout, max_shape)

    dtype_real = predict.dtype
    if "bf16" in str(dtype_real) or "bfloat16" in str(dtype_real):
        predicts = torch.from_numpy(predict.view(np.int16)).view(torch.bfloat16)
        target = torch.from_numpy(label.view(np.int16)).view(torch.bfloat16)
        grad_output = torch.from_numpy(dout.view(np.int16)).view(torch.bfloat16)
    else:
        predicts = torch.from_numpy(predict)
        target = torch.from_numpy(label)
        grad_output = torch.from_numpy(dout)

    ori_dtype = predicts.dtype

    if ori_dtype != torch.float32:
        predicts = predicts.to(torch.float32)
        target = target.to(torch.float32)
        grad_output = grad_output.to(torch.float32)

    with torch.enable_grad():
        predicts = predicts.detach().requires_grad_(True)
        loss = F.smooth_l1_loss(predicts, target, reduction=reduction, beta=sigma)
        (res,) = torch.autograd.grad(loss, predicts, grad_outputs=torch.ones_like(loss))
    if sigma == 0:
        # Public L1 backward uses sign(NaN)=0. Preserve the IR's documented NaN
        # propagation (including Inf-Inf), as the previous reference did.
        res = res.masked_fill(torch.isnan(predicts - target), torch.nan)
    # The IR allows a broadcastable per-element dout even for mean/sum. Applying
    # it after differentiating retains the broadcast output, not a scalar VJP.
    res = res * grad_output

    if ori_dtype != torch.float32:
        res = res.to(ori_dtype)

    if "bf16" in str(dtype_real) or "bfloat16" in str(dtype_real):
        return res.view(torch.int16).numpy().view(dtype_real)
    else:
        return res.numpy()


class SmoothL1LossGradV2TestSpec:
    """SmoothL1LossGradV2 test spec (kernel / GEIR flow, numpy in and out)."""

    golden = smooth_l1_loss_grad_v2_golden
