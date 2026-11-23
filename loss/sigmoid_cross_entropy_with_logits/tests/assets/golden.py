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
    "sigmoid_cross_entropy_with_logits": "SigmoidCrossEntropyWithLogitsTestSpec",
}

__golden__ = {
    "kernel": {
        "sigmoid_cross_entropy_with_logits": "sigmoid_cross_entropy_with_logits_golden"
    }
}


def sigmoid_cross_entropy_with_logits_golden(predict, target, **kwargs):
    """
    Golden function for sigmoid_cross_entropy_with_logits.
    All the parameters (names and order) follow
    @sigmoid_cross_entropy_with_logits_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Reference: the canonical CPU framework op
        torch.nn.functional.binary_cross_entropy_with_logits(..., reduction="none")
    which returns the element-wise loss (no reduction), equivalent to the stable
        loss = max(logits, 0) - logits * labels + log1p(exp(-|logits|))
    We call the framework op directly (not a self-written NumPy formula) so the
    golden matches the standard SigmoidCrossEntropyWithLogits contract. The op
    runs on CPU and is independent of the tested NPU operator. Using Torch also
    avoids importing TensorFlow into a worker that has already loaded torch_npu.

    Contract (ordinary op, NOT V2):
        - inputs are predict (logits) and target (labels), same shape/dtype;
        - output is the element-wise loss, shape == predict shape;
        - NO mean/sum reduction, NO weight / pos_weight / reduction attribute,
          and V2 (SigmoidCrossEntropyWithLogitsV2) semantics are not mixed in;
        - registration name stays sigmoid_cross_entropy_with_logits.

    NumPy is used only for dtype/shape adaptation around the framework op.
    Float16/bfloat16 inputs are upcast to float32; TTK-promoted double inputs
    return to float32, the highest floating dtype accepted by this IR.

    Args:
        predict: logits, dtype float16 / float / bfloat16.
        target:  labels, same shape and dtype as predict.
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [loss] following the IR output order; loss shape == predict shape,
        dtype == predict dtype.
    """
    import torch

    # Like ops-cv !2491: this IR excludes FP64; undo TTK's FP32 promotion.
    predict, target = (
        value.astype(np.float32) if value.dtype == np.float64 else value
        for value in (predict, target)
    )
    predict_dtype = predict.dtype
    p = torch.from_numpy(predict.astype(np.float32))
    t = torch.from_numpy(target.astype(np.float32))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(p, t, reduction="none")
    return [loss.numpy().astype(predict_dtype, copy=False)]


class SigmoidCrossEntropyWithLogitsTestSpec:
    """SigmoidCrossEntropyWithLogits test spec (kernel / GEIR flow, numpy in and out)."""

    golden = sigmoid_cross_entropy_with_logits_golden
