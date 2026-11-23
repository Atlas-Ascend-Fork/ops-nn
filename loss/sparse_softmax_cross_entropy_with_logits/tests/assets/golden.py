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
    "sparse_softmax_cross_entropy_with_logits": "SparseSoftmaxCrossEntropyWithLogitsTestSpec",
}

__golden__ = {
    "kernel": {
        "sparse_softmax_cross_entropy_with_logits": "sparse_softmax_cross_entropy_with_logits_golden"
    }
}


def sparse_softmax_cross_entropy_with_logits_golden(features, labels, **kwargs):
    """
    Golden function for sparse_softmax_cross_entropy_with_logits.
    All the parameters (names and order) follow
    @sparse_softmax_cross_entropy_with_logits_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Reference: CPU torch.nn.functional.cross_entropy(reduction="none") and its
    autograd gradient with respect to features. These implement the TensorFlow
    SparseSoftmaxCrossEntropyWithLogits (loss, backprop) contract:
        loss_n         = logsumexp(features_n) - features_{n, labels_n}
        backprop_{n,c} = softmax(features)_{n,c} - 1{c == labels_n}
    Both outputs come from independent CPU framework operations, without
    importing TensorFlow into a worker that has already loaded torch_npu.

    Contract checks enforced here:
        - class dimension is the last axis of features (num_classes = C);
        - the op requires 2-D features [N, C] and 1-D labels [N], so leading
          dims are flattened before the call and restored afterwards;
        - labels are integer class indices and must lie in [0, C);
        - output order is [loss, backprop] following the IR;
        - loss shape == labels shape, backprop shape == features shape,
          both dtype == features dtype.

    Args:
        features: logits, shape [..., C], dtype float16 / float / bfloat16.
        labels:   class indices, shape [...], dtype int32 / int64, value in [0, C).
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [loss, backprop] following the IR output order.
        loss shape == labels shape, backprop shape == features shape,
        both dtype == features dtype.
    """
    import torch

    features_dtype = features.dtype
    feat_shape = features.shape
    label_shape = labels.shape
    num_classes = feat_shape[-1]

    # Flatten leading dims to [N, C]. Upcast low precision, but preserve double
    # inputs (including promoted references from the test framework).
    compute_dtype = np.float64 if features_dtype == np.float64 else np.float32
    x = features.astype(compute_dtype).reshape(-1, num_classes)
    y = labels.astype(np.int64).reshape(-1)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"labels count {y.shape[0]} != features rows {x.shape[0]} "
            f"(features {feat_shape}, labels {label_shape})"
        )
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if y.size and (y.min() < 0 or y.max() >= num_classes):
        raise ValueError(
            f"labels out of range [0, {num_classes}); "
            f"got [{int(y.min())}, {int(y.max())}]"
        )

    with torch.enable_grad():
        logits = torch.from_numpy(x).requires_grad_()
        loss_tensor = torch.nn.functional.cross_entropy(
            logits, torch.from_numpy(y), reduction="none"
        )
        (backprop_tensor,) = torch.autograd.grad(loss_tensor.sum(), logits)
    loss = loss_tensor.detach().numpy()
    backprop = backprop_tensor.numpy()

    loss = loss.reshape(label_shape).astype(features_dtype, copy=False)
    backprop = backprop.reshape(feat_shape).astype(features_dtype, copy=False)
    return [loss, backprop]


class SparseSoftmaxCrossEntropyWithLogitsTestSpec:
    """SparseSoftmaxCrossEntropyWithLogits test spec (kernel / GEIR flow, numpy in and out)."""

    golden = sparse_softmax_cross_entropy_with_logits_golden
