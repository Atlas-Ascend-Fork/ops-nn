#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""sparse_segment_sum_grad 在 kernel/geir 测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，转 tf tensor 后调 tf.raw_ops.SparseSegmentSumGrad 计算，结果转回 numpy；
third_party 的 tf vendor 收到 tf.Tensor（框架按 vendor 自动转换），直接调同一接口，无需手动转换。
无 torch 现成接口，以 tf.raw_ops.SparseSegmentSumGrad 作为三方标杆。
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__spec__ = {
    "sparse_segment_sum_grad": "SparseSegmentSumGradKernelSpec",
    "tf.raw_ops.SparseSegmentSumGrad": "TfSparseSegmentSumGradSpec",
}

import tensorflow as tf


class SparseSegmentSumGradKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 tf.Tensor"""

    def golden(grad, indices, segment_ids, output_dim0, **kwargs):
        grad = tf.convert_to_tensor(grad)
        indices = tf.convert_to_tensor(indices)
        segment_ids = tf.convert_to_tensor(segment_ids)
        orig_dtype = grad.dtype
        if hasattr(output_dim0, "item"):
            dim0_val = output_dim0.item()
        else:
            dim0_val = int(output_dim0)

        if len(indices) == 0:
            inner_shape = grad.shape[1:]
            result = tf.zeros((dim0_val,) + inner_shape, dtype=orig_dtype)
        else:
            result = tf.raw_ops.SparseSegmentSumGrad(
                grad=grad,
                indices=tf.cast(indices, tf.int64),
                segment_ids=tf.cast(segment_ids, tf.int64),
                output_dim0=tf.constant(dim0_val, dtype=tf.int32),
            )

        return [result.numpy()]

    class ThirdPartyImpl:
        def __init__(self, grad, indices, segment_ids, output_dim0, **kwargs):
            self.grad = grad
            self.indices = indices
            self.segment_ids = segment_ids
            self.orig_dtype = grad.dtype
            self.output_dim0_t = output_dim0
            self._empty = indices.shape[0] == 0
            if self._empty:
                self._inner_shape = grad.shape[1:]

        def __call__(self, **kwargs):
            if self._empty:
                result = tf.zeros(
                    (self.output_dim0_t[0],) + self._inner_shape, dtype=self.orig_dtype
                )
                return [result]
            result = tf.raw_ops.SparseSegmentSumGrad(
                grad=self.grad,
                indices=tf.cast(self.indices, tf.int64),
                segment_ids=tf.cast(self.segment_ids, tf.int64),
                output_dim0=self.output_dim0_t[0],
            )
            return [result]

    third_party = {"tf": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }


class TfSparseSegmentSumGradSpec:
    """TensorFlow SparseSegmentSumGrad E2E TestSpec。"""

    @staticmethod
    def _run(grad, indices, segment_ids, output_dim0):
        return tf.raw_ops.SparseSegmentSumGrad(
            grad=tf.convert_to_tensor(grad),
            indices=tf.convert_to_tensor(indices),
            segment_ids=tf.convert_to_tensor(segment_ids),
            output_dim0=tf.cast(
                tf.reshape(tf.convert_to_tensor(output_dim0), []), tf.int32
            ),
        )

    @staticmethod
    def golden(grad, indices, segment_ids, output_dim0, name=None, **kwargs):
        del name, kwargs
        return [
            TfSparseSegmentSumGradSpec._run(grad, indices, segment_ids, output_dim0)
        ]

    class ThirdPartyImpl:
        def __init__(
            self, grad, indices, segment_ids, output_dim0, name=None, **kwargs
        ):
            del name, kwargs
            self.grad = grad
            self.indices = indices
            self.segment_ids = segment_ids
            self.output_dim0 = output_dim0

        def __call__(self):
            return [
                TfSparseSegmentSumGradSpec._run(
                    self.grad,
                    self.indices,
                    self.segment_ids,
                    self.output_dim0,
                )
            ]

    third_party = {"tf": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }
