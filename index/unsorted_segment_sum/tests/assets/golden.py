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

import torch

__spec__ = {
    "unsorted_segment_sum": "UnsortedSegmentSumKernelSpec",
    "aclnnUnsortedSegmentSum": "AclnnUnsortedSegmentSumSpec",
}

__golden__ = {
    "kernel": {"unsorted_segment_sum": "unsorted_segment_sum_golden"},
    "aclnn": {"aclnnUnsortedSegmentSum": "aclnn_unsorted_segment_sum_golden"},
}


def unsorted_segment_sum_golden(
    x, segment_ids, num_segments, *, is_preprocessed=False, check_ids=False, **kwargs
):
    """
    Golden function for unsorted_segment_sum.
    All the parameters (names and order) follow @unsorted_segment_sum_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    out = tf.unsorted_segment_sum(x, segment_ids, int(num_segments))
    with tf.Session() as sess:
        res = sess.run(out)
    return [res]


def aclnn_unsorted_segment_sum_golden(x, segmentIds, numSegments, out=None, **kwargs):
    """
    Aclnn golden for aclnnUnsortedSegmentSum.
    Parameters follow @aclnnUnsortedSegmentSumGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    x_dtype = x.dtype
    if "bfloat16" in str(x_dtype):
        x_np = x.to(torch.float32).numpy()
    else:
        x_np = x.numpy()
    ids_np = segmentIds.numpy()

    out_tf = tf.unsorted_segment_sum(x_np, ids_np, int(numSegments))
    with tf.Session() as sess:
        res = sess.run(out_tf)

    return [torch.from_numpy(res).to(x_dtype)]


def aclnn_unsorted_segment_sum_customize_inputs(
    x, segmentIds, numSegments, out=None, **kwargs
):
    """
    Customize inputs for aclnnUnsortedSegmentSum.
    All the input Tensors are torch.Tensor.
    """
    num_segments = int(numSegments)
    if num_segments > 0:
        new_ids = torch.randint(0, num_segments, segmentIds.shape).to(segmentIds.dtype)
        segmentIds.copy_(new_ids)


class UnsortedSegmentSumKernelSpec:
    golden = unsorted_segment_sum_golden


class _TorchUnsortedSegmentSum:
    """torch 侧没有原生 unsorted_segment_sum，用 index_add_ 组合实现作为 third_party。

    Spec 类形式的 third_party 由远端按 `__init__`/`__call__` 绑定 Parameter Name：
    `x`/`segmentIds` 两个输入 + `numSegments` 属性 (C 头文件参数序)。
    """

    def __init__(self, *, numSegments, **kwargs):
        self.num_segments = int(numSegments)

    def __call__(self, x, segmentIds, **kwargs):
        n_seg = self.num_segments
        ids = segmentIds
        # 语义对齐 tf.unsorted_segment_sum：ids 是 x 的前缀维度，
        # ids 每个元素(可 N 维)聚合一个 x 的后缀切片；ids 为标量时聚合整个 x。
        ids_rank = ids.dim()
        num_slices = max(ids.numel(), 1)
        ids_flat = ids.reshape(-1).to(torch.long)
        valid = (ids_flat >= 0) & (ids_flat < n_seg)
        x_flat = x.reshape(num_slices, -1)[valid]
        out = x.new_zeros([n_seg] + list(x.shape[ids_rank:]))
        out.reshape(n_seg, -1).index_add_(0, ids_flat[valid], x_flat.to(out.dtype))
        return [out]


class _TfUnsortedSegmentSum:
    """tf 侧直接调原生 API。远端输入是 torch.Tensor，先转 numpy。

    C 形参名是 x/segmentIds/numSegments；tf API 形参名是
    data/segment_ids/num_segments，由 __init__/__call__ 代理转发。
    """

    def __init__(self, *, numSegments, **kwargs):
        self.num_segments = int(numSegments)

    def __call__(self, x, segmentIds, **kwargs):
        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()

        def _to_np(t):
            # 远端 CPU 腿可能送 ml_dtypes.bfloat16 ndarray 或 torch.Tensor，统一转 numpy。
            # float16/bfloat16 提升为 float32 计算：tf 的 UnsortedSegmentSum CPU 侧
            # 未注册低精度 kernel，防御性提升避免运行期报错（L1 容差下不影响判定）。
            if isinstance(t, torch.Tensor):
                if t.dtype in (torch.bfloat16, torch.float16):
                    return t.detach().to(torch.float32).cpu().numpy()
                return t.detach().cpu().numpy()
            import numpy as _np

            if isinstance(t, _np.ndarray) and t.dtype in (
                _np.dtype("ml_dtypes.bfloat16"),
                _np.float16,
            ):
                return t.astype(_np.float32)
            return _np.asarray(t)

        x_np = _to_np(x)
        ids_np = _to_np(segmentIds)
        out = tf.unsorted_segment_sum(x_np, ids_np, self.num_segments)
        with tf.Session() as sess:
            res = sess.run(out)
        if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16:
            return [torch.from_numpy(res).to(torch.bfloat16)]
        if isinstance(x, torch.Tensor):
            return [torch.from_numpy(res).to(x.dtype)]
        return [res]


class AclnnUnsortedSegmentSumSpec:
    golden = aclnn_unsorted_segment_sum_golden
    customize_inputs = aclnn_unsorted_segment_sum_customize_inputs
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint32": {"standard": "binary_equal"},
        "uint64": {"standard": "binary_equal"},
    }
    third_party = {
        "torch": _TorchUnsortedSegmentSum,
        "tf": _TfUnsortedSegmentSum,
    }
