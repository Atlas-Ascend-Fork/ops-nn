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
"""ApplyFtrl：已授权的真实 TensorFlow ApplyFtrl V1 参考适配。

来源：operator-artifacts 58597ec06344f5920f49e26ba203d16ba7a47592，
InplaceApplyFtrl(ApplyFtrl)/Golden/golden_thirdparty.py。
8 个同 dtype 输入，use_locking=False，单显式输出 var；内部真实计算三态。
F16/BF16 reference 升 F32，实际 F32/F64 保持，尊重 TTK Promote 后的实际输入。
provider 保留 Torch 张量传输及 accum/linear 写回合同，计算调用真实 TF API。
TensorFlow 延迟导入；Graph/Session/RefVariable 均为局部，不改变全局 eager 状态。
"""

import numpy as np


__spec__ = {"apply_ftrl": "ApplyFtrlTestSpec"}

_INPUT_NAMES = ("var", "accum", "linear", "grad", "lr", "l1", "l2", "lr_power")
_DTYPES = ("float16", "bfloat16", "float32")


def _validate_numpy(inputs, use_locking):
    if use_locking:
        raise ValueError("ApplyFtrl only supports use_locking=False")
    dtype = inputs[0].dtype
    if dtype.name not in _DTYPES + ("float64",):
        raise TypeError("ApplyFtrl reference supports float16/bfloat16/float32/float64")
    if any(x.dtype != dtype for x in inputs):
        raise TypeError("All ApplyFtrl inputs must have the same dtype")
    if any(x.shape != inputs[0].shape for x in inputs[1:4]):
        raise ValueError("var, accum, linear and grad must have the same shape")
    if any(x.size != 1 for x in inputs[4:]):
        raise ValueError("FTRL scalar inputs must contain exactly one element")


def _compute_tf(inputs):
    """Run the real Ref ApplyFtrl once, then read its updated three states."""
    import tensorflow as tf

    var, accum, linear, grad, lr, l1, l2, lr_power = inputs
    tf_dtype = tf.as_dtype(var.dtype)
    graph = tf.compat.v1.Graph()
    with graph.as_default(), tf.device("/CPU:0"):
        v = tf.compat.v1.Variable(var, dtype=tf_dtype, use_resource=False)
        a = tf.compat.v1.Variable(accum, dtype=tf_dtype, use_resource=False)
        lin = tf.compat.v1.Variable(linear, dtype=tf_dtype, use_resource=False)
        var_op = tf.raw_ops.ApplyFtrl(
            var=v,
            accum=a,
            linear=lin,
            grad=tf.constant(grad, dtype=tf_dtype),
            lr=tf.constant(lr.reshape(()), dtype=tf_dtype),
            l1=tf.constant(l1.reshape(()), dtype=tf_dtype),
            l2=tf.constant(l2.reshape(()), dtype=tf_dtype),
            lr_power=tf.constant(lr_power.reshape(()), dtype=tf_dtype),
            use_locking=False,
            multiply_linear_by_lr=False,
        )
        init = tf.compat.v1.global_variables_initializer()
        config = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            sess.run(init)
            var_new = sess.run(var_op)
            accum_new, linear_new = sess.run((a, lin))
    return tuple(np.array(x, copy=True) for x in (var_new, accum_new, linear_new))


def _reference_states(
    var, accum, linear, grad, lr, l1, l2, lr_power, *, use_locking=False
):
    inputs = tuple(
        np.asarray(x) for x in (var, accum, linear, grad, lr, l1, l2, lr_power)
    )
    _validate_numpy(inputs, use_locking)
    input_dtype = inputs[0].dtype
    reference_dtype = (
        np.float32 if input_dtype.name in ("float16", "bfloat16") else input_dtype
    )
    # TTK 直接传 NumPy 输入：所有 TF 初值必须独立；不因模式标签再升/降精度。
    private_inputs = tuple(
        np.array(x, dtype=reference_dtype, order="C", copy=True) for x in inputs
    )
    return _compute_tf(private_inputs)


class ApplyFtrlTestSpec:
    """Kernel/GEIR 共用；三态计算只映射单个显式输出 var。"""

    @staticmethod
    def golden(
        var, accum, linear, grad, lr, l1, l2, lr_power, *, use_locking=False, **kwargs
    ):
        var_new, _, _ = _reference_states(
            var, accum, linear, grad, lr, l1, l2, lr_power, use_locking=use_locking
        )
        # 有 metadata 时按本次调用声明的输出 dtype 转换；直接调用回到 var dtype。
        # Promote 的原输出 dtype 由隔离 runtime 在上下文恢复后完整转换。
        output_dtypes = kwargs.get("output_dtypes")
        output_dtype = np.asarray(var).dtype
        if output_dtypes is not None:
            if len(output_dtypes) != 1:
                raise ValueError("ApplyFtrl requires exactly one output dtype")
            output_dtype = np.dtype(output_dtypes[0])
            if output_dtype.name not in _DTYPES + ("float64",):
                raise TypeError("Unsupported ApplyFtrl output dtype")
        return [var_new.astype(output_dtype, copy=False)]

    class ThirdPartyImpl:
        """Torch 张量桥接；数值来自真实 TF，而非 Torch 小算子复刻。"""

        def __init__(self, use_locking=False, **kwargs):
            if use_locking:
                raise ValueError("ApplyFtrl only supports use_locking=False")

        def __call__(
            self,
            var,
            accum,
            linear,
            grad,
            lr,
            l1,
            l2,
            lr_power,
            *,
            use_locking=False,
            **kwargs,
        ):
            import torch

            inputs = (var, accum, linear, grad, lr, l1, l2, lr_power)
            if any(not isinstance(x, torch.Tensor) for x in inputs):
                raise TypeError("ApplyFtrl provider inputs must be torch.Tensor")
            if var.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise TypeError("ApplyFtrl provider supports float16/bfloat16/float32")
            if any(x.dtype != var.dtype or x.device != var.device for x in inputs):
                raise TypeError(
                    "All ApplyFtrl inputs must have the same dtype and device"
                )
            if any(x.shape != var.shape for x in inputs[1:4]):
                raise ValueError("var, accum, linear and grad must have the same shape")
            if any(x.numel() != 1 for x in inputs[4:]):
                raise ValueError("FTRL scalar inputs must contain exactly one element")
            # NumPy 不原生承接 Torch BF16；此升 F32 与 reference 精度策略一致。
            numpy_inputs = tuple(
                (
                    x.detach().cpu().float()
                    if x.dtype == torch.bfloat16
                    else x.detach().cpu()
                ).numpy()
                for x in inputs
            )
            var_new, accum_new, linear_new = _reference_states(
                *numpy_inputs, use_locking=use_locking
            )
            # 只在最终输出/状态写回时恢复 provider 原 dtype/device。
            var_out = torch.from_numpy(var_new).to(dtype=var.dtype, device=var.device)
            with torch.no_grad():
                accum.copy_(
                    torch.from_numpy(accum_new).to(
                        dtype=accum.dtype, device=accum.device
                    )
                )
                linear.copy_(
                    torch.from_numpy(linear_new).to(
                        dtype=linear.dtype, device=linear.device
                    )
                )
            return [var_out]

    # 保留 TTK 既有 Torch 张量协议，以实现两个输入状态的原地写回。
    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
