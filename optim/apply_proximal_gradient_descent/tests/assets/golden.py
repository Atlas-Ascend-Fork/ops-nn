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

"""TensorFlow Golden 与独立 Torch provider，供 Kernel/GEIR 三方比对。

主Golden实际调用 tf.raw_ops.ResourceApplyProximalGradientDescent，
私有 tf.Variable 承接状态更新并读回结果，不修改调用方 NumPy 输入。
原 FP16 五输入全部升 FP32计算，原FP32/FP64计算保持；最终逐输出完整cast。
TTK cross_check 将 Golden 输入和输出 dtype 同步提升，provider 使用原 dtype。
TTK 的 output_dtypes 声明输出精度；直接调用缺省遵循单输出var的dtype。
高精度诊断须单独标识，不得把其未舍入结果作为本轮最终输出参考。
原库对标接口来源：ops-nn commit 9bead91b81171d946f93f2daa8988b5181af6a0d，
optim/apply_proximal_gradient_descent/tests/assets/golden.py。
Torch provider独立使用数学sign及l1>0/否则直接更新分支，保原dtype/device。
TensorFlow缺失或CPU原语不支持时直接报错，禁止公式回退或静默降精度。
"""

import numpy as np

__spec__ = {
    "apply_proximal_gradient_descent": "ApplyProximalGradientDescentTestSpec",
}


def _proximal_gradient_descent_compute(var, alpha, l1, l2, delta):
    """独立Torch provider；数学sign，保持来参设备和dtype，不修改输入。"""
    import torch

    inputs = (var, alpha, l1, l2, delta)
    if var.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError("目标仅支持 float16/float32；float64 仅供 Promote 参考")
    if any(x.dtype != var.dtype or x.device != var.device for x in inputs):
        raise ValueError("所有输入必须具有相同 dtype 和 device")
    if var.shape != delta.shape:
        raise ValueError("var 和 delta 的 shape 必须完全相同，不支持广播")
    for name, scalar in (("alpha", alpha), ("l1", l1), ("l2", l2)):
        if scalar.shape not in (torch.Size([]), torch.Size([1])):
            raise ValueError(f"{name} 必须为 0-D 或 shape=[1] 的标量张量")

    output_dtype = var.dtype
    if output_dtype == torch.float16:
        # 目标README规定FP16内部使用FP32；provider最后回原dtype。
        # Golden入口已提升原FP16；FP32/FP64均保持来参精度。
        var, alpha, l1, l2, delta = (x.to(torch.float32) for x in inputs)
    alpha, l1, l2 = (x.reshape(()) for x in (alpha, l1, l2))
    prox = var - alpha * delta
    threshold = torch.clamp_min(torch.abs(prox) - alpha * l1, 0)
    denominator = (alpha * l2) + prox.new_tensor(1)
    positive = (torch.sign(prox) * threshold) / denominator
    # TF功能分支；l1=0执行直接更新，保留其带符号零等数值语义。
    result = torch.where(l1 > 0, positive, prox / denominator)
    return result.to(output_dtype)


def _tf_proximal_gradient_descent_compute(var, alpha, l1, l2, delta, use_locking=False):
    """真实TF原语：私有资源更新后读回Tensor，禁止手写公式替代。"""
    import tensorflow as tf

    inputs = (var, alpha, l1, l2, delta)
    if var.dtype not in (tf.float32, tf.float64):
        raise TypeError("TF参考核心接收float32/float64；低精度由Golden入口提升")
    if any(x.dtype != var.dtype for x in inputs):
        raise ValueError("所有输入必须具有相同 dtype")
    if var.shape != delta.shape:
        raise ValueError("var 和 delta 的 shape 必须完全相同，不支持广播")
    for name, scalar in (("alpha", alpha), ("l1", l1), ("l2", l2)):
        if tuple(scalar.shape) not in ((), (1,)):
            raise ValueError(f"{name} 必须为 0-D 或 shape=[1] 的标量张量")

    with tf.device(var.device):
        resource = tf.Variable(tf.identity(var), trainable=False)
        tf.raw_ops.ResourceApplyProximalGradientDescent(
            var=resource.handle,
            alpha=tf.reshape(alpha, ()),
            l1=tf.reshape(l1, ()),
            l2=tf.reshape(l2, ()),
            delta=delta,
            use_locking=use_locking,
        )
        return resource.read_value()


class ApplyProximalGradientDescentTestSpec:
    """返回更新后的 var；原地别名由调用方的 output_inplace_indexes 绑定。"""

    @staticmethod
    def golden(var, alpha, l1, l2, delta, *, use_locking=False, **kwargs):
        inputs = tuple(np.asarray(x) for x in (var, alpha, l1, l2, delta))
        dtype = inputs[0].dtype
        if dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
            raise TypeError("目标仅支持 float16/float32；float64 仅供 Promote 参考")
        if any(x.dtype != dtype for x in inputs):
            raise ValueError("所有输入必须具有相同 dtype")
        if inputs[0].shape != inputs[4].shape:
            raise ValueError("var 和 delta 的 shape 必须完全相同，不支持广播")
        for name, scalar in zip(("alpha", "l1", "l2"), inputs[1:4]):
            if scalar.shape not in ((), (1,)):
                raise ValueError(f"{name} 必须为 0-D 或 shape=[1] 的标量张量")

        # Prox只输出var；原生TTK传入该输出的声明dtype。必须在内部提升前取值，
        # 不能从TF计算结果的dtype推断目标；无metadata的直接调用沿用var契约。
        output_dtypes = kwargs.get("output_dtypes")
        if output_dtypes is None:
            output_dtype = dtype
        else:
            if not isinstance(output_dtypes, (tuple, list)) or len(output_dtypes) != 1:
                raise ValueError("Prox要求恰好一个声明输出dtype")
            output_dtype = np.dtype(output_dtypes[0])
        if output_dtype not in (
            np.dtype("float16"),
            np.dtype("float32"),
            np.dtype("float64"),
        ):
            raise TypeError("最终输出仅支持float16/float32；float64仅供独立诊断")

        # 计算接口和精度保持：实际FP16提升FP32，实际FP32/FP64不变。
        if dtype == np.dtype("float16"):
            inputs = tuple(x.astype(np.float32) for x in inputs)

        import tensorflow as tf

        # 私有CPU输入副本；实际dtype决定精度，原语执行异常直接向调用方传播。
        with tf.device("/CPU:0"):
            tensors = [
                tf.convert_to_tensor(np.array(x, copy=True, order="C")) for x in inputs
            ]
            result = _tf_proximal_gradient_descent_compute(
                *tensors, use_locking=use_locking
            )
            # 唯一最终cast边界；有限舍入、溢出和NaN均由cast自然完成。
            return [result.numpy().astype(output_dtype, copy=False)]

    class ThirdPartyImpl:
        def __init__(self, use_locking=False, **kwargs):
            self.use_locking = use_locking

        def __call__(self, var, alpha, l1, l2, delta, **kwargs):
            # 直接使用原 dtype 的远端 Tensor；独立执行，禁止回调 Golden。
            return [_proximal_gradient_descent_compute(var, alpha, l1, l2, delta)]

    third_party = {"torch": ThirdPartyImpl}
    # cross_check 自动使用 Promote Golden；Torch provider 独立执行原 dtype 输入。
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
