#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""InplaceApplyAdagradV2 golden — Adagrad V2 优化器单步 inplace 更新参考实现。

公式:
    accum_new = (accum + grad ** 2) if update_slots else accum
    out_accum = accum_new
    out_var   = var - lr * grad / (sqrt(accum_new) + epsilon)

参考实现(oracle): TensorFlow tf.raw_ops.ResourceApplyAdagradV2。计算核在
进程内实调该算子(TF eager mode), 与被测 kernel 数值独立;
update_slots 两分支均显式传参比对。

"""

import contextlib

import numpy as np


# TestSpec 注册表: 查找名 → 类名
__spec__ = {
    "inplace_apply_adagrad_v2": "InplaceApplyAdagradV2Spec",
    "InplaceApplyAdagradV2": "InplaceApplyAdagradV2Spec",  # GE 类型名变体
    "tf.raw_ops.ApplyAdagradV2": "InplaceApplyAdagradV2TfSpec",  # TF 1.15 限定（V1 Ref 接口）
    "tf.raw_ops.ResourceApplyAdagradV2": "InplaceApplyAdagradV2TfSpec",  # TF 2.x 限定（Resource 接口）
}


# ============================================================================
# 辅助函数
# ============================================================================


def _to_np(x):
    """统一转 numpy array(兼容 torch.Tensor / tf.EagerTensor / numpy array / Python 标量)"""
    if hasattr(x, "detach"):
        # torch.Tensor → numpy(bfloat16 输入超出本算子 dtype 域, 不作处理)
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


# ============================================================================
# TF 计算核(进程内 lazy import 直调, 库内 golden 惯例形态)
# ============================================================================


def _compute_tf(var, accum, grad, lr, epsilon, update_slots):
    """TF ResourceApplyAdagradV2 eager mode 进程内直调。

    Args:
        var: numpy array, 模型参数
        accum: numpy array, 梯度平方累加器
        grad: numpy array, 当前梯度
        lr: float, 学习率
        epsilon: float, 数值稳定性小常数
        update_slots: bool, 是否更新 accum

    Returns:
        (var_out, accum_out): numpy arrays(输入 dtype 恒等透传, 零 cast 零 RNE)
    """
    import tensorflow as tf  # lazy import(见模块 docstring 导入纪律)

    # numpy dtype → TF 计算精度映射(恒等映射, 纯函数语义, 见模块 docstring):
    #   float32 → tf.float32(TF 原生 fp32 链)
    #   float64 → tf.float64(真值)
    _np_to_tf_dtype = {
        np.dtype(np.float32): tf.float32,
        np.dtype(np.float64): tf.float64,
    }

    tf_dtype = _np_to_tf_dtype[var.dtype]

    # rank-0 标量 → reshape 为 (1,), TF Variable 不支持 scalar shape
    orig_shape = var.shape
    is_scalar = len(orig_shape) == 0
    if is_scalar:
        var = var.reshape(1)
        accum = accum.reshape(1)
        grad = grad.reshape(1)

    var_tf = tf.Variable(tf.constant(var, dtype=tf_dtype))
    accum_tf = tf.Variable(tf.constant(accum, dtype=tf_dtype))

    tf.raw_ops.ResourceApplyAdagradV2(
        var=var_tf.handle,
        accum=accum_tf.handle,
        lr=tf.constant(float(lr), dtype=tf_dtype),
        grad=tf.constant(grad, dtype=tf_dtype),
        epsilon=tf.constant(float(epsilon), dtype=tf_dtype),
        update_slots=bool(update_slots),
        use_locking=False,
    )

    out_var = var_tf.numpy()
    out_accum = accum_tf.numpy()

    if is_scalar:
        out_var = out_var.reshape(orig_shape)
        out_accum = out_accum.reshape(orig_shape)

    del var_tf, accum_tf
    return (out_var, out_accum)


class _TfThirdParty:
    """TensorFlow XPU third-party implementation with native dual outputs."""

    def __call__(
        self,
        var,
        accum,
        lr,
        epsilon,
        grad,
        update_slots=True,
        use_locking=False,
        device=None,
        **kwargs,
    ):
        import tensorflow as tf

        del kwargs
        var_tf = tf.Variable(var)
        accum_tf = tf.Variable(accum)
        device_ctx = (
            tf.device(device) if device is not None else contextlib.nullcontext()
        )
        with device_ctx:
            tf.raw_ops.ResourceApplyAdagradV2(
                var=var_tf.handle,
                accum=accum_tf.handle,
                lr=tf.reshape(lr, []),
                epsilon=tf.reshape(epsilon, []),
                grad=grad,
                update_slots=bool(update_slots),
                use_locking=bool(use_locking),
            )
        return [var_tf.read_value(), accum_tf.read_value()]


# ============================================================================
# 主入口
# ============================================================================


def _inplace_apply_adagrad_v2_golden(*args, **kwargs):
    """InplaceApplyAdagradV2 golden。

    kernel/geir 模式: args = (var, accum, lr, epsilon, grad), 5 个 numpy
    数组按位置传入; update_slots 从 kwargs 取。
    use_locking 不消费(本算子仅支持 false, TF 侧固定传 false)。

    路由策略(零拦截, 无静默回退):
      任意输入按原生 dtype 直透 TF 计算核, golden 不设 dtype 白名单拦截
      (无意义拦截层禁, I-0326 —— 报错职能交框架原生裁决): bfloat16 在
      torch→numpy 转换层即原生报错, fp16 等其余 dtype 由计算核 numpy→TF
      dtype 映射表层原生裁决; TF 导入失败/计算异常显式上抛, 不做任何
      静默降级(ImportError/异常驱动回退属违禁形态)

    Returns:
        list of 2 numpy arrays: [var_out, accum_out](输入 dtype 恒等透传交付)
    """
    # 统一转 numpy(兼容 torch.Tensor / numpy / Python scalar)
    var = _to_np(args[0])
    accum = _to_np(args[1])
    lr_tensor = _to_np(args[2])
    epsilon_tensor = _to_np(args[3])
    grad = _to_np(args[4])

    # 提取标量 lr / epsilon(第 4 输入单元素 tensor, 镜像 lr 的 flat[0] 读取)
    lr = float(lr_tensor.flat[0]) if lr_tensor.size > 0 else 0.0
    epsilon = float(epsilon_tensor.flat[0]) if epsilon_tensor.size > 0 else 0.0

    # 提取 attributes(kernel/geir 模式从 kwargs 取)
    update_slots = kwargs.get("update_slots", kwargs.get("updateSlots", True))

    # lr/epsilon 按算子 FP32 参数契约量化，与框架无关。
    lr = float(np.float32(lr))
    epsilon = float(np.float32(epsilon))

    # TF 主路径: 计算核异常(导入失败/计算失败)显式上抛, 不做静默回退。
    out_var, out_accum = _compute_tf(
        var.copy(), accum.copy(), grad, lr, epsilon, update_slots
    )

    # 输入 dtype 恒等透传交付: TF 按到达 dtype 原生计算, 返回值已是
    # 输入 dtype, 零 cast 零还原(测试框架下发的 golden_mode 等 kwarg 由
    # **kwargs 吸收, 不消费)
    return [out_var, out_accum]


def tensorflow_apply_adagrad_v2_golden(
    var,
    accum,
    lr,
    epsilon,
    grad,
    use_locking=False,
    update_slots=True,
    name=None,
    **kwargs,
):
    """TF API 参数序入口(var, accum, lr, epsilon, grad)。

    按 tf.raw_ops.ApplyAdagradV2 的形参序接受输入, 复用同一计算核
    (实调 tf.raw_ops.ResourceApplyAdagradV2, 两算子数值语义一致);
    name/use_locking 不参与计算。
    """
    del name, use_locking
    return _inplace_apply_adagrad_v2_golden(
        var, accum, lr, epsilon, grad, update_slots=update_slots, **kwargs
    )


class InplaceApplyAdagradV2Spec:
    """算子原生参数序(var, accum, lr, epsilon, grad)的测试规范。"""

    golden = staticmethod(_inplace_apply_adagrad_v2_golden)
    third_party = {"tf": _TfThirdParty}
    tolerance = {"float32": {"standard": "cross_check", "level": "L1"}}


class InplaceApplyAdagradV2TfSpec:
    """TF API 参数序(var, accum, lr, epsilon, grad)的测试规范。

    经 __spec__ 的 tf.raw_ops.ApplyAdagradV2 / tf.raw_ops.ResourceApplyAdagradV2
    两键暴露(TF e2e 按 api_name 查 golden; 950 插件装载期 SoC 探测路由后
    TF 节点映射至本仓 InplaceApplyAdagradV2)。
    """

    golden = staticmethod(tensorflow_apply_adagrad_v2_golden)
    third_party = {"tf": _TfThirdParty}
    tolerance = {"float32": {"standard": "cross_check", "level": "L1"}}
