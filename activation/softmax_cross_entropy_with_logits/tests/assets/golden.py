#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""SoftmaxCrossEntropyWithLogits 的 TTK 多路径 TestSpec golden。

运行时依赖：加载此插件的 runner 和第三方参考进程均需安装 PyTorch；CPU 版即可运行 CPU 参考。

数学定义（op_api/aclnn_softmax_cross_entropy_with_logits.h、op_host def/infershape）：
    loss     = -Σ_i (y_i * log_softmax(x)_i)
    backprop = softmax(x)_i - y_i
其中 x=features、y=labels，最后一维为归约轴（2D 合同）。

广播域（infershape 白名单）：rank 组合 (2,2)/(2,1)/(1,2)/(4,4)，逐维一方为 1 可广播。
输出形状（infershape 契约）：
    2D: loss [d0], backprop [d0, d1]（广播后形状）
    4D: loss [d0, 1, d2, d3]（dim1 归约、置 1 保秩）, backprop [d0, d1, d2, d3]
    4D：本参考沿 dim1 计算，与 InferShape 的 loss 形状一致；arch35 非广播路径沿末轴计算。
    标准 ACLNN 仅接受同形的 2D 输入，不支持广播，不能用于裁定 4D 语义。在算子契约统一前，不将任一轴作为验收依据。

dtype 合同：FP16/BF16/FP32。Kernel/GEIR 的 CPU 参考使用 FP64，以避免长归约中的抵消误差；
返回请求的参考 dtype。ACLNN 遵循 TTK 的输入精度提升，独立校验设备输出 dtype。

通路注册（GOLDEN.01/1、GOLDEN.13/1、GOLDEN.16/1 整改）：
    Kernel/GEIR : softmax_cross_entropy_with_logits（numpy 入/出；GEIR 复用）
    ACLNN       : aclnnSoftmaxCrossEntropyWithLogits（torch 入/出，双输出）
    E2E         : tf.raw_ops.SoftmaxCrossEntropyWithLogits（TF 静态图，需 npu_device）
第三方参考（独立 Torch/TF 组合，不调用 DUT）：
    Kernel/GEIR : torch 组合 + tf 组合（gen_nn_ops，eager 直调、无全局 eager 开关）
    ACLNN / E2E : torch 组合
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _BF16
except ImportError:  # pragma: no cover
    _BF16 = None

__spec__ = {
    "softmax_cross_entropy_with_logits": "ScewlKernelSpec",
    "aclnnSoftmaxCrossEntropyWithLogits": "ScewlAclnnSpec",
    "tf.raw_ops.SoftmaxCrossEntropyWithLogits": "ScewlTfGraphSpec",
}

# 旧式注册保留（兼容仅支持 __golden__ 的旧 TTK；新 TTK 中 TestSpec 优先）。
__golden__ = {
    "kernel": {
        "softmax_cross_entropy_with_logits": "softmax_cross_entropy_with_logits_golden"
    },
}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _np_to_torch(array):
    """numpy → torch CPU 张量（bf16 经 uint16 位视图无损转换）。"""
    array = np.ascontiguousarray(array)
    if _BF16 is not None and array.dtype == np.dtype(_BF16):
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_np(tensor, dtype_name):
    """torch → numpy，cast 到输出 dtype（bf16 经 uint16 位视图）。"""
    if tensor.dtype == torch.bfloat16:
        if _BF16 is None:
            raise RuntimeError("ml_dtypes is required for bfloat16 golden output")
        return tensor.contiguous().view(torch.uint16).cpu().numpy().view(_BF16)
    values = tensor.detach().cpu().numpy()
    if values.dtype == np.float64 and dtype_name == "bfloat16":
        # ml_dtypes casts via float32. If that cast lands on a BF16 midpoint,
        # preserve which side the FP64 value was on before the final ties-to-even cast.
        with np.errstate(over="ignore", invalid="ignore"):
            f32 = values.astype(np.float32)
            midpoint = (f32.view(np.uint32) & 0xFFFF) == 0x8000
            direction = np.where(values > f32, np.float32(np.inf), np.float32(-np.inf))
            np.nextafter(
                f32,
                direction,
                out=f32,
                where=midpoint & (values != f32) & np.isfinite(f32),
            )
            return f32.astype(dtype_name)
    return values.astype(dtype_name, copy=False)


def _output_dtype_names(kwargs, fallback_tensor, output_count):
    """逐输出 dtype 名（numpy 可接受）：优先 TTK 注入的 output_dtypes，否则回退输入 dtype。"""
    out_dtypes = kwargs.get("output_dtypes")
    names = []
    for idx in range(output_count):
        raw = None
        if out_dtypes is not None and idx < len(out_dtypes):
            raw = out_dtypes[idx]
        if raw is None:
            names.append(str(fallback_tensor.dtype).split(".")[-1])
            continue
        if isinstance(raw, torch.dtype):
            names.append(str(raw).split(".")[-1])
        else:
            names.append(str(raw).split(".")[-1] if "." in str(raw) else str(raw))
    return names


def _scewl_core(features, labels):
    """torch 稳定实现（max 平移）；输入可为低精度（调用方负责提升）。

    归约轴：4D 输入按 infershape 合同取 dim=1（loss 置 1 保秩），其余取末轴。
    广播：labels 与 features 逐维广播到公共形状（覆盖 (2,2)/(2,1)/(1,2)）。
    返回 (loss, backprop) torch 张量。
    """
    reduce_dim = 1 if features.dim() == 4 else -1
    bcast_shape = torch.broadcast_shapes(features.shape, labels.shape)
    x = features.expand(bcast_shape)
    y = labels.expand(bcast_shape)

    if x.shape[reduce_dim] == 0:
        return x.sum(dim=reduce_dim, keepdim=x.dim() == 4), x - y
    x_max = x.amax(dim=reduce_dim, keepdim=True)
    x_sub = x - x_max
    exp_x = x_sub.exp()
    sum_exp = exp_x.sum(dim=reduce_dim, keepdim=True)
    softmax = exp_x / sum_exp
    log_softmax = x_sub - sum_exp.log()
    loss = -(y * log_softmax).sum(dim=reduce_dim)
    backprop = softmax - y
    if features.dim() == 4:
        loss = loss.unsqueeze(1)  # [d0,d2,d3] → [d0,1,d2,d3]（infershape 契约）
    return loss, backprop


def _promote_for_compute(tensor):
    """计算 dtype：低精度提升 FP32；fp64 保持（TTK Promote 语义防御性对齐）。"""
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(torch.float32)
    return tensor


# ---- 独立第三方参考（torch / tf 组合；不调用 DUT） ------------------------------


class _ScewlTorchCompose:
    """torch 组合参考：与 golden 相同数学定义的独立包装，按输入 dtype 原生计算。"""

    def __init__(self, **kwargs):
        pass  # 本算子无属性

    def __call__(self, *args, **kwargs):
        features = args[0] if args else kwargs.get("features")
        labels = args[1] if len(args) > 1 else kwargs.get("labels")
        loss, backprop = _scewl_core_native(features, labels)
        return [loss, backprop]


def _scewl_core_native(features, labels):
    """torch 组合独立计算路径（F.softmax/F.log_softmax + 归约），保留输入 dtype。

    log_softmax 到 loss 的乘加也必须在计算精度进行；提前转回 FP16 可导致 0 * -inf。
    """
    reduce_dim = 1 if features.dim() == 4 else -1
    bcast_shape = torch.broadcast_shapes(features.shape, labels.shape)
    x = _promote_for_compute(features).expand(bcast_shape)
    y = _promote_for_compute(labels).expand(bcast_shape)
    softmax = torch.nn.functional.softmax(x, dim=reduce_dim)
    log_softmax = torch.nn.functional.log_softmax(x, dim=reduce_dim)
    loss = -(y * log_softmax).sum(dim=reduce_dim)
    backprop = softmax - y
    if features.dim() == 4:
        loss = loss.unsqueeze(1)
    return loss.to(features.dtype), backprop.to(features.dtype)


class _ScewlTfCompose:
    """TF 组合参考（Kernel/GEIR 通路交叉验证）：gen_nn_ops eager 直调。

    gen_nn_ops.softmax_cross_entropy_with_logits 仅接受 2D（或可广播为 2D）输入；
    广播场景由 TF 原生处理（TF 2.19 实测）。rank>2 抛错（无 4D 用例，诚实暴露）。
    无 tf.compat.v1 session / disable_eager_execution 全局副作用。

    输入桥接（按实际传入类型分支，兼容 TTK remote XPU 的 tf provider 与本地直调）：
    - tf.Tensor：server 端 tf provider 经 tf.convert_to_tensor 构造（TTK commit
      202675b executor._to_vendor_tensor 的 tf 分支），直接以 TF 原生接口计算，
      不调用 torch 的 .detach()/.dim()；bf16 输入先 tf.cast 到 fp32（gen_nn_ops 的
      CPU bf16 路径以 fp32 计算语义对齐），输出 cast 回原 dtype。
    - torch.Tensor：本地直调场景；bf16 无 numpy 视图，先 .float() 再 .numpy()。
    - numpy.ndarray：直接使用。
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        import tensorflow as tf
        from tensorflow.python.ops import gen_nn_ops

        def compute(logits, targets):
            # The TF CPU primitive can abort on zero classes, before Python can catch it.
            if 0 in logits.shape or 0 in targets.shape:
                shape = np.broadcast_shapes(logits.shape, targets.shape)
                if len(shape) != 2:
                    raise ValueError(
                        "TF Softmax reference requires a rank-2 broadcast result"
                    )
                logits = tf.broadcast_to(logits, shape)
                return tf.reduce_sum(logits, axis=-1), logits - targets
            return gen_nn_ops.softmax_cross_entropy_with_logits(logits, targets)

        features = args[0] if args else kwargs.get("features")
        labels = args[1] if len(args) > 1 else kwargs.get("labels")

        if isinstance(features, tf.Tensor):
            if len(features.shape) > 2 or len(labels.shape) > 2:
                raise NotImplementedError(
                    "tf third_party for softmax_cross_entropy_with_logits supports rank<=2 "
                    "(gen_nn_ops contract); 4D contract pending device confirmation"
                )
            in_dtype = features.dtype
            f_t = tf.cast(features, tf.float32) if in_dtype != tf.float32 else features
            l_t = tf.cast(labels, tf.float32) if labels.dtype != tf.float32 else labels
            loss, backprop = compute(f_t, l_t)
            if in_dtype != tf.float32:
                loss = tf.cast(loss, in_dtype)
                backprop = tf.cast(backprop, in_dtype)
            return [loss, backprop]

        if isinstance(features, torch.Tensor):
            out_dtype = features.dtype
            f_np = (
                features.detach().cpu().float().numpy()
                if features.dtype in (torch.float16, torch.bfloat16)
                else features.detach().cpu().numpy()
            )
            l_np = (
                labels.detach().cpu().float().numpy()
                if labels.dtype in (torch.float16, torch.bfloat16)
                else labels.detach().cpu().numpy()
            )
            if features.dim() > 2 or labels.dim() > 2:
                raise NotImplementedError(
                    "tf third_party for softmax_cross_entropy_with_logits supports rank<=2"
                )
            loss, backprop = compute(f_np, l_np)
            # torch 输入路径：返回 torch 张量并 cast 回原 dtype（低精度经 torch cast，
            # 不经过 numpy bf16 视图）
            loss_t = torch.from_numpy(np.asarray(loss)).to(out_dtype)
            bp_t = torch.from_numpy(np.asarray(backprop)).to(out_dtype)
            return [loss_t, bp_t]

        # numpy 路径（本地直调）：低精度（fp16 / ml_dtypes bf16）→ fp32 计算，输出 cast 回
        f_np, l_np = np.asarray(features), np.asarray(labels)
        low_prec = f_np.dtype == np.float16 or "bfloat16" in str(f_np.dtype)
        if low_prec:
            f_np = f_np.astype("float32")
            l_np = l_np.astype("float32")
        loss, backprop = compute(f_np, l_np)
        if low_prec:
            out_torch_dtype = (
                torch.float16
                if np.asarray(features).dtype == np.float16
                else torch.bfloat16
            )
            loss_t = torch.from_numpy(np.asarray(loss)).to(out_torch_dtype)
            bp_t = torch.from_numpy(np.asarray(backprop)).to(out_torch_dtype)
            return [loss_t, bp_t]
        return [np.asarray(loss), np.asarray(backprop)]


# ---- Kernel / GEIR（numpy.ndarray） ----------------------------------------------


def softmax_cross_entropy_with_logits_golden(features, labels, **kwargs):
    """Kernel/GEIR 旧式 golden：numpy 入/出，双输出 [loss, backprop]。"""
    f_t = _np_to_torch(features)
    l_t = _np_to_torch(labels)
    loss, backprop = _scewl_core(f_t.double(), l_t.double())
    loss_name, bp_name = _output_dtype_names(kwargs, f_t, 2)
    return [_torch_to_np(loss, loss_name), _torch_to_np(backprop, bp_name)]


class ScewlKernelSpec:
    """Kernel/GEIR 流程 TestSpec（numpy 入/出）。"""

    @staticmethod
    def golden(features, labels, **kwargs):
        return softmax_cross_entropy_with_logits_golden(features, labels, **kwargs)

    @staticmethod
    def pre_compare(loss, backprop, expected_loss, expected_backprop):
        if (
            loss.shape != expected_loss.shape
            or backprop.shape != expected_backprop.shape
        ):
            raise ValueError("Softmax loss/backprop shape or order mismatch")

    third_party = {
        "torch": _ScewlTorchCompose,
        "tf": _ScewlTfCompose,
    }
    tolerance = dict(_TOL)


class ScewlTfGraphSpec:
    """TF 图输出顺序与 Kernel 一致；不把 TTK 的 CPU 回退计作 arch35 ST。"""

    @staticmethod
    def golden(features, labels, *, backend):
        if backend != "npu":
            raise RuntimeError("Softmax TF graph ST requires the npu_device backend")
        return softmax_cross_entropy_with_logits_golden(features, labels)

    pre_compare = staticmethod(ScewlKernelSpec.pre_compare)
    third_party = {"torch": _ScewlTorchCompose}
    tolerance = dict(_TOL)


# ---- ACLNN : aclnnSoftmaxCrossEntropyWithLogits(features, labels, loss, backprop) ----


class ScewlAclnnSpec:
    """ACLNN 流程 TestSpec（torch 入/出，双输出）。

    参数跟随 aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize（不含 workspaceSize/executor）；
    loss/backprop 为设备输出缓冲，不决定 Promote 模式下参考值的精度。
    """

    @staticmethod
    def golden(features, labels, loss=None, backprop=None, **kwargs):
        g_loss, g_backprop = _scewl_core(
            _promote_for_compute(features), _promote_for_compute(labels)
        )
        return [g_loss.to(features.dtype), g_backprop.to(features.dtype)]

    pre_compare = staticmethod(ScewlKernelSpec.pre_compare)
    third_party = {
        "torch": _ScewlTorchCompose,
    }
    tolerance = dict(_TOL)
