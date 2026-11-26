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
"""BNInferGrad 独立 Golden：Kernel、GEIR。"""

import torch


__spec__ = {
    # kernel / GEIR：CSV op_name 查找
    "bn_infer_grad": "BNInferGradKernelSpec",
    "BNInferGrad": "BNInferGradKernelSpec",
    # aclnn 通路：Dashboard 将 BNInferGrad 映射到 BatchNorm 反向的两个公开入口
    "aclnnBatchNormBackward": "AclnnBNInferGradSpec",
    "aclnnFastBatchNormBackward": "AclnnBNInferGradSpec",
    # torch 单算子 / 图模式：spec.yaml 声明的参考入口 native_batch_norm_backward(train=False)
    "torch.ops.aten.native_batch_norm_backward": "TorchBNInferGradE2ESpec",
    "torch.ops.aten.native_batch_norm_backward.default": "TorchBNInferGradE2ESpec",
    "torch.native_batch_norm_backward": "TorchBNInferGradE2ESpec",
    # tf 通路：TF plugin 以 BNInferGrad 为 OriginOpType；等价入口为 FusedBatchNormGradV3(is_training=False)
    "tf.raw_ops.FusedBatchNormGradV3": "TfBNInferGradE2ESpec",
    "tf.raw_ops.BNInferGrad": "TfBNInferGradE2ESpec",
}


_L0 = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
}


def _compute(grads, scale, batch_variance, epsilon, format_x):
    is_torch = isinstance(grads, torch.Tensor)
    if not is_torch:
        from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

        # TF e2e 通路下 golden 收到的是 tensorflow EagerTensor（框架原生张量，
        # needs_numpy_fallback=False）。先统一归一化为 numpy 再转 torch。
        grads = grads.numpy() if hasattr(grads, "numpy") else grads
        scale = scale.numpy() if hasattr(scale, "numpy") else scale
        batch_variance = (
            batch_variance.numpy()
            if hasattr(batch_variance, "numpy")
            else batch_variance
        )
        grads = numpy_to_torch_tensor(grads)
        scale = numpy_to_torch_tensor(scale)
        batch_variance = numpy_to_torch_tensor(batch_variance)
    shape_x = grads.shape
    output_dtype = grads.dtype

    if grads.numel() == 0 or scale.numel() == 0 or batch_variance.numel() == 0:
        result = torch.zeros(grads.shape, dtype=output_dtype, device=grads.device)
        return result if is_torch else torch_to_numpy_tensor(result.detach().cpu())

    if len(scale.shape) == 1 and format_x == "NCHW":
        channel_shape = (1, scale.shape[0], *((1,) * (grads.ndim - 2)))
        scale = scale.reshape(channel_shape)
        batch_variance = batch_variance.reshape(channel_shape)

    # 参考实现在 FP64 中计算后再回写输出 dtype：输入取到 ±FLT_MAX 时
    # grads*scale / scale*rsqrt 任一结合顺序在 FP32 中间域都会溢出为 inf，
    # 而真实结果可能仍是有限值（如 scale=-FLT_MAX、variance=FLT_MAX 时
    # 结果为 grads*1.8e19）。FP64 中间域无此溢出，作为参考口径更准确；
    # 输出仍按 dtype 舍入，常规值域与 FP32 计算逐位一致。
    grads_c = grads.to(torch.float64) if grads.is_floating_point() else grads
    scale_c = scale.to(torch.float64) if scale.is_floating_point() else scale
    var_c = (
        batch_variance.to(torch.float64)
        if batch_variance.is_floating_point()
        else batch_variance
    )

    data_add = torch.add(var_c, epsilon)
    data_abs = torch.abs(data_add)
    data_sqrt = torch.sqrt(data_abs)
    # 文档公式：grads * scale / sqrt(|variance + epsilon|)
    scale_broadcast = torch.broadcast_to(scale_c, shape_x)
    result = torch.divide(torch.multiply(grads_c, scale_broadcast), data_sqrt)
    result = result.to(output_dtype)
    return result if is_torch else torch_to_numpy_tensor(result.detach().cpu())


def bn_infer_grad_golden(grads, scale, batch_variance, epsilon=0.0001, **kwargs):
    input_formats = kwargs.get("input_formats") or ()
    format_x = str(input_formats[0]).upper() if input_formats else "NCHW"
    return [_compute(grads, scale, batch_variance, epsilon, format_x)]


class _TorchBNInferGrad:
    """Torch competitor used by three-party precision and performance runs."""

    def __call__(instance, grads, scale, batch_variance, epsilon=0.0001, **kwargs):
        input_formats = kwargs.get("input_formats") or ()
        format_x = str(input_formats[0]).upper() if input_formats else "NCHW"
        return [_compute(grads, scale, batch_variance, epsilon, format_x)]


def _as_torch_tensor(value):
    """torch.Tensor / TF EagerTensor / numpy → torch.Tensor。"""
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "numpy"):
        value = value.numpy()
    from ttk.utilities.dtypes import numpy_to_torch_tensor

    return numpy_to_torch_tensor(value)


def _bn_grad_full_outputs(y_backprop, x, scale, mean, variance, epsilon, format_x):
    """TF FusedBatchNormGradV3(is_training=False) 的 5 路参考输出。

    口径与原生 TF 一致：
      inv_std    = 1 / sqrt(|variance + epsilon|)
      x_backprop = y_backprop * scale * inv_std
      scale_bp   = sum_spatial(y_backprop * (x - mean) * inv_std)
      offset_bp  = sum_spatial(y_backprop)
    reserve_space_4/5 对本算子无意义，返回 shape(0,) 空张量以对齐输出路数。
    """
    g = _as_torch_tensor(y_backprop)
    x_t = _as_torch_tensor(x)
    s = _as_torch_tensor(scale)
    m = _as_torch_tensor(mean)
    v = _as_torch_tensor(variance)
    ndim = g.ndim
    axis = 1 if str(format_x).upper() == "NCHW" else ndim - 1

    def _per_channel(t):
        shape = [1] * ndim
        shape[axis] = -1
        return t.reshape(shape)

    g64 = g.to(torch.float64) if g.is_floating_point() else g
    x64 = x_t.to(torch.float64) if x_t.is_floating_point() else x_t
    s64 = s.to(torch.float64) if s.is_floating_point() else s
    m64 = m.to(torch.float64) if m.is_floating_point() else m
    v64 = v.to(torch.float64) if v.is_floating_point() else v

    inv_std = 1.0 / torch.sqrt(torch.abs(v64 + float(epsilon)))
    inv_std_b = _per_channel(inv_std)
    scale_b = _per_channel(s64)
    mean_b = _per_channel(m64)

    dx = (g64 * scale_b * inv_std_b).to(g.dtype)
    x_hat = (x64 - mean_b) * inv_std_b
    reduce_axes = tuple(i for i in range(ndim) if i != axis)
    dscale = torch.sum(g64 * x_hat, dim=reduce_axes).to(torch.float32)
    doffset = torch.sum(g64, dim=reduce_axes).to(torch.float32)
    empty = torch.empty(0, dtype=torch.float32, device=g.device)
    return [dx, dscale, doffset, empty, empty]


class _TfBNInferGrad:
    """TF 三方竞争类：入参名与 FusedBatchNormGradV3 对齐，返回 5 路输出。"""

    def __call__(
        instance,
        y_backprop=None,
        x=None,
        scale=None,
        reserve_space_1=None,
        reserve_space_2=None,
        reserve_space_3=None,
        epsilon=None,
        data_format=None,
        is_training=False,
        **kwargs,
    ):
        attrs = kwargs.get("attributes")
        attrs = attrs if isinstance(attrs, dict) else {}
        if y_backprop is None:
            y_backprop = kwargs.get("grads")
        if x is None:
            x = kwargs.get("x", kwargs.get("input"))
        if scale is None:
            scale = kwargs.get("weight")
        if reserve_space_1 is None:
            reserve_space_1 = kwargs.get("running_mean")
        if reserve_space_2 is None:
            reserve_space_2 = kwargs.get("batch_variance", kwargs.get("running_var"))
        # 解析顺序：显式入参 -> 扁平 kwargs -> attributes 嵌套 -> 内置默认值。
        # 形参默认值必须为 None，否则 attributes 分支恒不可达（属性形如 data_format/epsilon 会被忽略）。
        if data_format is None:
            data_format = kwargs.get("data_format")
            if data_format is None:
                data_format = attrs.get("data_format", "NHWC")
        if epsilon is None:
            epsilon = _resolve_epsilon(kwargs, default=0.0001)
        return _bn_grad_full_outputs(
            y_backprop,
            x,
            scale,
            reserve_space_1,
            reserve_space_2,
            float(epsilon),
            str(data_format),
        )


class BNInferGradKernelSpec:
    golden = staticmethod(bn_infer_grad_golden)
    third_party = {"torch": _TorchBNInferGrad}
    tolerance = _L0


# ─── 缺失通路（aclnn / torch / tf）的 golden 补齐 ──────────────────────
# 语义统一按 BNInferGrad 推理反向公式：
#   x_backprop = grads * scale / sqrt(|batch_variance + epsilon|)
# 各通路只是入口签名不同，计算核心复用 _compute。


def _resolve_epsilon(kwargs, default=0.0001):
    """epsilon/eps 兼容 CSV 属性、扁平 kwargs 与 attributes 嵌套三种下发形态。"""
    for name in ("epsilon", "eps"):
        value = kwargs.get(name)
        if value is not None:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            return float(value)
    attrs = kwargs.get("attributes")
    if isinstance(attrs, dict):
        for name in ("epsilon", "eps"):
            value = attrs.get(name)
            if value is not None:
                if isinstance(value, (list, tuple)) and value:
                    value = value[0]
                return float(value)
    return default


def _resolve_format(kwargs):
    """规整数据格式：aclnn 传 tensor_formats，kernel/tf 传 input_formats / data_format。

    与 _TfBNInferGrad 的 data_format 口径保持一致：扁平 kwargs 或 attributes 嵌套均可下发，
    统一剥离 "FORMAT_" 前缀；缺失时回退 NCHW。
    """
    fmts = kwargs.get("tensor_formats") or kwargs.get("input_formats") or ()
    if isinstance(fmts, str):
        fmts = (fmts,)
    try:
        if fmts:
            return str(fmts[0]).upper().replace("FORMAT_", "")
    except (TypeError, IndexError):
        pass
    data_format = kwargs.get("data_format")
    if data_format is None and isinstance(kwargs.get("attributes"), dict):
        data_format = kwargs["attributes"].get("data_format")
    if data_format:
        return str(data_format).upper().replace("FORMAT_", "")
    return "NCHW"


def _pick_scale_variance(args, kwargs):
    """从各通路的入参布局中取出 (scale, batch_variance)。

    - aclnnBatchNormBackward: grads, input, weight, running_mean, running_var, ...
    - native_batch_norm_backward: grad_out, input, weight, running_mean, running_var, ...
    - FusedBatchNormGradV3: y_backprop, x, scale, reserve_space_1, reserve_space_2, reserve_space_3
    - Kernel 兼容调用: grads, scale, batch_variance
    """
    scale = (
        kwargs.get("weight")
        if kwargs.get("weight") is not None
        else kwargs.get("scale")
    )
    variance = None
    for key in (
        "running_var",
        "batch_variance",
        "variance",
        "running_variance",
        "reserve_space_2",
    ):
        if kwargs.get(key) is not None:
            variance = kwargs[key]
            break

    if len(args) >= 4:
        # aclnn / native_batch_norm_backward / FusedBatchNormGradV3 统一布局：
        # (input|x, scale|weight, mean, variance, ...)
        if scale is None:
            scale = args[1]
        if variance is None:
            variance = args[3]
    elif len(args) == 2:
        # Kernel 兼容布局 (grads, scale, batch_variance)
        if scale is None:
            scale = args[0]
        if variance is None:
            variance = args[1]
    elif len(args) == 1 and scale is None:
        scale = args[0]
    return scale, variance


def _resolve_output_mask(kwargs):
    for name in ("output_mask", "outputMask"):
        value = kwargs.get(name)
        if value is None and isinstance(kwargs.get("attributes"), dict):
            value = kwargs["attributes"].get(name)
        if value is not None:
            if isinstance(value, str):
                import ast

                value = ast.literal_eval(value)
            return [bool(item) for item in value]
    return None


def _pathway_bn_infer_grad_golden(grads=None, *args, **kwargs):
    # TF e2e 按算子入参关键字名下发（y_backprop/x/scale/reserve_space_*），
    # 首个位置参数 grads 取不到，需从关键字入参兜底。
    if grads is None:
        for name in ("y_backprop", "grad_output", "grad_out"):
            if kwargs.get(name) is not None:
                grads = kwargs[name]
                break
    if grads is None:
        raise TypeError(
            "BNInferGrad golden: missing grads input "
            "(expected grads/y_backprop/grad_output/grad_out)"
        )
    scale, variance = _pick_scale_variance(args, kwargs)
    result = _compute(
        grads,
        scale,
        variance,
        _resolve_epsilon(kwargs),
        _resolve_format(kwargs),
    )
    # aclnn 全 10-tensor 布局：grads + (input, weight, mean, var, save_mean,
    # save_invstd, out_grad_input, out_grad_weight, out_grad_bias)。推理反向
    # outputMask=[True,False,False]，禁用的 grad_weight/grad_bias 直接透传
    # 调用方传入的输出缓冲，只有 grad_input 由 golden 改写。
    mask = _resolve_output_mask(kwargs)
    if len(args) >= 9 and mask is not None and len(mask) >= 3:
        if mask[0] and not mask[1] and not mask[2]:
            return [result, args[7], args[8]]
    return [result]


class AclnnBNInferGradSpec:
    """aclnn 通路别名：按推理反向口径给出 x_backprop（单输出）。"""

    golden = staticmethod(_pathway_bn_infer_grad_golden)
    third_party = {"torch": _TorchBNInferGrad}
    tolerance = _L0


class TorchBNInferGradE2ESpec:
    """torch 单算子 / 图模式：native_batch_norm_backward(train=False) 的 grad_input。"""

    golden = staticmethod(_pathway_bn_infer_grad_golden)
    third_party = {"torch": _TorchBNInferGrad}
    tolerance = _L0


class TfBNInferGradE2ESpec:
    """TF 通路：FusedBatchNormGradV3(is_training=False) 的 x_backprop。

    golden 只比对 x_backprop；三方竞争类按原生 TF 5 路输出实现，保证
    cross_check 的 third_party 输出路数与 NPU 侧一致。
    """

    golden = staticmethod(_pathway_bn_infer_grad_golden)
    third_party = {"torch": _TfBNInferGrad}
    tolerance = _L0
