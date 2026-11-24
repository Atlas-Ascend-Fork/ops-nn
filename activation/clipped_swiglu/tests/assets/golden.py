#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""clipped_swiglu 的 TTK 多路径 TestSpec golden。

运行时依赖：加载此插件的 runner 和第三方参考进程均需安装 PyTorch；CPU 版即可运行 CPU 参考。

参考 docs/aclnnClippedSwiglu.md / aclnnClippedSwigluV2.md：
    1. 按 dim 合轴 -> [pre, cut]
    2. group = min(sum(group_index), pre); 仅前 group 行参与计算
    3. interleaved: A=x[:,::2], B=x[:,1::2]; else 前后: A=x[:,:h], B=x[:,h:]  (h=cut//2)
    4. clamp_mode 0: A=clamp(A,None,limit), B=clamp(B,-limit,limit), y=A*sigmoid(alpha*A)*(B+bias)
       clamp_mode 1: B=clamp(B,-limit,limit), y=clamp(A*sigmoid(A),None,limit)*B

本版整改（issue-176）：
    - 数学核心由 NumPy 迁移至设备感知的 torch 组合（golden 与 third_party 均不回落 NumPy 计算；
      NumPy 仅作 Kernel 通路的容器桥接）。
    - 部分分组时，ACLNN V1/V2 和对应的 arch35 Kernel/GEIR 只校验参与行；无分组和全量分组比较完整输出。
      V2 明确声明尾行未清理；V1 文档未规定尾值，忽略尾行是本测试的比较策略，并非清零保证。
      Torch E2E 仍比较完整输出。
    - group_index 输入契约化（customize_inputs）：元素非负、和 > 0 且 ≤ pre（dim=0 时 ≤ 1）；
      全零分组不再作为合法默认样例。
    - 各通路 third_party 为独立 torch/tf 组合，不调用被测 cann_ops_nn.clipped_swiglu / ACLNN 自身。

通路注册（保留既有注册，新增 torch.ops 4 段式别名便于 E2E schema 解析）：
    Kernel/GEIR : clipped_swiglu（GEIR 复用）
    ACLNN       : aclnnClippedSwiglu（V1，无 clampMode，固定 clamp 在 silu 前）
                  aclnnClippedSwigluV2（clampMode ∈ {0,1}）
    E2E         : cann_ops_nn.clipped_swiglu / torch.ops.cann_ops_nn.clipped_swiglu
"""

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _BF16
except ImportError:  # pragma: no cover
    _BF16 = None

__spec__ = {
    "clipped_swiglu": "ClippedSwigluKernelSpec",
    "aclnnClippedSwiglu": "AclnnClippedSwigluSpec",
    "aclnnClippedSwigluV2": "AclnnClippedSwigluV2Spec",
    "cann_ops_nn.clipped_swiglu": "ClippedSwigluE2ESpec",
    "torch.ops.cann_ops_nn.clipped_swiglu": "ClippedSwigluE2ESpec",
    # 算子仓构建产物的包名变体（TORCH_EXTENSION_OPS 选择性构建时 setup.py 加 _custom 后缀），
    # 与 CANN 安装包内旧版 cann_ops_nn 共存时用于 E2E 实际可执行入口。
    "cann_ops_nn_custom.clipped_swiglu": "ClippedSwigluE2ESpec",
}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_DEFAULT_ALPHA = 1.702
_DEFAULT_LIMIT = 7.0
_DEFAULT_BIAS = 1.0


def _prod(seq):
    out = 1
    for value in seq:
        out *= int(value)
    return out


def _np_to_torch(array):
    """numpy → torch CPU 张量（bf16 经 uint16 位视图无损转换）。"""
    array = np.ascontiguousarray(array)
    if _BF16 is not None and array.dtype == np.dtype(_BF16):
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    if array.dtype == np.uint16:
        return torch.from_numpy(array).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_np(tensor, dtype_name):
    """torch → numpy，cast 到输出 dtype（bf16 经 uint16 位视图）。"""
    if tensor.dtype == torch.bfloat16:
        if _BF16 is None:
            raise RuntimeError("ml_dtypes is required for bfloat16 golden output")
        return tensor.contiguous().view(torch.uint16).cpu().numpy().view(_BF16)
    return tensor.detach().cpu().numpy().astype(dtype_name, copy=False)


def _group_rows(group_index, pre):
    """有效行数：group_index 缺省时全量；否则 min(sum, pre)（防御 clamping）。"""
    if group_index is None:
        return pre
    if isinstance(group_index, torch.Tensor):
        total = int(group_index.detach().cpu().sum().item())
    else:
        total = int(np.asarray(group_index).sum())
    return max(1, min(total, pre))


def _group_index_to_numpy(group_index):
    """group_index 归一为 numpy int64（torch 张量 / numpy / 标量兼容）。"""
    if isinstance(group_index, torch.Tensor):
        return group_index.detach().cpu().to(torch.int64).numpy()
    return np.asarray(group_index).astype(np.int64)


def _normalize_group_index(group_index, pre):
    """契约化 group_index：展平、元素非负、和 > 0 且 ≤ pre；返回 numpy int64。

    空/标量输入先重建为 1 维再规整；全零分组**保持原一维长度**（仅首元素置
    min(1, pre)、其余为 0），供 in-place copy_ 回原 tensor 时不触发广播
    （copy_ 对长度不匹配的源会广播到全量，导致 sum 超过 pre 违反合同）。
    """
    gi = np.asarray(group_index).astype(np.int64).reshape(-1).copy()
    gi = np.clip(gi, 0, pre)
    total = int(gi.sum())
    if total > pre:
        # 超出 pre 的分组按比例压缩到合法上界（保持"部分参与"语义而非全量）
        gi = (gi * pre) // max(total, 1)
        if int(gi.sum()) == 0:
            gi[0] = min(1, pre)
    elif gi.size == 0:
        gi = np.array([min(1, pre)], dtype=np.int64)  # 空分组不合法：至少 1 行参与
    elif total == 0:
        # 全零分组不合法：首元素置 1，其余保持 0（长度不变，sum=min(1,pre) 合法）
        gi[0] = min(1, pre)
    return gi


def _resolve_axis(shape, dim):
    """dim 归一化（支持负值），并返回合轴 (pre, cut)。"""
    ndim = len(shape)
    dim_pos = int(dim) % ndim if ndim > 0 else 0
    pre = _prod(shape[:dim_pos]) if dim_pos > 0 else 1
    cut = _prod(shape[dim_pos:])
    if ndim > 0 and shape[dim_pos] % 2 != 0:
        raise ValueError(
            f"clipped_swiglu dim axis size must be even: shape={tuple(shape)}, dim={dim} "
            f"(axis {dim_pos} size {shape[dim_pos]} is odd; device tiling rejects it likewise)"
        )
    return dim_pos, pre, cut


def _clipped_swiglu_core(
    x, dim_pos, group_rows, alpha, limit, bias, interleaved, clamp_mode
):
    """设备感知 torch 数学核心：仅计算前 group_rows 参与行，返回 [group_rows, cut//2]。

    x 为 torch 张量（任意设备、任意受支持 dtype）；不触碰非参与行。
    """
    shape = tuple(x.shape)
    _, pre, cut = _resolve_axis(shape, dim_pos)
    xf = x.reshape(pre, cut)
    xt = xf[:group_rows]
    if interleaved:
        a = xt[:, 0::2]
        b = xt[:, 1::2]
    else:
        half = cut // 2
        a = xt[:, :half]
        b = xt[:, half:]
    if int(clamp_mode) == 0:
        a = torch.clamp(a, max=limit)
        b = torch.clamp(b, min=-limit, max=limit)
        y = a * torch.sigmoid(alpha * a) * (b + bias)
    elif int(clamp_mode) == 1:
        b = torch.clamp(b, min=-limit, max=limit)
        y = torch.clamp(a * torch.sigmoid(a), max=limit) * b
    else:
        raise ValueError(f"clamp_mode must be 0 or 1, got {clamp_mode}")
    return y


def _build_full_output(core_result, x_shape, dim_pos, group_rows, dtype):
    """组装完整输出：全 0 占位 + 写前 group_rows 行 + reshape 回 dim 减半形状。

    非参与行填零；是否比较这些行取决于通路，见模块说明。
    """
    _, pre, cut = _resolve_axis(x_shape, dim_pos)
    out = torch.zeros((pre, cut // 2), dtype=dtype, device=core_result.device)
    out[:group_rows] = core_result.to(dtype)
    out_shape = list(x_shape)
    out_shape[dim_pos] = out_shape[dim_pos] // 2
    return out.reshape(out_shape)


def _partial_reference(array, valid_elements):
    # dtype metadata survives TTK's ndarray.copy() without changing any output bits.
    return array.view(
        np.dtype(
            array.dtype, metadata={"clipped_swiglu_valid_elements": valid_elements}
        )
    )


def _pre_compare_mask_invalid_rows(actual, golden):
    # The boundary belongs to this reference, never to a worker or a shape.
    # Replayed/third-party results without metadata are compared in full.
    if isinstance(golden, np.ndarray):
        valid = (golden.dtype.metadata or {}).get("clipped_swiglu_valid_elements")
    else:
        valid = None
    if actual.shape != golden.shape:
        raise ValueError(
            f"ClippedSwiglu shape mismatch: {actual.shape} vs {golden.shape}"
        )
    if valid is None:
        return [actual, golden]
    # A contiguous copy also covers strided tensor views and Fortran-order arrays.
    masked = (
        actual.contiguous().clone()
        if torch.is_tensor(actual)
        else actual.copy(order="C")
    )
    masked.reshape(-1)[valid:] = 0
    return [masked, golden]


# ---- 独立第三方参考（torch / tf 组合；不调用 DUT） ------------------------------


class _SwigluTorchCompose:
    """torch 组合参考：相同数学定义的独立 torch 包装，按输入 dtype 原生计算。"""

    def __init__(
        self,
        *,
        dim=-1,
        alpha=_DEFAULT_ALPHA,
        limit=_DEFAULT_LIMIT,
        bias=_DEFAULT_BIAS,
        interleaved=True,
        clamp_mode=0,
        clampMode=None,
        **kwargs,
    ):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.limit = float(limit)
        self.bias = float(bias)
        self.interleaved = bool(interleaved)
        self.clamp_mode = int(clampMode if clampMode is not None else clamp_mode)

    def __call__(self, x, group_index=None, *, groupIndexOptional=None, **kwargs):
        # third_party 在 Kernel/GEIR 通路由框架传入 torch.Tensor；本地直调兼容 numpy
        if group_index is None:
            group_index = groupIndexOptional
        if not isinstance(x, torch.Tensor):
            x = _np_to_torch(x)
        shape = tuple(x.shape)
        dim_pos, pre, _ = _resolve_axis(shape, self.dim)
        rows = _group_rows(group_index, pre)
        core = _clipped_swiglu_core(
            x,
            dim_pos,
            rows,
            self.alpha,
            self.limit,
            self.bias,
            self.interleaved,
            self.clamp_mode,
        )
        return [_build_full_output(core, shape, dim_pos, rows, x.dtype)]


class _SwigluTfCompose:
    """TF 组合参考（Kernel 通路交叉验证）：tf.clip_by_value / tf.sigmoid / tf.multiply 组合。"""

    def __init__(
        self,
        *,
        dim=-1,
        alpha=_DEFAULT_ALPHA,
        limit=_DEFAULT_LIMIT,
        bias=_DEFAULT_BIAS,
        interleaved=True,
        clamp_mode=0,
        **kwargs,
    ):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.limit = float(limit)
        self.bias = float(bias)
        self.interleaved = bool(interleaved)
        self.clamp_mode = int(clamp_mode)

    def __call__(self, x, group_index=None, **kwargs):
        """TF 组合参考（tf provider 传 tf.Tensor / torch provider 路径与本地直调传 torch/numpy）。

        输入桥接（按实际类型分支）：
        - tf.Tensor：TTK remote XPU 的 tf provider 经 tf.convert_to_tensor 构造
          （executor._to_vendor_tensor 的 tf 分支），以 TF 原生接口计算；
        - torch.Tensor：本地直调 / torch 通路；bf16 无 numpy 视图，先 .float() 再 .numpy()；
        - numpy.ndarray：直接使用。
        低精度统一提升 fp32 进 TF，输出 cast 回原 dtype（保留"低精度入、高精度算、
        原 dtype 出"的算子语义）。
        """
        import tensorflow as tf

        is_tf_tensor = False
        if isinstance(x, torch.Tensor):
            out_dtype = x.dtype
            x_np = (
                x.detach().cpu().float()
                if x.dtype == torch.bfloat16
                else x.detach().cpu()
            ).numpy()
        elif str(type(x).__module__).startswith("tensorflow"):
            # tf.Tensor（server 端 tf provider）：直接以 TF 原生计算，不做 torch 桥接
            is_tf_tensor = True
            out_dtype = x.dtype
            x_np = None
        else:
            x_np = np.asarray(x)
            out_dtype = (
                torch.float16
                if x_np.dtype == np.float16
                else torch.bfloat16
                if "bfloat16" in str(x_np.dtype)
                else torch.float32
            )
            if out_dtype != torch.float32:
                x_np = x_np.astype("float32")
        shape = (
            tuple(x_np.shape) if x_np is not None else tuple(int(s) for s in x.shape)
        )
        dim_pos, pre, cut = _resolve_axis(shape, self.dim)
        gi_np = None if group_index is None else _group_index_to_numpy(group_index)
        rows = _group_rows(gi_np, pre)
        if is_tf_tensor:
            xt_all = tf.reshape(
                tf.cast(x, tf.float32) if x.dtype != tf.float32 else x, (pre, cut)
            )
        else:
            xt_all = tf.reshape(tf.convert_to_tensor(x_np), (pre, cut))
        xt = xt_all[:rows]
        if self.interleaved:
            a = xt[:, 0::2]
            b = xt[:, 1::2]
        else:
            half = cut // 2
            a = xt[:, :half]
            b = xt[:, half:]
        limit_t = tf.cast(self.limit, xt.dtype)
        bias_t = tf.cast(self.bias, xt.dtype)
        alpha_t = tf.cast(self.alpha, xt.dtype)
        if self.clamp_mode == 0:
            a = tf.clip_by_value(a, xt.dtype.min, limit_t)
            b = tf.clip_by_value(b, -limit_t, limit_t)
            y = a * tf.sigmoid(alpha_t * a) * (b + bias_t)
        else:
            b = tf.clip_by_value(b, -limit_t, limit_t)
            y = tf.clip_by_value(a * tf.sigmoid(a), xt.dtype.min, limit_t) * b
        out = tf.concat([y, tf.zeros((pre - rows, cut // 2), dtype=xt.dtype)], axis=0)
        out_shape = list(shape)
        out_shape[dim_pos] //= 2
        out = tf.reshape(out, out_shape)
        if is_tf_tensor:
            # server 端 tf provider：返回 tf.Tensor（executor._outputs_to_numpy 负责 wire 转换）
            return [tf.cast(out, out_dtype) if out_dtype != tf.float32 else out]
        # 本地 / torch 通路：torch.from_numpy 接收 fp32/fp16；bf16 经 torch cast（不经过 numpy bf16 视图）
        return [torch.from_numpy(out.numpy()).to(out_dtype)]


# ---- Kernel / GEIR（numpy.ndarray）----------------------------------------------


class ClippedSwigluKernelSpec:
    """Kernel / GEIR 流程 TestSpec（numpy 入/出；GEIR 复用 op_name=clipped_swiglu）。"""

    @staticmethod
    def golden(
        x,
        group_index=None,
        *,
        dim=-1,
        alpha=_DEFAULT_ALPHA,
        limit=_DEFAULT_LIMIT,
        bias=_DEFAULT_BIAS,
        interleaved=True,
        clamp_mode=0,
        **kwargs,
    ):
        x_t = _np_to_torch(x)
        gi_t = (
            None
            if group_index is None
            else (
                group_index
                if isinstance(group_index, torch.Tensor)
                else torch.as_tensor(np.asarray(group_index), dtype=torch.int64)
            )
        )
        shape = tuple(x_t.shape)
        dim_pos, pre, _ = _resolve_axis(shape, int(dim))
        rows = _group_rows(gi_t, pre)
        # 低精度提升 FP32 计算（kernel 同为高精度累加），输出回 cast 输出 dtype
        compute = (
            x_t.to(torch.float32)
            if x_t.dtype in (torch.float16, torch.bfloat16)
            else x_t
        )
        core = _clipped_swiglu_core(
            compute,
            dim_pos,
            rows,
            float(alpha),
            float(limit),
            float(bias),
            bool(interleaved),
            int(clamp_mode),
        )
        out_dtypes = kwargs.get("output_dtypes")
        if out_dtypes is not None and len(out_dtypes) > 0 and out_dtypes[0] is not None:
            target = str(out_dtypes[0]).split(".")[-1]
        else:
            target = str(x_t.dtype).split(".")[-1]
        out_t = _build_full_output(core, shape, dim_pos, rows, _torch_dtype_of(target))
        out_np = _torch_to_np(out_t, target)
        if rows < pre and kwargs.get("short_soc_version") in ("Ascend950", "Ascend350"):
            out_np = _partial_reference(out_np, core.numel())
        return [out_np]

    @staticmethod
    def customize_inputs(x, group_index=None, **kwargs):
        """group_index 契约化：元素非负、和 > 0、和 ≤ pre（dim=0 时 ≤ 1）；None 原样传递。

        空/标量输入先重建再规整（检视意见：空数组取 gi[0] 会 IndexError）。
        """
        if group_index is None:
            return (x, group_index)
        dim = int(kwargs.get("dim", -1))
        _, pre, _ = _resolve_axis(np.shape(x), dim)
        return (x, _normalize_group_index(group_index, pre))

    third_party = {
        "torch": _SwigluTorchCompose,
        "tf": _SwigluTfCompose,
    }
    tolerance = dict(_TOL)
    pre_compare = _pre_compare_mask_invalid_rows


def _torch_dtype_of(name):
    return getattr(torch, str(name).split(".")[-1], torch.float32)


def _customize_group_torch_inplace(x, group_index, dim):
    """ACLNN/E2E 通路的 group_index 契约化（in-place：TTK 对这两路的 input 插件忽略返回值）。

    归一规则同 Kernel 通路 _normalize_group_index；torch 张量 in-place 写回
    （ACLNN 的 as_strided view / E2E 的 from_numpy 张量均与底层 storage 共享）。
    长度不匹配时跳过写回（torch copy_ 会把短源广播到全量，导致 sum 越界）。
    """
    if group_index is None:
        return
    _, pre, _ = _resolve_axis(tuple(np.shape(x)), int(dim))
    normalized = _normalize_group_index(group_index, pre)
    if isinstance(group_index, torch.Tensor):
        if int(np.prod(normalized.shape)) == int(np.prod(group_index.shape)):
            group_index.copy_(torch.from_numpy(normalized).to(group_index.dtype))
    else:
        np.asarray(group_index)[...] = normalized


# ---- ACLNN V1 : aclnnClippedSwiglu(x, groupIndexOptional, dim, alpha, limit, bias, interleaved, out) ----


class AclnnClippedSwigluSpec:
    """ACLNN V1 流程 TestSpec（torch 入/出；无 clampMode，固定 clamp 在 silu 前）。"""

    pre_compare = _pre_compare_mask_invalid_rows

    @staticmethod
    def golden(
        x,
        group_index=None,
        dim=-1,
        alpha=_DEFAULT_ALPHA,
        limit=_DEFAULT_LIMIT,
        bias=_DEFAULT_BIAS,
        interleaved=True,
        out=None,
        **kwargs,
    ):
        return _aclnn_golden_impl(
            x,
            group_index,
            dim,
            alpha,
            limit,
            bias,
            interleaved,
            0,
            out,
            kwargs,
            mask_tail=True,
        )

    @staticmethod
    def customize_inputs(
        x,
        group_index=None,
        dim=-1,
        alpha=_DEFAULT_ALPHA,
        limit=_DEFAULT_LIMIT,
        bias=_DEFAULT_BIAS,
        interleaved=True,
        out=None,
        **kwargs,
    ):
        """ACLNN V1：group_index 契约化（in-place，随机分组和约束到 ≤ pre）。"""
        _customize_group_torch_inplace(x, group_index, dim)
        return (x, group_index)

    third_party = {
        "torch": _SwigluTorchCompose,
    }
    tolerance = dict(_TOL)


# ---- ACLNN V2 : aclnnClippedSwigluV2(..., interleaved, clampMode, out) ----


class AclnnClippedSwigluV2Spec:
    """ACLNN V2 流程 TestSpec（torch 入/出；clampMode 驼峰参数名跟随 C 头签名）。"""

    @staticmethod
    def golden(
        x,
        group_index=None,
        dim=-1,
        alpha=_DEFAULT_ALPHA,
        limit=_DEFAULT_LIMIT,
        bias=_DEFAULT_BIAS,
        interleaved=True,
        clampMode=0,
        out=None,
        **kwargs,
    ):
        return _aclnn_golden_impl(
            x,
            group_index,
            dim,
            alpha,
            limit,
            bias,
            interleaved,
            int(clampMode),
            out,
            kwargs,
            mask_tail=True,
        )

    third_party = {
        "torch": _SwigluTorchCompose,
    }
    tolerance = dict(_TOL)
    pre_compare = _pre_compare_mask_invalid_rows

    @staticmethod
    def customize_inputs(
        x,
        group_index=None,
        dim=-1,
        alpha=_DEFAULT_ALPHA,
        limit=_DEFAULT_LIMIT,
        bias=_DEFAULT_BIAS,
        interleaved=True,
        clampMode=0,
        out=None,
        **kwargs,
    ):
        """ACLNN V2：group_index 契约化（in-place，随机分组和约束到 ≤ pre）。"""
        _customize_group_torch_inplace(x, group_index, dim)
        return (x, group_index)


def _aclnn_golden_impl(
    x,
    group_index,
    dim,
    alpha,
    limit,
    bias,
    interleaved,
    clamp_mode,
    out,
    kwargs,
    mask_tail=False,
):
    """ACLNN V1/V2 共用实现：torch 入/出（golden 在输入所在设备计算，无 NumPy 回落）。"""
    shape = tuple(x.shape)
    dim_pos, pre, _ = _resolve_axis(shape, int(dim))
    rows = _group_rows(group_index, pre)
    compute = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
    core = _clipped_swiglu_core(
        compute,
        dim_pos,
        rows,
        float(alpha),
        float(limit),
        float(bias),
        bool(interleaved),
        int(clamp_mode),
    )
    if isinstance(out, torch.Tensor):
        target_dt = out.dtype
    else:
        out_dtypes = kwargs.get("output_dtypes") or ()
        target_dt = (
            _torch_dtype_of(out_dtypes[0])
            if len(out_dtypes) > 0 and out_dtypes[0]
            else x.dtype
        )
    out_t = _build_full_output(core, shape, dim_pos, rows, target_dt)
    if mask_tail and rows < pre:
        # TTK accepts either carrier. Keep the mask with the returned reference,
        # including for CPU dumps; a tensor subclass is not serializable by TTK.
        array = _torch_to_np(out_t, str(target_dt).split(".")[-1])
        return [_partial_reference(array, core.numel())]
    return [out_t]


# ---- E2E : cann_ops_nn.clipped_swiglu(x, group_index=None, *, dim=-1, ..., clamp_mode=0) ----


class ClippedSwigluE2ESpec:
    """E2E 流程 TestSpec（torch 入/出；schema 见 torch_extension/clipped_swiglu.py）。

    third_party 为独立 torch 组合；禁止把被测 cann_ops_nn.clipped_swiglu / ACLNN 调用
    登记为参考。
    """

    @staticmethod
    def golden(
        x,
        group_index=None,
        *,
        dim=-1,
        alpha=_DEFAULT_ALPHA,
        limit=_DEFAULT_LIMIT,
        bias=_DEFAULT_BIAS,
        interleaved=True,
        clamp_mode=0,
        **kwargs,
    ):
        shape = tuple(x.shape)
        dim_pos, pre, _ = _resolve_axis(shape, int(dim))
        rows = _group_rows(group_index, pre)
        compute = (
            x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
        )
        core = _clipped_swiglu_core(
            compute,
            dim_pos,
            rows,
            float(alpha),
            float(limit),
            float(bias),
            bool(interleaved),
            int(clamp_mode),
        )
        out_t = _build_full_output(core, shape, dim_pos, rows, x.dtype)
        return [out_t]

    @staticmethod
    def customize_inputs(
        x,
        group_index=None,
        *,
        dim=-1,
        alpha=_DEFAULT_ALPHA,
        limit=_DEFAULT_LIMIT,
        bias=_DEFAULT_BIAS,
        interleaved=True,
        clamp_mode=0,
        **kwargs,
    ):
        """E2E：group_index 契约化（in-place，随机分组和约束到 ≤ pre）。"""
        _customize_group_torch_inplace(x, group_index, dim)
        return (x, group_index)

    third_party = {
        "torch": _SwigluTorchCompose,
    }
    tolerance = dict(_TOL)
