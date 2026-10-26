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
"""
数据排布与结构变换。

matmul golden 公共 util 的 rearrange 分类模块，包含：
  - transdata：FRACTAL_NZ ↔ ND 互转（nz_to_nd / nd_to_nz）
  - NZ 输入规范化：张量 storage 读写（overwrite_storage / nz_storage_to_nd）
  - transpose：末两维交换与多维转置轴序生成
  - 排布识别：从 strides 推断末两维是否转置

golden 资产请统一从总入口 ``matmul_golden_util`` 引用。
"""

import numpy as np
import torch

from matmul_dtype_util import dtype_to_np
from matmul_torch_util import torch_to_numpy


# ============================================================================
# transpose / axes
# ============================================================================


def transpose_last_two_dims(arr):
    """交换数组的最后两个维度（等价于矩阵转置）。"""
    return np.swapaxes(arr, -2, -1)


def gen_axes_for_transpose(offset, base):
    """生成多维转置的轴序列，保留前 offset 个维度，交换最后两个维度。"""
    return [x for x in range(offset)] + [x + offset for x in base]


def detect_transpose_from_strides(tensor):
    """从 tensor 的 strides 推断最后两维是否转置。

    支持 torch.Tensor 和 numpy.ndarray。
    """
    if tensor is None or tensor.ndim < 2:
        return False
    if isinstance(tensor, torch.Tensor):
        s = tensor.stride()
    else:
        itemsize = tensor.itemsize
        s = tuple(st // itemsize for st in tensor.strides)
    return s[-2] == 1 and s[-1] == tensor.shape[-2]


# ============================================================================
# NZ format utilities
# ============================================================================


def nz_to_nd(nz, ori_shape=None):
    """Convert FRACTAL_NZ format to ND format.

    NZ layout: (A0, A1, ..., An, N1, M1, M0, N0)
    ND layout: (A0, A1, ..., An, M, N)

    where M = (M1-1)*M0 + pad_m, N = (N1-1)*N0 + pad_n

    ori_shape 为 NZ 派生前的原始 ND 形状：非对齐场景由调用方传入，据此裁剪
    padding（须落在末块范围 (M1-1)*M0 < M <= M1*M0，N 同理，与框架一致）；
    为 None 时按对齐假设返回完整还原（A... + (M1*M0, N1*N0)，含 padding 区
    原始数据，仅对齐场景有意义）。
    """
    if nz.ndim < 4:
        raise ValueError(f"NZ input must be >= 4D, got {nz.ndim}D")
    batch = nz.shape[:-4]
    n1, m1, m0, n0 = nz.shape[-4:]
    offset = len(batch)
    # (A.., N1, M1, M0, N0) -> (A.., M1, M0, N1, N0) -> (A.., M1*M0, N1*N0)
    axes = list(range(offset)) + [offset + 1, offset + 2, offset, offset + 3]
    padded = nz.transpose(axes).reshape(batch + (m1 * m0, n1 * n0))
    if ori_shape is None:
        return np.ascontiguousarray(padded)
    ori_shape = tuple(ori_shape)
    m, n = ori_shape[-2:]
    if m > m1 * m0 or n > n1 * n0:
        raise ValueError(f"ori_shape M/N {m}/{n} 超出 NZ 容量 {m1 * m0}/{n1 * n0}")
    if m <= (m1 - 1) * m0 or n <= (n1 - 1) * n0:
        raise ValueError(
            f"ori_shape M/N {m}/{n} 与 NZ 块结构不一致"
            f"（须落在末块 ({(m1 - 1) * m0}, {m1 * m0}] x ({(n1 - 1) * n0}, {n1 * n0}]）"
        )
    out = padded[..., :m, :n]
    if out.shape != ori_shape:
        raise ValueError(
            f"ori_shape {ori_shape} 与输入 batch 维度不匹配（解码形状 {out.shape}）"
        )
    return np.ascontiguousarray(out)


def get_nz_n0(dtype_str):
    """根据 dtype 字符串返回 NZ 格式的 n0 (C0) 值。

    本表为 A5 packed-quant 约定（int32/float32 作为 4bit 打包载体时 n0=2；
    int8/uint8→32；其余→16）；经典 NZ 约定参考框架 PAD_C0_MAPPING
    （int32/float32→16、int64→4、bool→32 等）。
    """
    if dtype_str in ("int8", "uint8"):
        return 32
    elif dtype_str in ("float32", "int32"):
        return 2
    else:
        return 16


def nd_to_nz(nd, n0=None):
    """将 ND 数据（2 维~N 维，末两维为 (M, N)）重排为 FRACTAL_NZ 物理布局。

    ND:  (A0, .., An, M, N)
    NZ:  (A0, .., An, N1, M1, 16, N0)
    映射: nz[..., j//N0, i//16, i%16, j%N0] = nd[..., i, j]

    M/N 方向不足分块宽度时补 0；返回 C-contiguous 数组。n0 缺省按 dtype
    推导（get_nz_n0，A5 packed 约定）；经典 NZ 场景请显式传入 n0。
    """
    if nd.ndim < 2:
        raise ValueError(f"ND input must be >= 2D, got {nd.ndim}D")
    if n0 is None:
        n0 = get_nz_n0(np.dtype(nd.dtype).name)
    batch = nd.shape[:-2]
    m, n = nd.shape[-2], nd.shape[-1]
    m1 = (m + 15) // 16
    n1 = (n + n0 - 1) // n0
    padded = np.zeros(batch + (m1 * 16, n1 * n0), dtype=nd.dtype)
    padded[..., :m, :n] = nd
    blocks = padded.reshape(batch + (m1, 16, n1, n0))
    offset = len(batch)
    axes = list(range(offset)) + [offset + 2, offset, offset + 1, offset + 3]
    return np.ascontiguousarray(blocks.transpose(axes))


# ============================================================================
# NZ 输入规范化（张量底层 storage 读写）
# ============================================================================


def overwrite_storage(tensor, raw_array):
    """将 raw_array 的字节按 storage_offset 覆写进 torch 张量底层 storage。

    用于 NZ 输入规范化：框架按 storage shape 生成的随机 buffer 不是"某个 ND
    数据经 ND→NZ 的结果"，golden 侧需先取 view 形状数据、用 nd_to_nz 生成
    规范 NZ（零填充）后覆写，算子与 golden 才看到一致的排布。

    注：torch 2.x 的 UntypedStorage 不支持 numpy()/frombuffer，且
    ``storage[slice] = Tensor`` 会报 "can only set storage content with a int
    types"；统一走 ``set_`` 建的 uint8 全 buffer 视图写入。
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"overwrite_storage expects torch.Tensor, got {type(tensor)}")
    src = np.ascontiguousarray(raw_array)
    if src.dtype.itemsize != tensor.element_size():
        raise ValueError(
            f"raw_array itemsize {src.dtype.itemsize} != tensor element size {tensor.element_size()}"
        )
    storage = tensor.untyped_storage()
    offset = tensor.storage_offset() * tensor.element_size()
    if offset + src.nbytes > storage.nbytes():
        raise ValueError(
            f"raw_array nbytes {src.nbytes} exceeds storage space {storage.nbytes() - offset}"
        )
    flat = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
    flat[offset : offset + src.nbytes] = torch.from_numpy(
        src.view(np.uint8).reshape(-1)
    )


def nz_storage_to_nd(tensor, nd_shape=None, n0=None):
    """读取 NZ 张量底层 storage 并解码为 ND（ND→NZ 的逆；测例 golden 侧用）。

    nd_shape 为 NZ 对应的逻辑 ND 形状；缺省取张量自身 shape。n0 缺省按 dtype
    推导，并用 storage 元素数校验，不匹配时在候选中回搜（兼容 packed 等约定）。
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"nz_storage_to_nd expects torch.Tensor, got {type(tensor)}")
    if nd_shape is None:
        nd_shape = tuple(tensor.shape)
    nd_shape = tuple(nd_shape)
    if len(nd_shape) < 2:
        raise ValueError(f"nd_shape must be >= 2D, got {nd_shape}")
    itemsize = tensor.element_size()
    batch = nd_shape[:-2]
    m, n = nd_shape[-2:]
    elems = tensor.untyped_storage().nbytes() // itemsize
    batch_elems = elems // max(1, int(np.prod(batch))) if batch else elems

    def _nz_elems(n0cand):
        return ((m + 15) // 16) * 16 * ((n + n0cand - 1) // n0cand) * n0cand

    if n0 is None:
        dtype_name = str(tensor.dtype).replace("torch.", "")
        candidates = [get_nz_n0(dtype_name)] + [
            c for c in (64, 32, 16, 8, 4, 2) if c != get_nz_n0(dtype_name)
        ]
        for cand in candidates:
            if _nz_elems(cand) == batch_elems:
                n0 = cand
                break
        if n0 is None:
            raise ValueError(
                f"cannot infer NZ n0: m={m} n={n} batch_elems={batch_elems} candidates={candidates}"
            )
    m1 = (m + 15) // 16
    n1 = (n + n0 - 1) // n0
    np_dtype = dtype_to_np(tensor.dtype)
    if np_dtype is None:
        raise ValueError(
            f"unsupported torch dtype for NZ storage decode: {tensor.dtype}"
        )
    storage = tensor.untyped_storage()
    flat = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
    offset = tensor.storage_offset() * itemsize
    raw = flat[offset : offset + elems * itemsize].numpy()
    arr = raw.view(np_dtype).reshape(batch + (n1, m1, 16, n0))
    return nz_to_nd(arr, nd_shape)


def prepare_nz_input(tensor, tensor_formats, tensor_idx=1, n0=None):
    """NZ 输入规范化（customize_inputs 侧）：view 数据 -> nd_to_nz（零填充）
    -> 覆写 tensor storage；非 NZ / 非 torch 张量直接返回。

    框架按 storage shape 生成的随机 buffer 并非"某 ND 数据经 ND→NZ"的结果，
    非对齐场景下 view 只是线性重解释；改写为规范 NZ 后 NPU 与 golden 看到同一
    份排布。n0 缺省按 dtype 推导；内核有特定约定时（如 tqbmm c0=32）由调用方
    显式传入。
    """
    if not isinstance(tensor, torch.Tensor):
        return
    if len(tensor_formats) <= tensor_idx or tensor_formats[tensor_idx] != "FRACTAL_NZ":
        return
    nd_view = torch_to_numpy(tensor)
    overwrite_storage(tensor, nd_to_nz(np.ascontiguousarray(nd_view), n0=n0))


def nz_input_to_nd(tensor, tensor_formats, tensor_idx=1, n0=None):
    """NZ 张量 -> ND（返回与生成侧一致的 view 数值）；非 NZ / 非 torch 原样返回。"""
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if len(tensor_formats) <= tensor_idx or tensor_formats[tensor_idx] != "FRACTAL_NZ":
        return tensor
    return nz_storage_to_nd(tensor, nd_shape=tuple(tensor.shape), n0=n0)
