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
torch <-> numpy 桥接。

matmul golden 公共 util 的 torch 桥接叶子模块，包含：
  - torch_to_numpy：torch.Tensor → numpy（bf16 / fp8 / e8m0 等无原生 numpy()
    通道的 dtype 经关系表映射后按 uint8 字节视图降级）
  - to_numpy：多个输入批量转换（None 与 ndarray 原样返回）
  - write_back：customize_inputs 场景的 in-place 回写

本模块是叶子模块（只依赖 matmul_dtype_util 的名称转换），可被任意 util
模块在模块级导入；torch 为模块级依赖（与 util 层现状一致）。

golden 资产请统一从总入口 ``matmul_golden_util`` 引用。
"""

import numpy as np
import torch

from matmul_dtype_util import dtype_to_np


# ============================================================================
# torch<->numpy conversions
# ============================================================================


def torch_to_numpy(tensor):
    """将 PyTorch tensor 转换为 numpy array，处理 bfloat16 和 float8 等特殊类型。

    优先走 torch 原生 numpy() 通道；无原生通道的 dtype（bf16 / fp8 / e8m0）
    由关系表提供 numpy 实体后按**同宽整数视图**（bf16→int16；fp8/e8m0→uint8）
    reinterpret——同宽视图不要求末维 stride==1，空 / K=0 张量（ttk as_strided
    构造、stride 未归一）也能转换；表外 dtype 保留 torch 原始报错。
    """
    if tensor is None:
        return None

    # If already a numpy array, return as-is
    if isinstance(tensor, np.ndarray):
        return tensor

    tensor = tensor.detach().cpu().contiguous()
    try:
        return tensor.numpy()
    except TypeError:
        np_dtype = dtype_to_np(tensor.dtype)
        if np_dtype is None:
            raise
        raw_view = torch.int16 if np.dtype(np_dtype).itemsize == 2 else torch.uint8
        return tensor.view(raw_view).numpy().view(np_dtype)


# ============================================================================
# Input write-back
# ============================================================================


def write_back(dst, src):
    """将 src 数据 in-place 写回 dst（torch.Tensor 用 copy_，numpy 用 copyto）。

    用于 customize_inputs 场景：框架不捕获返回值时，需要将新生成的数据
    写回原始 tensor/array，使修改对后续 NPU 执行可见。
    """
    if isinstance(dst, torch.Tensor):
        if isinstance(src, torch.Tensor):
            dst.copy_(src.to(dst.dtype))
        else:
            dst.copy_(torch.from_numpy(src).to(dst.dtype))
    else:
        if isinstance(src, torch.Tensor):
            src = src.detach().cpu().numpy()
        np.copyto(dst, src.astype(dst.dtype))


def to_numpy(*tensors):
    """将多个输入统一转为 numpy（None 与 ndarray 原样返回）。

    等价于对每个元素调用 torch_to_numpy，便于一次转换多输入。
    """
    return tuple(torch_to_numpy(t) for t in tensors)
