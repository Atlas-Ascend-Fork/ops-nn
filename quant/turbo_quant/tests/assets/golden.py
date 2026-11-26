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
"""TurboQuant aclnn 用例的算子测试规范。

golden 与输入钩子用模块级 __golden__/__input__ 兼容约定;tolerance 与
pre_compare 由 __spec__ 注册的 TestSpec 类承载(这两项没有兼容约定通道)。

golden 按 kernel 运算序编写(reduce -> 1/norm(Div) -> mul -> 15 轮中点计数),L2 规约的
分块累加序与 kernel 一致(64 lane 一块、按块序累加),跨 lane vcadd 为硬件内部树序,以
对分树近似。y 的单字节打包两个 4bit 码本索引,比对前经 pre_compare 解包成索引序列,由
quant 判据按 |索引差| <= 1 判定:量化末步 round 是阶跃函数,跨 lane 树序近似的末位 ulp
差异可能把恰在中点边界上的值推到相邻档,昇腾算子精度标准 2.1 量化类通过线即整型输出
绝对误差 <= 1;差两档及以上仍失败。
"""

import numpy as np
import torch

__golden__ = {"aclnn": {"aclnnTurboQuant": "aclnn_turbo_quant_golden"}}
__input__ = {"aclnn": {"aclnnTurboQuant": "aclnn_turbo_quant_input"}}

# tolerance 只能经 TestSpec 类声明(quant 判据无 CLI 对应取值);路由为升格例外,
# CI 传入的 --compare close 对整数输出不连带覆盖。
__spec__ = {"aclnnTurboQuant": "TurboQuantAclnnTestSpec"}

NORM_EPS = np.float32(1e-16)


VL = 64  # fp32 lanes per vector chunk (one 256B register), the kernel's chunk granularity


def _reduce_sum_sq(latent):
    """L2 square-sum with the kernel's association (op_kernel/arch35/turbo_quant_regbase.h): the 64-lane
    chunks are accumulated lane-wise in chunk order (acc += x_chunk * x_chunk), then the 64 partial
    sums are cross-lane reduced by a single hardware vcadd. The chunk accumulation is mirrored
    exactly; vcadd's lane tree is hardware-internal and approximated here with a halving binary
    tree, so the golden stays deterministic across hosts (np.sum's association varies with the
    numpy build) and any residual last-ulp difference is covered by the quant criterion."""
    num_tok, head_dim = latent.shape
    chunks = head_dim // VL
    sq = (latent * latent).astype(np.float32).reshape(num_tok, chunks, VL)
    acc = sq[:, 0, :].copy()
    for c in range(1, chunks):
        acc = np.add(acc, sq[:, c, :], dtype=np.float32)
    while acc.shape[1] > 1:
        half = acc.shape[1] // 2
        acc = np.add(acc[:, :half], acc[:, half:], dtype=np.float32)
    return acc[:, 0]


def _turbo_quant_common(latent, centroids):
    """Reference implementation with the kernel's operation order (reduce -> reciprocal -> scale ->
    midpoint counting); the L2 reduction follows the kernel's chunk-accumulation association (see
    _reduce_sum_sq)."""
    latent = np.ascontiguousarray(latent, dtype=np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    red = _reduce_sum_sq(latent)
    norm = np.sqrt(red + NORM_EPS).astype(np.float32)
    inv = (np.float32(1.0) / norm).astype(np.float32)
    unit = (latent * inv[:, None]).astype(np.float32)
    bnd = ((centroids[:-1] + centroids[1:]) * np.float32(0.5)).astype(np.float32)
    # counting how many midpoint boundaries a value exceeds == index of the nearest centroid
    nib = (unit[:, :, None] >= bnd[None, None, :]).sum(axis=2).astype(np.uint8)
    y = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    scale = norm.astype(np.float16)
    return y, scale


def aclnn_turbo_quant_golden(latent, centroids, y=None, scale=None, **kwargs):
    """Aclnn golden for aclnnTurboQuant.

    Parameters follow @aclnnTurboQuantGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    y_np, scale_np = _turbo_quant_common(
        latent.to(torch.float32).numpy(), centroids.to(torch.float32).numpy()
    )
    return [torch.from_numpy(y_np), torch.from_numpy(scale_np)]


def _sorted_centroids(centroids):
    """算子契约要求升序码本;TTK 按 csv 值域生成的张量在此排序,数值不改。"""
    if isinstance(centroids, torch.Tensor):
        sorted_c, _ = torch.sort(centroids.to(torch.float32), dim=-1)
        return sorted_c
    return np.sort(np.asarray(centroids, dtype=np.float32), axis=-1)


def aclnn_turbo_quant_input(*args, **kwargs):
    """Input function for aclnnTurboQuant.

    TTK passes the tensors of the case (latent first, centroids second); only the
    centroids ordering is adjusted, everything else is passed through unchanged.
    """
    latent = args[0] if len(args) > 0 else kwargs.get("latent")
    centroids = args[1] if len(args) > 1 else kwargs.get("centroids")
    return [latent, _sorted_centroids(centroids)]


def _unpack_nibbles(y):
    arr = np.asarray(y)
    idx = np.empty(arr.shape[:-1] + (arr.shape[-1] * 2,), dtype=np.uint8)
    idx[..., 0::2] = arr & 0xF
    idx[..., 1::2] = arr >> 4
    return idx


def _turbo_quant_pre_compare(y, scale, y_golden, scale_golden):
    """把两侧 nibble 打包的 y 解包成码本索引(偶位取低半字节、奇位取高半字节),
    交给 quant 判据按 |索引差| <= 1 判定;scale 原样透传。"""
    return [_unpack_nibbles(y), scale, _unpack_nibbles(y_golden), scale_golden]


class TurboQuantAclnnTestSpec:
    """仅承载 tolerance 与 pre_compare;golden/input 走模块级兼容约定。"""

    tolerance = {"uint8": {"standard": "quant"}}
    pre_compare = staticmethod(_turbo_quant_pre_compare)
