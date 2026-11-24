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

"""Torch CPU golden for CosineEmbeddingLoss."""

import math

import numpy as np
import torch

__spec__ = {
    "cosine_embedding_loss": "CosineEmbeddingLossKernelSpec",
}

__golden__ = {"kernel": {"cosine_embedding_loss": "__golden_cosine_embedding_loss"}}

EPS = 1.0e-12

_KERNEL_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _resolve(kwargs, margin, reduction):
    attrs = kwargs.get("attrs")
    if isinstance(attrs, dict):
        margin = attrs.get("margin", margin)
        reduction = attrs.get("reduction", reduction)
    return float(margin), str(reduction)


def _torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype)
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }.get(name, torch.float32)


def _output_dtype(kwargs, index, default):
    output_dtypes = kwargs.get("output_dtypes") or []
    if index >= len(output_dtypes):
        return default
    dtype = output_dtypes[index]
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0]
    return _torch_dtype(dtype) if dtype is not None else default


def cosine_embedding_loss_golden(
    x1, x2, target, margin=0.0, reduction="mean", **kwargs
):
    margin, reduction = _resolve(kwargs, margin, reduction)
    # GE stores the Float attribute at FP32 precision before any FP64 computation.
    margin = float(np.float32(margin))
    a = torch.as_tensor(np.asarray(x1))
    b = torch.as_tensor(np.asarray(x2))
    t = torch.as_tensor(np.asarray(target))
    compute_dtype = torch.promote_types(torch.float32, a.dtype)
    out_dtype = _output_dtype(kwargs, 0, compute_dtype)
    compute_dtype = torch.promote_types(compute_dtype, out_dtype)
    if a.dtype == torch.int32 and b.dtype == torch.int32:
        # TTK does not promote integer inputs. Keep the CPU oracle independent of FP32 kernels.
        compute_dtype = torch.float64
    chunked = _maybe_chunked_loss(a, b, t, margin, reduction, compute_dtype, out_dtype)
    if chunked is not None:
        return chunked.numpy()

    a, b = torch.broadcast_tensors(a, b)
    a = a.to(compute_dtype)
    b = b.to(compute_dtype)

    dot = torch.sum(a * b, dim=1)
    s1 = torch.sum(a * a, dim=1)
    s2 = torch.sum(b * b, dim=1)
    eps = torch.tensor(EPS, dtype=compute_dtype)
    denom = torch.sqrt(s1 + eps) * torch.sqrt(s2 + eps)
    cos = dot / denom

    cos, t = torch.broadcast_tensors(cos, t)
    pos = torch.tensor(1.0, dtype=compute_dtype) - cos
    neg = torch.maximum(
        torch.tensor(0.0, dtype=compute_dtype),
        cos - torch.tensor(margin, dtype=compute_dtype),
    )
    loss = torch.where(
        t == 1.0, pos, torch.where(t == -1.0, neg, torch.zeros_like(pos))
    )

    if reduction == "none":
        return loss.to(out_dtype).numpy()
    if reduction == "sum":
        return torch.sum(loss).reshape(1).to(out_dtype).numpy()
    denom_n = loss.numel() if loss.numel() > 0 else 1
    return (torch.sum(loss) / denom_n).reshape(1).to(out_dtype).numpy()


def __golden_cosine_embedding_loss(x1, x2, target, **kwargs):
    return [cosine_embedding_loss_golden(x1, x2, target, **kwargs)]


_cosine_embedding_loss_spec_golden = __golden_cosine_embedding_loss


# Bound the number of elements in each broadcast intermediate, including the feature axis.
_CHUNK_THRESHOLD = 4_000_000


def _loss_tiles(shape, budget):
    steps = []
    for extent in reversed(shape):
        step = max(1, min(extent, budget))
        steps.append(step)
        budget = max(1, budget // step)
    steps.reverse()
    counts = tuple((n + s - 1) // s for n, s in zip(shape, steps))
    for coords in np.ndindex(counts):
        yield tuple(slice(i * s, (i + 1) * s) for i, s in zip(coords, steps))


def _broadcast_slice(tensor, index):
    # Missing leading dimensions and singleton axes are broadcast, not sliced.
    index = index[len(index) - tensor.ndim :]
    return tensor[
        tuple(slice(None) if n == 1 else s for n, s in zip(tensor.shape, index))
    ]


def _block_cosine(a, b, dtype):
    # The feature axis is last. Merge its partial sums before taking the norms.
    feature = a.shape[-1]
    for start in range(0, max(1, feature), _CHUNK_THRESHOLD):
        stop = start + _CHUNK_THRESHOLD
        x = a[..., start:stop].to(dtype)
        y = b[..., start:stop].to(dtype)
        dot = (x * y).sum(-1)
        s1 = (x * x).sum(-1)
        s2 = (y * y).sum(-1)
        if start == 0:
            total_dot, total_s1, total_s2 = dot, s1, s2
        else:
            total_dot += dot
            total_s1 += s1
            total_s2 += s2
    return total_dot / ((total_s1 + EPS).sqrt() * (total_s2 + EPS).sqrt())


def _merge_partial(partials, subtotal):
    # Binary-tree accumulation keeps the reference dtype and O(log(tiles)) scalars.
    for level, previous in enumerate(partials):
        if previous is None:
            partials[level] = subtotal
            return
        subtotal = previous + subtotal
        partials[level] = None
    partials.append(subtotal)


def _zero_negative_reduction(
    x1, x2, target, feature, margin, reduction, compute_dtype, out_dtype
):
    if reduction not in ("mean", "sum") or feature <= 0:
        return None
    if not math.isfinite(margin) or margin < 0:
        return None

    limits = torch.finfo(compute_dtype)
    if margin > limits.max or not (x1.device == x2.device == target.device):
        return None
    inputs = (x1, x2, target)
    supported = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    )
    if any(tensor.numel() == 0 or tensor.dtype not in supported for tensor in inputs):
        return None

    target_min, target_max = (value.item() for value in torch.aminmax(target))
    if target_min != -1 or target_max != -1:
        return None
    a_min, a_max = (value.item() for value in torch.aminmax(x1))
    b_min, b_max = (value.item() for value in torch.aminmax(x2))
    bounds = (a_min, a_max, b_min, b_max)
    if not all(math.isfinite(value) for value in bounds):
        return None
    if not ((a_min >= 0 and b_max <= 0) or (a_max <= 0 and b_min >= 0)):
        return None

    # Keep dot, squared norms, and the product of norms finite in the compute
    # dtype, with headroom for rounding during accumulation and normalization.
    max_abs = math.sqrt((limits.max / 16) / feature)
    if max(abs(value) for value in bounds) > max_abs:
        return None

    return torch.zeros((1,), dtype=out_dtype, device=x1.device)


def _maybe_chunked_loss(x1, x2, target, margin, reduction, compute_dtype, out_dtype):
    a, b = torch.broadcast_tensors(x1, x2)
    if a.ndim < 2:
        raise ValueError("broadcast rank of x1 and x2 must be at least 2")
    loss_shape = torch.broadcast_shapes(a.shape[:1] + a.shape[2:], target.shape)
    size = math.prod(loss_shape)
    if max(a.numel(), size) <= _CHUNK_THRESHOLD:
        return None

    zero = _zero_negative_reduction(
        x1, x2, target, a.shape[1], margin, reduction, compute_dtype, out_dtype
    )
    if zero is not None:
        return zero

    a, b = a.movedim(1, -1), b.movedim(1, -1)
    feature = a.shape[-1]
    if reduction == "none":
        out = torch.empty(loss_shape, dtype=out_dtype, device=a.device)
    else:
        partials = []
    tile_budget = max(1, _CHUNK_THRESHOLD // max(1, feature))
    for index in _loss_tiles(loss_shape, tile_budget):
        vectors = index + (slice(None),)
        cos = _block_cosine(
            _broadcast_slice(a, vectors), _broadcast_slice(b, vectors), compute_dtype
        )
        t = _broadcast_slice(target, index)
        loss = torch.where(
            t == 1, 1 - cos, torch.where(t == -1, (cos - margin).clamp_min(0), 0.0)
        )
        if reduction == "none":
            out[index] = loss
        else:
            subtotal = loss.sum()
            # One check after selecting labels avoids scanning a huge all-NaN aggregate.
            if not partials and torch.isnan(subtotal).item():
                return subtotal.to(out_dtype).reshape(1)
            _merge_partial(partials, subtotal)
    if reduction == "none":
        return out
    out = torch.zeros((), dtype=compute_dtype, device=a.device)
    for part in reversed(partials):
        if part is not None:
            out += part
    if reduction != "sum":
        out /= max(1, size)
    return out.to(out_dtype).reshape(1)


class _CosineEmbeddingLossCompose:
    """Third-party reference executed on the remote GPU server."""

    def __init__(self, margin=0.0, reduction="mean", **kwargs):
        self.margin, self.reduction = _resolve(kwargs, margin, reduction)

    def __call__(self, x1, x2, target, **kwargs):
        chunked = _maybe_chunked_loss(
            x1, x2, target, self.margin, self.reduction, torch.float32, torch.float32
        )
        if chunked is not None:
            return [chunked]
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        target = target.to(torch.float32)
        x1, x2 = torch.broadcast_tensors(x1, x2)
        dot = torch.sum(x1 * x2, dim=1, dtype=torch.float32)
        s1 = torch.sum(x1 * x1, dim=1, dtype=torch.float32)
        s2 = torch.sum(x2 * x2, dim=1, dtype=torch.float32)
        denom = torch.sqrt(s1 + EPS) * torch.sqrt(s2 + EPS)
        cos = dot / denom
        cos, target = torch.broadcast_tensors(cos, target)
        pos = 1.0 - cos
        neg = torch.maximum(
            torch.zeros((), dtype=torch.float32, device=cos.device),
            cos - self.margin,
        )
        loss = torch.where(
            target == 1.0,
            pos,
            torch.where(target == -1.0, neg, torch.zeros_like(pos)),
        )
        if self.reduction == "none":
            return [loss.to(torch.float32)]
        if self.reduction == "sum":
            return [torch.sum(loss, dtype=torch.float32).reshape(1)]
        denom_n = loss.numel() if loss.numel() > 0 else 1
        return [(torch.sum(loss, dtype=torch.float32) / denom_n).reshape(1)]


class CosineEmbeddingLossKernelSpec:
    golden = _cosine_embedding_loss_spec_golden
    third_party = {"torch": _CosineEmbeddingLossCompose}
    tolerance = _KERNEL_TOLERANCE


# 【不存在】aclnn 通路: CMakeLists.txt 使用 ACLNNTYPE aclnn_exclude.
# 【不存在】e2e 通路: 未发现 torch_npu eager/aten 绑定到 CosineEmbeddingLoss.
