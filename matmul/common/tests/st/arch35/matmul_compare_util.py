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
数据对比（ttk 自定义比较器）。

  - isclose_compare：基于 np.isclose 的通用比较器（复现 ttk --compare close，
    容差取自 CSV，整型输出按严格判定）
  - numeric_ulp_compare：基于 ULP / 绝对误差（阈值 1.0）的比较器
    （原 numeric_ulp.py，供 flat_quant 使用）

golden 资产请统一从总入口 ``matmul_golden_util`` 引用。
"""

import ast
import math

import numpy as np

from matmul_torch_util import torch_to_numpy

# 失败上报的 diff_indices 条数上限（与 ttk 侧 _CUSTOM_LIST_CAP 对齐）
_DIFF_INDEX_LIMIT = 20


def _aligned_operands(npu_flat, golden_flat):
    """按 dtype 对齐比较操作数：同 dtype 原样返回，跨 dtype 统一抬升 float32。

    跨 dtype 时 ml_dtypes/en_dtypes 的窄类型（bf16/fp16/fp8/fp4/hifloat8）
    之间直接 promote 会抛 DTypePromotionError（如 bf16 vs fp16），统一
    float32 可稳定且无损地比较（窄浮点均可精确表示为 float32）。
    """
    if npu_flat.dtype == golden_flat.dtype:
        return npu_flat, golden_flat
    return npu_flat.astype(np.float32), golden_flat.astype(np.float32)


def _fail(error_info, precision=0.0, diff_indices=None, metrics=None):
    """构造 ttk custom compare 的失败返回（保留实际匹配率与可选失配信息）。"""
    result = {"pass": False, "precision": precision, "error_info": error_info}
    if diff_indices is not None:
        result["diff_indices"] = diff_indices
    if metrics is not None:
        result["metrics"] = metrics
    return result


# ============================================================================
# isclose compare
# ============================================================================


def _compare_integer(npu_flat, golden_flat):
    """整型严格比较：全元素 |diff| <= 1 才 PASS，否则按失配比例上报。"""
    diff = np.abs(npu_flat.astype(np.int64) - golden_flat.astype(np.int64))
    match = diff <= 1
    precision = match.sum() / match.size
    if bool(match.all()):
        return {"pass": True, "precision": round(precision * 100, 4)}
    bad = np.flatnonzero(~match)
    max_diff = int(diff.max())
    return _fail(
        f"{bad.size}/{match.size} mismatched (integer |diff| > 1), max abs diff={max_diff}",
        precision=round(precision * 100, 4),
        diff_indices=[int(i) for i in bad[:_DIFF_INDEX_LIMIT]],
        metrics={"mismatch_count": int(bad.size), "max_abs_diff": max_diff},
    )


def _compare_float(npu_flat, golden_flat, compare_context):
    """浮点 np.isclose 比较：容差取自 CSV，按 dtype 对齐后逐元素判定。

    同 dtype 原样比较；跨 dtype 统一抬升 float32（窄类型间 promote 会抛
    DTypePromotionError）；同 dtype 窄类型在部分 numpy/ml_dtypes 组合下
    promote 失败时，退化为 float32 重试。CSV 缺容差时按 ttk 内置同款档位
    回退（float16 / bfloat16 / complex32 -> 0.001，其余 -> 0.0001；atol
    默认 1e-8）。
    """
    csv = compare_context.csv_fields if compare_context else {}

    ptol_raw = csv.get("precision_tolerances", "")
    atol_raw = csv.get("absolute_precision", "")

    if ptol_raw:
        rtol, ptol = ast.literal_eval(ptol_raw)[0]
    else:
        dtype_str = str(npu_flat.dtype).split(".")[-1]
        rtol = ptol = (
            0.001 if dtype_str in ("float16", "bfloat16", "complex32") else 0.0001
        )

    atol = ast.literal_eval(atol_raw) if atol_raw else 1e-8

    npu_cmp, golden_cmp = _aligned_operands(npu_flat, golden_flat)
    try:
        match = np.isclose(npu_cmp, golden_cmp, rtol=rtol, atol=atol, equal_nan=True)
    except (TypeError, ValueError):
        # 同 dtype 的窄类型在部分 numpy/ml_dtypes 组合下 promote 仍可能失败
        # （DTypePromotionError 属 TypeError）：退化为 float32 再比一次。
        npu_cmp = npu_cmp.astype(np.float32)
        golden_cmp = golden_cmp.astype(np.float32)
        match = np.isclose(npu_cmp, golden_cmp, rtol=rtol, atol=atol, equal_nan=True)
    precision = match.sum() / match.size
    if (1 - precision) <= ptol:
        return {"pass": True, "precision": round(precision * 100, 4)}

    bad = np.flatnonzero(~match)
    abs_diff = np.abs(npu_cmp.astype(np.float32) - golden_cmp.astype(np.float32))
    finite = abs_diff[np.isfinite(abs_diff)]
    max_diff = float(finite.max()) if finite.size else float("inf")
    return _fail(
        f"{bad.size}/{match.size} mismatched (rtol={rtol}, atol={atol}, ptol={ptol}), "
        f"max abs diff={max_diff:g}",
        precision=round(precision * 100, 4),
        diff_indices=[int(i) for i in bad[:_DIFF_INDEX_LIMIT]],
        metrics={"mismatch_count": int(bad.size), "max_abs_diff": max_diff},
    )


def isclose_compare(npu_out, golden_out, compare_context=None, **kwargs):
    """基于 np.isclose 的精度比较入口（ttk custom compare 契约）。

    负责与 dtype 无关的转换与守卫，并按输出 dtype 分发：
      - 整型输出 -> _compare_integer：严格判定，全元素 |diff| <= 1 才 PASS
      - 浮点输出 -> _compare_float：按 CSV 容差（rtol/ptol/atol）逐元素对比

    失败时返回 error_info / diff_indices / metrics 供 ttk 记录与落盘。

    Args:
        npu_out: NPU 输出数组
        golden_out: golden 参考数组
        compare_context: ttk 注入的 CompareContext，通过 csv_fields 访问 CSV 字段
        **kwargs: 吸收 ttk 传入的额外参数

    Returns:
        dict: {"pass": bool, "precision": float, [error_info/diff_indices/metrics]}
    """
    if hasattr(npu_out, "detach"):
        npu_out = torch_to_numpy(npu_out)
    if hasattr(golden_out, "detach"):
        golden_out = torch_to_numpy(golden_out)

    if npu_out is None or golden_out is None:
        return _fail(
            f"tensor is None: npu={npu_out is None}, golden={golden_out is None}"
        )

    # 空张量短路（对齐 ttk 内置比对语义）：都空->PASS；仅一方空->FAIL
    if npu_out.size == 0 and golden_out.size == 0:
        return {"pass": True, "precision": 100.0}
    if npu_out.size == 0 or golden_out.size == 0:
        return _fail(
            f"empty mismatch: npu size={npu_out.size}, golden size={golden_out.size}"
        )
    if npu_out.size != golden_out.size:
        return _fail(
            f"size mismatch: npu size={npu_out.size}, golden size={golden_out.size}"
        )

    # 与 ttk 内置比对一致按元素个数逐点比较（与形状无关）
    npu_flat = npu_out.ravel()
    golden_flat = golden_out.ravel()

    if np.issubdtype(npu_flat.dtype, np.integer):
        return _compare_integer(npu_flat, golden_flat)
    return _compare_float(npu_flat, golden_flat, compare_context)


# ============================================================================
# numeric ULP compare (standalone, no ttk dependency)
# ============================================================================


def _to_float(arr):
    dtype_str = str(arr.dtype).split(".")[-1]
    if "float4" in dtype_str:
        return arr.astype(np.float32, copy=False)
    if "float8_e8m0" in dtype_str:
        return arr.view(np.uint8).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def numeric_ulp_compare(npu_out, golden_out, compare_context=None, **kwargs):
    """基于 ULP 的自定义精度比较。

    将输出转为 float32 后逐元素比较绝对误差，阈值固定 1.0。

    Args:
        npu_out: NPU 输出数组
        golden_out: golden 参考数组
        compare_context: ttk 注入的 CompareContext（保留用于签名兼容，当前未使用）
        **kwargs: 吸收 ttk 传入的额外参数

    Returns:
        dict: {"pass": bool, "precision": float}
    """
    if hasattr(npu_out, "detach"):
        npu_out = torch_to_numpy(npu_out)
    if hasattr(golden_out, "detach"):
        golden_out = torch_to_numpy(golden_out)

    output = _to_float(npu_out)
    golden = _to_float(golden_out)

    output_flat = output.ravel()
    golden_flat = golden.ravel()
    diff_flat = np.abs(np.subtract(output_flat, golden_flat))
    diff_mask = diff_flat > 1.0
    diff_indices = np.where(diff_mask)[0]

    npu_nan = np.isnan(output_flat)
    golden_nan = np.isnan(golden_flat)
    both_nan = np.logical_and(npu_nan, golden_nan)
    diff_indices = np.setdiff1d(diff_indices, np.where(both_nan)[0])

    golden_size = golden_flat.size
    diff_size = diff_indices.size
    if golden_size == 0:
        return {"pass": diff_size == 0, "precision": 100.0}

    csv = compare_context.csv_fields if compare_context else {}
    ptol_raw = csv.get("precision_tolerances", "")
    if ptol_raw:
        _, ptol = ast.literal_eval(ptol_raw)[0]
    else:
        dtype_str = str(npu_out.dtype).split(".")[-1]
        ptol = 0.001 if dtype_str != "float32" else 0.0001

    precision = math.floor((golden_size - diff_size) / golden_size * 10000) / 100
    is_pass = diff_size / golden_size <= ptol
    return {"pass": bool(is_pass), "precision": precision}
