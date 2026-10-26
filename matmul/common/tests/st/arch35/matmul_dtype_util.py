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
dtype 常量、名称解析与转换。

matmul golden 公共 util 的 dtype 分类模块，包含：
  - numpy dtype 别名（ml_dtypes / en_dtypes 扩展类型）
  - dtype 关系表 DTypeTable：str / numpy / torch 三种表示，任一可检索另两者
    （dtype_to_str / dtype_to_np / get_torch_dtype）
  - dtype 相关的 cast 与精度截断动作（get_intermediate_dtype / truncate /
    hf32_truncate_np）

golden 资产请统一从总入口 ``matmul_golden_util`` 引用。
"""

import numpy as np
import en_dtypes
import ml_dtypes
import torch

# ============================================================================
# dtype definitions
# ============================================================================

np_fp4_e2m1 = en_dtypes.float4_e2m1
np_fp4_e1m2 = en_dtypes.float4_e1m2
np_hif8 = en_dtypes.hifloat8
np_fp8_e8m0 = en_dtypes.float8_e8m0
np_bfloat16 = ml_dtypes.bfloat16
np_fp8_e4m3 = ml_dtypes.float8_e4m3fn
np_fp8_e5m2 = ml_dtypes.float8_e5m2
np_int4 = ml_dtypes.int4


# ============================================================================
# dtype relationship table
# ============================================================================

# numpy 名字 -> 规范名（ml_dtypes 将 float8_e8m0fnu 注册为独立名字，需归一）
_NAME_CANON = {"float8_e8m0fnu": "float8_e8m0"}


class DTypeTable:
    """dtype 关系表：str / np / torch 三种表示，任一可检索另两者。

    新增 dtype 只需在 _TABLE 加一行 (规范名, numpy 实体, torch 实体)；
    无 torch 实体的 dtype 填 None。str <-> numpy 的解析由 numpy 注册表
    承担（规范名与 ``np.dtype(x).name`` 一致），torch 是唯一需显式建表的一侧。
    """

    _TABLE = (
        ("float16", np.float16, torch.float16),
        ("float32", np.float32, torch.float32),
        ("float64", np.float64, torch.float64),
        ("bfloat16", np_bfloat16, torch.bfloat16),
        ("int8", np.int8, torch.int8),
        ("int16", np.int16, torch.int16),
        ("int32", np.int32, torch.int32),
        ("int64", np.int64, torch.int64),
        ("uint8", np.uint8, torch.uint8),
        ("uint64", np.uint64, torch.uint64),
        ("bool", np.bool_, torch.bool),
        ("float8_e4m3fn", np_fp8_e4m3, torch.float8_e4m3fn),
        ("float8_e5m2", np_fp8_e5m2, torch.float8_e5m2),
        ("float8_e8m0", np_fp8_e8m0, torch.float8_e8m0fnu),
        ("hifloat8", np_hif8, None),
        ("int4", np_int4, None),
        ("float4_e2m1", np_fp4_e2m1, None),
        ("float4_e1m2", np_fp4_e1m2, None),
    )

    @classmethod
    def _find(cls, x):
        """任意表示 -> 命中行 (规范名, numpy 实体, torch 实体)；未命中 -> None。"""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.dtype
        elif isinstance(x, np.ndarray):
            x = x.dtype
        if isinstance(x, torch.dtype):
            for row in cls._TABLE:
                if row[2] is x:
                    return row
            return None
        try:
            name = np.dtype(x).name
        except TypeError:
            return None
        name = _NAME_CANON.get(name, name)
        for row in cls._TABLE:
            if row[0] == name:
                return row
        return None

    @classmethod
    def get_str(cls, x):
        """任意表示 -> 规范 dtype 名（None -> None）。

        支持 torch.Tensor / numpy.ndarray / torch.dtype / numpy dtype（类或实例）
        / 名称字符串；跨框架口径统一（如 torch.float8_e8m0fnu 与 en_dtypes 的
        float8_e8m0 均归一为 "float8_e8m0"）。未登记的 torch dtype 显式报错，
        未登记的 numpy dtype 回退 ``np.dtype(x).name``，映射表驱动，不做字符剥离。
        """
        if x is None:
            return None
        if isinstance(x, str):
            row = cls._find(x)
            if row is not None:
                return row[0]
            try:
                name = np.dtype(x).name
            except TypeError:
                return x
            return _NAME_CANON.get(name, name)
        if isinstance(x, torch.dtype):
            row = cls._find(x)
            if row is None:
                raise ValueError(f"Unsupported torch dtype: {x}")
            return row[0]
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            return cls.get_str(x.dtype)
        try:
            d = np.dtype(x)
        except TypeError:
            raise TypeError(f"Cannot resolve dtype from {x!r}") from None
        row = cls._find(d)
        return row[0] if row is not None else _NAME_CANON.get(d.name, d.name)

    @classmethod
    def get_np_dtype(cls, x):
        """任意表示 -> numpy dtype 实体（None / 无法解析 / 未登记 -> None）。

        torch 实体经关系表映射（如 float8_e8m0fnu -> en_dtypes.float8_e8m0）；
        无 numpy 实体的 torch 独有类型与表外输入返回 None，由调用方决定兜底
        （如 torch_to_numpy 在未命中时保留 torch 原始报错）。
        """
        if x is None:
            return None
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            return cls.get_np_dtype(x.dtype)
        row = cls._find(x)
        if row is not None:
            return row[1]
        if isinstance(x, torch.dtype):
            return None
        try:
            return np.dtype(x).type
        except TypeError:
            return None

    @classmethod
    def get_torch_dtype(cls, x):
        """任意表示 -> torch dtype 实体（无 torch 实体 / 未登记 -> None）。"""
        if x is None:
            return None
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            return cls.get_torch_dtype(x.dtype)
        row = cls._find(x)
        return row[2] if row is not None else None


dtype_to_str = DTypeTable.get_str
dtype_to_np = DTypeTable.get_np_dtype


# ============================================================================
# dtype cast
# ============================================================================


def cast_output_dtype(arr, dtype_name):
    """将数组转换为目标 dtype，支持字符串或 dtype 对象。"""
    target = dtype_to_np(dtype_name)
    if target is not None:
        return arr.astype(target)
    return arr.astype(dtype_name)


def e8m0_to_float(arr):
    """将 float8_e8m0（或等宽原始字节数组）解码为 float32：2^(raw-127)，255→NaN。

    与 en_dtypes float8_e8m0 的 astype(np.float32) 数值一致；
    fp32/fp64 输入直接透传（对齐上游平台归一新增的 scale 直传路径）。
    """
    dt = arr.dtype
    if dt == np.float32 or dt == np.float64:
        return arr.astype(np.float32)
    raw = np.ascontiguousarray(arr).view(np.uint8).astype(np.float32)
    result = np.power(2.0, raw - 127.0)
    result[raw == 255.0] = np.nan
    return result


# ============================================================================
# intermediate precision simulation
# ============================================================================


def get_intermediate_dtype(x_dtype_str):
    """根据输入 dtype 字符串返回对应的中间计算精度（fp16→float16, bf16→bfloat16）。"""
    if x_dtype_str == "bfloat16":
        return np_bfloat16
    elif x_dtype_str == "float16":
        return np.float16
    return None


def truncate(arr, inter_dtype):
    """将数组截断到指定中间精度（模拟硬件精度损失）。"""
    if inter_dtype is None:
        return arr
    return arr.astype(inter_dtype).astype(np.float32)


def hf32_truncate_np(arr):
    """Truncate a float32 numpy array to HF32 (19-bit mantissa).

    Simulates the NPU cube HF32 execution: each fp32 value's mantissa is rounded
    so that the low 12 bits of the 23-bit mantissa become 0 (round-half-up of
    the top surviving bit), keeping 11 mantissa bits.
    """
    int_view = arr.view(np.uint32)
    truncated = (((int_view >> 11) + 1) >> 1) << 12
    return truncated.view(np.float32)
