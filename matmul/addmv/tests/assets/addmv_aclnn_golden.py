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
"""ACLNN 层 golden 实现 for aclnnAddmv。

aclnnAddmv 语义：out = beta * self + alpha * (mat @ vec)
  self (n,), mat (n, m), vec (m,) -> out (n,)

计算流程：
  1. torch → numpy
  2. 整数/bool：int64 精确累加，一次取整（整数输出走二进制比对）
  3. 浮点：cubeMathType 精度模拟，fp16/bf16 → fp32 累加后一次 cast 到输出 dtype
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util

_FP32_EPS = 1e-6


def _apply_cube_type(mat, vec, cubeMathType):
    """cubeMathType 精度模拟（与 aclnnMv 一致）。

    - mode 2 (USE_FP16): fp32→fp16 截断，模拟 NPU fp16 计算
    - modes 1/3 (HF32): fp32→hf32 截断（19-bit mantissa）
    - mode 0 (KEEP_DTYPE): 无操作
    """
    if cubeMathType is None:
        return mat, vec
    cmt = int(cubeMathType)
    if cmt == 2:
        if mat is not None and mat.dtype == np.float32:
            mat = mat.astype(np.float16)
        if vec is not None and vec.dtype == np.float32:
            vec = vec.astype(np.float16)
    elif cmt in (1, 3):
        if mat is not None and mat.dtype == np.float32:
            mat = _util.hf32_truncate_np(mat)
        if vec is not None and vec.dtype == np.float32:
            vec = _util.hf32_truncate_np(vec)
    return mat, vec


class AclnnAddmvTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(self, mat, vec, alpha=1.0, beta=1.0, out=None, cubeMathType=0, **kwargs):
        """aclnnAddmv: out = beta * self + alpha * (mat @ vec)"""
        # 1) torch → numpy
        self_np, mat_np, vec_np = _util.to_numpy(self, mat, vec)
        alpha, beta = float(alpha), float(beta)
        out_dtype = _util.dtype_to_str(out if out is not None else mat.dtype)

        # 2) bool：cube 不支持（matmulDtypeSupportList 无 BOOL），算子走
        # GetMulResult 逻辑路径：
        #   mm_i = ANY_j(mat_ij AND vec_j)   (LogicalAnd + ReduceAny)
        #   out  = self OR mm                (LogicalOr)
        # 再 cast 到输出 dtype（True -> 1 / False -> 0）。
        if mat_np.dtype == np.bool_:
            if abs(alpha) <= _FP32_EPS or abs(beta) <= _FP32_EPS:
                base = np.zeros_like(self_np, dtype=bool)
                if abs(alpha) > _FP32_EPS:
                    base = np.any(mat_np & vec_np[np.newaxis, :], axis=-1)
                if abs(beta) > _FP32_EPS:
                    base = np.logical_or(base, self_np)
                return _util.cast_output_dtype(base, out_dtype) if out_dtype else base
            mm = np.any(mat_np & vec_np[np.newaxis, :], axis=-1)
            result = np.logical_or(self_np.astype(bool), mm)
            return _util.cast_output_dtype(result, out_dtype) if out_dtype else result

        # 3) 整数（非 bool）：int64 精确累加（输出为整型时按二进制比对）
        if np.issubdtype(mat_np.dtype, np.integer):
            ai, bi = int(alpha), int(beta)
            mm = np.matmul(mat_np.astype(np.int64), vec_np.astype(np.int64))
            result = bi * self_np.astype(np.int64) + ai * mm
            return _util.cast_output_dtype(result, out_dtype) if out_dtype else result

        # 4) cubeMathType 精度模拟
        mat_c, vec_c = _apply_cube_type(mat_np, vec_np, cubeMathType)

        # 5) 精度提升计算 (fp16/bf16 → fp32 累加，匹配 NPU 行为)
        if mat_c.dtype in (np.float16, _util.np_bfloat16):
            compute_dtype = np.float32
        else:
            compute_dtype = np.float64

        mm = np.matmul(mat_c.astype(compute_dtype), vec_c.astype(compute_dtype))

        # 6) beta * self + alpha * mm，单次 cast 到输出 dtype
        result = beta * self_np.astype(compute_dtype) + alpha * mm
        return _util.cast_output_dtype(result, out_dtype) if out_dtype else result


__spec__ = {
    "aclnnAddmv": "AclnnAddmvTestSpec",
}
