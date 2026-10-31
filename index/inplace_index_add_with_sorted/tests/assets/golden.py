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

import numpy as np
import torch


__golden__ = {
    "kernel": {"inplace_index_add_with_sorted": "inplace_index_add_with_sorted_golden"}
}

_LOW_PRECISION = {"fp16", "bf16", "float16", "bfloat16"}
_DTYPE_TOKENS = {"fp16", "bf16", "float16", "bfloat16", "float32"}
_declared_dtype_cache = {}


def _declared_dtype(testcase_name):
    """按 testcase_name 还原用例声明的输出 dtype 名（fp16 / bf16 / float32）。

    本算子走 mix_tolerance（浮点族默认判据），会触发 golden_mode=Promote：
    TTK 把 fp16/bf16 输入抬到 fp32 再交给 golden，并把 kwargs["output_dtypes"]
    同步抬成 fp32，因此无法从入参数组还原用例声明的类型，只能从 testcase_name
    兜底解析（用例命名规范为 IASorted_{fp16|bf16}_...）。
    """
    if testcase_name in _declared_dtype_cache:
        return _declared_dtype_cache[testcase_name]
    resolved = None
    for seg in str(testcase_name).split("_"):
        if seg in _DTYPE_TOKENS:
            resolved = seg
            break
    if resolved is None:
        resolved = "float32"
    _declared_dtype_cache[testcase_name] = resolved
    return resolved


def inplace_index_add_with_sorted_golden(
    var, value, sorted_indices, pos, alpha=None, *, axis=0, **kwargs
):
    """
    Golden function for inplace_index_add_with_sorted.
    All the parameters (names and order) follow @inplace_index_add_with_sorted_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Formula:
        var[sorted_indices[i], ..., :] += alpha * value[pos[i], ..., :]
    where sorted_indices are the sorted target indices along `axis`, and pos maps
    each sorted position back to the original update row of value:
        update_for_sorted_i = value[pos[i]]

    Equivalent to reordering value by pos, then index_add along axis:

        out = var.clone()
        out.index_add_(axis, sorted_indices, value[pos], alpha=alpha)

    精度说明：
        在高精度域（float64）累加、结束时只做一次 RNE 舍入回声明 dtype。
        - 直接用 fp16/bf16 调用 torch 会在中间结果处提前溢出（inf + x 恒为
          inf），即使最终数学和并未越过该类型上界。
        - 返回类型必须与用例声明一致（fp16/bf16），否则 TTK 会按
          finfo.max(65504) 重新裁 inf，把 (65504, 65520) 这段本应舍入回
          65504 的值误判为溢出——这正是极值用例
          (var=65504 + alpha*value) 精度误报的根因。

    Args:
        var: numpy.ndarray, tensor to be accumulated into (in-place semantics).
        value: numpy.ndarray, update tensor whose rows are referenced by pos.
        sorted_indices: numpy.ndarray (int32), sorted target indices along axis.
        pos: numpy.ndarray (int32), inverse permutation mapping sorted order to
            value rows.
        alpha: numpy.ndarray or None, scalar scaling factor on value. Default 1.
        axis: int, dimension of var along which indices apply, only 0.
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (numpy.ndarray) of the same shape and dtype as var.
    """
    dtype_name = _declared_dtype(kwargs.get("testcase_name"))
    # 高精度域累加、末尾单次 RNE 舍入到声明 dtype。
    acc_dtype = np.float64 if dtype_name in _LOW_PRECISION else np.asarray(var).dtype

    var_t = torch.from_numpy(np.asarray(var, dtype=acc_dtype))
    value_t = torch.from_numpy(np.asarray(value, dtype=acc_dtype))
    idx_t = torch.from_numpy(np.asarray(sorted_indices)).to(torch.int64)
    pos_t = torch.from_numpy(np.asarray(pos)).to(torch.int64)

    alpha_val = 1.0
    if alpha is not None:
        alpha_val = float(np.asarray(alpha).item())

    # 按 pos 重排 value，再按 sorted_indices 累加到 var
    value_reordered = value_t[pos_t] * alpha_val
    out = var_t.clone()
    out.index_add_(axis, idx_t, value_reordered)

    # 结束时仅舍入一次；fp16 下 |x| >= 65520 才溢出为 inf，其余（含 65504 附近的
    # 小增量）RNE 回 65504，与 NPU fp16 累加语义一致。
    with np.errstate(over="ignore"):
        if dtype_name in ("bf16", "bfloat16"):
            import ml_dtypes

            result_np = out.numpy().astype(ml_dtypes.bfloat16)
        elif dtype_name in ("fp16", "float16"):
            result_np = out.numpy().astype(np.float16)
        else:
            result_np = out.numpy().astype(np.dtype(dtype_name))
    return result_np
