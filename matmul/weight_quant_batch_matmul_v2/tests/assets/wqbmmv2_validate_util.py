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
WQBMMV2 950 (DAV_3510) 参数约束校验（op 专属，原 matmul_quant_util 内迁出）。

仅告警、不修改张量。
"""

import os
import sys
import warnings
from importlib import import_module

_common = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../common/tests/st/arch35"
)
if _common not in sys.path:
    sys.path.insert(0, _common)

_util = import_module("matmul_golden_util")


def validate_wqbmmv2_constraints(
    x,
    weight,
    antiquant_scale,
    antiquant_offset=None,
    quant_scale=None,
    quant_offset=None,
    bias=None,
    **kwargs,
):
    """Validate WQBMMV2 950 (DAV_3510) parameter constraints and emit warnings.

    Does not modify any tensors. Only warns about configurations that would
    cause NPU parameter validation failures on Ascend 950.

    Supports both kernel (input_formats) and aclnn (tensor_formats) kwargs.
    """
    testcase_name = kwargs.get("testcase_name", "unknown")
    x_dtype = _util.dtype_to_str(x)
    weight_dtype = _util.dtype_to_str(weight)
    scale_dtype = _util.dtype_to_str(antiquant_scale)
    offset_dtype = _util.dtype_to_str(antiquant_offset)
    bias_dtype = _util.dtype_to_str(bias)

    formats = kwargs.get("input_formats") or kwargs.get("tensor_formats")
    weight_format = formats[1] if formats and len(formats) > 1 else "ND"

    is_fp4_weight = weight_dtype in ("float32",)
    is_fp8_weight = weight_dtype in ("float8_e4m3fn", "hifloat8")
    is_int8_weight = weight_dtype == "int8"

    if quant_scale is not None:
        warnings.warn(
            f"[{testcase_name}] 950 does not support quantScale, must be null. "
            f"Got dtype={_util.dtype_to_str(quant_scale)}"
        )
    if quant_offset is not None:
        warnings.warn(
            f"[{testcase_name}] 950 does not support quantOffset, must be null. "
            f"Got dtype={_util.dtype_to_str(quant_offset)}"
        )

    if (is_fp4_weight or is_fp8_weight) and antiquant_offset is not None:
        warnings.warn(
            f"[{testcase_name}] 950 requires antiquantOffset=null when weight is "
            f"{weight_dtype}, got dtype={offset_dtype}"
        )

    if antiquant_scale is not None and scale_dtype != "float8_e8m0":
        if x_dtype is not None and scale_dtype != x_dtype:
            warnings.warn(
                f"[{testcase_name}] 950 requires antiquantScale dtype ({scale_dtype}) "
                f"to match x dtype ({x_dtype}) when not float8_e8m0"
            )

    if antiquant_offset is not None and x_dtype is not None:
        if offset_dtype != x_dtype:
            warnings.warn(
                f"[{testcase_name}] 950 requires antiquantOffset dtype ({offset_dtype}) "
                f"to match x dtype ({x_dtype})"
            )

    if bias is not None and x_dtype == "bfloat16":
        if (is_fp4_weight or is_fp8_weight) and bias_dtype != "bfloat16":
            warnings.warn(
                f"[{testcase_name}] 950 requires bias dtype=bfloat16 when weight is "
                f"{weight_dtype} and x=bfloat16, got bias dtype={bias_dtype}"
            )
        if is_int8_weight and weight_format == "FRACTAL_NZ" and bias_dtype != "float32":
            warnings.warn(
                f"[{testcase_name}] 950 requires bias dtype=float32 when weight is "
                f"int8+FRACTAL_NZ and x=bfloat16, got bias dtype={bias_dtype}"
            )
