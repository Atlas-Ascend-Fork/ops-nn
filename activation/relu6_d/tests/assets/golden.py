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

import math

import numpy as np

__golden__ = {"kernel": {"Relu6D": "relu6d_golden", "relu6_d": "relu6d_golden"}}

_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L0"},
    "float16": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
    "int32": {"standard": "binary_equal"},
}

_INT32_LIMIT = np.iinfo(np.int32)


def _upper_threshold(scale, int_path):
    """
    Upper bound 6*scale consumed by the kernel.

    The product is evaluated in float32 to mirror the operator: Tiling stores
    a float32 upperThreshold (6.0f * scale, see relu6_d_tiling_arch35.cpp) and
    the kernel casts it toward zero.  A float64 product can differ by one
    after truncation for scales whose 6*scale crosses an integer boundary
    under float32 rounding (e.g. 1.1666666).

    The README constrains the int32 threshold to truncation toward zero
    (``6*scale`` truncated).  A non-finite ``scale`` is rejected by Tiling
    (``std::isfinite`` check) for every dtype, so its threshold is never
    consumed; saturate it only so reference generation cannot raise for the
    framework's exploratory rows.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.float32(6.0) * np.float32(scale))
    if not int_path:
        return value
    if math.isnan(value):
        return 0
    if math.isinf(value):
        return int(_INT32_LIMIT.max if value > 0 else _INT32_LIMIT.min)
    return int(math.trunc(value))


def relu6d_golden(x, scale=1.0, **kwargs):
    """
    Kernel golden for relu6_d.
    All the parameters (names and order) follow @relu6d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: full_soc_version, short_soc_version, testcase_name,
        {input,output}_{dtypes,ori_shapes,formats,ori_formats}.

    Semantics: y = min(max(x, 0), 6*scale) (element-wise, same shape and dtype).
    FP16/BF16 are promoted to FP32 before the computation and cast back; INT32
    keeps its own precision and truncates 6*scale toward zero, matching
    source/activation/relu6_d/README.md.
    """
    import torch

    ori_dtype = x.dtype
    ori_dtype_str = str(ori_dtype).lower()
    is_bf16 = "bfloat16" in ori_dtype_str
    is_fp16 = ori_dtype == np.float16

    if is_bf16 or is_fp16:
        x_compute = x.astype(np.float32)
    else:
        x_compute = x

    upper = _upper_threshold(scale, "int32" in ori_dtype_str)

    x_torch = torch.from_numpy(np.ascontiguousarray(x_compute))
    result = torch.clamp(x_torch, min=0, max=upper)
    return [result.numpy().astype(ori_dtype)]


class _Relu6DCompose:
    """Torch reference for the kernel's scaled clamp semantics."""

    def __call__(self, x, scale=1.0, **kwargs):
        del kwargs
        import torch

        output_dtype = x.dtype
        compute = x
        if output_dtype in (torch.float16, torch.bfloat16):
            compute = x.to(torch.float32)
        upper = _upper_threshold(scale, output_dtype == torch.int32)
        return [torch.clamp(compute, min=0, max=upper).to(output_dtype)]


class Relu6DKernelSpec:
    @staticmethod
    def golden(x, scale=1.0, **kwargs):
        return relu6d_golden(x, scale=scale, **kwargs)

    third_party = {"torch": _Relu6DCompose}
    tolerance = _TOL_KERNEL


__spec__ = {"Relu6D": "Relu6DKernelSpec", "relu6_d": "Relu6DKernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)。
# aclnn: 未交付——算子 CMakeLists 设置 ACLNNTYPE aclnn_exclude 且无 op_api/ 目录。
# Torch E2E / torch 图模式 / TensorFlow E2E: 未交付——无 torch_extension、
#        无 framework tf plugin、已安装包无对应符号，故不声明 torch_graph / tf spec。
# 融合 pass / ONNX: 未交付。
