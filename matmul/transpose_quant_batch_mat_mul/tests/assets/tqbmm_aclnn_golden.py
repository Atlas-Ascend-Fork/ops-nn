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
"""ACLNN-level golden for TransposeQuantBatchMatMul.

Delegates core computation to ``tqbmm_kernel_golden.TransposeQuantBatchMatMulTestSpec``.
Handles torch -> numpy conversion, FRACTAL_NZ -> ND conversion for the WeightNz
variant, aclnn camelCase parameter mapping and output dtype inference.

Quantization semantics (MXFP8 layout, e8m0 NaN passthrough, uint64 decode)
all live in the kernel-level golden - no input data is regenerated here so
that the golden always consumes exactly what the NPU received.
"""

import os
import sys


sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import tqbmm_kernel_golden as _kernel


class AclnnTransposeQuantBatchMatMulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        x1Scale=None,
        x2Scale=None,
        dtype=0,
        groupSize=0,
        permX1=None,
        permX2=None,
        permY=None,
        batchSplitFactor=1,
        out=None,
        **kwargs,
    ):
        x2_nd = _util.nz_input_to_nd(x2, kwargs.get("tensor_formats", ()), 1, n0=32)
        x1_np, x2_np, bias_np, s1_np, s2_np = _util.to_numpy(
            x1, x2_nd, bias, x1Scale, x2Scale
        )

        out_dtype = _util.dtype_to_str(out)

        perm_x1 = tuple(permX1) if permX1 is not None else (1, 0, 2)
        perm_x2 = tuple(permX2) if permX2 is not None else (0, 1, 2)
        perm_y = tuple(permY) if permY is not None else (1, 0, 2)

        temp_kwargs = dict(kwargs)
        if out_dtype:
            temp_kwargs["output_dtypes"] = [out_dtype]

        return _kernel.TransposeQuantBatchMatMulTestSpec.golden(
            x1_np,
            x2_np,
            bias_np,
            s1_np,
            s2_np,
            dtype=dtype,
            group_size=groupSize,
            perm_x1=perm_x1,
            perm_x2=perm_x2,
            perm_y=perm_y,
            batch_split_factor=batchSplitFactor,
            **temp_kwargs,
        )


class AclnnTransposeQuantBatchMatMulWeightNzTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        bias=None,
        x1Scale=None,
        x2Scale=None,
        dtype=0,
        groupSize=0,
        permX1=None,
        permX2=None,
        permY=None,
        batchSplitFactor=1,
        out=None,
        **kwargs,
    ):
        """aclnn 输入定制：NZ x2 规范化（in-place 覆写 storage，框架不捕获返回值）。"""
        _util.prepare_nz_input(x2, kwargs.get("tensor_formats", ()), 1, n0=32)

    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        x1Scale=None,
        x2Scale=None,
        dtype=0,
        groupSize=0,
        permX1=None,
        permX2=None,
        permY=None,
        batchSplitFactor=1,
        out=None,
        **kwargs,
    ):
        return AclnnTransposeQuantBatchMatMulTestSpec.golden(
            x1,
            x2,
            bias=bias,
            x1Scale=x1Scale,
            x2Scale=x2Scale,
            dtype=dtype,
            groupSize=groupSize,
            permX1=permX1,
            permX2=permX2,
            permY=permY,
            batchSplitFactor=batchSplitFactor,
            out=out,
            **kwargs,
        )


__spec__ = {
    "aclnnTransposeQuantBatchMatMul": "AclnnTransposeQuantBatchMatMulTestSpec",
    "aclnnTransposeQuantBatchMatMulWeightNz": "AclnnTransposeQuantBatchMatMulWeightNzTestSpec",
}
