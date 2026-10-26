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
"""E2E-level golden for torch_npu.npu_fused_matmul."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util
from fmm_aclnn_golden import torch_fused_matmul_core

# HF32（hfloat32）用例：fp32 数据 + hf32 计算模式。torch 通路无 per-call
# cubeMathType 参数，需进程级开关 ALLOW_MATMUL_HF32（torch.npu.matmul.allow_hf32）
# 驱动 op-plugin 传 USE_HF32（否则 aclnnFusedMatmul 拒绝 fp32）；
# golden 侧同步按 cmt=3 做 hf32 截断建模。开关只影响 fp32，套件内其余
# fp16/bf16 用例为 no-op（CheckCubeMathTypeForMm 对 BF16+HF32 仅告警）。
_USE_HF32_FOR_FP32 = True


def _enable_hf32_for_fp32(*tensors):
    """fp32 输入时开启进程级 hf32 开关，返回 golden 应使用的 cube_math_type。"""
    if not _USE_HF32_FOR_FP32:
        return None
    if not any(t is not None and t.dtype == torch.float32 for t in tensors):
        return None
    try:
        import torch_npu  # noqa: F401

        torch.npu.matmul.allow_hf32 = True
    except Exception:  # noqa: BLE001  (CPU/离线环境无 torch_npu)
        pass
    return 3


class TorchNpuFusedMatmulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def customize_inputs(x1, x2, *, bias=None, x3=None, fused_op_type="", **kwargs):
        """执行前设置 hf32 开关：torch 通路无 per-call cubeMathType，只能靠进程级
        ALLOW_MATMUL_HF32 让 op-plugin 传 USE_HF32；该钩子在设备执行前被调用。"""
        _enable_hf32_for_fp32(x1, x2, bias, x3)

    @staticmethod
    def golden(x1, x2, *, bias=None, x3=None, fused_op_type="", **kwargs):
        """torch_npu.npu_fused_matmul: y = FUSED_OP(x1 @ x2 + bias, x3)."""
        out_dtype = torch.float32 if fused_op_type == "16cast32" else x1.dtype
        cube_math_type = _enable_hf32_for_fp32(x1, x2, bias, x3)
        return torch_fused_matmul_core(
            x1,
            x2,
            bias,
            x3,
            fused_op_type=fused_op_type,
            out_dtype=out_dtype,
            cube_math_type=cube_math_type,
        )


class CannOpsNnFusedMatmulTestSpec:
    @staticmethod
    def golden(
        x,
        x2,
        *,
        bias=None,
        x3=None,
        alpha=None,
        beta=None,
        fused_op_type="",
        **kwargs,
    ):
        """torch.ops.cann_ops_nn.fused_matmul: y = alpha*(x@x2) + beta*x3 (via aclnnFusedMatmulV2)."""
        out_dtype = torch.float32 if fused_op_type == "16cast32" else x.dtype
        alpha_value = 1.0 if alpha is None else float(alpha)
        beta_value = 1.0 if beta is None else float(beta)
        return torch_fused_matmul_core(
            x,
            x2,
            bias,
            x3,
            fused_op_type=fused_op_type,
            out_dtype=out_dtype,
            cube_math_type=None,
            alpha=alpha_value,
            beta=beta_value,
        )


__spec__ = {
    "torch_npu.npu_fused_matmul": "TorchNpuFusedMatmulTestSpec",
    "torch.ops.cann_ops_nn.fused_matmul": "CannOpsNnFusedMatmulTestSpec",
}
