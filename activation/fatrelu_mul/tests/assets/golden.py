#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

__spec__ = {
    "aclnnFatreluMul": "FatreluMulTestSpec",
    "fatrelu_mul": "FatreluMulTestSpec",
    "FatreluMul": "FatreluMulTestSpec",  # GE类型名变体(I-0028): board st-design按GE op type(REG_OP(FatreluMul), fatrelu_mul_proto.h:35)生成CamelCase op_name;只追加不删除
}

import numpy


def _to_torch(t):
    import torch

    if isinstance(t, torch.Tensor):
        return t
    if isinstance(t, numpy.ndarray):
        if "bfloat16" in str(t.dtype):
            return torch.from_numpy(t.view(numpy.uint16)).view(torch.bfloat16)
        return torch.from_numpy(t)
    return torch.as_tensor(t)


class FatreluMulTestSpec:
    def golden(x, threshold, out=None, **kwargs):
        import torch

        xt = _to_torch(x)
        tt = _to_torch(threshold)
        d = xt.shape[-1] // 2
        gate = xt[..., :d]
        up = xt[..., d:]
        if tt.numel() < 1:
            # 空threshold守卫(T14/golden_fix#3,对齐tiling契约: numel!=1一律拒绝,
            # fatrelu_mul_tiling_arch35.cpp:219-226): 非空x无可定义期望值→按拒绝语义
            # 报错; 空x短路路径(API成功,输出空)t不参与任何元素比较→按实际行为返回空输出
            if gate.numel() > 0:
                raise ValueError("threshold numel 0 must be 1 (single-element tensor)")
            return [gate * up]
        t = tt.reshape(-1)[0]
        # GT 极性(2026-10-24用户裁定对齐arch22与vLLM CUDA, Issue #7274): gate>t保留,否则置0;
        # NaN>t恒False→NaN置0; 非NaN域与原LE形逐位一致(x==t仍→0, x>t仍→x)
        gate_act = torch.where(gate > t, gate, torch.zeros_like(gate))
        return [gate_act * up]

    class TorchFatreluMul:
        def __call__(self, x, threshold, **kwargs):
            import torch

            gate, up = torch.chunk(x, 2, dim=-1)
            t = threshold.reshape(-1)[0]
            gate_act = torch.where(gate > t, gate, torch.zeros_like(gate))
            return [torch.mul(gate_act, up)]

    third_party = {"torch": TorchFatreluMul}

    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
    }
