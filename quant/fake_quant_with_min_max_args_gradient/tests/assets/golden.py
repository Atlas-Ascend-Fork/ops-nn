#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK 多通路 golden + third_party for fake_quant_with_min_max_args_gradient.

两条通路（geir/kernel + tf E2E）各含独立 golden + third_party + tolerance。
本算子 CMakeLists 标注 ACLNNTYPE aclnn_exclude（无 aclnn API），故无 aclnn 通路。

竞品与精度策略：
  - 竞品为 TensorFlow ``tf.quantization.fake_quant_with_min_max_args_gradient``（README
    §调用说明 + op_graph proto 均明确对齐）。
  - 精度敏感点（kernel regbase.h §5/§6/§11）：
    * §6  clamp 用 >=/<= + Select（NaN-safe），非 Min/Max —— torch 用 (ge)&(le) 对齐
    * §5  y = grad * mask 乘法门控（非 Select），NaN grad 经 NaN*0=NaN 传播
    * §11 ±0 sign-bit OR 修复：Ascend Mul 可能丢失 -0 符号位，kernel 用
       vOut |= (vG & 0x80000000) 重新注入 gradients 符号位。对正常数值/NaN 幂等无副作用，
       仅当 gradients=-0.0 时有差异（kernel 保留 -0，普通 mul 产出 +0）。
       golden 用 bitwise 操作对齐此行为。
  - third_party 的 tf provider 用 class 形式调用
    ``tf.quantization.fake_quant_with_min_max_args_gradient`` 作性能对照。
  - dtype 仅 float32（README 约束：gradients/x/y 仅 FLOAT32）。
  - 精度标准: cross_check L1（mare≤5 / mere≤1.5 / rmse≤1.5）。
"""

import math

import numpy as np

# 注意：torch 必须延迟导入——xpu-server 的 tf 端点（跑 _TfCompetitor）无 torch，
# 模块级 import torch / torch.tensor 常量会让整个 spec 模块加载失败；torch 仅在
# golden 腿（kernel/E2E golden）执行时需要，均在函数内导入。

__spec__ = {
    # geir / kernel 通路：ttk kernel | GEIR，CSV op_name 查找，golden 收 numpy
    "fake_quant_with_min_max_args_gradient": "FakeQuantWithMinMaxArgsGradientKernelSpec",
    # E2E 通路：ttk e2e，CSV api_name = tf 竞品 API 全路径，golden 收 torch.Tensor
    "tf.quantization.fake_quant_with_min_max_args_gradient": "TfFakeQuantWithMinMaxArgsGradientSpec",
}


# ─── Nudge 预计算（host 侧，与 forward 算子相同，README 公式对齐）──────────


def _round_half_away(z):
    """std::round 语义：round half away from zero（torch 无直接对应）."""
    if z >= 0:
        return math.floor(z + 0.5)
    return math.ceil(z - 0.5)


def _nudge(min_v, max_v, num_bits, narrow_range):
    q_min = 1 if narrow_range else 0
    q_max = 2**num_bits - 1
    scale = np.float32(max_v - min_v) / np.float32(q_max - q_min)
    zero_point_from_min = np.float32(q_min) - np.float32(min_v) / scale
    if zero_point_from_min <= q_min:
        nudged_zero_point = float(q_min)
    elif zero_point_from_min >= q_max:
        nudged_zero_point = float(q_max)
    else:
        nudged_zero_point = float(_round_half_away(float(zero_point_from_min)))
    nudged_min = float(np.float32(q_min - nudged_zero_point) * scale)
    nudged_max = float(np.float32(q_max - nudged_zero_point) * scale)
    return nudged_min, nudged_max


# ─── 前向计算（kernel regbase.h §5/§6/§11 逐点对齐）──────────────────────


def _forward(gradients, x, nudged_min, nudged_max):
    """梯度计算：mask = (nudgedMin <= x <= nudgedMax) ? 1 : 0; y = gradients * mask.

    NaN 语义（kernel §5/§6）：
      - x 为 NaN：比较结果 false → mask=0 → y = grad*0（若 grad 有限则为 0，若 grad 为 NaN 则为 NaN）
      - gradients 为 NaN：经 NaN*mask 传播（NaN*1=NaN, NaN*0=NaN）

    ±0 sign-bit 修复（kernel §11）：Ascend Mul 可能丢失 -0 符号位，kernel 用
      vOut |= (vG & 0x80000000) 重新注入 gradients 符号位。golden 用 bitwise 对齐。
    """
    import torch

    sign_mask = torch.tensor(0x80000000, dtype=torch.uint32)  # float32 符号位
    # §6: NaN-safe mask（NaN 比较结果 false）
    mask = ((x >= nudged_min) & (x <= nudged_max)).float()
    # §5: 乘法门控
    y = gradients * mask
    # §11: ±0 sign-bit OR 修复 —— 保留 gradients 的符号位
    #   vOut_u32 |= (vG_u32 & 0x80000000)
    #   对正常数值/NaN 幂等无副作用；仅修复 Mul 丢失的 -0 符号
    y_u32 = y.view(torch.uint32)
    g_sign = gradients.view(torch.uint32) & sign_mask
    y_u32 = y_u32 | g_sign
    return y_u32.view(torch.float32)


# ─── third_party TF 竞品（class form）──────────────────────────────
# 唯一竞品为 TensorFlow tf.quantization.fake_quant_with_min_max_args_gradient。
# min/max 是标量属性，直接传入 TF API。server 侧 import tensorflow（standalone）。


class _TfCompetitor:
    """third_party tf — 调用 tf.quantization.fake_quant_with_min_max_args_gradient."""

    def __init__(self, *, min=-6.0, max=6.0, num_bits=8, narrow_range=False, **kwargs):
        self.min = float(min)
        self.max = float(max)
        self.num_bits = int(num_bits)
        self.narrow_range = bool(narrow_range)

    def __call__(self, gradients=None, x=None, inputs=None, **kwargs):
        # 参数名兼容两条路由：kernel/geir 派发名为 x，e2e TF 派发名为 inputs
        # （xpu-server 按 __call__ 签名绑定关键字参数，缺名即 400）
        if x is None:
            x = inputs
        import tensorflow as tf

        return [
            tf.quantization.fake_quant_with_min_max_args_gradient(
                gradients,
                x,
                self.min,
                self.max,
                num_bits=self.num_bits,
                narrow_range=self.narrow_range,
            )
        ]


# ─── 通路 1: geir / kernel (GEIR) ─────────────────────────────────
# golden 收 numpy.ndarray，third_party 收 torch.Tensor
# GEIR 复用 kernel 的 spec（README：GEIR 复用 Kernel 的 spec，无需额外编写）。


class FakeQuantWithMinMaxArgsGradientKernelSpec:
    """geir / kernel 通路 — golden 收 numpy.ndarray."""

    def golden(
        gradients, x, min=-6.0, max=6.0, num_bits=8, narrow_range=False, **kwargs
    ):
        import torch

        nudged_min, nudged_max = _nudge(min, max, num_bits, narrow_range)
        y = _forward(
            torch.as_tensor(gradients, dtype=torch.float32),
            torch.as_tensor(x, dtype=torch.float32),
            nudged_min,
            nudged_max,
        )
        return [y.numpy()]

    third_party = {
        "tf": _TfCompetitor,
    }
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }


# ─── 通路 2: torch/tf (E2E) ────────────────────────────────────────
# golden / third_party 收 torch.Tensor
# 本算子无 aclnn API（CMakeLists ACLNNTYPE aclnn_exclude），故无 aclnn 通路 spec。


class TfFakeQuantWithMinMaxArgsGradientSpec:
    """E2E (TF) 通路 — golden 收 torch.Tensor."""

    def golden(
        gradients, inputs, min=-6.0, max=6.0, num_bits=8, narrow_range=False, **kwargs
    ):
        # TTK e2e 按 TF API 签名以 6 个位置实参调用（gradients/inputs/min/max/num_bits/narrow_range），
        # 标量必须可按位置接收（keyword-only 会触发 TypeError，FQR-F-13）
        import torch

        nudged_min, nudged_max = _nudge(min, max, num_bits, narrow_range)
        g = gradients.float() if gradients.dtype != torch.float32 else gradients
        xt = inputs.float() if inputs.dtype != torch.float32 else inputs
        return [_forward(g, xt, nudged_min, nudged_max)]

    third_party = {
        "tf": _TfCompetitor,
    }
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }
