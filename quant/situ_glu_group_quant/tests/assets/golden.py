#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""situ_glu_group_quant Golden（TTK __spec__ 模式；aclnn + kernel/geir 双通路）。

参考实现逐位复刻 kernel 语义（真机验证口径，200 例门槛 + 4 probe + 22 例 recipes 复检）：
  1. activate_left: true -> gate=前半/up=后半（recipes 原方向）；false 反之
  2. 激活链逐指令复刻 SituActivationVF：branch = fl32(s)/(1+Exp(v*fl32(-2/fl32(s))))*2
     + (-fl32(s))；sigmoid 用除法（不用倒数乘）；标量按 host fp32 精度预折算
  3. linear_beta <= 0 时 up 直通（kernel TPL_LINEAR_DISABLED 口径）
  4. weight 在 fp32 域相乘后做唯一的退出转换；exit cast = CAST_RINT + SatMode::SAT：
     NaN -> +0、±inf 与有限溢出 -> ±T 最大有限值（onboard probe + L1_140 实证）
  5. 每 32 元素一组取 amax（硬件归约忽略 NaN——NaN 已在 exit cast 清为 +0，天然
     不进 amax；混组按 normal 元素正常量化，全 NaN 组 amax=0）-> e8m0 =
     clamp(E-8, 0, 255)；E <= 8 -> recip = 0
  6. fp16 路径：half 值经 CAST_TRUNC 真值转 bf16 后取指数域（零 -> E=0，次正规
     -> E = 103+floor(log2(m))，正规 -> E = e16+112）
  7. y = e4m3fn（SAT 钳 ±448），NaN 元素逐个 -> +0（组不整体置零）；y_scale 每行
     gc 字节，奇 gc 行尾 pad 字节 kernel 不写（Compact CopyOut gap）：golden 按
     TTK 预填约定发 127，pre_compare 钩子再把两侧 pad 同置 0——判比与采集装置
     预填解耦（aclnn/kernel/GE 通用）

用例约束：EPR = x.shape[-1]/2 须为 32 的倍数（tiling 防呆口径）；group_index 由
customize_inputs 修为满额（padding 行契约由 bench 0xA5 哨兵对拍覆盖）。
"""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import numpy as np
import torch

__spec__ = {
    # aclnnSituGluGroupQuant: 真实 aclnn 接口符号（op_api/aclnn_situ_glu_group_quant.h）
    "aclnnSituGluGroupQuant": "AclnnSituGluGroupQuantSpec",
    # situ_glu_group_quant: kernel/L1 表（ST 单算子层）的算子名，属性为
    # proto snake_case 口径，参考与 aclnn 同源
    "situ_glu_group_quant": "SituGluGroupQuantKernelSpec",
    # e2e: torch 扩展（torch_extension/csrc 注册 cann_ops_nn.situ_glu_group_quant）。
    # TTK e2e 按 api_name 精确匹配，短名与 torch.ops 全名双注册
    # （e2e CSV 用 torch.ops.cann_ops_nn.situ_glu_group_quant；TorchAir 图模式
    #  转换器同样按 torch.ops 全名挂接）。
    "cann_ops_nn.situ_glu_group_quant": "SituGluGroupQuantE2ESpec",
    "torch.ops.cann_ops_nn.situ_glu_group_quant": "SituGluGroupQuantE2ESpec",
}

# 输入修补钩子（参照 fused_adam / apply_adam_w_quant 的 customize_inputs）
__input__ = {
    "aclnn": {"aclnnSituGluGroupQuant": "customize_inputs"},
    "kernel": {"situ_glu_group_quant": "customize_inputs"},
    "e2e": {
        "cann_ops_nn.situ_glu_group_quant": "customize_inputs_e2e",
        "torch.ops.cann_ops_nn.situ_glu_group_quant": "customize_inputs_e2e",
    },
}

_GROUP = 32  # MXFP8 group size
_E4M3_EMAX = 8  # OCP scale 偏移


def _to_torch(t):
    if isinstance(t, torch.Tensor):
        return t
    arr = np.asarray(t)
    if arr.dtype.name == "bfloat16":  # ml_dtypes bfloat16（kernel/geir 路径输入）
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.as_tensor(arr)


def _reference(x, group_index, beta, linear_beta, activate_left=False, weight=None):
    """Core reference. x: torch tensor on CPU of any rank; returns (y, y_scale)."""
    if not torch.is_tensor(x):
        x = _to_torch(x)
    orig_shape = tuple(x.shape)
    last = orig_shape[-1]
    epr = last // 2
    rows = x.numel() // last
    in_dtype = x.dtype

    xf = x.reshape(rows, last).float()
    # activate_left (owner decision, mirrors situ_mx_quant normal path):
    #   true  -> gate = first half, up = second half (recipes original)
    #   false -> gate = second half, up = first half (situ normal default, new)
    if activate_left:
        gate = xf[:, :epr]
        up = xf[:, epr:]
    else:
        gate = xf[:, epr:]
        up = xf[:, :epr]

    # Activation mirrors the kernel instruction chain EXACTLY (SituActivationVF):
    #   tmp = Exp(g * neg2OverBeta); tmp += 1
    #   branch = beta / tmp; branch *= 2; branch += negBeta     (= 2*beta/(1+e) - beta)
    #   sig = 1 + Exp(g * -1); branch = branch / sig            (DIV, not mul-recip)
    # Host scalars are fp32-rounded the same way the kernel computes them:
    #   neg2OverBeta = fl32(-2.0 / fl32(beta)), negBeta = -fl32(beta)
    # （sane 数据下与代数形态 beta*(2/(1+e)-1) 逐位一致；病态区间 |v|<<beta 的消除
    #  放大下指令序形态才与设备同源——200 例门槛分型验证）
    # KNOWN LIMITATION: torch.exp is correctly rounded while the device Exp
    # instruction is a hardware approximation; for inputs whose fp32 activation
    # lands within a few e-8 of an e4m3 rounding midpoint (verified example:
    # bench case c06/c07 element [25,553]) the exit cast can round one code
    # apart. Both the port and the recipes original produce the SAME byte on
    # device, so such elements are expected CPU golden vs NPU divergences.
    b32 = np.float32(beta)
    neg2OverBeta = np.float32(-2.0) / b32
    negBeta = -b32
    gt = 1.0 + torch.exp(gate * float(neg2OverBeta))
    gb = (torch.full_like(gate, float(b32)) / gt * 2.0) + float(negBeta)
    gb = gb / (1.0 + torch.exp(gate * -1.0))

    # linear_beta <= 0 disables the up-branch tanh soft-clamp: u = up
    # passthrough (kernel TPL_LINEAR_DISABLED caliber). Enabled path mirrors
    # the kernel: ub = fl32(lb)/(1+Exp(u*neg2OverLb))*2 + (-fl32(lb))
    if linear_beta <= 0:
        ub = up.reshape(rows, epr)
    else:
        a32 = np.float32(linear_beta)
        neg2OverAlpha = np.float32(-2.0) / a32
        negAlpha = -a32
        ut = 1.0 + torch.exp(up * float(neg2OverAlpha))
        ub = (torch.full_like(up, float(a32)) / ut * 2.0) + float(negAlpha)

    # weight scale is applied in fp32 BEFORE the single exit rounding to T
    act_f32 = gb.reshape(rows, epr) * ub.reshape(rows, epr)
    if weight is not None:
        w = _to_torch(weight).float().reshape(-1)
        act_f32 = act_f32.reshape(rows, epr) * w[:rows].unsqueeze(-1)

    # kernel 退出转换为 CAST_RINT + SatMode::SAT（onboard probe + L1_140 实证）：
    #   NaN -> +0（后续 amax 天然忽略、量化逐元素得 +0，混组不整体置零）
    #   ±inf 与有限溢出 -> ±T 最大有限值（w=±inf 退化路径 E=254 -> e8m0=246）
    # torch 直转的 inf 保持/NaN 传播 amax 的旧模型与设备不符。
    fin_max = torch.finfo(in_dtype).max
    act = (
        torch.nan_to_num(act_f32, nan=0.0, posinf=fin_max, neginf=-fin_max)
        .clamp(-fin_max, fin_max)
        .to(in_dtype)
    )
    actf = act.float()

    # group-wise amax -> bf16 exponent field E (bits[14:7], kernel mask 0x7F80).
    # NaN 已在 exit cast 清零，amax 无需（也不会）见到 NaN。
    gc = epr // _GROUP
    # scaleColNum = gc 偶数化（GM 行间 interleave-2）：奇 gc 时行尾 1 字节 pad
    # 由 Compact DataCopyPad 的 dstStride 留出、kernel 不写——保留输出 buffer
    # 初值；golden 按 TTK 预填约定发 127，pre_compare 再把两侧同置 0。
    scale_col = gc + (gc & 1)
    amax_g = actf.reshape(rows, gc, _GROUP).abs().amax(dim=-1)
    if in_dtype == torch.float16:
        # kernel half 分支 = Cast<half -> bf16, CAST_TRUNC> 后取 bf16 指数字段：
        #   零 (bits==0) -> bf16 0x0000 -> E=0（e8m0=0、recip=0）
        #   次正规 (e16==0, m!=0) -> 按值规范化 E = 103 + floor(log2(m))
        #   正规 -> E = e16 + 112（值保持；截断不进位不影响指数）
        bits16 = amax_g.to(torch.float16).view(torch.int16).to(torch.int32) & 0x7FFF
        e16 = (bits16 >> 10) & 0x1F
        m16 = bits16 & 0x03FF
        is_zero = bits16 == 0
        is_sub = (e16 == 0) & ~is_zero
        h = torch.floor(torch.log2(m16.to(torch.float32).clamp(min=1.0)))
        exp_bits = torch.where(
            is_zero,
            torch.zeros_like(e16),
            torch.where(is_sub, 103 + h.to(torch.int32), e16 + 112),
        )
    else:
        exp_bits = (
            amax_g.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0x7F80
        ) >> 7

    e8m0 = torch.clamp(exp_bits - _E4M3_EMAX, min=0, max=255).to(torch.uint8)

    # kernel halfScale = scaleBias(0x7f00, bf16 of 2^127) - sharedExp((E-8)<<7),
    # i.e. bf16 exponent field (255-E) -> value 2^(128-E)*2^7 = 2^(135-E).
    # Strictly positive shared exponent (E > 8) required, else recip = 0.
    e = exp_bits.to(torch.float32)
    recip = torch.where(
        exp_bits > _E4M3_EMAX,
        torch.exp2(127 + _E4M3_EMAX - e),
        torch.zeros_like(e),
    )  # (rows, gc)

    # quantize: bf16 path rounds the product back to bf16 (kernel T-domain mul);
    # fp16 path keeps the fp32 product (ComputeDataVF half branch casts to float)
    prod = actf.reshape(rows, gc, _GROUP) * recip.unsqueeze(-1)
    if in_dtype == torch.bfloat16:
        prod = prod.to(torch.bfloat16).float()
    # e4m3fn cast with SATURATION: the kernel quant cast runs in SatMode::SAT
    # (clamps to +-448 on overflow), while torch's fp32->float8_e4m3fn cast maps
    # out-of-range values to NaN. Clamp in the fp32 domain first; NaN cannot
    # occur here (act/recip 均已被上游清理为有限值)。
    y = prod.reshape(rows, epr)
    y = y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    # y_scale: scaleColNum e8m0 bytes per row, viewed as [scaleColNum/2, 2]. The
    # e8m0 byte IS the payload: reinterpret the bits (uint8 -> float8_e8m0fnu),
    # never a numeric cast (e.g. 118 would round to the nearest power of two,
    # corrupting it). Odd gc rows carry one unwritten tail pad byte: TTK 对 e8m0
    # 输出 host 数组按 1.0 哨兵预填（byte 0x7F），kernel 的 Compact DataCopyPad
    # 不写行尾 gap 字节，buffer 初值得以保留；判比侧由 pre_compare 掩膜解耦。
    if scale_col > gc:
        e8m0 = torch.cat(
            [e8m0, torch.full((rows, scale_col - gc), 127, dtype=e8m0.dtype)], dim=1
        )
    mx = e8m0.reshape(rows, scale_col // 2, 2).view(torch.float8_e8m0fnu)

    out_shape = orig_shape[:-1]
    y = y.reshape(*out_shape, epr)
    mx = mx.reshape(*out_shape, scale_col // 2, 2)
    return y, mx


def customize_inputs(x, weight, groupIndex=None, *unused_attrs, **kwargs):
    """TTK 输入修补钩子（参照 fused_adam 的 customize_inputs）。

    修补规则（随机输入 -> 合法输入）：
    - group_index 修为满额分布：总行数 rows = numel(x) / x.shape[-1]，
      按当前 token 槽位数均分（rows 均分有余则并入最后一个专家），
      保证 sum(tokens) == rows -> 无 padding 行（TTK 输出 buffer 不受
      稀疏契约控制，padding 场景由 bench 0xA5 哨兵对拍覆盖）。
    入参/返回均为 aclnn 头的输入序列（不含输出）；torch 路径原地 copy_，
    numpy 路径原地 [...] 写回。
    """
    xt = _to_torch(x)
    rows = xt.numel() // xt.shape[-1]

    if groupIndex is None:
        return x, weight, groupIndex
    tk = _to_torch(groupIndex)
    n = tk.numel()
    base = rows // n
    vals = [base] * n
    vals[-1] += rows - base * n
    tgt = torch.tensor(vals, dtype=tk.dtype)

    if torch.is_tensor(groupIndex):
        groupIndex.copy_(tgt.to(groupIndex.dtype))
        return x, weight, groupIndex
    arr = np.asarray(groupIndex)
    arr[...] = tgt.numpy().astype(arr.dtype)
    return x, weight, arr


def _pre_compare_mask_scale_pad(y=None, scale=None, *rest):
    """判比前豁免奇 gc 的 y_scale 行尾 pad 字节。

    算子契约：scaleColNum = gc 偶数化，kernel 仅写每行前 gc 字节，行尾 gap
    字节保留输出 buffer 初值——该初值随执行环境而异（TTK kernel 模式实测
    0x7F、GE 为分配器残留），任何固定建模都只对单一 harness 成立。此处将
    设备/golden 两侧的该字节同置 0，使判比与 buffer 预填解耦。gc 从 y 末维
    推得（epr = y.shape[-1] -> gc = epr/32）；偶 gc 无 pad，原样返回。
    契约：入参 (y, y_scale, y_golden, y_scale_golden)，返回同长度序列。
    """
    if y is None or scale is None or len(rest) < 2:
        return None
    y_g, s_g = rest[-2], rest[-1]
    try:
        epr = int(y.shape[-1])
    except Exception:
        return None
    gc = epr // _GROUP
    scn = gc + (gc & 1)
    if scn == gc:
        return [y, scale, y_g, s_g]

    def _mask(arr):
        # 判比缓冲区为只读，生成掩膜后的可写副本返回（pre_compare 契约：
        # 返回变换序列，原地改写不可行）
        if isinstance(arr, torch.Tensor):
            u8 = arr.detach().view(torch.uint8).clone().reshape(-1, scn)
            u8[:, gc:] = 0
            return u8.reshape(arr.shape).view(arr.dtype)
        a = np.asarray(arr)
        u8 = np.ascontiguousarray(a.view(np.uint8)).reshape(-1, scn).copy()
        u8[:, gc:] = 0
        return u8.reshape(a.shape).view(a.dtype)

    try:
        scale_m, s_g_m = _mask(scale), _mask(s_g)
    except Exception:
        return None
    return [y, scale_m, y_g, s_g_m]


class AclnnSituGluGroupQuantSpec:
    """ACLNN 路径 — (x, group_index) -> (y, y_scale)。"""

    @staticmethod
    def golden(
        x,
        weight=None,
        groupIndex=None,
        beta=1.0,
        linear_beta=0.0,
        activateLeft=False,
        axis=-1,
        quantMode=0,
        dstType=36,
        dstTypeMax=0.0,
        roundModeOptional="rint",
        groupListType=1,
        yOut=None,
        yScaleOut=None,
        **kwargs,
    ):
        if axis != -1:
            raise ValueError(f"axis only supports -1 currently, but got {axis}")
        if quantMode != 0:
            raise ValueError(
                f"quant_mode only supports 0 currently, but got {quantMode}"
            )
        if dstType != 36:
            raise ValueError(f"dst_type only supports 36 currently, but got {dstType}")
        if dstTypeMax < 0:
            raise ValueError(f"dst_type_max must be >= 0, but got {dstTypeMax}")
        if roundModeOptional != "rint":
            raise ValueError(
                f"round_mode only supports rint currently, but got {roundModeOptional}"
            )
        if groupListType != 1:
            raise ValueError(
                "group_list_type=0 (cumsum mode) is reserved and not supported yet, pass 1, but got "
                f"{groupListType}"
            )
        y, mx = _reference(
            x,
            groupIndex,
            float(beta),
            float(linear_beta),
            bool(activateLeft),
            weight,
        )
        return [y, mx]

    # 逐位复刻参考（与原 recipes 算子 22 case 字节对拍验证）：量化输出按位比对，
    # 不走 mix_tolerance 的 Promote 升精度 golden（bf16->fp32 会在 amax 指数
    # 边界引入假差异，weight 逐 token 缩放下被放大）。
    tolerance = {
        "float8_e4m3fn": {"standard": "binary_equal"},
        "float8_e8m0": {"standard": "binary_equal"},
    }
    customize_inputs = customize_inputs
    pre_compare = staticmethod(_pre_compare_mask_scale_pad)


def customize_inputs_e2e(x, weight=None, group_index=None, **kwargs):
    """E2E 输入修补钩子（torch 扩展形参名 group_index，与 aclnn 的 groupIndex 区分）。"""
    return customize_inputs(x, weight, group_index, **kwargs)


class SituGluGroupQuantE2ESpec(AclnnSituGluGroupQuantSpec):
    """E2E 路径（torch.ops.cann_ops_nn.situ_glu_group_quant）— torch.Tensor 入参。

    torch 扩展签名（torch_extension/csrc/situ_glu_group_quant.cpp + .py schema）：
        situ_glu_group_quant(x, weight=None, group_index=None, *,
            beta=1.0, linear_beta=0.0, activate_left=False, axis=-1,
            quant_mode=0, dst_type=36, dst_type_max=0.0,
            round_mode="rint", group_list_type=0)
    形参口径为 snake_case 且与 aclnn 顺序不同（weight/group_index 位置一致），
    统一转发到 aclnn 参考实现；group_list_type 默认 0（cumsum 预留）会被算子
    拒收，e2e 用例须显式传 1。
    """

    @staticmethod
    def golden(
        x,
        weight=None,
        group_index=None,
        beta=1.0,
        linear_beta=0.0,
        activate_left=False,
        axis=-1,
        quant_mode=0,
        dst_type=36,
        dst_type_max=0.0,
        round_mode="rint",
        group_list_type=1,
        **kwargs,
    ):
        return AclnnSituGluGroupQuantSpec.golden(
            x,
            weight,
            group_index,
            beta=beta,
            linear_beta=linear_beta,
            activateLeft=activate_left,
            axis=axis,
            quantMode=quant_mode,
            dstType=dst_type,
            dstTypeMax=dst_type_max,
            roundModeOptional=round_mode,
            groupListType=group_list_type,
        )

    # 判据与掩膜与 aclnn 同源（e4m3/e8m0 binary_equal + 奇 gc pad 掩膜）
    tolerance = AclnnSituGluGroupQuantSpec.tolerance
    pre_compare = staticmethod(_pre_compare_mask_scale_pad)
    customize_inputs = staticmethod(customize_inputs_e2e)


class SituGluGroupQuantKernelSpec(AclnnSituGluGroupQuantSpec):
    """Kernel/L1 路径 — (x, weight, group_index) -> (y, y_scale)。

    L1 表（ST 单算子层）的属性走 proto snake_case 口径（linear_beta /
    activate_left / dst_type / ...），与 aclnn 层 camelCase 不同；未知键由
    **attrs 吸收（TTK kernel 路径以 case attributes 为 kwargs 注入）。
    """

    # kernel/geir 路由按 CSV 原名透传属性：ST(L1) 表为 proto snake_case，
    # aclnn 表为 camelCase——双拼写取并集（None 哨兵区分"未给"与 0/False）
    _ATTR_ALIASES = {
        "beta": ("beta",),
        "linear_beta": ("linear_beta", "linearBeta"),
        "activate_left": ("activate_left", "activateLeft"),
        "axis": ("axis",),
        "quant_mode": ("quant_mode", "quantMode"),
        "dst_type": ("dst_type", "dstType"),
        "dst_type_max": ("dst_type_max", "dstTypeMax"),
        "round_mode": ("round_mode", "roundMode", "roundModeOptional"),
        "group_list_type": ("group_list_type", "groupListType"),
    }

    @staticmethod
    def golden(x, weight=None, group_index=None, groupIndex=None, **attrs):
        if group_index is None:
            group_index = groupIndex

        def pick(name, default):
            for key in SituGluGroupQuantKernelSpec._ATTR_ALIASES[name]:
                if key in attrs and attrs[key] is not None:
                    return attrs[key]
            return default

        return AclnnSituGluGroupQuantSpec.golden(
            x,
            weight,
            group_index,
            beta=pick("beta", 1.0),
            linear_beta=pick("linear_beta", 0.0),
            activateLeft=bool(pick("activate_left", False)),
            axis=pick("axis", -1),
            quantMode=pick("quant_mode", 0),
            dstType=pick("dst_type", 36),
            dstTypeMax=pick("dst_type_max", 0.0),
            roundModeOptional=pick("round_mode", "rint"),
            groupListType=pick("group_list_type", 1),
        )
