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


_TANH_C1 = np.float32(-0.333327681)
_TANH_C2 = np.float32(0.133152977)
_TANH_C3 = np.float32(-0.0523039624)
_TANH_C4 = np.float32(0.0157396831)
_TANH_THRESHOLD = np.float32(0.6)


def _fma32(a, b, c):
    """IEEE fp32 fused multiply-add emulation: fp64 product of fp32 inputs is
    exact (<=48 bits); the fp64 sum is exact whenever operand exponents differ
    by <=5 (Horner's tightly-bounded |x|<=0.6 value domain guarantees this),
    so the final fp32 cast is the sole rounding — matching hardware
    FusedMulDstAdd. (Theoretical double-rounding requires an exponent gap >5,
    unreachable in this polynomial's value range.)"""
    ad = (
        a.double()
        if torch.is_tensor(a)
        else torch.tensor(float(np.float32(a))).double()
    )
    bd = (
        b.double()
        if torch.is_tensor(b)
        else torch.tensor(float(np.float32(b))).double()
    )
    cd = (
        c.double()
        if torch.is_tensor(c)
        else torch.tensor(float(np.float32(c))).double()
    )
    return (ad * bd + cd).float()


def _two_path_tanh(x):
    """Mirror kernel VFTanhTwoPathVf (tanh.h reference, as in situ_glu /
    situ_mx_quant): |x| < 0.6 -> degree-9 polynomial, FMA Horner
    x*(1 + c1*x^2 + c2*x^4 + c3*x^6 + c4*x^8) (bit-exact via _fma32);
    |x| >= 0.6 -> sigmoid decomposition 2/(1+exp(-2x)) - 1. No cancellation
    anywhere; the polynomial region is exp-free (replaces the recipes naive
    exp-subtraction form whose beta-scale cancellation amplified ULP(beta)
    noise and the hardware Exp approximation at tiny |gate/beta|)."""
    x2 = x * x
    p = x2 * _TANH_C4
    p = p + _TANH_C3
    p = _fma32(x2, p, _TANH_C2)
    p = _fma32(x2, p, _TANH_C1)
    p = p * x2
    p = _fma32(x, p, x)
    e = torch.exp(x * np.float32(-2.0))
    t = np.float32(1.0) / (np.float32(1.0) + e)
    t = t * np.float32(2.0) - np.float32(1.0)
    return torch.where(torch.abs(x) >= _TANH_THRESHOLD, t, p)


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

    # Activation mirrors the kernel instruction chain EXACTLY (SituActivationVF,
    # two-path tanh caliber aligned with situ_glu / situ_mx_quant, i.e. the
    # tanh.h reference):
    #   x = g * invBeta;  tanh = VFTanhTwoPathVf(x):
    #     |x| < 0.6:  degree-9 polynomial, FMA Horner (bit-exact via _fma32)
    #     |x| >= 0.6: 2/(1+Exp(-2x)) - 1 (sigmoid decomposition)
    #   sig = 1/(1+Exp(-g)); situ_a = (tanh * sig) * beta
    # Host scalars are fp32-rounded the same way the kernel computes them:
    #   invBeta = SafeReciprocal(fl32(beta)) (clamped to +-FLOAT_MAX for
    #   subnormal beta, preventing 0 * inf -> NaN)
    # This replaces the recipes naive exp-subtraction form 2*beta/(1+e)-beta,
    # whose beta-scale cancellation amplified ULP(beta) noise and the hardware
    # Exp approximation into visible divergence at tiny |gate| (200-case menkan
    # threshold-classification evidence: 44 FAILs, all in extreme beta x
    # tiny-gate). The two-path form has no cancellation anywhere and the
    # polynomial region (|x| < 0.6) is exp-free.
    # KNOWN LIMITATION: torch.exp is correctly rounded while the device Exp
    # instruction is a hardware approximation; exp survives only in the tanh
    # sigmoid region (|x| >= 0.6, tanh saturated — error impact negligible)
    # and in sigmoid(gate) (~1e-7, unamplified). Residual divergence is
    # boundary-straddle lottery only (expected CPU golden vs NPU divergences).
    b32 = np.float32(beta)
    # Plain reciprocal (kernel-aligned, situ_mx_quant caliber): subnormal beta
    # overflows 1/beta to +-inf -> two-path sigmoid branch consumes it
    # (tanh -> +-1); gate == 0 * inf = NaN -> exit cast NaN -> +0 (modeled
    # below by nan_to_num).
    with np.errstate(all="ignore"):
        inv_beta = np.float32(1.0) / b32
    tanh_g = _two_path_tanh(gate * float(inv_beta))
    sig_g = np.float32(1.0) / (np.float32(1.0) + torch.exp(gate * np.float32(-1.0)))
    # 硬件 FTZ 口径（onboard 实证）：除法产生的次正规结果在 NPU 被冲刷为 0。
    # rangebound 用例 up=-inf × gate≈-87.5 时：exp(87.5)=1.0e38 有限 → CPU 得次正规
    # sigmoid 9.98e-39 → act = tiny*(-inf) = +inf → 饱和 448；设备冲 0 →
    # act = 0*(-inf) = NaN → CAST_RINT+SAT 的 NaN→+0。逐位对齐须同样冲刷
    # 绝对值 < 2^-126（fp32 最小正规数）的次正规。
    sig_g = torch.where(
        sig_g.abs() < np.float32(1.17549435e-38), torch.zeros_like(sig_g), sig_g
    )
    gb = (tanh_g * sig_g) * float(b32)

    # linear_beta <= 0 disables the up-branch tanh soft-clamp: u = up
    # passthrough (kernel TPL_LINEAR_DISABLED caliber). Enabled path mirrors
    # the kernel: ub = linearBeta * VFTanhTwoPathVf(up * invLinearBeta)
    if linear_beta <= 0:
        ub = up.reshape(rows, epr)
    else:
        a32 = np.float32(linear_beta)
        with np.errstate(all="ignore"):
            inv_alpha = np.float32(1.0) / a32
        ub = _two_path_tanh(up * float(inv_alpha)) * float(a32)

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


_E8M0_BIAS = 127
_OFF_RTOL = 2.0**-2  # 与 y 侧 isclose 同口径（e4m3 两次独立舍入上界）
_OFF_ATOL = 2.0**-6  # 几个 subnormal 步长，豁免 0<->次正规翻转
_OFF_GROUP_BAD_MAX = 0.125  # 组内超差元素比例上限（真踩线 0~2/32，伪 ±1 全组）


def _payload_f32(t):
    """e4m3 payload -> 展平 float32 numpy（torch / ml_dtypes numpy 双路径）。"""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().reshape(-1).float().numpy()
    return np.asarray(t).reshape(-1).astype(np.float32)


def _group_dequant_equiv(y, y_g, s_dev_u8, s_gold_u8, gc, epr):
    """逐组反量化等价校验（② 豁免的成因判据）。

    对每个 (row, group) 槽位计算两侧反量化值 payload·2^(E-127) 并按 y 侧
    同口径容差比对；返回 (rows, gc) bool——该组超差元素比例 <= 1/8 时认为
    等价（真踩线 payload 配套差 2 倍、乘积两侧都 ≈ act；伪 ±1 的 y 逐位
    不动、乘积差 2 倍全组超差）。e8m0 无 0 值表示、全字节合法，无需特判。
    """
    try:

        def _to_np_u8(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return np.asarray(t)

        rows = s_dev_u8.shape[0]
        s_dev_np, s_gold_np = _to_np_u8(s_dev_u8), _to_np_u8(s_gold_u8)
        p_dev = _payload_f32(y).reshape(rows, gc, 32).astype(np.float64)
        p_gold = _payload_f32(y_g).reshape(rows, gc, 32).astype(np.float64)
        f_dev = np.exp2(s_dev_np[:, :gc].astype(np.float64) - _E8M0_BIAS)
        f_gold = np.exp2(s_gold_np[:, :gc].astype(np.float64) - _E8M0_BIAS)
        v_dev = p_dev * f_dev[:, :, None]
        v_gold = p_gold * f_gold[:, :, None]
        bad = np.abs(v_dev - v_gold) > _OFF_ATOL + _OFF_RTOL * np.maximum(
            np.abs(v_dev), np.abs(v_gold)
        )
        return bad.mean(axis=2) <= _OFF_GROUP_BAD_MAX
    except Exception:
        # 校验失败（形状/dtype 异常）时保守返回 False——不豁免，交由
        # binary_equal 判比器处置（宁可误杀不可放水）
        return np.zeros(s_dev_u8[:, :gc].shape, dtype=bool)


def _pre_compare_mask_scale_pad(y=None, scale=None, *rest):
    """判比前豁免两类已声明的 y_scale 非契约差异（家族先例：clipped_swiglu
    的 _pre_compare_mask_invalid_rows 同款模式——判比器只看真正该比的东西）。

    ① 奇 gc 行尾 pad 字节：算子契约 scaleColNum = gc 偶数化，kernel 仅写每行
    前 gc 字节，行尾 gap 字节保留输出 buffer 初值——该初值随执行环境而异
    （TTK kernel 模式实测 0x7F、GE 为分配器残留），任何固定建模都只对单一
    harness 成立。两侧该字节同置 0，使判比与 buffer 预填解耦。gc 从 y 末维
    推得（epr = y.shape[-1] -> gc = epr/32）；偶 gc 无 pad。

    ② amax 骑 2^k 边界的 ±1 指数差（KNOWN LIMITATION 的判比侧豁免，带成因
    校验）：sigmoid(gate) 链路 exp 在 NPU 为硬件近似指令、CPU golden 为精确
    舍入，~1e-7 无放大噪声进入组 amax——当某组 amax 恰落在 2 的幂分界 ±1e-7
    内时两侧 e8m0 记档差 1。e8m0 被 TTK _BIN_DTYPES 类别强制逐位判比，容差
    不可声明，故在此豁免。豁免条件（PR#16059 检视意见修正——仅字节差 ±1
    是症状不是成因，须校验组级反量化等价以区分真踩线与"y 不动 scale 偏"
    的伪 ±1）：
      真踩线：payload 配套差 2 倍（payload = act/scale），两侧反量化值
              payload·2^E 都 ≈ act → 组内超差元素 0~2/32 → 豁免
      伪 ±1：y 逐位相同、scale 整体偏 1 → 反量化值差 2 倍 → 组内 32/32
              全超差 → 不豁免（"y 不变 + 有效 scale 整体偏 1"必须失败）
    判比口径与 y 侧同源：RTOL=2^-2（e4m3 两次独立舍入的相对差上界）、
    ATOL=2^-6（几个 subnormal 步长，避免 0↔次正规翻转误伤）、组内超差
    ≤ 1/8（真踩线与伪 ±1 之间隔 8 倍，阈值不敏感）。

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

    def _u8_np(arr):
        """统一转 numpy u8 (rows, scn) 视图（torch / numpy 双路径）。

        判比算术全部在 numpy 域进行——TTK 判比实测设备侧与 golden 侧类型可
        不同（geir: dev=numpy(ml_dtypes)、golden=torch(float8)），按设备侧
        单一类型选分支会在还原段对 torch dtype 调 numpy view 而炸（外层
        except 吞掉后连 ① pad 掩膜一并丢失）。"""
        if isinstance(arr, torch.Tensor):
            return arr.detach().view(torch.uint8).reshape(-1, scn).numpy().copy()
        return np.array(np.asarray(arr).view(np.uint8).reshape(-1, scn), copy=True)

    try:
        s_dev_m = _u8_np(scale)
        s_gold_m = _u8_np(s_g)
        if s_dev_m.shape != s_gold_m.shape:
            return None
        # ① pad 掩膜（numpy 域）
        if scn != gc:
            s_dev_m[:, gc:] = 0
            s_gold_m[:, gc:] = 0
        # ② 逐位差 == ±1 且组级反量化等价的槽位：同置为 golden 值。
        # 仅字节差 ±1 是症状（真踩线与伪 ±1 共有），成因校验按组对比
        # 反量化值 payload·2^E——真踩线两侧都 ≈ act（payload 配套补偿），
        # 伪 ±1（y 不动 scale 偏）差 2 倍全组超差，不豁免。
        diff = np.asarray(s_dev_m, dtype=np.int64) - np.asarray(
            s_gold_m, dtype=np.int64
        )
        off_by_one = np.abs(diff) == 1
        if off_by_one.any() and gc > 0:
            # ② 独立兜底：成因校验异常时退化为"不豁免"（返回 False 矩阵），
            # ① 的 pad 掩膜不受影响——奇 gc 场景 off_by_one 为 (rows, scn)、
            # 校验结果为 (rows, gc)，须按前 gc 列对齐后按位与
            try:
                equiv = _group_dequant_equiv(y, y_g, s_dev_m, s_gold_m, gc, epr)
                merged = np.zeros_like(off_by_one)
                merged[:, :gc] = off_by_one[:, :gc] & equiv
                off_by_one = merged
            except Exception:
                off_by_one = np.zeros_like(off_by_one)
        if off_by_one.any():
            s_dev_m[off_by_one] = s_gold_m[off_by_one]
        # 还原形状与 dtype：按各自来源类型恢复（dev/golden 类型可不同）
        if isinstance(scale, torch.Tensor):
            scale_m = torch.from_numpy(s_dev_m).reshape(scale.shape).view(scale.dtype)
        else:
            scale_m = s_dev_m.reshape(scale.shape).view(scale.dtype)
        if isinstance(s_g, torch.Tensor):
            s_g_m = torch.from_numpy(s_gold_m).reshape(s_g.shape).view(s_g.dtype)
        else:
            s_g_m = s_gold_m.reshape(s_g.shape).view(s_g.dtype)
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

    # 逐位复刻参考（与原 recipes 算子 22 case 字节对拍验证）；判比口径：
    # e4m3fn 用 close（rtol=2^-2 吸收 ±1 码骑线元素，ptol=1e-3 容忍 0.1%
    # 失配——两路口径下正常区间实测逐位一致，此容差仅兜底边界彩票）；
    # e8m0 声明 close 但被 TTK _BIN_DTYPES 类别强制忽略，实际仍逐位
    # binary_equal（声明保留作口径文档）。不走 mix_tolerance 的 Promote
    # 升精度 golden（bf16->fp32 会在 amax 指数边界引入假差异，weight 逐
    # token 缩放下被放大）。
    tolerance = {
        "float8_e4m3fn": {"standard": "close", "rtol": 2**-2, "ptol": 1e-3},
        "float8_e8m0": {"standard": "close"},
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

    # 判据与掩膜与 aclnn 同源（继承上方 close 口径 + 奇 gc pad 掩膜）
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
