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
"""AddLayerNormQuantV2 多通路 golden（TestSpec 范式）。

通路支持面（见 01_requirement.md §3.3）：
  kernel ✅ / geir ❌（未注册 V2 infershape，图模式不可编译）/ aclnn ✅ /
  e2e ❌（torch_npu 二进制未引用 aclnnAddLayerNormQuantV2，对照 aclnnAddRmsNormQuant 有命中）。
"""

__spec__ = {
    # kernel（GEIR 复用同一 op 蛇形名，但本算子 GE 通路不适用）
    "add_layer_norm_quant_v2": "AddLayerNormQuantV2KernelSpec",
    # aclnn
    "aclnnAddLayerNormQuantV2": "AddLayerNormQuantV2AclnnSpec",
}

import logging

import numpy as np
import torch

# 两条通路（kernel / aclnn）共用同一套判据：浮点 cross_check L1、int8 quant。
# cross_check 依赖 third_party（两个 Spec 都挂 torch 参考）；三方不可用时浮点判 GOLDEN_FAILURE。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    # 量化输出判据：绝对误差 <= 1 LSB。ptol 取 0.1%（TTK 文档中 PerTokenQuant 的推荐值）：
    # 量化末步是阶跃函数，x 与 mean 在 fp32 上恰好相等时 (x-mean) 精确为 0 而参考侧非 0，
    # 量化后必然 0 vs ±127，任何 fp32 实现都无法规避（实测 L0_119 约 39M 元素里有 3 个）。
    "int8": {"standard": "quant", "ptol": 0.001},
}


def _attr(kwargs, name, default, *aliases):
    for n in (name,) + aliases:
        if n in kwargs:
            v0 = kwargs[n]
            if isinstance(v0, (list, tuple)) and v0:
                v0 = v0[0]
            if isinstance(v0, str):
                s = v0.strip().strip('"').strip("'").lower()
                if s in ("true", "false", "yes", "no", "1", "0"):
                    return s in ("true", "yes", "1")
                try:
                    return type(default)(s)
                except Exception:
                    return v0.strip().strip('"').strip("'")
            return v0
    return default


def _quant_mode(kwargs):
    """quant_mode 归一化（CSV 属性名 kernel 用 quant_mode、aclnn 用 quantMode）。"""
    return (
        str(_attr(kwargs, "quant_mode", "dynamic", "quantMode"))
        .strip()
        .strip('"')
        .strip("'")
        .lower()
    )


def _validate_contract(x1, gamma, quant_mode):
    """入口契约校验：dynamic 只支持 perChannel，perTensor 形状一律拒。

    依据（算子契约，非本文件的实现选择）：
      · README.md「约束说明」：`quant_mode="dynamic"` **仅**支持 perChannel 模式；
        「形状约束」：gamma 的 shape 需等于 x1 的后 gammaDimNum 维，
        dynamic 时 `out_scales1/2` 为「x1 去掉 gamma 所占尾维后的前缀」。
      · op_api/aclnn_add_layer_norm_quant_v2.cpp `CheckShapeRelV2Regbase`：dynamic 分支
        `x1DimNums >= 2` 且逐维校验 outScales 前缀 == x1 前缀、gamma 尾维逐维相等；
        `gamma=[1,N]` 这类 perTensor 形状被当场拒绝（实测报
        "Shape of gamma should be equal to the last 2 dims of x1"）。

    此前 golden 不做该校验，非法域会以两种形式失败：x1/gamma 秩相同则静默产出
    错口径 out_scales，秩更高则 `os_shape = x.shape[:-gamma.dim()]` 与 `os_` 元素数
    对不上 → `RuntimeError: shape ... is invalid for input of size ...`，把"用例
    非法"误报成"golden 崩了"。
    ⚠ 不要用 `x1.shape[:-1]` 之类的写法绕过：那会违反 tiling
    （norm/add_layer_norm_quant/op_host/add_layer_norm_quant_v2_tiling_arch35.cpp
    dynamic 分支 outScales 前缀逐维校验）与 aclnn 的同一套校验。
    """
    if quant_mode != "dynamic" or x1 is None or gamma is None:
        return
    x_shape = tuple(x1.shape)
    g_shape = tuple(gamma.shape)
    if len(g_shape) < 1:
        # rank-0 gamma 会被 tail==g_shape 静默放行，但 tiling 拒（0 == weightDimNum）
        raise ValueError(
            "AddLayerNormQuantV2: gamma must have rank >= 1 "
            f"(got gamma.shape={list(g_shape)}). Ref: README.md 形状约束「gamma 的 shape 需等于 "
            "x1 的后 gammaDimNum 维」."
        )
    if len(x_shape) < 2:
        raise ValueError(
            "AddLayerNormQuantV2: quant_mode='dynamic' requires x1.dim() >= 2 "
            f"(got x1.shape={list(x_shape)}). "
            'Ref: README.md 约束说明「quant_mode="dynamic"时，x1 的维数需不小于2」.'
        )
    tail = x_shape[len(x_shape) - len(g_shape) :]
    if tail != g_shape:
        raise ValueError(
            "AddLayerNormQuantV2: quant_mode='dynamic' only supports perChannel gamma; "
            f"gamma.shape must equal x1.shape[-gamma.dim():], got gamma{list(g_shape)} vs "
            f"x1{list(x_shape)} (last dims are {list(tail)}). "
            'Ref: README.md 约束说明「quant_mode="dynamic"仅支持perChannel模式」+ 形状约束'
            "「gamma 的 shape 需等于 x1 的后 gammaDimNum 维」. "
            "perTensor 量化（gamma=[1,N]、scales1/zp 为 [1]）请改用 quant_mode='static'."
        )


def _as_tensor(a):
    """numpy/torch -> torch.Tensor；缺失可选输入(None/object 占位) -> None；fp16/bf16 抬到 fp32。

    抬精度仅向上兜底：内核 VF 全程 fp32、出口才窄回；golden 若留在 fp16/bf16 逐步 round
    会比内核糙，边缘元素差几 ULP → 判据在容差边缘随机翻红。
    """
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        return a.to(torch.float32) if a.dtype in (torch.bfloat16, torch.float16) else a
    arr = np.asarray(a)
    if arr.dtype == object:
        flat = [x for x in arr.reshape(-1).tolist() if x is not None]
        if not flat:
            return None
        arr = np.asarray(flat[0])
    if str(arr.dtype) in ("bfloat16", "float16"):
        arr = arr.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _empty_outputs(x, gamma, quant_mode, has_scales2, additional_output):
    """空 tensor 语义（A5 口径，与 V1 对齐）：y1/y2/x/layernorm_res 为空；
    dynamic 且归一化轴为 0（cols==0 && rows!=0）时，走 empty-input kernel，out_scales 填 -inf。

    aclnn 判据：dynamic 且 outScales1 为空 → no-op；static 且 x1 为空 → no-op；
    其余空 tensor 继续下发，由 empty kernel（key=9）写 dynamic 的 out_scales。
    注意 A2/A3 无 empty kernel，一律 no-op，out_scales 内容未定义。

    契约上「无实际意义」的输出返回 None（TTK 判 SUPPRESSED→PASS，不参与比对）。
    """
    dev = x.device
    x_shape = tuple(x.shape)
    base_dtype = x.dtype
    g_rank = gamma.dim() if gamma is not None else 1
    prefix = tuple(x_shape[: len(x_shape) - g_rank]) if len(x_shape) > g_rank else ()
    y1 = torch.zeros(x_shape, dtype=torch.int8, device=dev)
    y2 = torch.zeros(x_shape, dtype=torch.int8, device=dev) if has_scales2 else None
    xo = (
        torch.zeros(x_shape, dtype=base_dtype, device=dev)
        if additional_output
        else None
    )
    ln = torch.zeros(x_shape, dtype=base_dtype, device=dev)
    if quant_mode == "dynamic":
        os_shape = prefix if prefix else (1,)
        os1 = torch.full(os_shape, float("-inf"), dtype=torch.float32, device=dev)
        os2 = os1.clone() if has_scales2 else None
    else:
        os1 = None
        os2 = None
    return [y1, y2, xo, ln, os1, os2]


def _compute(*tensors, **kwargs):
    """x1,x2,gamma,beta,bias?,scales1?,scales2?,zero_points1?,zero_points2?
    -> y1,y2,x,layernorm_res,out_scales1,out_scales2（顺序照 def.cpp）。"""
    t = list(tensors) + [None] * 9
    x1, x2, gamma, beta, bias, scales1, scales2, zp1, zp2 = t[:9]

    quant_mode = _quant_mode(kwargs)
    epsilon = float(_attr(kwargs, "epsilon", 1e-5))
    div_mode = bool(_attr(kwargs, "div_mode", True, "divMode"))

    # 无意义输出判定之一：x 是否写出（def.cpp `additional_output` 默认 false）。
    # 取不到属性时按 True（参与比对）：宁可多一条假失败，也不静默跳过真实失败。
    additional_output = _attr(kwargs, "additionalOutput", None, "additional_output")
    if additional_output is None:
        logging.warning(
            "AddLayerNormQuantV2 golden: additionalOutput not found; "
            "falling back to True (compare the x slot) to avoid silently skipping a real failure."
        )
        additional_output = True
    additional_output = bool(additional_output)

    # 非法域先于任何计算显式报错（dynamic+perTensor），避免以 reshape 崩溃呈现
    _validate_contract(x1, gamma, quant_mode)

    x = x1 + x2 if bias is None else x1 + x2 + bias
    if x.numel() == 0 or gamma.numel() == 0:
        return _empty_outputs(
            x, gamma, quant_mode, scales2 is not None, additional_output
        )

    row = int(gamma.reshape(-1).numel())
    xf = x.reshape(-1, row)
    # 统计量按「先除后求和」两遍归约（与内核 VFCalcMeanVar 及三方参考同算法）：
    # 直接 xf.mean() 是先求和再除，x1 接近 dtype 上界时 n*|x1| 会在 fp32 里溢出成 inf，
    # 继而使 (x-mean)=nan、整行输出失真（实测 L0_005: x1≡3.39e38、row=286 → 820624/820624 行 inf）。
    # 先减去行首元素 x0 再归约（与内核同一中心化口径）：x 量级远大于行内波动时（实测 L0_126
    # x1≡-65504、波动 ~1e-3），x-mean 的减法会在 x 的量级上舍入，误差盖过波动本身。
    nf = float(row)
    xd = xf - xf[:, :1]
    # 统计量取「fp64 精确行和 → 窄回 fp32」：与 kernel 侧 welford 补偿累加 / full-load 精化同口径
    # 先除以行宽再精确求和（与 kernel 的 Div→Reduce 同口径）：累加原始和会在行值接近 dtype
    # 上限时于 fp32 溢出（n*|x| > FLT_MAX）
    x_mean = _exact_row_sum(xd / nf).to(torch.float32)
    x_dev = xd - x_mean
    x_var = _exact_row_sum((x_dev * x_dev) / nf).to(torch.float32)
    g = gamma.reshape(1, -1).to(xf.dtype)
    b = beta.reshape(1, -1).to(xf.dtype)
    y = x_dev / torch.sqrt(x_var + epsilon) * g + b
    y = y.reshape(x.shape)
    # layernorm_res 出口饱和（kernel 同口径）：仅钳有限值，±inf/NaN 原样保留
    y = _saturate_finite_inplace(y, _ln_res_sat_kwargs(kwargs))

    layernorm_res = y

    def _round_i8(v):
        # NaN → 0（与 kernel `Round2Int8` 的「先清 NaN 再 RINT+窄回」同口径）：
        # 全行 NaN 的输入（bias≡nan → x≡nan）下，硬件与 torch 对 NaN→int8 的结果不同，
        # 兄弟算子 add_rms_norm_quant 的参考也按 nan_to_num(nan=0) 处理。
        return torch.clamp(torch.round(torch.nan_to_num(v, nan=0.0)), -128, 127).to(
            torch.int8
        )

    os_shape = tuple(x.shape[: -gamma.dim()]) if gamma.dim() < x.dim() else ()

    if quant_mode == "static":
        # 静态：per-channel（scales==gamma）或 per-tensor（scales 单元素）
        s1 = scales1.reshape(-1)[0] if scales1.numel() == 1 else scales1
        y1 = (
            _round_i8(y / s1 + (0.0 if zp1 is None else zp1))
            if div_mode
            else _round_i8(y * s1 + (0.0 if zp1 is None else zp1))
        )
        if scales2 is not None:
            s2 = scales2.reshape(-1)[0] if scales2.numel() == 1 else scales2
            y2 = (
                _round_i8(y / s2 + (0.0 if zp2 is None else zp2))
                if div_mode
                else _round_i8(y * s2 + (0.0 if zp2 is None else zp2))
            )
        else:
            y2 = None  # scales2 缺省：y2 无实际意义
        # static 下 out_scales1/2 无实际意义（README.md:27）
        out_scale1 = None
        out_scale2 = None
    else:
        yf = y.reshape(-1, row)

        def _dyn(sc):
            if sc is None:
                tmp = yf
            elif sc.numel() == 1:
                tmp = yf * sc.reshape(-1)[0].to(yf.dtype)
            else:
                # scales shape == gamma（可为多维，如 [4, 64]）；yf 已展平成 [rows, prod(gamma)]，
                # 必须同步展平，否则多维 gamma 会按尾部维广播失败（实测 2x2x2x2x2x2x4x64）。
                tmp = yf * sc.reshape(-1).to(yf.dtype)
            m = torch.abs(tmp).amax(dim=-1)
            # 与内核同算法走乘法（Muls 1/127.0），不是除法：除法会差 ≤1 ULP
            os_ = (m * (1.0 / 127.0)).to(torch.float32)
            # 全零行（row_max=0）在核心里是 0/0；这里置 0 避免 NaN 污染，正常数据不会命中
            denom = os_.reshape(-1, 1).clone()
            denom[denom == 0] = 1.0
            out = _round_i8(tmp / denom).reshape(x.shape)
            return out, (os_.reshape(os_shape) if os_shape else os_)

        y1, out_scale1 = _dyn(scales1)
        if scales2 is not None:
            y2, out_scale2 = _dyn(scales2)
        else:
            y2 = None  # scales2 缺省：y2 无实际意义
            out_scale2 = None  # scales2 缺省：out_scales2 无实际意义

    # x 仅在 additional_output=true 时写出；false 时输出为脏数据（算子不写）
    x_out = x if additional_output else None
    return [y1, y2, x_out, layernorm_res, out_scale1, out_scale2]


_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _od(kwargs):
    """输出 dtype 列表（dtype 名字符串）。

    TTK 下发的 tensor_dtypes / output_dtypes 有两种形态：CSV 写 `(a,b,c)` 时是**扁平字符串序列**，
    写 `((a,),(b,))` 时是嵌套单元素。旧实现只按嵌套取 `d[0]`，遇到扁平序列会把 'bfloat16'
    取成首字母 'b' —— 于是 `_cast_out*` 的窄回和出口饱和**整体静默失效**（实测：内核已饱和而
    golden 未饱和，rowmax 一个有限一个 inf，int8 量化输出整片超差、百万级 exceed_count）。
    """
    od = kwargs.get("output_dtypes") or []
    res = []
    for d in od:
        if isinstance(d, (list, tuple)):
            d = d[0] if d else ""
        res.append(str(d))
    return res


def _ln_res_sat_kwargs(kwargs):
    """layernorm_res 出口的窄回饱和上限（dtype 名）。

    layernorm_res 的 dtype 与 x1 声明 dtype 相同（见 docs 数据类型表）。取两者中更窄的一个，
    这样即使框架把 output_dtypes 抬成 Promote 档（fp32），也不会漏掉算子声明 dtype
    （bf16/fp16）的上溢钳位。
    """
    od = _od(kwargs)
    idt = kwargs.get("input_dtypes") or []
    cands = []
    if len(od) > 3 and od[3] in _TORCH_DTYPE:
        cands.append(od[3])
    if idt and not isinstance(idt, str):
        d0 = idt[0]
        if isinstance(d0, (list, tuple)):
            d0 = d0[0] if d0 else ""
        d0 = str(d0)
        if d0 in _TORCH_DTYPE:
            cands.append(d0)
    cands = [c for c in cands if _TORCH_DTYPE[c].is_floating_point]
    if not cands:
        return None
    return min(cands, key=lambda c: torch.finfo(_TORCH_DTYPE[c]).max)


def _saturate_finite_inplace(v, sat_dtype):
    """窄回到 sat_dtype 时的饱和：有限值钳到 ±finfo.max，±inf/NaN 原样保留。

    kernel 侧 `SaturateFiniteToOutputRange<X1_TYPE>` 与之逐位一致（同 mask、同上下界、同 NaN/Inf 语义）。
    动机：y 依赖行统计量，kernel 与参考的归约次序不同（实测相对差 ~1e-5），恰压在输出 dtype
    上溢点上的元素会在 inf / max 之间翻转；TTK 的 NaN/Inf 判据对「NPU 与 golden、竞品都不一致」
    直接判 FAIL（实测 L0_146 有 1288/221109120 个元素）。钳位幂等：两侧都钳到同一 ±max，
    翻转消失；真正的 ±inf/NaN 语义不变（mask 取 |v| < +inf，NaN 与 ±inf 都被排除）。
    """
    if (
        not isinstance(v, torch.Tensor)
        or not v.is_floating_point()
        or sat_dtype is None
    ):
        return v
    return _saturate_finite_torch(v, _TORCH_DTYPE[sat_dtype])


def _saturate_finite_torch(v, tgt):
    """`_saturate_finite_inplace` 的 torch.dtype 版（三方竞品腿按 out_dtype 调用）。"""
    if (
        not isinstance(v, torch.Tensor)
        or not v.is_floating_point()
        or not isinstance(tgt, torch.dtype)
        or not tgt.is_floating_point
    ):
        return v
    fmax = torch.finfo(tgt).max
    if fmax >= torch.finfo(v.dtype).max:
        return v
    # torch.clamp：±inf → ±fmax，NaN 原样透传（与 kernel 侧「y==y」掩码 + Mins/Maxs 逐位一致）
    return torch.clamp(v, -fmax, fmax)


def _exact_row_sum(v):
    """按行（最后一维）求**精确和**：fp64 分块累加，避免整行 fp64 临时量。

    用途：让参考侧的 mean/var 等于 `fl32(精确和/n)`（正确舍入），与 kernel
    `VFRefineMeanFromAcc` / welford 补偿累加得到的 mean 逐位一致。这一步是必要的：
    y = (x-mean)*rstd*gamma 在 |gamma|≈dtype上限时会把 mean 最后一位的 1ULP 差异放大到
    1e31 量级（实测 L0_055），只有两侧 mean 完全相同，亚 ULP 偏差元素的判定才稳定。
    """
    acc = torch.zeros(v.shape[0], 1, dtype=torch.float64, device=v.device)
    step = 1 << 18
    for k in range(0, v.shape[1], step):
        acc += v[:, k : k + step].double().sum(dim=-1, keepdim=True)
    return acc


def _cast_out(outs, od):
    """KernelSpec 出口：按【下发】的 output_dtypes 窄回，返 numpy；None 槽原样透传。"""
    res = []
    for i, o in enumerate(outs):
        if o is None:
            res.append(None)
            continue
        arr = o.detach().cpu().numpy()
        if i < len(od) and od[i]:
            arr = arr.astype(od[i])
        res.append(arr)
    return res


def _cast_out_tensor(outs, od):
    """AclnnSpec 出口：按【下发】的 output_dtypes 窄回，返 torch.Tensor（模板要求返 Tensor）。

    Promote 下 od 已是抬高档（fp16→fp32、fp32→fp64）→ 本步退化为 no-op，等价规范的「零 cast」；
    两方（无 Promote）下 od 是算子声明 dtype → 与 NPU 出口窄回行为一致。
    """
    res = []
    for i, o in enumerate(outs):
        if o is None:
            res.append(None)
            continue
        if not isinstance(o, torch.Tensor):
            o = torch.as_tensor(o)
        if i < len(od) and od[i] and od[i] in _TORCH_DTYPE:
            o = o.to(_TORCH_DTYPE[od[i]])
        res.append(o)
    return res


class _AddLayerNormQuantV2Compose:
    """三方竞品标杆：与 arch35 kernel **同算法**的 torch 拼接（远端 GPU server 执行）。

    铁律：本类**不复用** `_compute`。`_compute` 是数学最优写法（mean/var 库实现 + `1/sqrt`），
    只作 CPU 真值；本类逐条对齐 kernel 的运算序列，只作 cross_check 的竞品标杆。

    与 kernel 的对齐点（证据见 GOLDEN_REF.md §3）：
      · Add 结合序：`(x1 + x2) + bias`。
      · LayerNorm 统计量：mean-centered 两遍归约，**逐元素先除以 cols 再 ReduceSum**
        （VFCalcMeanVarFast/VFCalcMeanVar: Div→ReduceSum）；全程 fp32。
      · rstd：`torch.rsqrt(var + eps)`，对齐 kernel 的 Newton-Raphson rsqrt（NR 结果即 rsqrt 近似）。
      · y：`(x - mean) * rstd * gamma + beta`（Sub→Mul(rstd)→Mul(gamma)→Add(beta)）。
      · static：`round(y/sc + zp)` / `round(y*sc + zp)`，RINT 取整 + clamp[-128,127]。
      · dynamic：`tmp=y*sc → rowmax(|tmp|) → os = m * (1/127) → round(tmp/os)`
        （kernel `Muls(outScale, outScale, 1/127f)`，不是 m/127）。
      · **cast 矩阵（按 golden_standard.md §一/§四：GPU 腿按 NPU 加宽行为同步 cast、出口对应窄回）**：
          入口：fp16/bf16 → fp32（kernel VF 全程 fp32）；fp32 不升档（kernel 累加也是 fp32）。
          出口：x/layernorm_res → x1.dtype（NPU 窄回）；out_scales → fp32；
                **y1/y2 → int8**（torch 整型中间量会升 int32，必须显式 .to(torch.int8)，否则漏成 int32）。
        int8 出参用 clamp[-128,127]+RINT 对齐内核饱和 Cast（±1 LSB 判据）。
        本类不做整型除法，故不涉及 torch INT_MIN/-1 的 SIGFPE 陷阱。

    已知差异（无法用 torch 完全复刻，选择依据=不高于 kernel 精度）：
      · 大列（welford 通路）内核用并行 Welford + 二分合并；本类统一用 mean-centered 两遍归约，
        归约合并次序不同，误差同为 fp32 量级、不优于 Welford，不会把竞品凭空变准。
      · kernel 的 NR 含 2 次修正，`torch.rsqrt` 与之差 ≤1 ULP。
      · kernel int8 前多一次 fp32→fp16 舍入；本类直接 round→int8（±1 LSB 判据覆盖）。
    """

    def __init__(
        self,
        quant_mode="dynamic",
        epsilon=1e-5,
        additional_output=False,
        div_mode=True,
        quantMode=None,
        additionalOutput=None,
        divMode=None,
        **kwargs,
    ):
        def _one(v):
            if isinstance(v, (list, tuple)) and v:
                return v[0]
            return v

        # 属性别名：TTK 按算子接口名下发（quantMode/additionalOutput/divMode）
        if quantMode is not None:
            quant_mode = quantMode
        if additionalOutput is not None:
            additional_output = additionalOutput
        if divMode is not None:
            div_mode = divMode

        qm = _one(quant_mode)
        self.quant_mode = (
            str(qm).strip().strip('"').strip("'") if qm is not None else "dynamic"
        )
        self.epsilon = float(_one(epsilon))
        ao = _one(additional_output)
        self.additional_output = (
            ao if isinstance(ao, bool) else str(ao).lower() in ("true", "1", "yes")
        )
        dm = _one(div_mode)
        self.div_mode = (
            dm if isinstance(dm, bool) else str(dm).lower() in ("true", "1", "yes")
        )

    def __call__(
        self,
        x1,
        x2,
        gamma,
        beta,
        bias=None,
        scales1=None,
        scales2=None,
        zero_points1=None,
        zero_points2=None,
        input_dtypes=None,
        output_dtypes=None,
        # TTK 按算子接口的参数名下发（xxxOptional），与本地 C 顺序签名并用：
        # 远端三方腿只按名字绑定，不做位置兜底，故两种命名都要接。
        biasOptional=None,
        scales1Optional=None,
        scales2Optional=None,
        zeroPoints1Optional=None,
        zeroPoints2Optional=None,
        **kwargs,
    ):
        # 别名归一：位置名优先，缺失时取 xxxOptional 形式
        bias = bias if bias is not None else biasOptional
        scales1 = scales1 if scales1 is not None else scales1Optional
        scales2 = scales2 if scales2 is not None else scales2Optional
        zero_points1 = zero_points1 if zero_points1 is not None else zeroPoints1Optional
        zero_points2 = zero_points2 if zero_points2 is not None else zeroPoints2Optional
        # 声明 dtype 窄回目标：三方腿收到的是 **Promote 后**的 fp32 输入（TTK 现状），
        # 若只按 x1.dtype 窄回会漏掉算子声明的低精度输出（bf16/fp16）。若框架下发了
        # input_dtypes/output_dtypes（逻辑 dtype），优先用它；否则退回 x1.dtype。
        dt_tab = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        declared = None
        for cand in (output_dtypes, input_dtypes):
            if cand:
                c0 = cand[0] if isinstance(cand, (list, tuple)) else cand
                if isinstance(c0, str) and c0 in dt_tab:
                    declared = dt_tab[c0]
                    break
        # ⚠ 出口 dtype 必须在入口加宽**之前**取：x1.dtype 是 NPU 的输出 dtype（bf16/fp16/fp32）。
        # 若先 map(f32)，x1 已被改成 fp32，出口就漏窄回（GPU 停在 fp32，与 NPU 低精度输出不同精度 →
        # cross_check 的 rmse 比值被放大成假红）。
        out_dtype = declared or x1.dtype

        def f32(t):
            if t is None:
                return None
            return (
                t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t
            )

        x1, x2, gamma, beta = map(f32, (x1, x2, gamma, beta))
        bias, s1, s2, z1, z2 = map(
            f32, (bias, scales1, scales2, zero_points1, zero_points2)
        )

        x = (x1 + x2) if bias is None else (x1 + x2 + bias)

        n = int(gamma.numel())
        prefix = tuple(x.shape[: x.shape.__len__() - gamma.dim()])
        if x.numel() == 0:
            y1 = torch.zeros(x.shape, dtype=torch.int8, device=x.device)
            y2 = (
                torch.zeros(x.shape, dtype=torch.int8, device=x.device)
                if s2 is not None
                else torch.zeros((1,), dtype=torch.int8, device=x.device)
            )
            xo = torch.zeros(x.shape, dtype=out_dtype, device=x.device)
            ln = torch.zeros(x.shape, dtype=out_dtype, device=x.device)
            if self.quant_mode == "static":
                # static 的 out_scales 无实际意义（见 _compute 同位置注释）：占位统一 0
                os1 = torch.zeros((1,), dtype=torch.float32, device=x.device)
                os2 = torch.zeros((1,), dtype=torch.float32, device=x.device)
            else:
                os_shape = prefix if prefix else (1,)
                os1 = torch.full(
                    os_shape, float("-inf"), dtype=torch.float32, device=x.device
                )
                os2 = (
                    os1.clone()
                    if s2 is not None
                    else torch.zeros((1,), dtype=torch.float32, device=x.device)
                )
            return [y1, y2, xo, ln, os1, os2]

        # ---- LayerNorm：mean-centered 两遍归约（fp32，逐元素先除 cols 再求和）----
        # 竞品腿刻意保持「与 golden 同阶的 fp32 归约」而不是精确归约：它只用于 cross_check 的
        # ratios/计数分母，若做得比 NPU 明显更准，小值域 ErrorCount 比值会被放大成假红
        # （实测 exact 版：L0_057 cnt_third=0 → ratio 1.3e4；L0_149 全量 2.6e8）。
        # 内核侧 mean 已改为正确舍入，参考侧（_compute）也是精确统计量，竞品这条腿无需再拔高。
        xf = x.reshape(-1, n)
        nf = float(n)
        xd = xf - xf[:, :1]

        mean = (xd / nf).sum(dim=-1, keepdim=True)
        dev = xd - mean
        var = ((dev * dev) / nf).sum(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + self.epsilon)
        y = (dev * rstd * gamma.reshape(1, -1) + beta.reshape(1, -1)).reshape(x.shape)
        # 与 kernel/golden 同口径的 layernorm_res 出口饱和（仅钳有限值）
        y = _saturate_finite_torch(y, out_dtype)

        def _i8(v):
            return torch.clamp(torch.round(torch.nan_to_num(v, nan=0.0)), -128, 127).to(
                torch.int8
            )

        if self.quant_mode == "static":
            sc1 = (
                s1.reshape(-1)[0] if s1.numel() == 1 else s1.reshape(tuple(gamma.shape))
            )
            # perTensor 下 zero_points 与 scales 同为 [1]：单元素先取标量，
            # 否则 reshape 到 gamma 形状（如 [1, 128]）会因元素数不足报错。
            off1 = (
                0.0
                if z1 is None
                else (
                    z1.reshape(-1)[0]
                    if z1.numel() == 1
                    else z1.reshape(tuple(gamma.shape))
                )
            )
            y1 = _i8(y / sc1 + off1) if self.div_mode else _i8(y * sc1 + off1)
            if s2 is not None:
                sc2 = (
                    s2.reshape(-1)[0]
                    if s2.numel() == 1
                    else s2.reshape(tuple(gamma.shape))
                )
                off2 = (
                    0.0
                    if z2 is None
                    else (
                        z2.reshape(-1)[0]
                        if z2.numel() == 1
                        else z2.reshape(tuple(gamma.shape))
                    )
                )
                y2 = _i8(y / sc2 + off2) if self.div_mode else _i8(y * sc2 + off2)
            else:
                y2 = torch.zeros((1,), dtype=torch.int8, device=x.device)
            # static 的 out_scales 无实际意义（见 _compute 同位置注释）：占位统一 0
            os1 = torch.zeros((1,), dtype=torch.float32, device=x.device)
            os2 = torch.zeros((1,), dtype=torch.float32, device=x.device)
        else:
            yf = y.reshape(-1, n)

            def _dyn(sc):
                tmp = (
                    yf
                    if sc is None
                    else yf * (sc.reshape(-1)[0] if sc.numel() == 1 else sc.reshape(-1))
                )
                m = torch.abs(tmp).amax(dim=-1, keepdim=False)
                os_ = (m * (1.0 / 127.0)).to(torch.float32)  # kernel Muls(1/127f)
                out = _i8(tmp / os_.reshape(-1, 1)).reshape(x.shape)
                return out, (os_.reshape(prefix) if prefix else os_)

            y1, os1 = _dyn(s1)
            if s2 is not None:
                y2, os2 = _dyn(s2)
            else:
                y2 = torch.zeros((1,), dtype=torch.int8, device=x.device)
                os2 = torch.zeros((1,), dtype=torch.float32, device=x.device)

        xo = x.to(out_dtype)
        ln = y.to(out_dtype)
        return [y1, y2, xo, ln, os1, os2]


class AddLayerNormQuantV2KernelSpec:
    """kernel；收 numpy.ndarray，返 numpy.ndarray。"""

    def golden(*inputs, **kwargs):
        od = _od(kwargs)
        t = [_as_tensor(a) for a in inputs]
        outs = _compute(*t, **kwargs)
        return _cast_out(outs, od)

    # 三方竞品标杆（kernel/GEIR 通路取用；aclnn 与 kernel 共用同一 torch 参考）
    third_party = {"torch": _AddLayerNormQuantV2Compose}
    tolerance = _TOL


class AddLayerNormQuantV2AclnnSpec:
    """aclnn；golden 收已 H2D 的 torch.Tensor，【直接返 Tensor】（规范 §6 / 模板要求）。"""

    def golden(*inputs, **kwargs):
        # ⚠ TTK aclnn 按 C 头**参数顺序**把标量属性 positional 传入（不是 kwargs）：
        #   [0..8] 9 个输入 tensor / [9] quantMode / [10] epsilon / [11] additionalOutput /
        #   [12] divMode / [13..18] 6 个输出 tensor / [19] workspaceSize / [20] executor。
        # 只按 kwargs 取属性会让 quantMode 落回默认 "dynamic" → static 用例 golden 全错
        # （实测 y1/y2 与 NPU 差 ~94%，x/layernorm_res 正常）。这里两路都接。
        n_in = 9
        t = list(inputs[:n_in])
        merged = dict(kwargs)
        if len(inputs) >= n_in + 4 and isinstance(inputs[n_in], str):
            merged.update(
                {
                    "quantMode": inputs[n_in],
                    "epsilon": inputs[n_in + 1],
                    "additionalOutput": inputs[n_in + 2],
                    "divMode": inputs[n_in + 3],
                }
            )
        # 输出 dtype：TTK 传 tensor_dtypes（9 入 + 6 出，C 顺序），用它还原出口窄回
        td = merged.get("tensor_dtypes")
        if td and len(td) >= n_in + 6 and not merged.get("output_dtypes"):
            merged["output_dtypes"] = list(td)[n_in : n_in + 6]
        od = _od(merged)
        # fp16/bf16 输入抬到 fp32 再算（与 KernelSpec 的 _as_tensor 一致）：仅向上兜底，不向下 cast；
        # 镜像 NPU 内核的加宽行为（VF 全程 fp32、出口才窄回）。Promote 开启时输入已被 TTK 抬过 → no-op。
        t = [
            a.to(torch.float32)
            if isinstance(a, torch.Tensor)
            and a.dtype in (torch.bfloat16, torch.float16)
            else a
            for a in t
        ]
        outs = _compute(*t, **merged)
        return _cast_out_tensor(outs, od)

    tolerance = _TOL

    # 三方标杆：复用 kernel 通路同一 torch 参考；浮点已用 cross_check，会实际走三方比对。
    third_party = {"torch": _AddLayerNormQuantV2Compose}


# 【不适用】geir 通路：本算子未注册 AddLayerNormQuantV2 infershape（arch910b/canndev/built-in 三处均无，
# 见 01 §3.3），GE 图模式不可编译（Cannot find Node op1[...] custom infer_datatype func）。
# 【不适用】e2e/torch 通路：torch_npu 未引用 aclnnAddLayerNormQuantV2（strings 查 0 命中，
# 对照 aclnnAddRmsNormQuant 有命中，方法有效）。
