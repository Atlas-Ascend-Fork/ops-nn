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
TTK TestSpec for multi_add_rms_norm_dynamic_quant (kernel / GEIR 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— 真值，torch 算子拼接（add/mean/rsqrt/round/clamp/amax）
                   计算精度跟下发 dtype（零 cast，抬档由 TTK 的 Promote 负责，非 golden）
    third_party  —— 三方标杆，竞品融合 API `F.rms_norm` + 动态量化，在远端 GPU 上执行
                   （torch.compile 融合，竞品最优形态；**不与 golden 同构**，见 ）
    tolerance    —— int8 输出(y1/y2) 配 quant（绝对误差 ≤1 LSB；.5 边界舍入方向属实现
                   自由度，逐位相等=要求复现浮点运算次序，会大面积误判）；浮点输出
                   (x/y 为 fp16/bf16，scale1/scale2 为 fp32) 配 cross_check

Canonical IO order (multi_add_rms_norm_dynamic_quant_def.cpp):
    inputs : x1(TensorList, DYNAMIC), x2, gamma, smooth_scale1(OPT), smooth_scale2(OPT)
    outputs: y1(int8), y2(int8), x(同 x1 dtype), y(同 x1 dtype), scale1(fp32), scale2(fp32)
    attrs  : epsilon(OPTIONAL float=1e-6)

语义（**加法顺序**是语义，它决定舍入点；计算精度不是，见 _compute）:
    x = (x1[0]+x2) + x1[1] + ...     (A2 顺序:x2 第二个加)
    rstd = 1/sqrt(mean(x^2)+eps)     (末轴)
    normed = x * rstd * gamma        => y 输出
    对每个 smooth_scale_j:
      q = normed * smooth_j
      scale_j = max(|q|)/127         (逐行)
      y_j = round_half_even(q/scale_j) -> clamp(-128,127) -> int8
    无 smooth 时 q = normed。
浮点输出的出口 dtype 一律按**下发的** output_dtypes 取（Promote 档它已被抬高）
不按算子声明的 dtype —— 按声明取会把 Promote 抬上来的精度砍回去。
y2/scale2 有效 <=> smooth_scale2 存在(smoothNum==2)。对齐 A2/arch35 内核(hasY2Scale2_=hasSmoothScale2_,
smoothNum<2 时内核不写 y2/scale2、留 buf 初值)——golden 对它们返回 None 哨兵,TTK 比对跳过
(core_modules/npu/op/comparison.py: None/str 哨兵 continue),避免拿内核未写的垃圾值误判 FAIL。
"""

import numpy as np
import torch
import torch.nn.functional as F

# Spec.tolerance 只认官方四标准：stat_rel_err / binary_equal / cross_check / quant
# （close、requant 是 CLI 专用别名，写进 Spec 会 InvalidSpecError）。
# int8 量化输出必须显式配 quant——不声明则整数短路 binary_equal，round 在 .5 边界的
# ±1 LSB tie 会大面积误判红（gnsq 实证）。
_TOL = {
    "int8": {"standard": "quant"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
}


def _attr(kwargs, name, default):
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    return default if v is None else v


def _to_torch(arr):
    """numpy(含 ml_dtypes.bfloat16) -> torch tensor(原 dtype,按位 view 无损机械转换)。"""
    name = arr.dtype.name
    if name == "bfloat16":
        return torch.from_numpy(arr.view(np.uint16).view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(arr))


def _work_dtype(*tensors):
    """计算 dtype：任一输入已是 fp64（即 TTK 的 Promote 抬过）则全程 fp64，否则 fp32。

    **golden 永不自行抬档**——抬真值是 TTK 的职责（ + 三方那一格
    golden 零 cast，入口出口都不做）。这里的 fp32 不是精度决策，是跟随**内核的加宽行为**
    （两方泛化档 TTK 不抬，规范要求 golden 跟 NPU 加宽到哪就到哪）。
    """
    for t in tensors:
        if t is not None and getattr(t, "dtype", None) == torch.float64:
            return torch.float64
    return torch.float32


# ---- mean(x^2) 的规范归约 -------------------------------------------------
# 拓扑是「两两配对 + 奇数进位」的平衡二叉树 —— 与任何实现无关的规范形态, 不含内核的
# lane 宽度 / 链长 / 分核切分。照抄被测切分会让「规约拓扑不当」这类缺陷在比对里恒等
# 消失。仓内既有规范实现: norm/group_norm_silu_quant/tests/assets/golden.py。
# 不是抬档: TwoSum / TwoProd 在原 dtype 下精确, 全程仍是 _work_dtype 决定的精度。
# 换掉 torch.mean 的理由: 它内部是分块 pairwise, 既非内核拓扑也非规范拓扑。


def _clamp_err(e):
    """残差非有限即置 0。和溢出成 inf 时残差为 inf-inf=NaN, 加回去会把本该是 inf 的
    和污染成 NaN, 使 rstd 由 0 变 NaN。内核侧 DfClampResidual 同一处理, 两侧须一致。"""
    return torch.where(torch.isfinite(e), e, torch.zeros_like(e))


def _two_sum(a, b):
    """Knuth TwoSum: a + b = s + err, 在 a/b 的 dtype 下精确。"""
    s = a + b
    bb = s - a
    return s, _clamp_err((a - (s - bb)) + (b - bb))


def _split(a):
    """Dekker 拆分, 常量 = 2^ceil(p/2) + 1(fp32: 4097, fp64: 134217729)。"""
    p = 24 if a.dtype == torch.float32 else 53
    c = a * float((1 << ((p + 1) // 2)) + 1)
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    """精确乘积 a*b = p + err。"""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, _clamp_err((((ah * bh - p) + ah * bl) + al * bh) + al * bl)


def _df_add(ah, al, bh, bl):
    s, e = _two_sum(ah, bh)
    return _two_sum(s, e + (al + bl))


def _df_tree_sum(hi, lo):
    """沿末轴做双浮点两两树求和, 每层一次张量运算, 共 ceil(log2(n)) 层。"""
    while hi.shape[-1] > 1:
        n = hi.shape[-1]
        half = n // 2
        sh, sl = _df_add(
            hi[..., 0 : 2 * half : 2],
            lo[..., 0 : 2 * half : 2],
            hi[..., 1 : 2 * half : 2],
            lo[..., 1 : 2 * half : 2],
        )
        if n % 2:  # 节点数为奇: 落单的一条原样进下一层
            sh = torch.cat([sh, hi[..., -1:]], dim=-1)
            sl = torch.cat([sl, lo[..., -1:]], dim=-1)
        hi, lo = sh, sl
    return hi[..., 0], lo[..., 0]


def _mean_square(x):
    """mean(x^2)(末轴): 精确平方 -> 双浮点平衡树求和 -> 双浮点除以 n。
    除法先取商再用余项修正, 避免 (hi+lo) 先折回再除的双重舍入。"""
    ph, pl = _two_prod(x, x)
    sh, sl = _df_tree_sum(ph, pl)
    n = torch.tensor(float(x.shape[-1]), dtype=x.dtype)
    q = sh / n
    qh, ql = _two_prod(q, n)
    rh, rl = _df_add(sh, sl, -qh, -ql)
    # 修正项非有限即丢弃: sh 溢出成 inf 时 sh-q*n = inf-inf = NaN(这次 NaN 落在和上,
    # 不是残差上, _clamp_err 在 _two_sum 内部拦不住), 加回去会把本该是 inf 的 var 变 NaN。
    # var=inf -> rstd=0 才与内核一致(内核 1/sqrt(inf)=0)。
    return (q + _clamp_err((rh + rl) / n)).unsqueeze(-1)


def _to_np(t, dtype_name):
    """torch -> numpy 的目标 dtype。

    dtype_name 取自**下发的** output_dtypes(Promote 档已被抬高), 不是算子声明值。
    bf16 经 ml_dtypes 按位 view —— numpy/torch 的 bf16 载体不互通, 这是载体转换
    不是精度决策。
    """
    if dtype_name == "bfloat16":
        import ml_dtypes

        return (
            t.to(torch.bfloat16)
            .view(torch.int16)
            .numpy()
            .view(np.uint16)
            .view(ml_dtypes.bfloat16)
        )
    return t.to(getattr(torch, dtype_name)).numpy()


def _compute(x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, **kwargs):
    """torch.Tensor 进 / 出（x1 为 list[Tensor]），返回 [y1, y2, x, y, scale1, scale2]，
    顺序照 def.cpp；y2/scale2 在 smooth_scale2 缺省时为 None 哨兵。

    计算精度 = `_work_dtype` —— **golden 绝不自行抬档**:
      · 三方档(cross_check): TTK 设 `golden_mode_override="Promote"` 已把输入抬一档
        (fp16/bf16→fp32、fp32→fp64), 抬真值是**工具的职责**, golden 照单全收即规范
         的"零 cast";
      · 两方泛化档: TTK 不抬, golden 按**内核的加宽行为**走(arch35 内核通篇
        `LocalTensor<float>`/`RegTensor<float>` ⇒ fp16/bf16 抬 fp32、fp32 不再抬)。
    两档合起来就是"任一输入已是 fp64 则全程 fp64, 否则 fp32"——没有任何一条路径需要
    golden 自己 `.to(float64)`。
    加法**顺序**严格照 A2(它决定舍入点, 属语义)。
    出口窄回交由各路壳按**下发的** output_dtypes 处理, 此处不窄。
    """
    epsilon = float(_attr(kwargs, "epsilon", 1e-6))

    x1_list = list(x1) if isinstance(x1, (list, tuple)) else [x1]
    w = _work_dtype(*x1_list, x2, gamma, smooth_scale1, smooth_scale2)

    # A2 权威的加法顺序(对齐 canndev normal/single_row_kernel 的注释
    # "1.将x1的第一个tensor和x2相加 2.将x1剩余的相加"):
    # x = (x1[0]+x2) + x1[1]+...+x1[n-1]
    xw = x1_list[0].to(w) + x2.to(w)
    for t in x1_list[1:]:
        xw = xw + t.to(w)

    var = _mean_square(xw)
    rstd = torch.rsqrt(var + epsilon)
    normed = xw * rstd * gamma.to(w)

    def quant(smooth):
        q = normed if smooth is None else normed * smooth.to(w)
        amax = q.abs().amax(dim=-1, keepdim=True)
        scale = amax / 127.0
        # amax == 0 <=> 整行 q 恒为 0(amax 是 |q| 的最大值), 此时 scale = 0, 公式的
        # round(q/scale) 退化为 0/0 无定义。而 q 恒为 0 意味着任何非零分母都给出同一个
        # 结果 0, 故分母取 1 是该退化点唯一自洽的定义; scale 仍按定义式输出 0。
        yq = torch.round(q / torch.where(amax == 0, torch.ones_like(scale), scale))
        # int8 无法表示 NaN/inf, 而量化输出的 dtype 契约是 int8, 故必须定义落位:
        # NaN 取 0(对称量化的零点); ±inf 经 nan_to_num 换成 ±FLT_MAX, 再由下面的
        # clamp 落到 +127/-128。显式写出, 不依赖 torch `.to(torch.int8)` 对 NaN 的
        # 实现定义行为(实测给 0, 但文档未作保证)。两种写法在
        # {nan, ±inf, ±200, ±0, 63.5} 上逐位相等, 本改动不改变任何已有判据结果。
        yq = torch.nan_to_num(yq, nan=0.0)
        yq = torch.clamp(yq, -128, 127).to(torch.int8)
        return yq, scale.squeeze(-1)

    y1_t, scale1_t = quant(smooth_scale1)
    if smooth_scale2 is not None:
        y2_t, scale2_t = quant(smooth_scale2)
    else:
        y2_t, scale2_t = None, None
    return [y1_t, y2_t, xw, normed, scale1_t, scale2_t]


class _MarndqCompose:
    """三方标杆：竞品实现在远端 GPU 执行。参数名与 def.cpp 逐字一致
    （x1/x2/gamma/smooth_scale1/smooth_scale2 + attr epsilon）。

    竞品最优形态（性能对标铁律）：torch.compile 融合（eager 逐算子 launch 全是开销
    NPU 侧是单个融合内核，拿未融合 eager 比不是同一层次）；编译失败回落 eager。
    x/y 输出 cast 回 NPU 的输出 dtype（fp16/bf16）——竞品留在 fp32 会让 cross_check
    的 ratio 凭空爆表（gnsq 实测 mare 961→1.0 的教训）。
    smoothNum<2 时 y2/scale2 返回空占位（golden 侧 None 哨兵使该输出整列跳过比对
    三方值不会被读取；空 tensor 仅为远端序列化安全）。
    """

    def __init__(self, *, epsilon=1e-6, **_):
        self.epsilon = float(epsilon)
        self._compiled = None

    def _impl(self, x1, x2, gamma, smooth_scale1=None, smooth_scale2=None):
        x1_list = list(x1) if isinstance(x1, (list, tuple)) else [x1]
        # 三方腿按 NPU 的加宽行为同步 cast(内核 U=fp32), 出口对应窄回
        x_fp32 = x1_list[0].to(torch.float32) + x2.to(torch.float32)
        for t in x1_list[1:]:
            x_fp32 = x_fp32 + t.to(torch.float32)
        # 归一化用竞品的**融合 API** F.rms_norm, 不把 golden 的 mean/rsqrt 表达式抄一遍
        # —— 规范 : "三方腿与 golden 同构" 是退化的首因, 修法在三方腿侧。
        # 语义已实测与 x*rsqrt(mean(x^2)+eps)*g 一致(D=64/512/15360 三档 maxrel=0)。
        # 它在 CPU 上与手写链逐位相同(V5 的告警), 独立性要在 GPU 腿上验。
        normed = F.rms_norm(
            x_fp32,
            (x_fp32.shape[-1],),
            weight=gamma.to(torch.float32),
            eps=self.epsilon,
        )

        def quant(smooth):
            q = normed if smooth is None else normed * smooth.to(torch.float32)
            amax = q.abs().amax(dim=-1, keepdim=True)
            scale = amax / 127.0
            denom = torch.where(amax == 0, torch.ones_like(scale), scale)
            # NaN 落位显式化, 理由同 _compute.quant —— 这一腿跑在 GPU 上, CUDA 的
            # `.to(torch.int8)` 对 NaN 未必与 CPU 同值, 不能依赖实现定义行为。
            yq = torch.nan_to_num(torch.round(q / denom), nan=0.0)
            yq = torch.clamp(yq, -128, 127).to(torch.int8)
            return yq, scale.squeeze(-1).to(torch.float32)

        y1, scale1 = quant(smooth_scale1)
        x_out = x_fp32.to(x1_list[0].dtype)
        y_out = normed.to(x1_list[0].dtype)
        if smooth_scale2 is not None:
            y2, scale2 = quant(smooth_scale2)
        else:
            y2 = torch.zeros((0,), dtype=torch.int8, device=x2.device)
            scale2 = torch.zeros((0,), dtype=torch.float32, device=x2.device)
        return [y1, y2, x_out, y_out, scale1, scale2]

    def __call__(self, x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, **_):
        if self._compiled is None:
            try:
                self._compiled = torch.compile(self._impl)
            except Exception:
                self._compiled = self._impl
        try:
            return self._compiled(x1, x2, gamma, smooth_scale1, smooth_scale2)
        except Exception:
            return self._impl(x1, x2, gamma, smooth_scale1, smooth_scale2)


def _inject_nonfinite(x1, testcase_name=""):
    """错误场景档 nonfinite 的定点注入: 只对用例名含 `_nonfinite` 的用例生效。

    为什么不能靠 CSV 的 `input_data_ranges` 写 inf/nan: TTK 的 RandomData 会把值域里的
    inf/nan **钳到 dtype 极值**(`ttk/utilities/data.py` 的 `_digitize_inf_nan`), 写了也
    造不出非有限数据 —— 那一档会变成"名字叫 nonfinite、数据却全是普通数"的空跑。

    只注入 x1[0](数据面的首个张量), **不动 gamma / smooth_scale**(权重面): 权重里放 inf
    会让整行 normed 全非有限, 测的是"全行污染"而不是"非有限值在末轴归约里的传播", 两码事。
    位置固定不随机(随机会让复现依赖 seed, 小 shape 时还可能一个都注不进去):
    首元素 +inf、第 2 个 -inf、第 3 个 nan。元素数 < 4 不注入。

    预期行为: 按 IEEE 语义自然传播(`mean(x^2)` 含 inf -> rstd=0 或 nan), 仍走正常精度判据,
    不是拒收档。
    """
    if "_nonfinite" not in (testcase_name or ""):
        return x1
    lst = list(x1) if isinstance(x1, (list, tuple)) else [x1]
    if not lst:
        return x1
    a = np.array(lst[0], copy=True)
    if a.size < 4:
        return x1
    f = a.reshape(-1)
    f[0], f[1], f[2] = np.inf, -np.inf, np.nan
    out = [a] + lst[1:]
    return out if isinstance(x1, (list, tuple)) else out[0]


class MultiAddRmsNormDynamicQuantSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray（x1 为 list[np.ndarray]）、返 list。"""

    def golden(x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, **kwargs):
        x1_list = list(x1) if isinstance(x1, (list, tuple)) else [x1]
        # "已连接但 0 元素"的 smooth 按**不存在**处理, 与 host 判据对齐:
        # tiling_arch35.cpp:102-106 与 infershape.cpp:59-65 都用 GetShapeSize()>0 判存在,
        # 即空 smooth 走"无 smooth"的量化公式。此前 golden 直接 normed*smooth, 遇 shape (0,)
        # 广播报错 ⇒ GOLDEN_FAILURE(用例 supp2_smooth_zero_elem_f16 实测)。
        smooth_scale1 = (
            None
            if smooth_scale1 is not None and np.asarray(smooth_scale1).size == 0
            else smooth_scale1
        )
        smooth_scale2 = (
            None
            if smooth_scale2 is not None and np.asarray(smooth_scale2).size == 0
            else smooth_scale2
        )
        outs = _compute(
            [_to_torch(a) for a in x1_list],
            _to_torch(x2),
            _to_torch(gamma),
            _to_torch(smooth_scale1) if smooth_scale1 is not None else None,
            _to_torch(smooth_scale2) if smooth_scale2 is not None else None,
            **kwargs,
        )
        y1_t, y2_t, x_t, y_t, scale1_t, scale2_t = outs
        # 出口按下发的 output_dtypes 取: 三方档它已被 Promote 抬高(x/y fp16/bf16→fp32
        # scale fp32→fp64), 两方泛化档则是算子声明值(golden 跟内核的加宽行为窄回)。
        # 取不到时退回输入 dtype / fp32, 与 Promote 前的行为一致。
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        in_name = x1_list[0].dtype.name

        def dt(i, fallback):
            return od[i] if len(od) > i and od[i] else fallback

        # y1/y2 的 int8 是**量化契约**而非精度落点, 不随 Promote(整型无 Promote 条目)。
        y1 = y1_t.numpy()
        y2 = y2_t.numpy() if y2_t is not None else None
        x = _to_np(x_t, dt(2, in_name))
        y = _to_np(y_t, dt(3, in_name))
        scale1 = _to_np(scale1_t, dt(4, "float32"))
        scale2 = _to_np(scale2_t, dt(5, "float32")) if scale2_t is not None else None
        return [y1, y2, x, y, scale1, scale2]

    def customize_inputs(
        x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, **kwargs
    ):
        return (
            _inject_nonfinite(x1, kwargs.get("testcase_name", "")),
            x2,
            gamma,
            smooth_scale1,
            smooth_scale2,
        )

    third_party = {"torch": _MarndqCompose}
    tolerance = _TOL


def multi_add_rms_norm_dynamic_quant_golden(
    x1, x2, gamma, smooth_scale1=None, smooth_scale2=None, **kwargs
):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return tuple(
        MultiAddRmsNormDynamicQuantSpec.golden(
            x1, x2, gamma, smooth_scale1, smooth_scale2, **kwargs
        )
    )


__spec__ = {"multi_add_rms_norm_dynamic_quant": "MultiAddRmsNormDynamicQuantSpec"}
__golden__ = {
    "kernel": {
        "multi_add_rms_norm_dynamic_quant": "multi_add_rms_norm_dynamic_quant_golden"
    }
}

# 【不存在】aclnn / e2e / tf / onnx 通路：A2 仅 GE 图 + 融合 pass（无 op_api/torch 绑定/
# 框架插件），A5 对齐不新增（01_requirement.md §3.3：通路 = GE 图 + 融合 pass）。
