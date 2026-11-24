#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK golden plugin for DeformableConv2D (kernel/geir flow).

参考实现按算子分解语义计算：DeformableOffsets + Conv2D(stride = kernel_size)。

offsets 通道为 value-block 布局 ``[dx block | dy block | mask block]``，每个 block
``deformable_groups * kh * kw`` 个通道（组主序，组内 kernel 行主序）；``modulated``
为 False 时无 mask block。卷积 padding 为零填充：双线性插值的四个邻域只在原始图像
``[0, ih-1] x [0, iw-1]`` 内取值，越界邻域贡献 0。
"""

__spec__ = {"deformable_conv2_d": "DeformableConv2DTestSpec"}

import numpy as np


def _bilinear_sample(x, ph, pw):
    """按分数坐标采样 x ``[n, c, ih, iw]``。

    ``ph`` / ``pw`` 为原始（未 pad）坐标系下的采样位置，形状 ``[n, oh, ow]``；
    邻域越出 ``[0, ih-1] x [0, iw-1]`` 时取 0，与零填充卷积语义一致。
    """
    n, c, ih, iw = x.shape
    oh, ow = ph.shape[1], pw.shape[2]
    h0 = np.floor(ph).astype(np.int64)
    w0 = np.floor(pw).astype(np.int64)
    vh = (ph - h0)[:, None, :, :]
    vw = (pw - w0)[:, None, :, :]
    sample = np.zeros((n, c, oh, ow), dtype=np.float64)
    flat = x.reshape(n, c, ih * iw)
    for ha, wa, fa, fb in (
        (h0, w0, 1 - vh, 1 - vw),
        (h0, w0 + 1, 1 - vh, vw),
        (h0 + 1, w0, vh, 1 - vw),
        (h0 + 1, w0 + 1, vh, vw),
    ):
        valid = (ha >= 0) & (ha <= ih - 1) & (wa >= 0) & (wa <= iw - 1)
        if not valid.any():
            continue
        gidx = (np.clip(ha, 0, ih - 1) * iw + np.clip(wa, 0, iw - 1)).reshape(
            n, 1, oh * ow
        )
        gathered = np.take_along_axis(
            flat, np.broadcast_to(gidx, (n, c, oh * ow)), axis=2
        ).reshape(n, c, oh, ow)
        sample += np.where(valid[:, None, :, :], gathered * fa * fb, 0.0)
    return sample


def _output_dtype(x):
    """fp16/bf16 输入以 fp32 输出参与比对（容差覆盖低精度舍入），其余与输入一致。"""
    if x.dtype in (np.float16,):
        return np.float32
    if str(x.dtype) == "bfloat16":
        return np.float32
    if x.dtype in (np.int8, np.int16):
        return np.int32
    return x.dtype


class DeformableConv2DTestSpec:
    @staticmethod
    def golden(x, w, *args, **kwargs):
        offsets = args[0] if len(args) > 0 else None
        bias = args[1] if len(args) > 1 else None

        attrs = kwargs.get("attributes", kwargs) or {}
        strides = list(attrs.get("strides", [1, 1, 1, 1]))
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        dilations = list(attrs.get("dilations", [1, 1, 1, 1]))
        groups = int(attrs.get("groups", 1) or 1)
        dfm = int(attrs.get("deformable_groups", 1) or 1)
        modulated = bool(attrs.get("modulated", True))
        data_format = str(attrs.get("data_format", "NCHW") or "NCHW").upper()

        out_dtype = _output_dtype(x)
        if x is None or w is None or x.ndim < 4 or w.ndim < 4:
            return [np.zeros((0,), dtype=out_dtype)]

        x = np.ascontiguousarray(x).astype(np.float64)
        w = np.ascontiguousarray(w).astype(np.float64)
        off = None
        if offsets is not None:
            off = np.ascontiguousarray(offsets).astype(np.float64)

        nhwc = DeformableConv2DTestSpec._resolve_nhwc(
            x, w, off, data_format, dfm, modulated
        )
        if nhwc:
            x = np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
            if off is not None:
                off = np.transpose(off, (0, 3, 1, 2))  # NHWC -> NCHW
            w = np.transpose(w, (3, 2, 0, 1))  # HWCN -> NCHW
            sh, sw = int(strides[1]), int(strides[2])
            dh, dw = int(dilations[1]), int(dilations[2])
        else:
            sh, sw = int(strides[2]), int(strides[3])
            dh, dw = int(dilations[2]), int(dilations[3])
        pt, pb, pl, pr = int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])

        n, ic, ih, iw = x.shape
        kn, kc, kh, kw = w.shape
        oh = (ih + pt + pb - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + pl + pr - dw * (kw - 1) - 1) // sw + 1
        oh, ow = max(oh, 0), max(ow, 0)

        def _empty_like():
            shape = (n, oh, ow, kn) if nhwc else (n, kn, oh, ow)
            return [np.zeros(shape, dtype=out_dtype)]

        if (
            oh == 0
            or ow == 0
            or kn == 0
            or kh == 0
            or kw == 0
            or ic == 0
            or x.size == 0
        ):
            return _empty_like()

        kk = kh * kw
        if off is None:
            off = np.zeros(
                (n, (3 if modulated else 2) * dfm * kk, oh, ow), dtype=np.float64
            )
            if modulated:
                off[:, 2 * dfm * kk :, :, :] = 1.0
        elif off.shape[1] == 2 * dfm * kk:
            modulated = False  # 2-block offsets 即 DCNv1，与 attr 缺省无关

        dx = off[:, 0 : dfm * kk, :, :]
        dy = off[:, dfm * kk : 2 * dfm * kk, :, :]
        mk = off[:, 2 * dfm * kk : 3 * dfm * kk, :, :] if modulated else None

        base_h = np.arange(oh).reshape(oh, 1) * sh - pt
        base_w = np.arange(ow).reshape(1, ow) * sw - pl
        cg = ic // dfm
        ocg = kn // groups
        deformed = np.zeros((n, dfm, cg, kh, oh, kw, ow), dtype=np.float64)
        for g in range(dfm):
            for i in range(kh):
                for j in range(kw):
                    kidx = g * kk + i * kw + j
                    ph = np.broadcast_to(base_h, (n, oh, ow)) + i * dh + dy[:, kidx]
                    pw = np.broadcast_to(base_w, (n, oh, ow)) + j * dw + dx[:, kidx]
                    sample = _bilinear_sample(x[:, g * cg : (g + 1) * cg], ph, pw)
                    if mk is not None:
                        sample = sample * mk[:, kidx][:, None, :, :]
                    deformed[:, g, :, i, :, j, :] = sample
        # [n, dfm, cg, kh, oh, kw, ow] -> [n, ic, oh*kh, ow*kw]
        deform = deformed.transpose(0, 1, 2, 4, 3, 6, 5).reshape(
            n, ic, oh * kh, ow * kw
        )
        blocks = deform.reshape(n, ic, oh, kh, ow, kw)

        y = np.zeros((n, kn, oh, ow), dtype=np.float64)
        for og in range(groups):
            for oci in range(ocg):
                oc = og * ocg + oci
                acc = np.zeros((n, oh, ow), dtype=np.float64)
                for ci in range(kc):
                    icc = og * kc + ci
                    for i in range(kh):
                        for j in range(kw):
                            acc += blocks[:, icc, :, i, :, j] * w[oc, ci, i, j]
                y[:, oc, :, :] = acc
        if bias is not None:
            y += np.ascontiguousarray(bias).astype(np.float64).reshape(1, kn, 1, 1)

        if nhwc:
            y = np.transpose(y, (0, 2, 3, 1))  # NCHW -> NHWC
        return [y.astype(out_dtype)]

    @staticmethod
    def _resolve_nhwc(x, w, off, data_format, dfm, modulated):
        """判定输入实际排布。

        ``data_format`` 属性与实际张量可能冲突（如属性写 NHWC 而张量为 NCHW），
        此时按张量实际排布计算，与 infershape 以输入 origin format 优先的规则一致。
        """
        if data_format != "NHWC" or x.ndim != 4 or w.ndim != 4:
            return False
        kk = int(w.shape[0]) * int(w.shape[1])  # HWCN: 前两维是 kernel
        exp_off_c = (3 if modulated else 2) * dfm * kk
        c_ok = x.shape[3] == int(w.shape[2])  # x.C == filter.C
        off_ok = off is None or (off.ndim == 4 and off.shape[3] == exp_off_c)
        return bool(c_ok and off_ok)


def _legacy_to_np(t):
    """torch.Tensor（含 bf16）→ float64 numpy；None 透传。"""
    if t is None:
        return None
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy().astype(np.float64)
    except ImportError:
        pass
    return np.ascontiguousarray(t).astype(np.float64)


def _legacy_deformable_conv2d(x, w, offsets, bias, attrs, output_dtypes):
    """旧 __golden__ 约定的 aclnn/e2e 参考实现。

    与设备数据流一致（CASE-2026-009）：deformOut 先按输出 dtype（低精度）物化，
    再做最终卷积；返回 (out, deformOut) 两个输出，均按声明 dtype 舍入。
    参数排布与张量格式按 aclnnDeformableConv2d / torch_npu.npu_deformable_conv2d
    的 NCHW 语义（strides/pads/dilations 取后两维）。
    """
    strides = list(attrs.get("strides", attrs.get("stride", [1, 1, 1, 1])))
    pads = list(attrs.get("pads", attrs.get("padding", [0, 0, 0, 0])))
    dilations = list(attrs.get("dilations", attrs.get("dilation", [1, 1, 1, 1])))
    groups = int(attrs.get("groups", 1) or 1)
    dfm = int(attrs.get("deformable_groups", attrs.get("deformableGroups", 1)) or 1)
    modulated = bool(attrs.get("modulated", True))

    if len(strides) == 2:
        strides = [1, 1, strides[0], strides[1]]
    if len(pads) == 2:
        pads = [pads[0], pads[0], pads[1], pads[1]]
    if len(dilations) == 2:
        dilations = [1, 1, dilations[0], dilations[1]]

    out_dtype_str = output_dtypes[0] if output_dtypes else None
    deform_dtype_str = output_dtypes[1] if len(output_dtypes) > 1 else None

    x64 = _legacy_to_np(x)
    w64 = _legacy_to_np(w)
    off64 = _legacy_to_np(offsets) if offsets is not None else None
    bias64 = _legacy_to_np(bias) if bias is not None else None

    # ---- 阶段 1：DeformableOffsets（双线性 + mask），得 deformOut ----
    # 复用 spec golden 的采样与卷积核，但以“低精度物化 deformOut”为分界：
    # 先在 fp64 下求 deformOut（含 mask），物化后再进卷积。
    n, ic, ih, iw = x64.shape
    kn, kc, kh, kw = w64.shape
    sh, sw = int(strides[2]), int(strides[3])
    dh, dw = int(dilations[2]), int(dilations[3])
    pt, pb = int(pads[0]), int(pads[1])
    pl, pr = int(pads[2]), int(pads[3])
    oh = (ih + pt + pb - dh * (kh - 1) - 1) // sh + 1
    ow = (iw + pl + pr - dw * (kw - 1) - 1) // sw + 1
    oh, ow = max(oh, 0), max(ow, 0)

    kk = kh * kw
    if off64 is None:
        off64 = np.zeros(
            (n, (3 if modulated else 2) * dfm * kk, oh, ow), dtype=np.float64
        )
        if modulated:
            off64[:, 2 * dfm * kk :, :, :] = 1.0
    elif off64.shape[1] == 2 * dfm * kk:
        modulated = False

    # 旧 aclnn/e2e 表的 offsets 为 value-block 布局（与 NPU DeformableOffsets
    # kernel 一致）：[w 块 | h 块 | (mask 块)]，每块 dfm*kk 通道，块内
    # deformable-group 主序、组内 kernel 行主序。区别于 spec golden 的
    # group 主序切分。
    w_blk = off64[:, 0 : dfm * kk, :, :]
    h_blk = off64[:, dfm * kk : 2 * dfm * kk, :, :]
    mk = off64[:, 2 * dfm * kk : 3 * dfm * kk, :, :] if modulated else None

    base_h = np.arange(oh).reshape(oh, 1) * sh - pt
    base_w = np.arange(ow).reshape(1, ow) * sw - pl
    cg = ic // dfm
    # deformOut: [n, ic, oh*kh, ow*kw]，行 = oh*kH + ki（oh 外层、ki 内层），
    # 列 = ow*kW + kj；通道按 deformable-group 连续分组（dg*k_per_group + c）。
    deform = np.zeros((n, ic, oh * kh, ow * kw), dtype=np.float64)
    for g in range(dfm):
        ch = g * cg
        for i in range(kh):
            for j in range(kw):
                kidx = g * kk + i * kw + j
                ph = np.broadcast_to(base_h, (n, oh, ow)) + i * dh + h_blk[:, kidx]
                pw = np.broadcast_to(base_w, (n, oh, ow)) + j * dw + w_blk[:, kidx]
                sample = _bilinear_sample(x64[:, ch : ch + cg], ph, pw)
                if mk is not None:
                    sample = sample * mk[:, kidx][:, None, :, :]
                for ohp in range(oh):
                    for owp in range(ow):
                        deform[:, ch : ch + cg, ohp * kh + i, owp * kw + j] = sample[
                            :, :, ohp, owp
                        ]
    # deformOut 物化：与设备数据流一致（CASE-2026-009）——低精度输出声明
    # 时 deformOut 先舍入到声明 dtype 再进卷积。fp64 中间量转 fp32 后经
    # torch 物化（RNE 与设备一致）。
    deform_np = _materialize(deform, deform_dtype_str)

    # ---- 阶段 2：Conv2D(stride=kernel)，消费物化后的 deformOut ----
    # 卷积窗取 deformOut 的 [oh*kh:(oh+1)*kh, ow*kw:(ow+1)*kw] 块
    y = np.zeros((n, kn, oh, ow), dtype=np.float64)
    ocg = kn // groups
    for og in range(groups):
        for oci in range(ocg):
            oc = og * ocg + oci
            acc = np.zeros((n, oh, ow), dtype=np.float64)
            for ci in range(kc):
                icc = og * kc + ci
                for i in range(kh):
                    for j in range(kw):
                        acc += deform_np[:, icc, i::kh, j::kw] * w64[oc, ci, i, j]
            y[:, oc, :, :] = acc
    if bias64 is not None:
        y += bias64.reshape(1, kn, 1, 1)

    # ---- 输出按声明 dtype 舍入（同 _materialize）----
    out_np = _materialize(y, out_dtype_str)
    deform_result = _materialize(deform_np, deform_dtype_str)
    return [out_np, deform_result]


def _materialize(a, dtype_str):
    """fp64 中间量按声明 dtype 物化（低精度舍入后回 fp64 载体）。

    bf16/fp16 舍入经 torch 完成（round-to-nearest-even，与设备一致）；
    torch 不可用时回退 numpy（fp16 精确，bf16 经 fp32 截断近似）。
    dtype_str 为 None/float32 时仅转 fp32 载体（无损）。
    """
    a32 = a.astype(np.float32)
    if dtype_str in ("bfloat16", "float16"):
        try:
            import torch

            t = torch.from_numpy(a32)
            t = t.to(torch.bfloat16 if dtype_str == "bfloat16" else torch.float16)
            return t.float().numpy().astype(np.float64)
        except ImportError:
            if dtype_str == "float16":
                return a32.astype(np.float16).astype(np.float64)
            return a32.astype(np.float64)
    return a32.astype(np.float64)


def _legacy_output_dtypes(kwargs):
    od = kwargs.get("output_dtypes")
    if od:
        return [str(d) for d in od]
    td = kwargs.get("tensor_dtypes")
    if td and len(td) >= 6:
        return [str(td[4]) if td[4] else None, str(td[5]) if td[5] else None]
    return [None, None]


def deformable_conv2d_golden(
    x,
    weight,
    offset,
    bias_optional=None,
    kernel_size: list = None,
    stride: list = None,
    padding: list = None,
    dilation: list = None,
    groups: int = 1,
    deformable_groups: int = 1,
    modulated: bool = True,
    out=None,
    deform_out_optional=None,
    **kwargs,
):
    """旧 __golden__ 约定入口：aclnnDeformableConv2d / torch_npu.npu_deformable_conv2d。

    RT2.0 迁移后 kernel/geir 走 DeformableConv2DTestSpec；本函数按旧约定为
    opstest 既有 ttk_aclnn/e2e_deformable_conv2d 表（双输出 out+deformOut，
    低精度声明）提供 golden，参数顺序遵循 aclnnDeformableConv2dGetWorkspaceSize
    的 C 头签名。精度口径与设备数据流一致：deformOut 按声明 dtype 物化后再进
    卷积（CASE-2026-009）。
    """
    attrs = {
        "kernel_size": kernel_size,
        "strides": stride if stride is not None else [1, 1, 1, 1],
        "pads": padding if padding is not None else [0, 0, 0, 0],
        "dilations": dilation if dilation is not None else [1, 1, 1, 1],
        "groups": groups,
        "deformable_groups": deformable_groups,
        "modulated": modulated,
    }
    attrs.update(
        {k: v for k, v in (kwargs.get("attributes") or {}).items() if v is not None}
    )
    output_dtypes = _legacy_output_dtypes(kwargs)
    results = _legacy_deformable_conv2d(
        x, weight, offset, bias_optional, attrs, output_dtypes
    )

    # 按旧约定的返回 dtype（np 载体）：声明 bf16 时以 ml_dtypes.bfloat16 返回
    out_dtype_str = output_dtypes[0]
    out_np, deform_np = results
    if out_dtype_str == "bfloat16":
        try:
            import ml_dtypes

            out_np = out_np.astype(ml_dtypes.bfloat16)
            deform_np = deform_np.astype(ml_dtypes.bfloat16)
        except ImportError:
            pass  # 无 ml_dtypes 时以 fp64 载体返回（数值已是 bf16 舍入后）
    elif out_dtype_str == "float16":
        out_np = out_np.astype(np.float16)
        deform_np = deform_np.astype(np.float16)
    return [out_np, deform_np]


__golden__ = {
    "aclnn": {"aclnnDeformableConv2d": "deformable_conv2d_golden"},
    "e2e": {"torch_npu.npu_deformable_conv2d": "deformable_conv2d_golden"},
}
