#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

__spec__ = {
    "dilation2_d": "Dilation2DKernelSpec",
}

__golden__ = {"kernel": {"dilation2_d": "dilation2_d_golden"}}

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
}


class _Dilation2DTorchCompose:
    """Third-party reference executed on the remote GPU server.

    torch 组合实现：按 kernel 语义手工 pad(-inf) 后对每个 (fh, fw) 位移取 max。
    """

    def __call__(
        self,
        x=None,
        filter=None,
        *tensors,
        strides=None,
        rates=None,
        padding_mode="SAME",
        pads=None,
        ceil_mode=False,
        data_format="NHWC",
        **kwargs,
    ):
        import torch

        if x is None:
            x = tensors[0]
        if filter is None and len(tensors) > 1:
            filter = tensors[1]
        if strides is None:
            strides = [1, 1, 1, 1]
        if rates is None:
            rates = [1, 1, 1, 1]

        orig_torch_dtype = torch.as_tensor(x if x is not None else tensors[0]).dtype
        if data_format == "NCHW":
            x_t = torch.as_tensor(x).permute(0, 2, 3, 1)
            filter_t = torch.as_tensor(filter).permute(1, 2, 0)
            stride_h, stride_w = strides[2], strides[3]
            rate_h, rate_w = rates[2], rates[3]
        else:
            x_t = torch.as_tensor(x)
            filter_t = torch.as_tensor(filter)
            stride_h, stride_w = strides[1], strides[2]
            rate_h, rate_w = rates[1], rates[2]

        N, H, W, C = x_t.shape
        fH, fW, _ = filter_t.shape
        filter_h_eff = fH + (fH - 1) * (rate_h - 1)
        filter_w_eff = fW + (fW - 1) * (rate_w - 1)

        if padding_mode == "SAME":
            oH = -(-H // stride_h)
            oW = -(-W // stride_w)
            pad_h = max((oH - 1) * stride_h + filter_h_eff - H, 0)
            pad_w = max((oW - 1) * stride_w + filter_w_eff - W, 0)
            pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
            pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        elif padding_mode == "VALID":
            oH = max(0, -(-(H - filter_h_eff + 1) // stride_h))
            oW = max(0, -(-(W - filter_w_eff + 1) // stride_w))
            pad_top = pad_bottom = pad_left = pad_right = 0
        else:
            p = list(pads) if pads is not None else [0, 0, 0, 0]
            pad_top, pad_bottom, pad_left, pad_right = p[0], p[1], p[2], p[3]
            h_num = H + pad_top + pad_bottom - filter_h_eff
            w_num = W + pad_left + pad_right - filter_w_eff
            if ceil_mode:
                oH = max(0, -(-h_num // stride_h) + 1)
                oW = max(0, -(-w_num // stride_w) + 1)
            else:
                oH = max(0, h_num // stride_h + 1)
                oW = max(0, w_num // stride_w + 1)
        compute_dtype = (
            torch.float64
            if not x_t.dtype.is_floating_point
            else torch.promote_types(x_t.dtype, torch.float32)
        )
        if oH == 0 or oW == 0:
            y = torch.empty((N, oH, oW, C), dtype=compute_dtype)
        else:
            xp = x_t.to(compute_dtype).permute(0, 3, 1, 2)
            ft = filter_t.to(compute_dtype)
            # 覆盖全部窗口所需的最小 padded 高宽；不足处补 -inf（与 kernel 越界跳过等价）
            need_h = max(
                H + pad_top + pad_bottom, (oH - 1) * stride_h + (fH - 1) * rate_h + 1
            )
            need_w = max(
                W + pad_left + pad_right, (oW - 1) * stride_w + (fW - 1) * rate_w + 1
            )
            xp = torch.nn.functional.pad(
                xp,
                (pad_left, need_w - W - pad_left, pad_top, need_h - H - pad_top),
                value=float("-inf"),
            )
            out = torch.full(
                (N, C, oH, oW), float("-inf"), dtype=compute_dtype, device=xp.device
            )
            for fh in range(fH):
                h_start = fh * rate_h
                h_end = h_start + (oH - 1) * stride_h + 1
                for fw in range(fW):
                    w_start = fw * rate_w
                    w_end = w_start + (oW - 1) * stride_w + 1
                    window = xp[:, :, h_start:h_end:stride_h, w_start:w_end:stride_w]
                    cand = window + ft[fh, fw, :].view(1, C, 1, 1)
                    # 与 TF/kernel 语义一致：NaN 候选不参与 max（NaN > out 恒为 False）
                    out = torch.where(cand > out, cand, out)
            y = out.permute(0, 2, 3, 1)
        if data_format == "NCHW":
            y = y.permute(0, 3, 1, 2)
        if not x_t.dtype.is_floating_point:
            info = torch.iinfo(orig_torch_dtype)
            y = torch.clamp(torch.round(y), info.min, info.max)
        return y.to(orig_torch_dtype)


class Dilation2DKernelSpec:
    """kernel spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(x, filter, **kwargs):
        return dilation2_d_golden(x, filter, **kwargs)

    third_party = {"torch": _Dilation2DTorchCompose()}
    tolerance = _TOLERANCE


def dilation2_d_golden(
    x,
    filter,
    *,
    strides=None,
    rates=None,
    padding_mode="SAME",
    pads=None,
    ceil_mode=False,
    data_format="NHWC",
    **kwargs,
):
    """
    Golden function for dilation2_d.
    All the parameters (names and order) follow @dilation2_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Uses TensorFlow's tf.nn.dilation2d as the reference computation engine, but
    manually computes padding and output size per the SE document formulas to
    ensure consistency with the kernel implementation.

    Morphological dilation 2D operation.
    y[b, oh, ow, c] = max_{fh, fw} (x[b, oh*stride_h + fh*rate_h, ow*stride_w + fw*rate_w, c] + filter[fh, fw, c])

    Args:
        x: numpy array, shape (N, H, W, C) for NHWC or (N, C, H, W) for NCHW
        filter: numpy array, shape (fH, fW, C)
        strides: list of 4 ints, [stride_n, stride_h, stride_w, stride_c] (NHWC) or
                 [stride_n, stride_c, stride_h, stride_w] (NCHW)
        rates: list of 4 ints, [rate_n, rate_h, rate_w, rate_c] (NHWC) or
               [rate_n, rate_c, rate_h, rate_w] (NCHW)
        padding_mode: str, "SAME", "VALID", or "CALCULATED"
        pads: list of 4 non-negative ints, [H_top, H_bottom, W_left, W_right] (CALCULATED mode)
        ceil_mode: bool, use ceil or floor for output size (CALCULATED mode only)
        data_format: str, "NHWC" or "NCHW"
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (numpy.ndarray) in the same data_format as input.
    """
    if strides is None:
        strides = [1, 1, 1, 1]
    if rates is None:
        rates = [1, 1, 1, 1]

    orig_dtype = x.dtype

    # ------------------------------------------------------------------
    # Step 1: NCHW -> NHWC conversion
    #         tf.nn.dilation2d only supports NHWC data_format.
    # ------------------------------------------------------------------
    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
        filter = np.transpose(filter, (1, 2, 0))
        # NCHW strides/rates: [N, C, H, W] -> extract H, W from index 2, 3
        stride_h = strides[2]
        stride_w = strides[3]
        rate_h = rates[2]
        rate_w = rates[3]
    else:
        # NHWC strides/rates: [N, H, W, C] -> extract H, W from index 1, 2
        stride_h = strides[1]
        stride_w = strides[2]
        rate_h = rates[1]
        rate_w = rates[2]

    N, H, W, C = x.shape
    fH, fW, fC = filter.shape

    # Effective filter size (with rates)
    filter_h_eff = fH + (fH - 1) * (rate_h - 1)
    filter_w_eff = fW + (fW - 1) * (rate_w - 1)

    # tf.nn.dilation2d expects NHWC-order strides and dilations
    tf_strides = [1, stride_h, stride_w, 1]
    tf_dilations = [1, rate_h, rate_w, 1]

    # ------------------------------------------------------------------
    # Step 2: Determine compute dtype
    #         tf.nn.dilation2d natively supports float16/32/64, int8/16/32/64,
    #         uint8/16/32/64, bfloat16. For integer types, use float64 for
    #         computation to avoid overflow in x+filter addition, then cast back.
    # ------------------------------------------------------------------
    compute_dtype = np.float64

    # ------------------------------------------------------------------
    # Step 3: Manually compute output size and padding per canndev (SE doc)
    #         formulas.
    #
    # WHY not tf.nn.dilation2d's built-in SAME/VALID:
    #
    # 1. TF SAME padding distribution differs from canndev when rates > 1.
    #    canndev: pad_top = pad_h // 2, pad_bottom = pad_h - pad_top
    #    TF internally may distribute padding differently for dilated filters.
    #
    # 2. Output size formulas differ in CALCULATED ceil_mode:
    #    canndev:  oH = ceil((H_padded - filter_eff) / stride) + 1
    #    TF VALID: oH = floor((H_padded - filter_eff) / stride) + 1
    #    When (H_padded - filter_eff) is not divisible by stride, canndev
    #    produces one extra row/col whose windows sample beyond the padded
    #    input (the kernel treats those out-of-bounds samples as skipped).
    #
    # SOLUTION: Compute padding and output size manually (per canndev), then
    # extend the bottom/right padding with -inf up to the deepest sample of
    # the largest window (need_h/need_w below). On that padded tensor TF's
    # VALID output size equals the canndev oH x oW exactly, so a single
    # tf.nn.dilation2d VALID call computes every output element - including
    # the ceil_mode extra rows/cols - from one reference engine with TF's own
    # NaN semantics (strict "candidate > accumulator" comparisons drop NaN,
    # matching the kernel). Appended -inf cells never win the max, which is
    # equivalent to the kernel skipping out-of-bounds positions.
    #
    # Window positions are consistent because both canndev and TF use:
    #   ih = oh * stride_h + fh * rate_h - pad_top
    # ------------------------------------------------------------------
    if padding_mode == "SAME":
        oH = int(np.ceil(H / stride_h))
        oW = int(np.ceil(W / stride_w))
        pad_h = max((oH - 1) * stride_h + filter_h_eff - H, 0)
        pad_w = max((oW - 1) * stride_w + filter_w_eff - W, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    elif padding_mode == "VALID":
        oH = int(np.ceil((H - filter_h_eff + 1) / stride_h))
        oW = int(np.ceil((W - filter_w_eff + 1) / stride_w))
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
    else:
        # CALCULATED mode: pads = [H_top, H_bottom, W_left, W_right] (canndev compatible)
        pad_top = pads[0] if pads is not None and len(pads) > 0 else 0
        pad_bottom = pads[1] if pads is not None and len(pads) > 1 else 0
        pad_left = pads[2] if pads is not None and len(pads) > 2 else 0
        pad_right = pads[3] if pads is not None and len(pads) > 3 else 0
        H_padded = H + pad_top + pad_bottom
        W_padded = W + pad_left + pad_right
        if ceil_mode:
            oH = int(np.ceil((H_padded - filter_h_eff) / stride_h)) + 1
            oW = int(np.ceil((W_padded - filter_w_eff) / stride_w)) + 1
        else:
            oH = int(np.floor((H_padded - filter_h_eff) / stride_h)) + 1
            oW = int(np.floor((W_padded - filter_w_eff) / stride_w)) + 1

    oH = max(oH, 0)
    oW = max(oW, 0)

    # ------------------------------------------------------------------
    # Step 4: Handle empty output (oH=0 or oW=0) without calling TF
    # ------------------------------------------------------------------
    if oH == 0 or oW == 0:
        y = np.empty((N, oH, oW, C), dtype=compute_dtype)
    else:
        # TF 仅在此分支惰性导入：TTK geir worker 预载 torch/tbe 后顶层导入 TF 会触发
        # libtensorflow 静态初始化 SIGSEGV（ENV-GEIR-TF），延迟到真正需要 TF 时导入可规避
        import tensorflow as tf

        # Convert numpy arrays to TensorFlow tensors
        x_tf = tf.constant(x.astype(compute_dtype))
        filter_tf = tf.constant(filter.astype(compute_dtype))

        # Pad top/left with the mode-computed padding, and extend bottom/right
        # with -inf so that even the largest window's deepest sample lies
        # inside the padded tensor: positions beyond the original input hold
        # -inf and never win the max, which is equivalent to the kernel
        # skipping out-of-bounds samples.
        need_h = max(
            H + pad_top + pad_bottom, (oH - 1) * stride_h + (fH - 1) * rate_h + 1
        )
        need_w = max(
            W + pad_left + pad_right, (oW - 1) * stride_w + (fW - 1) * rate_w + 1
        )
        paddings = tf.constant(
            [
                [0, 0],  # N dimension: no padding
                [pad_top, need_h - H - pad_top],  # H dimension
                [pad_left, need_w - W - pad_left],  # W dimension
                [0, 0],  # C dimension: no padding
            ],
            dtype=tf.int32,
        )
        x_padded = tf.pad(
            x_tf, paddings, mode="CONSTANT", constant_values=float("-inf")
        )

        # ----------------------------------------------------------------
        # Step 5: Execute dilation via tf.nn.dilation2d with VALID padding.
        #         On the extended padded tensor TF's VALID output size is
        #         exactly oH x oW (ceil((need_h - filter_eff + 1) / stride)
        #         == oH), so the ceil_mode extra rows/cols are computed by
        #         TF itself; the slice below is only a defensive crop.
        # ----------------------------------------------------------------
        y_tf = tf.nn.dilation2d(
            input=x_padded,
            filters=filter_tf,
            strides=tf_strides,
            padding="VALID",
            data_format="NHWC",
            dilations=tf_dilations,
        )
        y = y_tf.numpy()[:, :oH, :oW, :]

        # ----------------------------------------------------------------
        # Step 6: Restore -inf for windows whose candidates were all dropped.
        #
        # tf.nn.dilation2d initializes its internal max accumulator to
        # finfo(compute_dtype).min. When every candidate of a window is
        # dropped (all window positions are -inf padding cells, or all
        # candidates are NaN - TF compares with strict ">", so NaN never
        # replaces the accumulator), the output retains finfo.min instead of
        # the kernel's -inf (GetMinInit). Legitimate fp16 results computed in
        # float64 can never reach finfo.min, so replacing any output value
        # <= finfo.min with -inf is safe and restores the correct semantics.
        # ----------------------------------------------------------------
        compute_finfo_min = np.finfo(compute_dtype).min
        y = np.where(y <= compute_finfo_min, -np.inf, y)

    # ------------------------------------------------------------------
    # Step 7: Convert back to original dtype
    # ------------------------------------------------------------------
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        y = np.clip(np.round(y), info.min, info.max).astype(orig_dtype)
    else:
        y = y.astype(orig_dtype)

    # ------------------------------------------------------------------
    # Step 8: NHWC -> NCHW conversion (if original data_format was NCHW)
    # ------------------------------------------------------------------
    if data_format == "NCHW":
        y = np.transpose(y, (0, 3, 1, 2))

    return y
