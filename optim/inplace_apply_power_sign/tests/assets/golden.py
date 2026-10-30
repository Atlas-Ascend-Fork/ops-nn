#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TensorFlow PowerSign Golden 与独立 Torch provider，供 Kernel/GEIR 三方比对。

主计算只调用tf.raw_ops.ResourceApplyPowerSign，不提供公式fallback。
每次调用建立私有var/m资源，API更新后读取两状态；不污染TTK原始输入。
实际float16/bfloat16计算提升float32，实际float32/float64计算保持。
最终两份结果分别完整cast到原始var/m的实际dtype，不另做溢出或非有限值处理。
cross_check 自动提升 Golden 精度；Torch provider 使用原 dtype/device 独立计算。
"""

import numpy as np


__spec__ = {"inplace_apply_power_sign": "InplaceApplyPowerSignTestSpec"}


class InplaceApplyPowerSignTestSpec:
    """Kernel/GEIR三方参考；七输入、use_locking属性、[new_var,new_m]。"""

    def golden(
        var, m, lr, logbase, sign_decay, beta, grad, *, use_locking=False, **kwargs
    ):
        raw_inputs = (var, m, lr, logbase, sign_decay, beta, grad)
        # Check before bf16 -> fp32 adaptation, which could hide mixed dtypes.
        if any(value.dtype != var.dtype for value in raw_inputs):
            raise ValueError("all seven inputs must have the same dtype")
        if var.dtype.name not in ("float16", "bfloat16", "float32", "float64"):
            raise TypeError("PowerSign reference requires floating-point input tensors")
        if var.shape != m.shape or var.shape != grad.shape:
            raise ValueError("var, m and grad must have the same shape")
        if any(value.ndim > 8 for value in raw_inputs):
            raise ValueError("input rank must not exceed 8")
        for name, value in zip(
            ("lr", "logbase", "sign_decay", "beta"), raw_inputs[2:6]
        ):
            if tuple(value.shape) not in ((), (1,)):
                raise ValueError(f"{name} must have shape [] or [1]")

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "PowerSign Golden requires TensorFlow ResourceApplyPowerSign; "
                "run it in the tester-managed .conda-envs/golden-tf environment"
            ) from exc

        # Reference precision follows actual numpy inputs, never mode labels or
        # input/output dtype metadata. Resources and results remain private.
        compute_dtype = (
            np.float32 if var.dtype.name in ("float16", "bfloat16") else var.dtype
        )
        arrays = [
            np.array(value, dtype=compute_dtype, copy=True) for value in raw_inputs
        ]
        with tf.device("/CPU:0"):
            var_state = tf.Variable(arrays[0], trainable=False)
            m_state = tf.Variable(arrays[1], trainable=False)
            scalar_tensors = [
                tf.convert_to_tensor(value.reshape(())) for value in arrays[2:6]
            ]
            update = tf.raw_ops.ResourceApplyPowerSign(
                var=var_state.handle,
                m=m_state.handle,
                lr=scalar_tensors[0],
                logbase=scalar_tensors[1],
                sign_decay=scalar_tensors[2],
                beta=scalar_tensors[3],
                grad=tf.convert_to_tensor(arrays[6]),
                use_locking=use_locking,
            )
            # The resource op has no Tensor return. Eager executes immediately;
            # an explicit dependency also orders resource reads in graph mode.
            with tf.control_dependencies([] if update is None else [update]):
                new_var = var_state.read_value()
                new_m = m_state.read_value()
            # Final output precision follows the original in-place states.
            # Cast every element once; rounding and overflow follow dtype conversion.
            return [
                new_var.numpy().astype(var.dtype, copy=True),
                new_m.numpy().astype(m.dtype, copy=True),
            ]

    class ThirdPartyImpl:
        def __init__(self, use_locking=False, **kwargs):
            # All results are private tensors; no shared state requires locking.
            self.use_locking = use_locking

        def __call__(self, var, m, lr, logbase, sign_decay, beta, grad, **kwargs):
            import torch

            inputs = (var, m, lr, logbase, sign_decay, beta, grad)
            if any(not isinstance(value, torch.Tensor) for value in inputs):
                raise TypeError("PowerSign provider inputs must be torch.Tensor")
            if var.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ):
                raise TypeError(
                    "PowerSign provider requires floating-point input tensors"
                )
            if any(
                value.dtype != var.dtype or value.device != var.device
                for value in inputs
            ):
                raise ValueError("all seven inputs must have the same dtype and device")
            if var.shape != m.shape or var.shape != grad.shape:
                raise ValueError("var, m and grad must have the same shape")
            if any(value.ndim > 8 for value in inputs):
                raise ValueError("input rank must not exceed 8")
            for name, value in zip(
                ("lr", "logbase", "sign_decay", "beta"), inputs[2:6]
            ):
                if tuple(value.shape) not in ((), (1,)):
                    raise ValueError(f"{name} must have shape [] or [1]")

            output_dtypes = (var.dtype, m.dtype)
            # README specifies FP32 intermediates for FP16/BF16. Preserve wider inputs.
            if var.dtype in (torch.float16, torch.bfloat16):
                var, m, lr, logbase, sign_decay, beta, grad = (
                    value.to(torch.float32) for value in inputs
                )
            lr, logbase, sign_decay, beta = (
                value.reshape(()) for value in (lr, logbase, sign_decay, beta)
            )
            # Update m first, then use its sign for the var update (README formula).
            new_m = beta * m + (m.new_tensor(1) - beta) * grad
            # TensorFlow propagates NaN through sign; torch.sign alone maps NaN to zero.
            sign_m = torch.where(torch.isnan(new_m), new_m, torch.sign(new_m))
            sign_grad = torch.where(torch.isnan(grad), grad, torch.sign(grad))
            exponent = logbase * sign_decay * sign_m * sign_grad
            new_var = var - lr * torch.exp(exponent) * grad
            return [new_var.to(output_dtypes[0]), new_m.to(output_dtypes[1])]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        # FP64 is a diagnostic reference dtype; cross_check supports only the three above.
        "float64": {"standard": "stat_rel_err"},
    }
