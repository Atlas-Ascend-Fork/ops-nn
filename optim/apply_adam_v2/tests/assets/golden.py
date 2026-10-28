# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import numpy as np
import torch


__spec__ = {"apply_adam_v2": "ApplyAdamV2TestSpec"}

_L0 = {
    "float32": {"standard": "cross_check", "level": "L0"},
    "float16": {"standard": "cross_check", "level": "L0"},
}


def _to_scalar(x):
    if isinstance(x, np.ndarray):
        return float(x.flat[0]) if x.size > 0 else 0.0
    return float(x)


def _promote_dtype(out_dtype):
    if out_dtype == np.float16 or str(out_dtype) == "bfloat16":
        return np.float32, torch.float32
    return np.float64, torch.float64


def _resolve_adam_mode(**kwargs):
    if "adam_mode" in kwargs:
        return kwargs["adam_mode"]
    if "attributes" in kwargs:
        attrs = kwargs["attributes"]
        if isinstance(attrs, dict):
            return attrs.get("adam_mode", "adam")
    return "adam"


def _compute_adam(
    var_f, m_f, v_f, g_t, lr, beta1, beta2, epsilon, weight_decay, compute_td, mgn, ggn
):
    if mgn > 0.0 and ggn > 1.0:
        g_t = torch.mul(g_t, 1.0 / ggn)

    b1 = torch.tensor(beta1, dtype=compute_td)
    b2 = torch.tensor(beta2, dtype=compute_td)
    eps = torch.tensor(epsilon, dtype=compute_td)
    lr_t = torch.tensor(lr, dtype=compute_td)
    wd = torch.tensor(weight_decay, dtype=compute_td)

    one = torch.tensor(1.0, dtype=compute_td)
    m_t = torch.add(torch.mul(b1, m_f), torch.mul(torch.sub(one, b1), g_t))
    v_t = torch.add(
        torch.mul(b2, v_f), torch.mul(torch.sub(one, b2), torch.mul(g_t, g_t))
    )

    sqrt_v = torch.sqrt(v_t)
    u = torch.div(m_t, torch.add(sqrt_v, eps))

    if weight_decay > 0.0:
        u = torch.add(u, torch.mul(wd, var_f))

    var_out = torch.sub(var_f, torch.mul(lr_t, u))
    return var_out, m_t, v_t


def _compute_mbart_adam(
    var_f, m_f, v_f, g_t, lr, beta1, beta2, epsilon, weight_decay, step_size, compute_td
):
    b1 = torch.tensor(beta1, dtype=compute_td)
    b2 = torch.tensor(beta2, dtype=compute_td)
    eps = torch.tensor(epsilon, dtype=compute_td)
    lr_t = torch.tensor(lr, dtype=compute_td)
    wd = torch.tensor(weight_decay, dtype=compute_td)
    ss = torch.tensor(step_size, dtype=compute_td)

    one = torch.tensor(1.0, dtype=compute_td)
    m_t = torch.add(torch.mul(b1, m_f), torch.mul(torch.sub(one, b1), g_t))
    v_t = torch.add(
        torch.mul(b2, v_f), torch.mul(torch.sub(one, b2), torch.mul(g_t, g_t))
    )

    sqrt_v = torch.sqrt(v_t)
    u = torch.div(m_t, torch.add(sqrt_v, eps))

    u_st = torch.mul(ss, u)

    if weight_decay > 0.0:
        u_st = torch.add(u_st, torch.mul(wd, torch.mul(lr_t, var_f)))

    var_out = torch.sub(var_f, u_st)
    return var_out, m_t, v_t


def _golden_adam(
    var,
    m,
    v,
    lr,
    beta1,
    beta2,
    epsilon,
    grad,
    max_grad_norm,
    global_grad_norm,
    weight_decay,
):
    out_dtype = var.dtype
    compute_np, compute_td = _promote_dtype(out_dtype)

    ggn = float(global_grad_norm)
    mgn = float(max_grad_norm)

    if math.isnan(ggn) or math.isinf(ggn):
        ggn = 1.0
    elif ggn < 1e-10:
        ggn = 1.0

    var_f = torch.from_numpy(var.astype(compute_np))
    m_f = torch.from_numpy(m.astype(compute_np))
    v_f = torch.from_numpy(v.astype(compute_np))
    g_t = torch.from_numpy(grad.astype(compute_np))

    var_out, m_t, v_t = _compute_adam(
        var_f,
        m_f,
        v_f,
        g_t,
        lr,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        compute_td,
        mgn,
        ggn,
    )

    return (
        var_out.numpy().astype(out_dtype),
        m_t.numpy().astype(out_dtype),
        v_t.numpy().astype(out_dtype),
    )


def _golden_mbart_adam(
    var, m, v, lr, beta1, beta2, epsilon, grad, weight_decay, step_size
):
    out_dtype = var.dtype
    compute_np, compute_td = _promote_dtype(out_dtype)

    var_f = torch.from_numpy(var.astype(compute_np))
    m_f = torch.from_numpy(m.astype(compute_np))
    v_f = torch.from_numpy(v.astype(compute_np))
    g_f = torch.from_numpy(grad.astype(compute_np))

    var_out, m_t, v_t = _compute_mbart_adam(
        var_f,
        m_f,
        v_f,
        g_f,
        lr,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        step_size,
        compute_td,
    )

    return (
        var_out.numpy().astype(out_dtype),
        m_t.numpy().astype(out_dtype),
        v_t.numpy().astype(out_dtype),
    )


class ApplyAdamV2TestSpec:
    def golden(
        var,
        m,
        v,
        lr,
        beta1,
        beta2,
        epsilon,
        grad,
        max_grad_norm=None,
        global_grad_norm=None,
        weight_decay=None,
        step_size=None,
        **kwargs,
    ):
        if var.size == 0:
            return [var.copy(), m.copy(), v.copy()]

        adam_mode = _resolve_adam_mode(**kwargs)

        lr_s = _to_scalar(lr)
        b1_s = _to_scalar(beta1)
        b2_s = _to_scalar(beta2)
        eps_s = _to_scalar(epsilon)
        wd_s = _to_scalar(weight_decay) if weight_decay is not None else 0.0
        mgn_s = _to_scalar(max_grad_norm) if max_grad_norm is not None else 0.0
        ggn_s = _to_scalar(global_grad_norm) if global_grad_norm is not None else 1.0
        ss_s = _to_scalar(step_size) if step_size is not None else lr_s

        if adam_mode == "adam":
            result = _golden_adam(
                var, m, v, lr_s, b1_s, b2_s, eps_s, grad, mgn_s, ggn_s, wd_s
            )
        elif adam_mode == "mbart_adam":
            result = _golden_mbart_adam(
                var, m, v, lr_s, b1_s, b2_s, eps_s, grad, wd_s, ss_s
            )
        else:
            raise ValueError(f"Unknown adam_mode: {adam_mode}")

        return list(result)

    class ThirdPartyImpl:
        def __init__(
            self,
            var,
            m,
            v,
            lr,
            beta1,
            beta2,
            epsilon,
            grad,
            max_grad_norm=None,
            global_grad_norm=None,
            weight_decay=None,
            step_size=None,
            **kwargs,
        ):
            self.var = var
            self.m = m
            self.v = v
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.grad = grad
            self.max_grad_norm = max_grad_norm
            self.global_grad_norm = global_grad_norm
            self.weight_decay = weight_decay
            self.step_size = step_size
            self.adam_mode = _resolve_adam_mode(**kwargs)

        def __call__(self):
            # GEIR/Kernel 三方通路入参为 torch.Tensor（nd亦可），统一 as_tensor；
            # Tensor 的 .size 是方法，空判断须用 numel()
            var_t = torch.as_tensor(self.var)
            m_t = torch.as_tensor(self.m)
            v_t = torch.as_tensor(self.v)
            g_t = torch.as_tensor(self.grad)
            if var_t.numel() == 0:
                return [var_t.clone(), m_t.clone(), v_t.clone()]

            out_dtype = self.var.dtype

            lr_s = _to_scalar(self.lr)
            b1_s = _to_scalar(self.beta1)
            b2_s = _to_scalar(self.beta2)
            eps_s = _to_scalar(self.epsilon)
            wd_s = (
                _to_scalar(self.weight_decay) if self.weight_decay is not None else 0.0
            )
            mgn_s = (
                _to_scalar(self.max_grad_norm)
                if self.max_grad_norm is not None
                else 0.0
            )
            ggn_s = (
                _to_scalar(self.global_grad_norm)
                if self.global_grad_norm is not None
                else 1.0
            )
            ss_s = _to_scalar(self.step_size) if self.step_size is not None else lr_s

            if self.adam_mode == "adam":
                var_out, m_out, v_out = _compute_adam(
                    var_t,
                    m_t,
                    v_t,
                    g_t,
                    lr_s,
                    b1_s,
                    b2_s,
                    eps_s,
                    wd_s,
                    var_t.dtype,
                    mgn_s,
                    ggn_s,
                )
            elif self.adam_mode == "mbart_adam":
                var_out, m_out, v_out = _compute_mbart_adam(
                    var_t,
                    m_t,
                    v_t,
                    g_t,
                    lr_s,
                    b1_s,
                    b2_s,
                    eps_s,
                    wd_s,
                    ss_s,
                    var_t.dtype,
                )
            else:
                raise ValueError(f"Unknown adam_mode: {self.adam_mode}")

            # 三方在 GPU 侧计算，返回 CUDA tensor（.numpy() 不可用）；
            # server 负责序列化回传，dtype 归一回 out_dtype
            return [var_out.to(out_dtype), m_out.to(out_dtype), v_out.to(out_dtype)]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = _L0
