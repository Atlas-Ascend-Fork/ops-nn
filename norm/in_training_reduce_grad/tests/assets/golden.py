#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""INTrainingReduceGrad golden（kernel / GEIR 通路，TestSpec 格式）。

InstanceNorm 训练反向 ReduceGrad：由 dy 与前向缓存的统计量计算 pd_x（对输入 x 的梯度）。
无 PyTorch 竞品单算子，golden 与 third_party 均以 torch 小算子拼接实现。

Canonical IO order (in_training_reduce_grad_def.cpp):
    inputs : dy, x (fp16/fp32 同 shape 同 dtype), variance, mean,
             res_gamma, res_beta, gamma (fp32, NCHW=[N,C,1,1] / NHWC=[N,1,1,C])
    outputs: pd_x (同 dy shape/dtype)
    attrs  : 无（eps=1e-6 为算子定义常量）

计算公式（对齐 arch35 kernel，统计量按尾部补 1 广播）：
    data_sqrt   = sqrt(variance + 1e-6)
    diff        = x - mean
    tds         = (res_gamma * diff) / data_sqrt * (-1/num)
    coef        = dy + tds + (-res_beta / num)
    pd_x        = (coef * gamma) / data_sqrt

结合次序与 kernel 一致（res_gamma*diff 先行、先除 data_sqrt 后乘 -1/num）：
避免预生成逐通道 multiplier（= -res_gamma/num/data_sqrt）在 res_gamma 极小
时的次正规中间量（设备 FTZ 下冲零，multiplier * (±inf diff) 会得 NaN 而非
±inf）；也避免先乘 -1/num 使 res_gamma=FLT_MIN 时 t 落入次正规区被冲零
（L0_030 输出偏离真值 6 个数量级）。

其中 num = dy.size // variance.size（NCHW/NHWC 下即 H*W）。
空 Tensor（num==0）返回与 dy 同 shape 的空输出。

精度策略：
    golden      —— 真值。计算链全程 fp64（含开方），出口窄回输入 dtype：
                   极值中间量（res_gamma*diff ~1e39）在 fp32 会上溢为 ±inf
                   并与 -inf 相加得 NaN，偏离精确结果；fp64 链保持精确。
    third_party —— 三方标杆。torch 拼接，镜像 kernel 的 fp32 逐算术路径，
                   出口窄回 dy dtype（竞品自然精度）。
    tolerance   —— 浮点输出 cross_check（NPU/竞品 相对 golden 的误差比值）。
"""

import numpy
import torch


# 双拼写注册：opInterface 名（安装包 op-info 键，TTK kernel/geir 通路按 CSV op_name
# 精确匹配）与 GE 算子类型名（生成侧 op_name 拼写）均可直接解析，无需执行副本归一。
__spec__ = {
    "in_training_reduce_grad": "INTrainingReduceGradTestSpec",
    "INTrainingReduceGrad": "INTrainingReduceGradTestSpec",
}

__golden__ = {
    "kernel": {
        "in_training_reduce_grad": "in_training_reduce_grad_golden",
        "INTrainingReduceGrad": "in_training_reduce_grad_golden",
    },
}


_TOL = {
    "float32": {"standard": "cross_check", "level": "L0"},
    "float16": {"standard": "cross_check", "level": "L0"},
}


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    return numpy.asarray(x)


def _derive_num(dy_np, variance_np):
    if dy_np.ndim != 4:
        raise ValueError(f"rank 仅支持 4（NCHW / NHWC），实际 {dy_np.ndim}")
    if dy_np.size == 0 or variance_np.size == 0:
        return 0
    return dy_np.size // variance_np.size


def _align_stat(t, dy_ndim, dtype=torch.float32):
    t = torch.as_tensor(t, dtype=dtype)
    while t.ndim < dy_ndim:
        t = t.unsqueeze(-1)
    return t


class INTrainingReduceGradTestSpec:
    def golden(dy, x, variance, mean, res_gamma, res_beta, gamma, num=None, **kwargs):
        dy_np = _to_numpy(dy)
        x_np = _to_numpy(x)
        variance_np = _to_numpy(variance)
        mean_np = _to_numpy(mean)
        res_gamma_np = _to_numpy(res_gamma)
        res_beta_np = _to_numpy(res_beta)
        gamma_np = _to_numpy(gamma)

        # float64 仅在 golden_mode=Promote 下出现（fp32 被框架抬一档），属合法输入
        if str(dy_np.dtype) not in ("float16", "float32", "float64"):
            raise ValueError(f"dy 仅支持 float16/float32，实际 {dy_np.dtype}")
        if str(x_np.dtype) != str(dy_np.dtype):
            raise ValueError(
                f"x 与 dy 必须同 dtype，实际 x={x_np.dtype} dy={dy_np.dtype}"
            )
        if x_np.shape != dy_np.shape:
            raise ValueError(
                f"x 与 dy 必须同 shape，实际 x={x_np.shape} dy={dy_np.shape}"
            )

        if num is None:
            num = _derive_num(dy_np, variance_np)
        num = int(num)
        if num == 0:
            return [numpy.empty(dy_np.shape, dtype=dy_np.dtype)]

        orig_dtype = dy_np.dtype
        # 真值链全程 fp64：极值中间量（res_gamma*diff 可达 ~1e39）在 fp32 计算
        # 会上溢为 ±inf，再与 -inf 相加得 NaN，偏离精确结果（±inf），把算子与
        # 精确值一致的输出误判为失败（GEIR L0_022 实测）。fp64 链在同一输入下
        # 保持精确。ThirdPartyImpl 保持 fp32 逐算术路径（镜像内核/竞品语义），
        # cross_check 以两侧相对真值的误差比值判定，精度档位不受影响。
        dy_t = torch.from_numpy(dy_np.astype(numpy.float64))
        x_t = torch.from_numpy(x_np.astype(numpy.float64))
        variance_t = _align_stat(variance_np, dy_t.ndim, torch.float64)
        mean_t = _align_stat(mean_np, dy_t.ndim, torch.float64)
        res_gamma_t = _align_stat(res_gamma_np, dy_t.ndim, torch.float64)
        res_beta_t = _align_stat(res_beta_np, dy_t.ndim, torch.float64)
        gamma_t = _align_stat(gamma_np, dy_t.ndim, torch.float64)

        eps = 1e-6
        data_sqrt = torch.sqrt(variance_t + eps)
        num_rec = 1.0 / num
        offset_inv_reverse = res_beta_t * (-num_rec)
        diff = x_t - mean_t
        tds = ((res_gamma_t * diff) / data_sqrt) * (-num_rec)
        coef = dy_t + tds + offset_inv_reverse
        pd_x = (coef * gamma_t) / data_sqrt

        return [pd_x.numpy().astype(orig_dtype)]

    class ThirdPartyImpl:
        def __init__(self, **kwargs):
            pass

        def __call__(
            self, dy, x, variance, mean, res_gamma, res_beta, gamma, num=None, **kwargs
        ):
            dy_c = dy.to(torch.float32) if dy.dtype != torch.float32 else dy
            x_c = x.to(torch.float32) if x.dtype != torch.float32 else x

            if num is None:
                if dy_c.ndim != 4:
                    raise ValueError(f"rank 仅支持 4（NCHW / NHWC），实际 {dy_c.ndim}")
                if dy_c.numel() == 0 or variance.numel() == 0:
                    num = 0
                else:
                    num = dy_c.numel() // variance.numel()
            num = int(num)
            if num == 0:
                return [torch.empty_like(dy)]

            var_c = variance.to(torch.float32)
            mean_c = mean.to(torch.float32)
            rg_c = res_gamma.to(torch.float32)
            rb_c = res_beta.to(torch.float32)
            ga_c = gamma.to(torch.float32)

            while var_c.ndim < dy_c.ndim:
                var_c = var_c.unsqueeze(-1)
                mean_c = mean_c.unsqueeze(-1)
                rg_c = rg_c.unsqueeze(-1)
                rb_c = rb_c.unsqueeze(-1)
                ga_c = ga_c.unsqueeze(-1)

            eps = 1e-6
            data_sqrt = torch.sqrt(var_c + eps)
            num_rec = 1.0 / num
            offset_inv_reverse = rb_c * (-num_rec)
            diff = x_c - mean_c
            tds = ((rg_c * diff) / data_sqrt) * (-num_rec)
            coef = dy_c + tds + offset_inv_reverse
            pd_x = (coef * ga_c) / data_sqrt

            return [pd_x.to(dy.dtype)]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = _TOL


def in_training_reduce_grad_golden(
    dy, x, variance, mean, res_gamma, res_beta, gamma, num=None, *args, **kwargs
):
    """历史 ``__golden__`` kernel 加载器兼容入口（签名照 def.cpp），与 Spec 共用同一实现。"""
    del args
    return tuple(
        INTrainingReduceGradTestSpec.golden(
            dy, x, variance, mean, res_gamma, res_beta, gamma, num=num, **kwargs
        )
    )


# 【不存在】aclnn 通路：CMakeLists.txt 声明 ``ACLNNTYPE aclnn_exclude``，
# 仓库无 op_api 文件与 aclnn 接口文档。
# 【不存在】e2e 通路：PyTorch 无 INTrainingReduceGrad 单算子（数学等价物为
# InstanceNorm backward autograd），torch_npu 无单算子调用；该算子仅由 GE 图
# 在 InstanceNorm 训练反向时内部构造。
