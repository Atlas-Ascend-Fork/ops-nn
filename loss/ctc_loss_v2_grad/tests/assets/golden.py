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
"""CTCLossV2Grad / aclnnCtcLossBackward 测试规范（TestSpec）。

真值来源：竞品接口 ``torch.ops.aten._ctc_loss_backward``（与前向 golden
``loss/ctc_loss_v2`` 使用 ``torch.ops.aten._ctc_loss`` 一致，非 numpy 纯公式）。
两个入口共享唯一计算核心 ``_ctc_loss_backward_core``，各自只负责参数包装。

两个入口的协议差异是实质性的，不用别名掩盖：

====================  ==========================  =============================
项                    kernel / GEIR               aclnn
====================  ==========================  =============================
注册 key              ctc_loss_v2_grad            aclnnCtcLossBackward
参数名来源            ctc_loss_v2_grad_def.cpp    aclnn_ctc_loss_backward.h
长度参数              张量【输入】                aclIntArray【属性】
长度参数名            input_lengths/target_lengths inputLengths/targetLengths
归零属性名            zero_infinity               zeroInfinity
reduction             有（不参与计算，见下）      无此参数
入/出容器             numpy.ndarray               CPU torch.Tensor
====================  ==========================  =============================

语义核对（对照 op_kernel/arch35/ctc_loss_v2_grad.h）：
  - grad[t,b,c] = (exp(lp) - exp(res + nll - lp)) * grad_out[b]，即 grad_out 为逐 batch
    上游梯度；这正是 aten._ctc_loss_backward 的定义。
  - reduction 在 kernel 中【不参与计算】（tiling 仅读取 blank/zero_infinity），
    grad_out 已在框架侧按 reduction 完成缩放。故计算核心【签名上就不接收】
    reduction，从接口层面杜绝重复缩放。
  - zero_infinity=True 时，对 neg_log_likelihood 为 ±inf 的 batch 输出 0 梯度
    （由 aten 内核保证，与 NPU kernel 一致）。

dtype：fp16/bf16 在 CPU aten 上不支持（实测 ``"ctc_loss_cpu" not implemented for
'Half'/'BFloat16'``），统一抬 fp32 计算后按各入口契约舍回目标 dtype（与 NPU 内核
fp32 累加一致，不抬 fp64）。整型 targets 统一转 int64 供 aten 使用。
"""

import hashlib

import numpy as np
import torch

try:
    import ml_dtypes  # noqa: F401
except ImportError:  # pragma: no cover
    ml_dtypes = None

__spec__ = {
    "ctc_loss_v2_grad": "CtcLossV2GradTestSpec",
    "aclnnCtcLossBackward": "AclnnCtcLossBackwardTestSpec",
    "torch.ops.aten._ctc_loss_backward": "CtcLossBackwardE2eSpec",
}

# CTC 中"不可达状态"的数学定义值（概率 0 的对数）
_NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def _np_to_torch(array):
    """numpy → CPU torch 张量（bf16 经 uint16 位视图无损转换）。"""
    array = np.ascontiguousarray(array)
    if ml_dtypes is not None and array.dtype == np.dtype(ml_dtypes.bfloat16):
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy(tensor, dtype_name):
    """CPU torch 张量 → numpy，并 cast 到目标输出 dtype（bf16 经位视图）。"""
    if tensor.dtype == torch.bfloat16:
        if ml_dtypes is not None and np.dtype(dtype_name) == np.dtype(
            ml_dtypes.bfloat16
        ):
            array = tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            return np.ascontiguousarray(array)
        tensor = tensor.to(torch.float32)
    array = tensor.detach().cpu().numpy()
    return np.ascontiguousarray(array.astype(dtype_name, copy=False))


def _int_list(values):
    """长度容器（numpy 张量 / list / tuple）→ aten 需要的 python int list。

    list / tuple 刻意走**纯 Python** 路径，不经 numpy：在 torch.compile 下
    ``np.asarray(...).tolist()`` 会被 Dynamo 追踪成 tensor 运算，``tolist()``
    退化为 ``item()`` 产出 unbacked SymInt，最终以
    ``aten._local_scalar_dense`` 让后端编译失败并断图。纯 Python 下这里是
    编译期常量，长度参数得以正常常量化。
    """
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    elif isinstance(values, (list, tuple)):
        flat = []
        for v in values:
            if isinstance(v, (list, tuple)):
                flat.extend(_int_list(v))
            elif isinstance(v, torch.Tensor):
                flat.extend(_int_list(v))
            else:
                flat.append(int(v))
        return flat
    else:
        values = np.asarray(values).reshape(-1).tolist()
    return [int(v) for v in values]


def _compute_dtype(dtype):
    """计算精度：fp16/bf16 抬 fp32（CPU aten 不支持低精度）；fp64 保持 fp64。"""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def _seed_from_name(name):
    return int(hashlib.md5(str(name).encode()).hexdigest(), 16) % (2**32)


def _label_pool(num_classes, blank, dtype):
    """按【目标 dtype 实际可表示】的类别构造合法标签池。

    aclnnCtcLossBackward 的 targets 支持 INT64/INT32/BOOL/FLOAT/FLOAT16
    （见 ``aclnn_ctc_loss_backward.h``），算子内部把非 INT32/INT64 的 targets
    Cast 成 INT64 后参与计算。因此"合法标签"必须以**经目标 dtype 往返后的值**
    为准，而不是采样时的 int64 原值：

      - BOOL：任何非零标签落地都变 True，转 INT64 即类别 1。故 ``blank=0``
        时唯一合法标签是 1；``blank=1`` 时唯一合法标签是 0（需 C 允许）。
      - FLOAT/FLOAT16：小整数可精确表示，但超出尾数范围的类别会被舍入到
        别的类别，同样要剔除。

    做法：对候选类别做一次 ``int64 -> dtype -> int64`` 往返，只保留**往返后
    仍等于自身**的类别，从而保证落地标签仍在 ``[0, C)`` 且不等于 blank。

    Returns:
        list[int]: 合法类别列表（已排除 blank 与会被 dtype 改写的类别）。

    Raises:
        ValueError: 该 (C, blank, dtype) 组合下不存在任何合法非 blank 标签。
    """
    blank_i = int(blank)
    classes = int(num_classes)
    candidates = [c for c in range(classes) if c != blank_i]
    if candidates:
        probe = torch.tensor(candidates, dtype=torch.int64)
        landed = probe.to(dtype).to(torch.int64).tolist()
        pool = [c for c, got in zip(candidates, landed) if got == c]
    else:
        pool = []
    if not pool:
        raise ValueError(
            f"targets dtype={dtype} 在 C={classes}、blank={blank_i} 下不存在合法"
            f"非 blank 标签（候选 {candidates} 经该 dtype 往返后均被改写或为空）；"
            f"请调整用例的类别数或 blank，不做静默回退"
        )
    return pool


# --------------------------------------------------------------------------- #
# 唯一计算核心：复用 aten._ctc_loss_backward，不重写 CTC 公式
# --------------------------------------------------------------------------- #
def _ctc_loss_backward_core(
    grad_out,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank,
    zero_infinity,
):
    """CTC 反向真值核心。入参均为 CPU torch.Tensor，长度为 python int list。

    【注意】签名不含 reduction：grad_out 已在框架侧按 reduction 缩放完毕，
    kernel 与 aten 定义一致，再乘一次就是重复缩放。

    Returns:
        torch.Tensor: grad，shape = log_probs = (T, N, C)，dtype = 计算精度
        （fp16/bf16 入参时为 fp32），由调用方按自身契约舍回目标 dtype。
    """
    compute = _compute_dtype(log_probs.dtype)
    return torch.ops.aten._ctc_loss_backward(
        grad_out.to(compute),
        log_probs.to(compute),
        targets.to(torch.int64),
        input_lengths,
        target_lengths,
        neg_log_likelihood.to(compute),
        log_alpha.to(compute),
        int(blank),
        bool(zero_infinity),
    )


def _sanitize_log_alpha(log_alpha, input_lengths, target_lengths):
    """把每个 batch 有效时间步、有效状态之外的区域统一填 -inf。

    ``aten._ctc_loss`` 只初始化 ``t < input_lengths[b]`` 且
    ``s < 2*target_lengths[b]+1`` 的区域。无效行、列都需要显式初始化，
    避免向设备下发分配器遗留的数据；零 input_length 的 batch 整块清理。

    填 -inf 是该位置的**数学定义值**（不可达状态概率为 0，对数为 -inf），
    不是为了让测试通过而放宽输入；真实框架调用也不会把未初始化内存交给算子。
    """
    log_alpha = log_alpha.clone()
    max_len = log_alpha.shape[-1]
    for batch_idx, length in enumerate(target_lengths):
        valid = 2 * int(length) + 1
        if valid < max_len:
            log_alpha[batch_idx, :, valid:] = float("-inf")
        log_alpha[batch_idx, int(input_lengths[batch_idx]) :, :] = float("-inf")
    return log_alpha


def _forward_for_consistency(
    log_probs, targets, input_lengths, target_lengths, blank, zero_infinity
):
    """用一次【合法前向】产出与输入自洽的 (neg_log_likelihood, log_alpha)。

    nll 与 log_alpha 是 CTC 前向的中间产物，与 (log_probs, targets, lengths, blank)
    存在严格数学约束。随机填充会让反向公式中的 ``res + nll - lp`` 完全失真，
    真值与设备都算出无意义结果，无法形成有效比较。

    log_alpha 的无效行、列按数学定义清为 -inf，见 ``_sanitize_log_alpha``。
    """
    compute = _compute_dtype(log_probs.dtype)
    neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(
        log_probs.to(compute),
        targets.to(torch.int64),
        input_lengths,
        target_lengths,
        int(blank),
        bool(zero_infinity),
    )
    return neg_log_likelihood, _sanitize_log_alpha(
        log_alpha, input_lengths, target_lengths
    )


# --------------------------------------------------------------------------- #
# kernel / GEIR 入口
# --------------------------------------------------------------------------- #
class CtcLossV2GradTestSpec:
    """CTCLossV2Grad 测试规范（kernel / GEIR 流程，numpy 入/出）。

    自定义输入生成仍由同目录 ``input.py`` 的 ``__input__`` 提供（本类不声明
    customize_inputs，Loader 会回落到该插件），kernel 输入生成行为保持不变。
    """

    def golden(
        grad_out,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        *,
        blank=0,
        reduction="mean",
        zero_infinity=False,
        **kwargs,
    ):
        """参数名/顺序跟随 ctc_loss_v2_grad_def.cpp 的输入（不含输出）。

        Args:
            grad_out: (N,) 逐 batch 上游梯度。
            log_probs: (T, N, C) 对数概率。
            targets: (N, S) 或 (sum(target_lengths),) 目标序列。
            input_lengths: (N,) 每个 batch 的有效输入长度【张量输入】。
            target_lengths: (N,) 每个 batch 的有效目标长度【张量输入】。
            neg_log_likelihood: (N,) 前向损失。
            log_alpha: (N, T, 2*max(target_lengths)+1) 前向 alpha。
            blank: 空白标识，0 <= blank < C。
            reduction: 算子属性，【不参与计算】（grad_out 已按 reduction 缩放）。
            zero_infinity: 是否对 ±inf 损失的 batch 输出 0 梯度。
            **kwargs: TTK 注入的元信息（{input,output}_{dtypes,ori_shapes,...}、
                testcase_name 等）。

        Returns:
            list[numpy.ndarray]: [grad]，shape=(T, N, C)，dtype 按输出契约。
        """
        grad = _ctc_loss_backward_core(
            _np_to_torch(grad_out),
            _np_to_torch(log_probs),
            _np_to_torch(targets),
            _int_list(input_lengths),
            _int_list(target_lengths),
            _np_to_torch(neg_log_likelihood),
            _np_to_torch(log_alpha),
            blank,
            zero_infinity,
        )
        # 输出 dtype 解包 list/tuple；缺失或无法解析时回退 log_probs。
        out_dtypes = kwargs.get("output_dtypes") or ()
        declared_dtype = out_dtypes[0] if out_dtypes else None
        if isinstance(declared_dtype, (list, tuple)):
            declared_dtype = declared_dtype[0] if declared_dtype else None
        target_dtype = np.asarray(log_probs).dtype
        if declared_dtype is not None:
            try:
                target_dtype = np.dtype(declared_dtype)
            except (TypeError, ValueError):
                pass
        return [_torch_to_numpy(grad, target_dtype)]

    class ThirdPartyImpl:
        """torch 三方标杆（kernel 参数包装）：长度为张量输入，属性 snake_case。

        与 golden 共享 ``_ctc_loss_backward_core``，仅参数包装不同。
        框架将 numpy 转 torch 后调用，按实际设备处理张量，不做 CPU numpy 往返。
        """

        def __init__(self, *, blank=0, reduction="mean", zero_infinity=False, **kwargs):
            # reduction 接收但不使用：grad_out 已按 reduction 缩放
            self.blank = int(blank)
            self.zero_infinity = bool(zero_infinity)

        def __call__(
            self,
            grad_out,
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            neg_log_likelihood,
            log_alpha,
            **kwargs,
        ):
            grad = _ctc_loss_backward_core(
                grad_out,
                log_probs,
                targets,
                _int_list(input_lengths),
                _int_list(target_lengths),
                neg_log_likelihood,
                log_alpha,
                self.blank,
                self.zero_infinity,
            )
            return [grad.to(log_probs.dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }


# --------------------------------------------------------------------------- #
# 输入构造（E2E / aclnn 共用）
# --------------------------------------------------------------------------- #
def _fill_targets(targets, num_classes, blank, generator, mode="random"):
    """在 ``[0, C) \\ {blank}`` 内生成合法标签，并【原地】写入 targets。

    mode 控制标签形态（覆盖 CTC 的重复标签路径，确定性可复现）：
      - ``random``：均匀随机采样。
      - ``repeat_adjacent``：相邻两两重复（如 [1,1,2,2]），命中 target_prime 中
        ``s`` 与 ``s+2`` 同类的转移受限分支。
      - ``all_same``：全部同一标签，重复路径的极端情形。

    采样池按 ``targets.dtype`` 实际可表示的类别取（见 ``_label_pool``），
    因此 BOOL / FLOAT16 等窄 dtype 下落地标签仍落在 ``[0, C)`` 且不等于 blank。

    Returns:
        torch.Tensor: **从 targets 读回**的落地标签（int64，形状同入参）。
        必须用它而不是采样时的 int64 原值去算前向中间量——窄 dtype 下两者不等
        （BOOL 会把所有非零标签坍缩为 True），否则 nll / log_alpha 与设备实际
        消费的标签不自洽。
    """
    count = targets.numel()
    if count == 0:
        return targets.detach().to(torch.int64)
    pool = _label_pool(num_classes, blank, targets.dtype)
    pool_t = torch.tensor(pool, dtype=torch.int64)
    if mode == "all_same":
        picks = pool_t[torch.randint(len(pool), (1,), generator=generator)].repeat(
            count
        )
    elif mode == "repeat_adjacent":
        half = (count + 1) // 2
        base = pool_t[torch.randint(len(pool), (half,), generator=generator)]
        picks = base.repeat_interleave(2)[:count]
    else:
        picks = pool_t[torch.randint(len(pool), (count,), generator=generator)]
    targets.copy_(picks.reshape(targets.shape).to(targets.dtype))
    # 读回落地值作为唯一事实来源：窄 dtype 往返后若与采样值不符即为用例构造错误
    landed = targets.detach().to(torch.int64)
    _assert_landed_labels(landed, num_classes, blank)
    return landed


def _assert_landed_labels(landed, num_classes, blank):
    """校验**实际落地**的标签仍在 ``[0, C)`` 且不等于 blank。

    放在输入生成侧而非比较侧：落地标签一旦越界或撞上 blank，nll / log_alpha
    与设备消费的标签就不再自洽，此时"用例通过"没有意义。宁可显式失败。
    """
    blank_i = int(blank)
    flat = landed.reshape(-1).tolist()
    bad_range = [v for v in flat if v < 0 or v >= int(num_classes)]
    hit_blank = [v for v in flat if v == blank_i]
    if bad_range or hit_blank:
        raise ValueError(
            f"落地标签非法：越界项 {sorted(set(bad_range))}、"
            f"撞 blank({blank_i}) 项 {sorted(set(hit_blank))}；"
            f"C={int(num_classes)}，实际标签={flat}"
        )


def _align_log_alpha(log_alpha_c, target_shape, compute):
    """把前向产出的 log_alpha 对齐到用例声明的形状，空缺按 -inf 补齐。

    ``aten._ctc_loss`` 返回 ``(N, T, 2*max(target_lengths)+1)``；用例声明与之
    不一致时按 -inf 对齐（-inf 是不可达状态的数学定义值，见 ``_sanitize_log_alpha``）。
    """
    if tuple(log_alpha_c.shape) == tuple(target_shape):
        return log_alpha_c
    aligned = torch.full(tuple(target_shape), float("-inf"), dtype=compute)
    slices = tuple(
        slice(0, min(a, b)) for a, b in zip(aligned.shape, log_alpha_c.shape)
    )
    aligned[slices] = log_alpha_c[slices]
    return aligned


# --------------------------------------------------------------------------- #
# aclnn 入口
# --------------------------------------------------------------------------- #
class AclnnCtcLossBackwardTestSpec:
    """aclnnCtcLossBackward 测试规范（aclnn 流程，CPU torch.Tensor 入/出）。

    参数名/顺序跟随 aclnnCtcLossBackwardGetWorkspaceSize（不含 workspaceSize
    与 executor）。长度为 aclIntArray【属性】（CSV attributes 填 list），
    与 kernel 的张量输入协议不同；归零属性名为驼峰 zeroInfinity；无 reduction。
    """

    def golden(
        gradOut,
        logProbs,
        targets,
        inputLengths,
        targetLengths,
        negLogLikelihood,
        logAlpha,
        blank=0,
        zeroInfinity=False,
        out=None,
        **kwargs,
    ):
        """aclnn 反向真值。

        Args:
            gradOut: (N,) 逐 batch 上游梯度（dtype 与 logProbs 一致）。
            logProbs: (T, N, C) 对数概率。
            targets: (N, S) 或 (sum(targetLengths),) 目标序列。
            inputLengths: 长度为 N 的 int 序列【aclIntArray 属性】。
            targetLengths: 长度为 N 的 int 序列【aclIntArray 属性】。
            negLogLikelihood: (N,) 前向损失。
            logAlpha: (N, T, 2*max(targetLengths)+1) 前向 alpha。
            blank: 空白标识，0 <= blank < C。
            zeroInfinity: 是否对 ±inf 损失的 batch 输出 0 梯度。
            out: (T, N, C) 输出张量，用于确定输出 dtype 契约。
            **kwargs: tensor_dtypes、tensor_formats、scalar_dtypes、use_torch、
                short_soc_version、testcase_name 等。

        Returns:
            list[torch.Tensor]: [grad]，shape=(T, N, C)，dtype 同输出契约。
        """
        grad = _ctc_loss_backward_core(
            gradOut,
            logProbs,
            targets,
            _int_list(inputLengths),
            _int_list(targetLengths),
            negLogLikelihood,
            logAlpha,
            blank,
            zeroInfinity,
        )
        target_dtype = out.dtype if out is not None else logProbs.dtype
        return [grad.to(target_dtype)]

    def customize_inputs(
        gradOut,
        logProbs,
        targets,
        inputLengths,
        targetLengths,
        negLogLikelihood,
        logAlpha,
        blank=0,
        zeroInfinity=False,
        out=None,
        **kwargs,
    ):
        """原地改写输入，保证 negLogLikelihood / logAlpha 与前向自洽。

        【为何必须自定义】negLogLikelihood 与 logAlpha 是 CTC 前向的中间产物，
        彼此以及与 (logProbs, targets, lengths, blank) 之间存在严格数学约束——
        随机填充会让反向公式 grad=(exp(lp)-exp(res+nll-lp))*gradOut 中的
        ``res+nll-lp`` 完全失真，真值与设备都会算出无意义结果，无法形成有效比较。

        【为何原地改写】TTK aclnn 流程的 ``_call_custom_input`` 丢弃返回值，
        只有 ``copy_()`` 原地写入才能让设备与 golden 观测到同一组输入。

        【确定性】seed = md5(testcase_name)，可复现。
        """
        name = kwargs.get("testcase_name", "aclnnCtcLossBackward")
        generator = torch.Generator().manual_seed(_seed_from_name(name))

        time_steps, batch, num_classes = (
            int(logProbs.shape[0]),
            int(logProbs.shape[1]),
            int(logProbs.shape[2]),
        )
        compute = _compute_dtype(logProbs.dtype)

        input_lengths_l = [min(v, time_steps) for v in _int_list(inputLengths)]
        target_lengths_l = _int_list(targetLengths)

        # logProbs：随机张量做 log_softmax(dim=2) 得到合法对数概率
        raw = torch.randn(
            time_steps, batch, num_classes, generator=generator, dtype=torch.float32
        )
        log_probs_c = torch.log_softmax(raw, dim=2).to(compute)
        logProbs.copy_(log_probs_c.to(logProbs.dtype))

        # targets：在【目标 dtype 可表示】的 [0, C) \ {blank} 内采样，
        # 二维 (N,S) 与一维拼接两种形式都支持。本接口 targets 合法 dtype 含
        # BOOL/FLOAT/FLOAT16，窄 dtype 下采样值与落地值不等（BOOL 会把所有
        # 非零标签坍缩为 True），故统一走 _fill_targets 并取其【读回的落地值】。
        blank_i = int(blank)
        targets_landed = _fill_targets(targets, num_classes, blank_i, generator)

        # 前向：用落地后的 logProbs/targets 产生自洽的 nll / logAlpha。
        # 必须传 targets_landed（实际 targets 转 int64），不能传采样时的原值，
        # 否则 BOOL 等 dtype 下中间量对应的是另一组类别编号，输入不自洽。
        nll_c, log_alpha_c = _forward_for_consistency(
            logProbs.to(compute),
            targets_landed,
            input_lengths_l,
            target_lengths_l,
            blank_i,
            zeroInfinity,
        )
        negLogLikelihood.copy_(nll_c.to(negLogLikelihood.dtype))

        # logAlpha 声明第三维为 2*S+1；与前向输出不一致时按 -inf 对齐填充
        expected = log_alpha_c
        if tuple(expected.shape) != tuple(logAlpha.shape):
            aligned = torch.full(tuple(logAlpha.shape), float("-inf"), dtype=compute)
            slices = tuple(
                slice(0, min(a, b)) for a, b in zip(aligned.shape, expected.shape)
            )
            aligned[slices] = expected[slices]
            expected = aligned
        logAlpha.copy_(expected.to(logAlpha.dtype))

        # gradOut：逐 batch 非均匀上游梯度（真实上游梯度，非全 1）
        grad_out_c = (
            torch.rand(batch, generator=generator, dtype=torch.float32) * 2.0 - 1.0
        )
        gradOut.copy_(grad_out_c.to(gradOut.dtype))

    class ThirdPartyImpl:
        """torch 三方标杆（aclnn 参数包装）：长度为 aclIntArray 属性，驼峰属性名。

        与 golden 共享 ``_ctc_loss_backward_core``，仅参数包装不同。
        入参已是目标设备上的 torch 张量，直接按设备处理，不做 numpy 往返。
        """

        def __init__(
            self,
            *,
            inputLengths=None,
            targetLengths=None,
            blank=0,
            zeroInfinity=False,
            **kwargs,
        ):
            self.input_lengths = (
                _int_list(inputLengths) if inputLengths is not None else None
            )
            self.target_lengths = (
                _int_list(targetLengths) if targetLengths is not None else None
            )
            self.blank = int(blank)
            self.zero_infinity = bool(zeroInfinity)

        def __call__(
            self,
            gradOut,
            logProbs,
            targets,
            negLogLikelihood,
            logAlpha,
            out=None,
            **kwargs,
        ):
            grad = _ctc_loss_backward_core(
                gradOut,
                logProbs,
                targets,
                self.input_lengths,
                self.target_lengths,
                negLogLikelihood,
                logAlpha,
                self.blank,
                self.zero_infinity,
            )
            target_dtype = out.dtype if out is not None else logProbs.dtype
            return [grad.to(target_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }


# --------------------------------------------------------------------------- #
# 独立参考实现（E2E third_party 专用，不调用被测 aten._ctc_loss_backward）
# --------------------------------------------------------------------------- #
def _batch_targets(targets, batch_idx, target_lengths):
    """取第 b 个 batch 的标签序列，兼容 ``(N,S)`` 与 ``(sum(tl),)`` 两种形态。"""
    length = int(target_lengths[batch_idx])
    if targets.dim() == 1:
        offset = sum(int(v) for v in target_lengths[:batch_idx])
        return targets[offset : offset + length].to(torch.int64)
    return targets[batch_idx, :length].to(torch.int64)


def _target_prime(labels, blank, device):
    """``target_prime = [blank, l0, blank, l1, ...]``，长度 ``2*S+1``。"""
    size = 2 * int(labels.numel()) + 1
    prime = torch.full((size,), int(blank), dtype=torch.int64, device=device)
    if labels.numel() > 0:
        prime[1::2] = labels
    return prime


def _log_beta(log_probs_b, prime, input_length, blank):
    """反向递推 ``log_beta``，形状 ``(input_length, 2*S+1)``。

    转移与 CTC 定义一致：``s`` / ``s+1`` 恒允许；``s+2`` 仅当 ``prime[s]`` 非
    blank 且 ``prime[s] != prime[s+2]``（相邻重复标签必须经 blank 分隔）。
    """
    width = int(prime.numel())
    device, dtype = log_probs_b.device, log_probs_b.dtype
    beta = torch.full((input_length, width), _NEG_INF, dtype=dtype, device=device)

    last = input_length - 1
    beta[last, width - 1] = log_probs_b[last, prime[width - 1]]
    if width >= 2:
        beta[last, width - 2] = log_probs_b[last, prime[width - 2]]

    for step in range(input_length - 2, -1, -1):
        nxt = beta[step + 1]
        shift1 = torch.full((width,), _NEG_INF, dtype=dtype, device=device)
        shift1[: width - 1] = nxt[1:]
        shift2 = torch.full((width,), _NEG_INF, dtype=dtype, device=device)
        if width > 2:
            allowed = (prime[: width - 2] != int(blank)) & (
                prime[: width - 2] != prime[2:]
            )
            shift2[: width - 2] = torch.where(
                allowed, nxt[2:], torch.full_like(nxt[2:], _NEG_INF)
            )
        merged = torch.logsumexp(torch.stack([nxt, shift1, shift2], dim=0), dim=0)
        beta[step] = merged + log_probs_b[step, prime]
    return beta


def _ctc_loss_backward_independent(
    grad_out,
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    neg_log_likelihood,
    log_alpha,
    blank,
    zero_infinity,
):
    """用基础算子独立重算 CTC 反向，**不调用** ``aten._ctc_loss_backward``。

    【为何需要独立实现】``_ctc_loss_backward_core`` 直接复用
    ``aten._ctc_loss_backward``，按设备分派：入参在 NPU 上时它就落回**被测
    kernel**，三方标杆会退化成自比对。此处从 ``log_beta`` 递推重算后验并组合
    梯度，与被测实现无共享代码路径。

    【设备语义】全程用入参所在设备上的张量运算，输出设备 = ``log_probs.device``，
    不做隐式 H2D/D2H，满足"保留计算设备"的标杆要求。

    公式与 ATen CPU 实现一致：
        ``grad[t,b,c] = (exp(lp) - exp(logsumexp_{s: prime[s]=c}(alpha+beta)
                         + nll - lp)) * grad_out[b]``
    ``t >= input_lengths[b]`` 处梯度为 0；``zero_infinity`` 时 nll 非有限的
    batch 整体输出 0。

    Returns:
        torch.Tensor: ``(T, N, C)``，dtype = ``log_probs.dtype``。
    """
    device, dtype = log_probs.device, log_probs.dtype
    time_steps, batch, num_classes = (int(v) for v in log_probs.shape)
    grad = torch.zeros((time_steps, batch, num_classes), dtype=dtype, device=device)

    for batch_idx in range(batch):
        valid_t = min(int(input_lengths[batch_idx]), time_steps)
        if valid_t <= 0:
            continue
        labels = _batch_targets(targets, batch_idx, target_lengths)
        prime = _target_prime(labels, blank, device)
        width = int(prime.numel())

        log_probs_b = log_probs[:, batch_idx, :]
        beta = _log_beta(log_probs_b, prime, valid_t, blank)
        alpha = log_alpha[batch_idx, :valid_t, :width]
        nll = neg_log_likelihood[batch_idx]

        # 同一 class 可能对应多个 s（如 blank 占所有偶数位），按 class 做 logsumexp
        posterior = alpha + beta
        acc = torch.full((valid_t, num_classes), _NEG_INF, dtype=dtype, device=device)
        for cls in torch.unique(prime).tolist():
            acc[:, int(cls)] = torch.logsumexp(posterior[:, prime == cls], dim=1)

        lp_valid = log_probs_b[:valid_t, :]
        term = torch.exp(acc + nll - lp_valid)
        grad_b = (torch.exp(lp_valid) - term) * grad_out[batch_idx]
        if zero_infinity and not bool(torch.isfinite(nll)):
            grad_b = torch.zeros_like(grad_b)
        grad[:valid_t, batch_idx, :] = grad_b

    return grad


# --------------------------------------------------------------------------- #
# Torch E2E 入口（框架 API，eager 与 graph 共用）
# --------------------------------------------------------------------------- #
class CtcLossBackwardE2eGraph(torch.nn.Module):
    """图模式待编译模块（``torch_graph`` 协议：必须是 nn.Module 子类）。

    TTK ``split_params`` 按 ``__init__`` / ``forward`` 签名拆参，再以
    ``model(**fwd_kwargs)`` 全关键字调用，故形参名必须与算子参数名一致。

    分工刻意如此：
      - ``blank`` / ``zero_infinity`` 放 ``__init__`` → 成为显式图常量（编译期固定）。
      - ``input_lengths`` / ``target_lengths`` 留在 ``forward`` → 保持 ``int[]``
        入参形态，正是图模式要考察的 **list 常量化** 路径（动态 shape 下是否被
        特化、是否触发重新编译）。
    """

    def __init__(self, blank=0, zero_infinity=False):
        super().__init__()
        self.blank = int(blank)
        self.zero_infinity = bool(zero_infinity)

    def forward(
        self,
        grad,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
    ):
        """直调被测 int[] overload，返回裸张量（图协议不接受 list）。"""
        return torch.ops.aten._ctc_loss_backward(
            grad,
            log_probs,
            targets,
            _int_list(input_lengths),
            _int_list(target_lengths),
            neg_log_likelihood,
            log_alpha,
            self.blank,
            self.zero_infinity,
        )


class CtcLossBackwardE2eSpec:
    """``torch.ops.aten._ctc_loss_backward`` 测试规范（E2E 框架 API 流程）。

    注册 key 为 CSV ``api_name`` 原始字符串 ``torch.ops.aten._ctc_loss_backward``
    （TTK 对带点 api_name 的命名约定会产出非法类名，必须经 ``__spec__`` 显式注册）。

    **入口为 int[] overload，不是别名**。三个 overload 中只有它带 NPU 注册
    （``RegisterNPU.cpp`` 的 PrivateUse1 键），实测落点
    ``aclnnCtcLossBackward_CtcLossV2GradAiCore_CTCLossV2Grad``，即本仓交付实现；
    ``.Tensor`` overload 在 NPU 上是**静默 CPU 回退**（``VariableFallbackKernel``
    告警），``.out`` 同样无 NPU 注册——两者都不能用来验证本算子。

    与另两个入口的协议差异（不用别名掩盖，签名按 ATen schema 写）：

    ====================  ==========================  =============================
    项                    aclnn                       torch E2E
    ====================  ==========================  =============================
    注册 key              aclnnCtcLossBackward        torch.ops.aten._ctc_loss_backward
    参数名来源            aclnn_ctc_loss_backward.h   aten schema（_schemas）
    上游梯度参数名        gradOut                     **grad**
    长度参数              aclIntArray【属性】         int[]【属性】
    长度参数名            inputLengths/targetLengths  input_lengths/target_lengths
    归零属性名            zeroInfinity                zero_infinity
    输出契约              out 张量定 dtype            **无 out**，dtype 随 log_probs
    ====================  ==========================  =============================

    reduction：ATen ``_ctc_loss_backward`` 本身**不含** reduction——上层
    ``ctc_loss`` 的 mean/sum 缩放发生在传入 ``grad`` **之前**。故本入口与 golden
    都只做"按给定 grad 反传"，两侧均不再执行任何归一化，结构上杜绝重复缩放
    （计算核心 ``_ctc_loss_backward_core`` 签名上就不接收 reduction）。

    CSV 张量池必须**正好 5 个**（grad / log_probs / targets / neg_log_likelihood /
    log_alpha）：``match_overload`` 实测 5 个张量命中 overload 0，6 或 7 个均匹配
    失败；``input_lengths`` / ``target_lengths`` / ``blank`` / ``zero_infinity``
    一律进 CSV ``attributes``，不声明 ``out``，输出取 API 返回值。
    """

    def golden(
        grad,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        blank=0,
        zero_infinity=False,
        **kwargs,
    ):
        """CPU 独立参考计算（入参为 CPU torch.Tensor，参数名同 aten schema）。

        Args:
            grad: (N,) 逐 batch 真实上游梯度（非全 1，见 customize_inputs）。
            log_probs: (T, N, C) 对数概率。
            targets: (N, S) 或 (sum(target_lengths),) 目标序列。
            input_lengths: 长度为 N 的 int 序列【int[] 属性】。
            target_lengths: 长度为 N 的 int 序列【int[] 属性】。
            neg_log_likelihood: (N,) 前向损失（与输入自洽）。
            log_alpha: (N, T, 2*max(target_lengths)+1) 前向 alpha。
            blank: 空白标识，0 <= blank < C。
            zero_infinity: 是否对 ±inf 损失的 batch 输出 0 梯度。
            **kwargs: TTK 注入的元信息（tensor_dtypes、testcase_name 等）。

        Returns:
            list[torch.Tensor]: [grad_log_probs]，shape=(T, N, C)，
            dtype 舍回 log_probs 的 dtype。

        低精度策略：fp16/bf16 在 CPU aten 上不支持（实测
        ``"ctc_loss_backward_cpu" not implemented for 'Half'/'BFloat16'``），
        故**参考计算**抬 fp32 后舍回被测 dtype。升精度只服务参考端，
        NPU 侧全程以原 dtype 下发（实测 fp16/bf16 各自落 AiCore 且输出 dtype 保持），
        不静默改变被测 dtype。
        """
        result = _ctc_loss_backward_core(
            grad,
            log_probs,
            targets,
            _int_list(input_lengths),
            _int_list(target_lengths),
            neg_log_likelihood,
            log_alpha,
            blank,
            zero_infinity,
        )
        return [result.to(log_probs.dtype)]

    def customize_inputs(
        grad,
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        blank=0,
        zero_infinity=False,
        **kwargs,
    ):
        """原地改写输入，保证 log_probs 合法、中间量与前向自洽。

        【为何必须自定义】``neg_log_likelihood`` 与 ``log_alpha`` 是 CTC 前向的
        中间产物，与 (log_probs, targets, lengths, blank) 存在严格数学约束。
        独立随机填充会让反向公式中的 ``res + nll - lp`` 完全失真，参考与设备
        都算出无意义结果，无法形成有效比较。随机 log_probs 也不满足接口要求
        （必须是对数概率）。

        【为何原地改写】TTK E2E 的 ``generate_inputs`` 以
        ``input_func(*args, **kwargs)`` 调用并**丢弃返回值**，随后从同一组
        ``np_storages`` 重建张量。``torch.from_numpy`` 与 ``as_strided`` 共享
        该 storage（bf16 经 int16 位视图同样共享），故只有 ``copy_()`` 原地写入
        才能让设备侧与 golden 观测到**完全相同**的输入；重新绑定变量名会丢失。

        【targets 形态】``targets_mode`` 不是 schema 参数名，经 ``extra_attrs``
        注入，用于确定性地构造重复标签，取值见 ``_fill_targets``。

        【确定性】seed = md5(testcase_name)，可复现。
        """
        name = kwargs.get("testcase_name", "torch.ops.aten._ctc_loss_backward")
        generator = torch.Generator().manual_seed(_seed_from_name(name))

        time_steps, batch, num_classes = (
            int(log_probs.shape[0]),
            int(log_probs.shape[1]),
            int(log_probs.shape[2]),
        )
        compute = _compute_dtype(log_probs.dtype)
        blank_i = int(blank)

        # input_lengths 的接口约束是 <= T；按约束截断，避免用例笔误产生非法输入
        input_lengths_l = [min(v, time_steps) for v in _int_list(input_lengths)]
        target_lengths_l = _int_list(target_lengths)

        # log_probs：log_softmax(dim=2) 保证是合法对数概率（满足接口要求）
        raw = torch.randn(
            time_steps, batch, num_classes, generator=generator, dtype=torch.float32
        )
        log_probs.copy_(torch.log_softmax(raw, dim=2).to(log_probs.dtype))

        # targets：合法标签（∈ [0,C) 且 != blank），形态由 targets_mode 控制。
        # 返回值是【读回的落地标签】（窄 dtype 下与采样原值可能不等）。
        targets_landed = _fill_targets(
            targets,
            num_classes,
            blank_i,
            generator,
            kwargs.get("targets_mode", "random"),
        )

        # 前向：用【落地后的】log_probs/targets 产出自洽的 nll / log_alpha
        nll_c, log_alpha_c = _forward_for_consistency(
            log_probs.to(compute),
            targets_landed,
            input_lengths_l,
            target_lengths_l,
            blank_i,
            zero_infinity,
        )
        neg_log_likelihood.copy_(nll_c.to(neg_log_likelihood.dtype))
        log_alpha.copy_(
            _align_log_alpha(log_alpha_c, log_alpha.shape, compute).to(log_alpha.dtype)
        )

        # grad：逐 batch 非均匀真实上游梯度（含正负，不是全 1）。
        # 不在此处做 mean/sum 缩放——ATen 反向不含 reduction，缩放属上层职责。
        grad_c = torch.rand(batch, generator=generator, dtype=torch.float32) * 2.0 - 1.0
        grad.copy_(grad_c.to(grad.dtype))

    class ThirdPartyImpl:
        """E2E 三方标杆：torch 基础算子独立重算，**不调用被测 aten 入口**。

        【为何不能复用 golden 的计算核心】``_ctc_loss_backward_core`` 直调
        ``torch.ops.aten._ctc_loss_backward``；E2E 的 third_party 入参是**设备侧**
        张量，该调用会按 PrivateUse1 键分派回本仓待测 kernel，退化为自比对。
        故此处改用 ``_ctc_loss_backward_independent``（logsumexp/where/gather 等
        基础算子重算 log_beta 与后验），与被测实现无共享代码路径。

        【参数与输出契约】按 ATen ``_ctc_loss_backward`` 的 **int[] overload**：
        长度参数名 ``input_lengths``/``target_lengths`` 且为属性（经 ``__init__``
        注入），归零属性名 ``zero_infinity``；上游梯度参数名为 ``grad``（非
        ``gradOut``）；**无 out 参数**，输出 dtype 随 ``log_probs``。

        【设备语义】输出设备恒等于 ``log_probs.device``，不做隐式 H2D/D2H；
        低精度仅在参考侧抬到 fp32 计算后舍回被测 dtype，不改变被测 dtype。
        """

        def __init__(
            self,
            *,
            input_lengths=None,
            target_lengths=None,
            blank=0,
            zero_infinity=False,
            **kwargs,
        ):
            self.input_lengths = (
                _int_list(input_lengths) if input_lengths is not None else None
            )
            self.target_lengths = (
                _int_list(target_lengths) if target_lengths is not None else None
            )
            self.blank = int(blank)
            self.zero_infinity = bool(zero_infinity)

        def __call__(
            self,
            grad,
            log_probs,
            targets,
            neg_log_likelihood,
            log_alpha,
            **kwargs,
        ):
            compute = _compute_dtype(log_probs.dtype)
            result = _ctc_loss_backward_independent(
                grad.to(compute),
                log_probs.to(compute),
                targets,
                self.input_lengths,
                self.target_lengths,
                neg_log_likelihood.to(compute),
                log_alpha.to(compute),
                self.blank,
                self.zero_infinity,
            )
            return [result.to(log_probs.dtype)]

    torch_graph = CtcLossBackwardE2eGraph
    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }
