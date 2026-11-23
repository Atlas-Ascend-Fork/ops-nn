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

import numpy as np
import torch

__spec__ = {
    "mse_loss": "MseLossTestSpec",
    "aclnnMseLoss": "AclnnMseLossTestSpec",
    "aclnnMseLossOut": "MseLossOutAclnnSpec",
    "torch.nn.functional.mse_loss": "TorchMseLossTestSpec",
}

__golden__ = {
    "kernel": {"mse_loss": "mse_loss_golden"},
    "aclnn": {"aclnnMseLoss": "aclnn_mse_loss_golden"},
}

_REDUCTION_INT_TO_STR = {0: "none", 1: "mean", 2: "sum"}

# aclnnMseLossOut 只经 __spec__ 注册（loader 的 TestSpec 优先于 __golden__，
# 双注册会产生绕过 spec 的死配置）。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _demote_promoted(value):
    """Like ops-cv !2491: IR/ACLNN exclude FP64 tensors, so undo TTK promotion.

    Do not use this for Torch E2E, which accepts genuine FP64 inputs.
    """
    if hasattr(value, "detach"):
        return value.float() if str(value.dtype) == "torch.float64" else value
    arr = np.asarray(value)
    return arr.astype(np.float32) if arr.dtype == np.float64 else value


def _red_str(v):
    """reduction 归一到字符串：kernel/e2e 通路本来就是 str，aclnn 通路是 int64。"""
    if isinstance(v, str):
        return v.strip().lower()
    return _REDUCTION_INT_TO_STR.get(int(v), "mean")


def aclnn_mse_loss_golden(self_t, target_t, reduction, out_t=None, **kwargs):
    import torch

    self_t, target_t = map(_demote_promoted, (self_t, target_t))
    red = _REDUCTION_INT_TO_STR.get(int(reduction), "mean")
    # Compute from supported input precision; do not inspect DUT output values.
    compute = torch.promote_types(self_t.dtype, target_t.dtype)
    if compute in (torch.float16, torch.bfloat16):
        compute = torch.float32
    diff = self_t.to(compute) - target_t.to(compute)
    sqr = diff * diff
    if red == "sum":
        result = sqr.sum()
    elif red == "mean":
        result = sqr.mean()
    else:
        result = sqr
    if out_t is not None and hasattr(out_t, "dtype"):
        result = result.to(out_t.dtype)
    return result


def aclnn_mse_loss_out_golden(
    self_t, target_t, reduction, out_t=None, out_dtype=None, **kwargs
):
    """aclnnMseLossOut：mean/sum 归约 0 轴（broadcast 去首维），none 逐元素
    （语义见 docs/aclnnMseLossOut.md 与 aclnn_mse_loss_out.cpp）。

    FP64 交付输入按接口支持范围回到 FP32；末位照声明 out dtype 落取值网格与
    溢出语义（fp16 out 放不下时 golden 与 NPU 同为 ±inf），容器回到计算档
    （直返 fp16 会在 np.promote_types(fp16, bfloat16) 抛 TypeError）。
    """
    self_t, target_t = map(_demote_promoted, (self_t, target_t))
    red = _red_str(reduction)
    diff = self_t - target_t
    sqr = diff * diff
    # 按公式归约：空的 0 轴 sum 为 0、mean 为 NaN；其他轴为空时输出仍为空。
    # API 的空路径不写 out；测试框架的预填值不是数学参考或输出豁免契约。
    if red == "sum":
        result = sqr.sum(dim=0)
    elif red == "mean":
        result = sqr.sum(dim=0) / sqr.shape[0]
    else:
        result = sqr
    dt = out_dtype
    if dt is None and out_t is not None and hasattr(out_t, "dtype"):
        dt = out_t.dtype
    if dt is not None:
        result = result.to(_to_torch_dtype(dt)).to(diff.dtype)
    return result


def _flatten(nested):
    """递归平铺嵌套 tuple/list（tensor_dtypes 可能按 TensorList 嵌套）。"""
    if isinstance(nested, (str, bytes)):  # 裸字符串是单元素，不可逐字符迭代
        yield nested
        return
    for x in nested:
        if isinstance(x, (tuple, list)):
            yield from _flatten(x)
        else:
            yield x


def _to_torch_dtype(dt):
    """dtype 归一到 torch.dtype：接受 torch.dtype 与 CSV 声明字符串（'float16' 等）。"""
    if isinstance(dt, torch.dtype):
        return dt
    return {
        "float": torch.float32,
        "float32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "double": torch.float64,
        "float64": torch.float64,
    }[str(dt).strip().lower()]


def _third_party_mse_out(self_t, target_t, reduction, out_t=None):
    """三方腿：torch 原生接口、原生 dtype，不自行抬 fp32。
    与 NPU 链路（MSELossV2→ReduceMean/Sum→Cast，中间均落输入 dtype）逐算子
    同构；抬 fp32 跳过中间舍入会让三方误差虚低、cross_check 三比值误判 NPU。
    空张量同样按框架语义归约，不复制设备空路径保留的输出初值。"""
    red = _red_str(reduction)
    # F.mse_loss 要求同 dtype：混合输入按 result_type 对齐 NPU PromoteType
    work = torch.result_type(self_t, target_t)
    d = torch.nn.functional.mse_loss(
        self_t.to(work), target_t.to(work), reduction="none"
    )
    if red == "mean":
        out = d.mean(dim=0)
    elif red == "sum":
        out = d.sum(dim=0)
    else:
        out = d
    if out_t is not None and hasattr(out_t, "dtype"):
        out = out.to(out_t.dtype)
    return out


class _MseLossOutCompose:
    """aclnnMseLossOut 三方标杆：入参按 aclnn 头文件顺序 (self, target, reduction, out)。

    ⚠️ `self` 仅位置参数：aclnn 首参名就叫 self，服务端按名绑定时 self=<tensor>
    会以关键字进 **_；positional-only 声明后关键字 self 落进 **kw，不撞实例参数。
    """

    def __init__(self, /, reduction=1, **_):
        self._red = _red_str(reduction)

    def __call__(self, /, *args, **kw):
        x = args[0] if len(args) > 0 else kw.get("self", kw.get("input"))
        t = args[1] if len(args) > 1 else kw.get("target")
        red = args[2] if len(args) > 2 else kw.get("reduction", self._red)
        out_t = args[3] if len(args) > 3 else kw.get("out")
        return [_third_party_mse_out(x, t, red, out_t)]


class MseLossOutAclnnSpec:
    """aclnn 通路 spec（aclnnMseLossOut）：golden 收 CPU torch.Tensor、返 Tensor。

    签名取 aclnn_mse_loss_out.h 参数序（self/target/reduction/out）。
    golden 末位照声明 out dtype 落网格+溢出语义，dtype 来源按可靠性：
    out 实张量 dtype（纯输出不被 Promote 抬档，探针实证以声明 dtype 送达）
    → kwargs tensor_dtypes 末位。不取 kwargs["output_dtypes"]：aclnn 通路
    不下发该键。不砍回时 fp16 out 溢出位 golden 有限、NPU ±inf，
    cross_check Path A 必误判 NPU（L0_003）。
    """

    def golden(self, target, reduction=1, out=None, **kwargs):
        # 可靠性排序: out 实张量 dtype → tensor_dtypes 末位(纯输出声明档)
        out_dtype = getattr(out, "dtype", None)
        if out_dtype is None:
            td = kwargs.get("tensor_dtypes")
            if td is not None:
                flat = [x for x in _flatten(td) if x is not None]
                if flat:
                    out_dtype = flat[-1]
        return [
            aclnn_mse_loss_out_golden(self, target, reduction, out, out_dtype=out_dtype)
        ]

    third_party = {"torch": _MseLossOutCompose}
    tolerance = _TOL


def mse_loss_golden(input0, input1, *, reduction="mean", **kwargs):
    """
    Golden function for mse_loss.
    All the parameters (names and order) follow @mse_loss_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    For reduction="none" the output keeps the input shape, as required by
    @mse_loss_infershape.cpp. (The reference previously flattened the inputs and
    returned a 1-D result, which mismatched the declared output shape.)

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [y]: shape == input shape for reduction="none", scalar otherwise.
    """
    import torch
    import torch.nn.functional as F

    input0, input1 = map(_demote_promoted, (input0, input1))
    ori_dtype = input0.dtype
    predict = input0
    label = input1

    if "float16" in ori_dtype.name:
        predict = predict.astype("float32")
        label = label.astype("float32")

    y = F.mse_loss(
        torch.from_numpy(predict), torch.from_numpy(label), reduction=reduction
    )
    return [y.numpy().astype(ori_dtype, copy=False)]


def torch_mse_loss_golden(
    input,
    target,
    size_average=None,
    reduce=None,
    reduction="mean",
    weight=None,
    **kwargs,
):
    """E2E golden for torch.nn.functional.mse_loss.

    The parameter list mirrors the full torch signature positionally, because the
    E2E flow fills every parameter (inputs and attributes alike) by position.
    Match the legacy CPU framework reference at the input dtype, including
    intermediate rounding and overflow; do not promote the reference to float64.
    """
    import torch

    # Older Torch versions lack weight; do not pass an absent optional argument.
    weight_kwargs = {"weight": weight.detach().cpu()} if weight is not None else {}
    result = torch.nn.functional.mse_loss(
        input.detach().cpu(),
        target.detach().cpu(),
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
        **weight_kwargs,
    )
    return [result]


class MseLossTestSpec:
    """MseLoss test spec (kernel / GEIR flow, numpy in and out)."""

    golden = mse_loss_golden
    third_party = {"torch": "torch.nn.functional.mse_loss"}


class AclnnMseLossTestSpec:
    """MseLoss test spec (aclnn flow, torch.Tensor in and out)."""

    golden = aclnn_mse_loss_golden


class TorchMseLossTestSpec:
    """MseLoss test spec (e2e flow).

    Dispatch depends on the framework/CANN version and execution mode. With
    Torch 2.10/CANN 9.3, eager reaches MSELossV2, while GE graphs reach MseLoss.
    Profile each mode before counting it as coverage of the target kernel;
    framework numerical agreement alone does not establish that coverage.
    """

    golden = torch_mse_loss_golden
