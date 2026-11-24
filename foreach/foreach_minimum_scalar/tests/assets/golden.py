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
"""
foreach_minimum_scalar 两方 Golden（无 GPU 版）

由用户 golden 适配而来，用于"NPU vs CPU golden"两方精度比对：
来源: /data/w50064389/SIMT-LongTails/ForeachMinimumScalar/deliverables/thirdparty_golden/golden.py

适配内容（相对原文件）：
1. tolerance: 浮点 cross_check/L1（生态统一口径，需三方参考），整型 binary_equal
2. kernel golden 增加 bf16 (ml_dtypes 载体 / numpy 'V' dtype) 处理：
   view 还原 → fp32 计算 → 输出按用例 output_dtypes 转回；修复原版 bf16 TypeError
3. 补充 aclnnForeachMinimumScalarV2 注册（原版只注册 V1）
4. third_party 保留并按名绑定类形式声明（Kernel/ACLNN/E2E 三通路均有 torch 第三方参考；
   浮点 cross_check 轮次消费三方输出，需具备 XPU endpoint）
5. golden 统一超范围标量回退（Kernel/ACLNN/E2E）：低精度张量叠加超出该精度的标量时 torch
   直接调用会 RuntimeError，回退为 fp32 计算后按原 dtype 回写；仅对“浮点标量转换溢出”回退，
   整数溢出/长度/形状/其它错误原样抛出。third_party 保持 torch 原语义、不做回退（三方参考独立）
6. E2E 类形态 third_party 按池中参数分派 torch._foreach_minimum 的三种重载：bare api 名下
   torchapi/torchair 用例同时覆盖 Scalar（attributes.scalar）/ ScalarList（attributes.scalars）
   / List.List（第二输入 other）；原先 __init__ 强制要求 scalar，另两种重载在远端按名绑定时
   直接 400（parameter 'scalar' of __init__ has no value，仅 int32 因 binary_equal 不走三方
   而通过）。函数形态 golden 与 E2E 三方统一经 _foreach_minimum_dispatch 按实参类型分派；
   E2E 三方 __init__ 显式接收 scalar/scalars/other，避免 **kwargs 静默吞掉重载参数；
   空 TensorList 返回空输出；ScalarList 回退前校验等长，长度不一致时重抛。

公式: y[i] = minimum(x[i], scalar)
"""

__spec__ = {
    "foreach_minimum_scalar": "ForeachMinimumScalarKernelSpec",
    "aclnnForeachMinimumScalar": "AclnnForeachMinimumScalarSpec",
    "aclnnForeachMinimumScalarV2": "AclnnForeachMinimumScalarSpec",
    "torch._foreach_minimum": "TorchForeachMinimumScalarSpec",
}

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:  # pragma: no cover
    _bf16 = None

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


def _to_scalar(value):
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item()
    return value


def _per_tensor_dtypes(output_dtypes):
    if output_dtypes is None:
        return None
    flat = []
    for entry in output_dtypes:
        if isinstance(entry, (tuple, list)):
            flat.extend(entry)
        else:
            flat.append(entry)
    return flat


def _to_torch_tensor(value):
    array = np.asarray(value)
    if array.dtype.kind == "V" and _bf16 is not None:
        array = array.view(_bf16)
    if _bf16 is not None and array.dtype == _bf16:
        return torch.from_numpy(array.astype(np.float32))
    return torch.from_numpy(np.ascontiguousarray(array))


def _target_dtype(entry):
    if entry is None:
        return None
    return entry[0] if isinstance(entry, (tuple, list)) else str(entry)


def _cast_out(tensor, target):
    if target == "bfloat16":
        return tensor.numpy().astype(_bf16) if _bf16 is not None else tensor.numpy()
    return tensor.numpy().astype(target)


def _is_scalar_overflow_error(exc):
    """torch 把标量转换到低精度 dtype 失败（超出可表示范围）时的固定报错。"""
    return "without overflow" in str(exc)


def _all_float_tensors(x):
    """回退只对浮点张量生效：整数标量超范围不是精度问题，不能编造 f32 参考值。"""
    return all(isinstance(t, torch.Tensor) and t.is_floating_point() for t in x)


def _foreach_minimum_safe(x, scalar_value):
    """Golden 用 Scalar 参考：超范围标量（低精度张量）回退 fp32 计算后按原 dtype 回写。

    仅重试“标量转换溢出”这一类 RuntimeError；长度/形状/其它错误原样抛出，
    避免把非法输入静默算成结果；空 TensorList 返回空输出。
    """
    if isinstance(x, (list, tuple)) and not x:
        return []
    try:
        return list(torch._foreach_minimum(x, scalar_value))
    except RuntimeError as exc:
        if not _is_scalar_overflow_error(exc) or not _all_float_tensors(x):
            raise
        scalar_f32 = float(np.float32(scalar_value))
        results = []
        for tensor in x:
            if tensor.dtype == torch.float32:
                result = torch._foreach_minimum([tensor], scalar_f32)[0]
            else:
                source_dtype = tensor.dtype
                result = torch._foreach_minimum([tensor.to(torch.float32)], scalar_f32)[
                    0
                ].to(source_dtype)
            results.append(result)
        return results


def _foreach_minimum_scalars_safe(x, scalars):
    """Golden 用 ScalarList 参考：逐张量对标量；超范围标量同样回退 fp32 后回写。

    回退前校验标量列表与张量列表等长：长度不一致时原样重抛（zip 会把短列表静默
    截断，导致比对按缺失项判过）；仅重试“标量转换溢出”，其它 RuntimeError 原样抛出。
    两侧同为空列表（合法空 TensorList）返回空输出。
    """
    if isinstance(x, (list, tuple)) and not x:
        return []
    values = [_to_scalar(value) for value in scalars]
    try:
        return list(torch._foreach_minimum(x, values))
    except RuntimeError as exc:
        if (
            len(values) != len(x)
            or not _is_scalar_overflow_error(exc)
            or not _all_float_tensors(x)
        ):
            raise
        results = []
        for tensor, value in zip(x, values):
            source_dtype = tensor.dtype
            result = torch._foreach_minimum(
                [tensor.to(torch.float32)], [float(np.float32(value))]
            )[0].to(source_dtype)
            results.append(result)
        return results


def _foreach_minimum_dispatch(x, value=None, *, safe=True):
    """按第二个实参的类型分派 torch._foreach_minimum 的三种重载。

    value 为标量 → Scalar；标量列表 → ScalarList；张量列表 → List.List。
    safe=True（golden）对超范围标量走 fp32 回退；safe=False（third_party）保持
    torch 原语义、不做回退，确保三方参考独立。空 TensorList 返回空输出。
    """
    if isinstance(x, (list, tuple)) and not x:
        return []
    if isinstance(value, (list, tuple)):
        if value and torch.is_tensor(value[0]):
            return list(torch._foreach_minimum(x, value))
        if safe:
            return _foreach_minimum_scalars_safe(x, value)
        return list(torch._foreach_minimum(x, [_to_scalar(v) for v in value]))
    if value is None:
        raise TypeError(
            "foreach_minimum: none of 'scalar'/'scalars'/'other' is available "
            "(Scalar / ScalarList / List.List overload cannot be determined)"
        )
    if safe:
        return _foreach_minimum_safe(x, _to_scalar(value))
    return list(torch._foreach_minimum(x, _to_scalar(value)))


class _TorchForeachMinimumScalarImpl:
    """Kernel/ACLNN 通路三方参考：按 TTK 具名绑定（__init__ 取 scalar，__call__ 取 x）。

    scalar 在 Kernel/GEIR 经 attributes、ACLNN V1 经同名张量输入、V2 经 aclScalar
    属性到达，恒在绑定池内；__init__ 给默认值只为避免远端按名绑定时无默认值即 400，
    真缺参在 __call__ 以显式错误暴露。third_party 保持 torch 原语义，不做超范围回退。
    """

    def __init__(self, scalar=None, **kwargs):
        self._scalar = None if scalar is None else _to_scalar(scalar)

    def __call__(self, x, scalar=None, **kwargs):
        value = self._scalar if self._scalar is not None else scalar
        if value is None:
            raise TypeError("_TorchForeachMinimumScalarImpl: 'scalar' is not provided")
        return list(torch._foreach_minimum(x, _to_scalar(value)))


class _TorchForeachMinimumScalarE2EImpl:
    """E2E 通路三方参考：*args 按池顺序取首参（torch API 首参名为 self），
    scalar/scalars/other 按名绑定后分派对应重载。__init__ 不得强制要求只属于
    某一重载的参数，且须显式接收 scalars/other，避免 **kwargs 静默吞掉重载参数。"""

    def __init__(self, scalar=None, scalars=None, other=None, **kwargs):
        self._scalar = None if scalar is None else _to_scalar(scalar)
        self._scalars = scalars
        self._other = other

    def __call__(self, *args, scalar=None, scalars=None, other=None, **kwargs):
        x = args[0] if args else kwargs.pop("x", None)
        if x is None:
            raise TypeError("_TorchForeachMinimumScalarE2EImpl: first input is missing")
        scalar = self._scalar if self._scalar is not None else scalar
        scalars = self._scalars if scalars is None else scalars
        other = self._other if other is None else other
        if scalar is not None:
            value = scalar
        elif scalars is not None:
            value = scalars
        elif other is not None:
            value = other
        else:
            raise TypeError(
                "_TorchForeachMinimumScalarE2EImpl: none of 'scalar'/'scalars'/'other' is provided"
            )
        return _foreach_minimum_dispatch(x, value, safe=False)


_THIRD_PARTY = {"torch": _TorchForeachMinimumScalarImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachMinimumScalarE2EImpl}


class ForeachMinimumScalarKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。y[i] = minimum(x[i], scalar)"""

    def golden(x, scalar, **kwargs):
        output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))
        s = _to_scalar(scalar)
        results = []
        for index, a in enumerate(x):
            ta = _to_torch_tensor(a)
            out = _foreach_minimum_safe([ta], s)[0]
            target = None
            if output_dtypes is not None and index < len(output_dtypes):
                target = _target_dtype(output_dtypes[index])
            if target is None:
                target = str(np.asarray(a).dtype)
            results.append(_cast_out(out, target))
        return results

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachMinimumScalarSpec:
    """ACLNN 路径（V1/V2）— list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, scalar, out=None, **kwargs):
        s = _to_scalar(scalar)
        result = _foreach_minimum_safe(x, s)
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachMinimumScalarSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x=None, scalar=None, self=None, **kwargs):
        actual_x = self if self is not None else x
        value = scalar
        if value is None:
            value = kwargs.get("scalars")
        if value is None:
            value = kwargs.get("other")
        return _foreach_minimum_dispatch(actual_x, value, safe=True)

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
