# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import ClassVar

import numpy as np
import torch

__spec__ = {
    "hard_swish_grad_v2": "HardSwishGradV2KernelSpec",
    "aclnnHardswishBackwardV2": "HardSwishGradV2AclnnSpec",
}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _numpy_to_torch_tensor(array):
    if "bfloat16" in array.dtype.name:
        return torch.from_numpy(array.view(dtype=np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy_tensor(tensor):
    if tensor.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return tensor.view(torch.int16).numpy().view(dtype=bfloat16)
    return tensor.numpy()


def _prepare_hard_swish_grad_v2_inputs(grad_output, self_x):
    source_tensor = (
        _numpy_to_torch_tensor(np.asarray(grad_output))
        if not isinstance(grad_output, torch.Tensor)
        else grad_output
    )
    compute_dtype = (
        torch.float32
        if source_tensor.dtype in (torch.float16, torch.bfloat16)
        else source_tensor.dtype
    )
    g = source_tensor.to(compute_dtype)
    x_source = (
        _numpy_to_torch_tensor(np.asarray(self_x))
        if not isinstance(self_x, torch.Tensor)
        else self_x
    )
    x = x_source.to(compute_dtype)
    return source_tensor, g, x, compute_dtype


def _hard_swish_grad_v2_compute(grad_output, self_x):
    source_tensor, g, x, compute_dtype = _prepare_hard_swish_grad_v2_inputs(
        grad_output, self_x
    )

    # in-place 等价改写: 原实现非 in-place, val/两次 where 输出等多个 full-size
    # 临时张量同时驻留, 大 shape 用例下推高 host 峰值内存. masked_fill_ 与
    # torch.where 填充字面常量逐位一致 (mul 掩码在 val 为负/-inf 时会产生
    # -0.0/NaN, 不等价); 对任意输入 (含 NaN/±inf):
    #   x <= -3 (含 NaN) -> 0;  x >= 3 (含 NaN) -> 1;  (-3, 3) -> x/3 + 0.5
    val = torch.div(x, 3)
    val.add_(0.5)
    bound = torch.logical_not(x > -3)
    val.masked_fill_(bound, 0.0)
    torch.logical_not(x < 3, out=bound)
    val.masked_fill_(bound, 1.0)
    del bound

    # 结果复用 val 缓冲, fp32 下 .to() 无拷贝
    return torch.mul(g, val, out=val).to(source_tensor.dtype)


# 三方(GPU)分块参数: 2^27 元素以下直调保持原行为; 超限时按 flat 维度分块,
# 临时显存预算 ~1 GiB(覆盖 aten 内部临时 + 结果 + 掩码 + 两次 where 的 ~8 份
# 等长临时)。纯 elementwise 算子分块与整块逐位一致, 输入/结果整块驻留不可省。
_THIRD_PARTY_CHUNK_MIN_NUMEL = 134217728
_THIRD_PARTY_CHUNK_TEMP_BYTES = 1 << 30


def _hard_swish_grad_v2_third_party_direct(native_op, grad_tensor, self_tensor):
    """aten 直调 + 端点覆盖(与原实现逐行一致, 供直调与分块两路共用)。"""
    native_result = native_op.default(grad_tensor, self_tensor)
    # aten uses a non-strict endpoint convention. The operator contract
    # requires self == -3 -> 0 and self == 3 -> gradOutput, so override only those
    # exact boundary elements while retaining the native result elsewhere.
    lower = self_tensor == torch.tensor(
        -3.0, dtype=self_tensor.dtype, device=self_tensor.device
    )
    upper = self_tensor == torch.tensor(
        3.0, dtype=self_tensor.dtype, device=self_tensor.device
    )
    native_result = torch.where(lower, torch.zeros_like(native_result), native_result)
    native_result = torch.where(upper, grad_tensor, native_result)
    return native_result


class _HardSwishGradV2Compose:
    def __call__(_instance, gradOutput, self, **kwargs):
        del kwargs
        self_x = self
        native_op = getattr(torch.ops.aten, "hardswish_backward", None)
        if native_op is None or not hasattr(native_op, "default"):
            raise RuntimeError(
                "torch.ops.aten.hardswish_backward.default is unavailable; "
                "native third-party reference cannot be provided"
            )
        grad_tensor = (
            gradOutput
            if isinstance(gradOutput, torch.Tensor)
            else _numpy_to_torch_tensor(np.asarray(gradOutput))
        )
        self_tensor = (
            self_x
            if isinstance(self_x, torch.Tensor)
            else _numpy_to_torch_tensor(np.asarray(self_x))
        )
        # 大张量分块: 整块三方(aten + 端点覆盖)的 full-size 临时显存需求
        # (~40 GiB 起) 超过常见 40G 卡容量, 必然 CUDA OOM。分块后峰值
        # = 输入 + 结果 + O(1 GiB)。
        if (
            grad_tensor.is_cuda
            and grad_tensor.is_contiguous()
            and self_tensor.is_contiguous()
            and grad_tensor.numel() > _THIRD_PARTY_CHUNK_MIN_NUMEL
        ):
            chunk = max(
                1, _THIRD_PARTY_CHUNK_TEMP_BYTES // (8 * grad_tensor.element_size())
            )
            grad_flat = grad_tensor.view(-1)
            self_flat = self_tensor.view(-1)
            result = torch.empty_like(grad_flat)
            for start in range(0, grad_flat.numel(), chunk):
                stop = min(start + chunk, grad_flat.numel())
                result[start:stop] = _hard_swish_grad_v2_third_party_direct(
                    native_op, grad_flat[start:stop], self_flat[start:stop]
                )
            return [result.view(grad_tensor.shape)]
        return [
            _hard_swish_grad_v2_third_party_direct(native_op, grad_tensor, self_tensor)
        ]


def _tensorflow_hard_swish_grad_v2(gradOutput, self, **kwargs):
    """TensorFlow third-party reference for the grad formula."""
    import tensorflow as tf

    del kwargs
    grad_output = tf.convert_to_tensor(gradOutput)
    self_x = tf.convert_to_tensor(self)
    source_dtype = grad_output.dtype
    compute_dtype = (
        tf.float32 if source_dtype in (tf.float16, tf.bfloat16) else source_dtype
    )
    grad = tf.cast(grad_output, compute_dtype)
    self_value = tf.cast(self_x, compute_dtype)
    value = self_value / tf.constant(3.0, dtype=compute_dtype)
    value = value + tf.constant(0.5, dtype=compute_dtype)
    value = tf.where(
        self_value > tf.constant(-3.0, dtype=compute_dtype),
        value,
        tf.constant(0.0, dtype=compute_dtype),
    )
    value = tf.where(
        self_value < tf.constant(3.0, dtype=compute_dtype),
        value,
        tf.constant(1.0, dtype=compute_dtype),
    )
    return [tf.cast(grad * value, source_dtype)]


class HardSwishGradV2KernelSpec:
    @staticmethod
    def golden(gradOutput, self, **kwargs):
        del kwargs
        out = _hard_swish_grad_v2_compute(gradOutput, self)
        return [_torch_to_numpy_tensor(out)]

    third_party: ClassVar[dict] = {
        "torch": _HardSwishGradV2Compose,
        "tf": _tensorflow_hard_swish_grad_v2,
    }
    tolerance = _TOL


class HardSwishGradV2AclnnSpec:
    @staticmethod
    def golden(gradOutput, self, out=None, **kwargs):
        return [_hard_swish_grad_v2_compute(gradOutput, self)]

    third_party: ClassVar[dict] = {
        "torch": _HardSwishGradV2Compose,
        "tf": _tensorflow_hard_swish_grad_v2,
    }
    tolerance = _TOL
