#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FastGeluV2 独立 Golden：raw/GEIR Kernel。"""

import torch

__spec__ = {
    "fast_gelu_v2": "FastGeluV2KernelSpec",
}

_L0 = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
}


def _torch_fast_gelu_v2(x):
    """Compose the CANN FastGeluV2 polynomial from independent Torch ops."""
    eps = torch.as_tensor(1.0e-12, dtype=x.dtype, device=x.device)
    x_eps = x + eps
    sign = torch.clamp(x_eps / torch.abs(x_eps), min=-1.0, max=1.0)

    clipped = torch.clamp(0.7071 * torch.abs(x), max=1.769)
    inner = -0.1444 * (clipped - 1.769) ** 2 + 0.5
    return x * (sign * inner + 0.5)


def _compute(x):
    from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor

    is_torch = isinstance(x, torch.Tensor)
    tensor = x if is_torch else numpy_to_torch_tensor(x)
    output_dtype = tensor.dtype
    compute = (
        tensor.to(torch.float32)
        if output_dtype in (torch.float16, torch.bfloat16)
        else tensor
    )
    result = _torch_fast_gelu_v2(compute).to(output_dtype)
    return result if is_torch else torch_to_numpy_tensor(result.detach().cpu())


def fast_gelu_v2_golden(x, **kwargs):
    del kwargs
    return [_compute(x)]


def _get_torch_input(args, kwargs):
    """从位置参数中取第一个 torch.Tensor, 否则按别名取, 均无则报错"""
    x = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
    if x is not None:
        return x
    for name in ("x", "self", "input", "input_x"):
        x = kwargs.get(name)
        if x is not None:
            return x
    raise ValueError(
        "Missing tensor input; expected positional Tensor or one of: x, self, input, input_x"
    )


class _TorchFastGeluV2:
    """Torch composition of the CANN-specific FastGeluV2 formula."""

    def __call__(instance, *args, **kwargs):
        import torch

        x = _get_torch_input(args, kwargs)
        output_dtype = x.dtype
        compute = (
            x.to(torch.float32)
            if output_dtype in (torch.float16, torch.bfloat16)
            else x
        )
        return [_torch_fast_gelu_v2(compute).to(output_dtype)]


class FastGeluV2KernelSpec:
    golden = staticmethod(fast_gelu_v2_golden)
    third_party = {"torch": _TorchFastGeluV2}
    tolerance = _L0
