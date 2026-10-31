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
"""foreach_lerp_list 三方 Golden — 线性插值。"""

__spec__ = {
    "foreach_lerp_list": "ForeachLerpListKernelSpec",
    "aclnnForeachLerpList": "AclnnForeachLerpListSpec",
    "torch._foreach_lerp": "TorchForeachLerpListSpec",
}

import numpy as np
import torch

try:
    import ml_dtypes as _ml_dtypes
except ImportError:
    _ml_dtypes = None

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L1"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _remote_third_party(x1, x2, weight, **kwargs):
    return list(torch._foreach_lerp(x1, x2, weight))


class _TorchForeachLerpListE2EImpl:
    def __call__(self, *args, **kwargs):
        return _remote_third_party(*args, **kwargs)


_THIRD_PARTY = {"torch": _remote_third_party}
_E2E_THIRD_PARTY = {"torch": _TorchForeachLerpListE2EImpl}


def _is_ml_bf16(a):
    return _ml_dtypes is not None and a.dtype == _ml_dtypes.bfloat16


def _np_to_torch(a):
    a = np.ascontiguousarray(a)
    if _is_ml_bf16(a):
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(a)


def _torch_to_np(t):
    if t.dtype == torch.bfloat16:
        if _ml_dtypes is not None:
            return t.float().numpy().astype(_ml_dtypes.bfloat16)
        return t.float().numpy().astype(np.float16)
    return t.cpu().numpy()


class ForeachLerpListKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。公式: y[i]=x1[i]+weight[i]*(x2[i]-x1[i])"""

    def golden(x1, x2, weight, **kwargs):
        x1_list = [_np_to_torch(t) for t in x1]
        x2_list = [_np_to_torch(t) for t in x2]
        w_list = [_np_to_torch(t) for t in weight]
        result = torch._foreach_lerp(x1_list, x2_list, w_list)
        return [_torch_to_np(t) for t in result]

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachLerpListSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: aclnnForeachLerpListGetWorkspaceSize(x1, x2, weight, out, workspaceSize, executor)
    """

    def golden(x1, x2, weight, out=None, **kwargs):
        result = torch._foreach_lerp(x1, x2, weight)
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachLerpListSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x1, x2, weight, **kwargs):
        return list(torch._foreach_lerp(x1, x2, weight))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
