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

"""SoftShrinkGrad 独立 Golden：Kernel、ACLNN、Torch。

The operator computes the softshrink backward gradient:
``y = gradOutput`` when ``|self| > lambd`` else ``0``, with NaN positions
passing ``gradOutput`` through (aligned with PyTorch ATen semantics).

Reference oracle: ``torch.ops.aten.softshrink_backward``.
"""

import torch

__spec__ = {
    "soft_shrink_grad": "SoftShrinkGradKernelSpec",
    "aclnnSoftshrinkBackward": "AclnnSoftshrinkBackwardSpec",
    "torch.ops.aten.softshrink_backward": "TorchSoftshrinkBackwardSpec",
}


_L2 = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
}


def _aten_softshrink_backward(grad_output, x, lambd=0.5):
    """Call the ATen competitor after adapting TTK's scalar tensor."""

    return torch.ops.aten.softshrink_backward(grad_output, x, float(lambd))


def aclnn_softshrink_backward_golden(
    gradOutput, self, lambd=0.5, gradInput=None, **kwargs
):
    """Parameters follow ``aclnnSoftshrinkBackwardGetWorkspaceSize``."""

    del kwargs
    result = _aten_softshrink_backward(gradOutput, self, lambd)
    if gradInput is not None:
        result = result.reshape(gradInput.shape).to(gradInput.dtype)
    return [result]


class SoftShrinkGradGraph(torch.nn.Module):
    """Graph module for the ATen softshrink backward path."""

    def forward(self, grad_out, x, lambd=0.5):
        return _aten_softshrink_backward(grad_out, x, lambd)


class SoftShrinkGradKernelThirdParty:
    """third_party (Kernel/GEIR): parameter names match soft_shrink_grad def.cpp
    (input_grad / input_x / lambd)."""

    def __call__(self, input_grad, input_x, lambd=0.5, **kwargs):
        del kwargs
        return [_aten_softshrink_backward(input_grad, input_x, lambd)]


class SoftShrinkGradAclnnThirdParty:
    """third_party (ACLNN): parameter names match aclnnSoftshrinkBackward's
    header (gradOutput / self / lambd). ``self`` is declared as an explicit
    parameter: on a bound ``__call__`` the framework binds the pool entry named
    ``self`` by parameter name (aten-style reserved-name handling)."""

    def __call__(instance, gradOutput, self=None, lambd=0.5, **kwargs):
        del kwargs
        return [_aten_softshrink_backward(gradOutput, self, lambd)]


class SoftShrinkGradTorchThirdParty:
    """third_party (E2E): parameter names match the aten schema
    (grad_output / self / lambd); ``self`` is declared as an explicit
    parameter, bound by name like the ACLNN adapter."""

    def __call__(instance, grad_output, self=None, lambd=0.5, **kwargs):
        del kwargs
        return [_aten_softshrink_backward(grad_output, self, lambd)]


class SoftShrinkGradKernelSpec:
    """Shared GEIR/Kernel spec registered by CSV ``op_name``."""

    golden = "torch.ops.aten.softshrink_backward"
    third_party = {"torch": SoftShrinkGradKernelThirdParty}
    tolerance = _L2


class AclnnSoftshrinkBackwardSpec:
    """ACLNN spec registered by CSV ``api_name``."""

    golden = staticmethod(aclnn_softshrink_backward_golden)
    third_party = {"torch": SoftShrinkGradAclnnThirdParty}
    tolerance = _L2


class TorchSoftshrinkBackwardSpec:
    """Shared Torch Single/Graph spec registered by CSV ``api_name``."""

    def golden(grad_out, x, lambd=0.5, **kwargs):
        del kwargs
        return [_aten_softshrink_backward(grad_out, x, lambd)]

    third_party = {"torch": SoftShrinkGradTorchThirdParty}
    torch_graph = SoftShrinkGradGraph
    tolerance = _L2
