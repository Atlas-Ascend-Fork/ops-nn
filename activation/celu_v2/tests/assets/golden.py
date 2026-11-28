#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK golden plugin for the CeluV2 operator.

Registered names (CSV op_name / api_name dialects):
  - "celu_v2" / "CeluV2" : Kernel / GEIR path ("CeluV2" is the GE op type
    name, REG_OP(CeluV2), used by graph-engine case generation)
  - "aclnnCelu" / "aclnnInplaceCelu" : ACLNN path (all aclnn APIs declared
    in op_api/aclnn_celu.h GetWorkspaceSize entries)
  - "torch.nn.functional.celu" : Torch E2E path
  - "torch.celu" : Torch E2E path (alias of "torch.nn.functional.celu",
    same underlying aten op; both names resolve to the same spec)

Import discipline: torch is imported lazily inside function bodies only --
TTK loads this file inside processes where aclrt may already be initialized,
and a module-level framework import risks SIGSEGV.

The Aclnn golden follows aclnnCeluGetWorkspaceSize(self, alpha, out) and
aclnnInplaceCeluGetWorkspaceSize(selfRef, alpha): the out form reshapes the
result to the declared output shape, the inplace form (no out parameter)
returns the freshly computed result. The computation runs at the received
(framework-promoted) precision and the result is returned without casting
back, so the golden side stays the higher-precision reference value.

Tolerance is keyed by OUTPUT dtype (TTK convention). The pass criteria are
the three-way ratio metrics of the "cross_check" comparator (level presets
L0/L1/L2: mare_ratio / mere_ratio / rmse_ratio limits, plus small_value
handling); the numerical accuracy budget per dtype is documented in the
operator's spec.
"""

__spec__ = {
    "celu_v2": "CeluV2KernelSpec",
    "CeluV2": "CeluV2KernelSpec",  # GE op type name (REG_OP(CeluV2))
    "aclnnCelu": "CeluV2AclnnSpec",
    "aclnnInplaceCelu": "CeluV2AclnnSpec",
    "torch.nn.functional.celu": "CeluV2TorchSpec",
    "torch.celu": "CeluV2TorchSpec",
}

_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
}


def aclnn_celu_golden(selfT, alpha=1.0, out=None, **kwargs):
    """CPU golden for aclnnCelu(self, alpha, out) and aclnnInplaceCelu(selfRef, alpha).

    Parameters follow the aclnn_celu.h GetWorkspaceSize entries without
    workspaceSize & executor; the first tensor is the input (self / selfRef).
    Under framework input promotion the tensors arrive at the next higher
    float precision and the result is returned at that received precision
    (no cast back to the native output dtype).
    """
    import torch  # lazy import: TTK process may have aclrt initialized

    result = torch.nn.functional.celu(selfT, alpha=float(alpha))
    if out is not None:
        result = result.reshape(out.shape)
    return [result]


class CeluV2KernelSpec:
    """Kernel / GEIR spec registered by CSV ``op_name`` ("celu_v2" / "CeluV2").

    The golden is the aten OpOverload form "torch.ops.aten.celu": its schema
    (celu(Tensor self, Scalar alpha=1.) -> Tensor) lets the runner forward
    operator attributes by schema matching, so the non-default alpha declared
    on a case reaches the reference. A plain torch function string (e.g.
    "torch.nn.functional.celu") carries no schema and silently computed the
    reference at the default alpha instead.
    """

    golden = "torch.ops.aten.celu"
    third_party = {"torch": "torch.nn.functional.celu"}
    tolerance = _TOLERANCE


class CeluV2AclnnSpec:
    """ACLNN spec registered by CSV ``api_name`` ("aclnnCelu" / "aclnnInplaceCelu")."""

    golden = aclnn_celu_golden
    third_party = {"torch": "torch.nn.functional.celu"}
    tolerance = _TOLERANCE


class CeluV2TorchSpec:
    """Torch E2E spec registered by CSV ``api_name`` ("torch.nn.functional.celu"
    / "torch.celu").

    Graph mode (torch.compile) uses the framework's generic GraphNetwork
    wrapping the registered API.

    Path-domain note: the string golden form is safe here because the e2e
    runner feeds attributes through the API's signature plan; kernel / GEIR
    specs must keep the aten OpOverload form for schema-backed attribute
    passthrough (a plain function string loses non-default attributes there).
    """

    golden = "torch.nn.functional.celu"
    third_party = {"torch": "torch.nn.functional.celu"}
    tolerance = _TOLERANCE
