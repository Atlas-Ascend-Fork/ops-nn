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
"""Single-operator Torch golden for ConfidenceHeadFused."""

import os
import sys

# Register the custom torch.ops namespace when the extension is installed.
# Kernel/ACLNN testing can load these assets without the optional Torch extension.
try:
    import cann_ops_nn  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence_head_fused_aclnn_golden as _aclnn


class CannOpsNnConfidenceHeadFusedTestSpec:
    compare = _aclnn.AclnnConfidenceHeadFusedTestSpec.compare

    @staticmethod
    def golden(
        hidden,
        weight,
        *,
        markov_emb=None,
        temperature=None,
        bias=None,
        clamp_low=None,
        clamp_high=None,
        **kwargs,
    ):
        return _aclnn.AclnnConfidenceHeadFusedTestSpec.golden(
            hidden,
            weight,
            markovEmbOptional=markov_emb,
            temperatureOptional=temperature,
            biasOptional=bias,
            clampLowOptional=clamp_low,
            clampHighOptional=clamp_high,
        )


__spec__ = {
    "cann_ops_nn.confidence_head_fused": "CannOpsNnConfidenceHeadFusedTestSpec",
    "torch.ops.cann_ops_nn.confidence_head_fused": "CannOpsNnConfidenceHeadFusedTestSpec",
}
