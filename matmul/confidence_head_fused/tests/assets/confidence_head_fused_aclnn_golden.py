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
"""ACLNN-level golden for aclnnConfidenceHeadFused."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import confidence_head_fused_kernel_golden as _kernel
import matmul_golden_util as _util


class AclnnConfidenceHeadFusedTestSpec:
    compare = _kernel.ConfidenceHeadFusedKernelTestSpec.compare

    @staticmethod
    def golden(
        hidden,
        weight,
        markovEmbOptional=None,
        temperatureOptional=None,
        biasOptional=None,
        clampLowOptional=None,
        clampHighOptional=None,
        out=None,
        **kwargs,
    ):
        """Keep Tensor and Scalar positions in the ACLNN C signature order."""
        tensors = tuple(
            (
                tensor
                if tensor is None or isinstance(tensor, np.ndarray)
                else _util.torch_to_numpy(tensor)
            )
            for tensor in (
                hidden,
                weight,
                markovEmbOptional,
                temperatureOptional,
                biasOptional,
            )
        )
        return _kernel._confidence_head_compute(
            *tensors, clamp_low=clampLowOptional, clamp_high=clampHighOptional
        )


__spec__ = {
    "aclnnConfidenceHeadFused": "AclnnConfidenceHeadFusedTestSpec",
}
