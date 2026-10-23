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
import torch
from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor


__golden__ = {"kernel": {"threshold_grad_v2": "threshold_grad_v2_golden"}}


def threshold_grad_v2_golden(gradients, features, threshold, **kwargs):
    """
    Kernel golden for threshold_grad_v2.
    All the parameters follow @threshold_grad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
             input_formats, output_formats, input_ori_formats, output_ori_formats,
             input_dtypes, output_dtypes.
    """
    del kwargs
    gradients_t = numpy_to_torch_tensor(gradients)
    features_t = numpy_to_torch_tensor(features)
    threshold_t = numpy_to_torch_tensor(threshold)
    gradients_t, features_t, threshold_t = torch.broadcast_tensors(
        gradients_t, features_t, threshold_t
    )
    output_dtype = gradients_t.dtype
    # The kernel promotes every supported dtype to float32 for comparison and
    # selection, then casts the selected gradient back to the output dtype.
    gradients_f32 = gradients_t.to(torch.float32)
    mask = features_t.to(torch.float32) > threshold_t.to(torch.float32)
    result = torch.where(mask, gradients_f32, torch.zeros_like(gradients_f32))
    return torch_to_numpy_tensor(result.to(output_dtype).cpu())
