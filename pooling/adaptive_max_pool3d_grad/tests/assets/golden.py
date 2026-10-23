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
from functools import reduce
import operator
from copy import deepcopy


__golden__ = {
    "aclnn": {
        "aclnnAdaptiveMaxPool3dBackward": "aclnn_adaptive_max_pool3d_backward_golden",
    },
    "kernel": {"adaptive_max_pool3d_grad": "adaptive_max_pool3d_grad_golden"},
}

__input__ = {
    "aclnn": {
        "aclnnAdaptiveMaxPool3dBackward": "aclnn_adaptive_max_pool3d_backward_input"
    },
    "kernel": {"adaptive_max_pool3d_grad": "adaptive_max_pool3d_grad_input"},
}


def adaptive_max_pool3d_grad_input(x, grad, argmax, **kwargs):
    """
    Input function for adaptive_max_pool3d_grad kernel.
    All the parameters (names and order) follow @adaptive_max_pool3d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Following opstest approach: generate random x, run forward adaptive max pool,
    use forward output as grad, forward indices as argmax.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    import torch.nn.functional as F

    x_dtype = x.dtype
    argmax_dtype = argmax.dtype

    # Generate random x data (like opstest: random int8 cast to float)
    input_shape = np.s_[x.shape]
    ele_num = reduce(operator.mul, input_shape)
    random_array = np.random.randint(
        low=np.iinfo(np.int8).min,
        high=np.iinfo(np.int8).max + 1,
        size=(ele_num,),
        dtype=np.int8,
    )
    input_x = random_array.reshape(input_shape).astype(np.float32)
    x_torch = torch.from_numpy(input_x)

    # Output spatial size equals grad's trailing 3 dims (D, H, W)
    output_size = list(grad.shape)[-3:]

    max_out, max_indices = F.adaptive_max_pool3d(
        x_torch, output_size, return_indices=True
    )

    # Convert back to the placeholder dtypes (supports bfloat16 via ml_dtypes)
    input_x = x_torch.numpy().astype(x_dtype)
    max_out = max_out.numpy().astype(x_dtype)
    max_indices = max_indices.numpy().astype(argmax_dtype)

    return [input_x, max_out, max_indices]


def adaptive_max_pool3d_grad_golden(x, grad, argmax, **kwargs):
    """
    Kernel golden for adaptive_max_pool3d_grad.
    All the parameters (names and order) follow @adaptive_max_pool3d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    input_x = deepcopy(x)
    x_dtype = input_x.dtype
    out_grad = deepcopy(grad)
    max_indices = deepcopy(argmax).astype(np.int64)

    # torch cpu 不支持 fp16/bf16, 先升精度到 fp32
    if x_dtype == np.float16 or x_dtype.name == "bfloat16":
        input_x = input_x.astype(np.float32)
        out_grad = out_grad.astype(np.float32)

    backward_out = torch.ops.aten.adaptive_max_pool3d_backward(
        torch.from_numpy(out_grad),
        torch.from_numpy(input_x),
        torch.from_numpy(max_indices),
    ).numpy()

    return backward_out.astype(x_dtype)


def aclnn_adaptive_max_pool3d_backward_input(*args, **kwargs):
    """
    Input function for aclnnAdaptiveMaxPool3dBackward.
    Generate valid gradOutput, self and indices via forward adaptive max pool.
    Tensors: gradOutput(0), self(1), indices(2), gradInput(3)

    Following opstest approach: generate random self, run forward adaptive max pool,
    use forward output as gradOutput, forward indices as indices.
    """
    import torch.nn.functional as F

    grad_output = args[0]
    self_input = args[1]
    indices = args[2]

    # Generate random self data (like opstest: random int8 cast to float)
    self_shape = self_input.shape
    ele_num = reduce(operator.mul, self_shape)
    random_array = np.random.randint(
        low=np.iinfo(np.int8).min,
        high=np.iinfo(np.int8).max + 1,
        size=(ele_num,),
        dtype=np.int8,
    )
    input_x = random_array.reshape(self_shape).astype(np.float32)
    x_torch = torch.from_numpy(input_x)

    # Output spatial size equals gradOutput's trailing 3 dims (D, H, W), supports 4D/5D input
    output_size = list(grad_output.shape)[-3:]

    max_out, max_indices = F.adaptive_max_pool3d(
        x_torch, output_size, return_indices=True
    )

    # Convert back to original dtype
    orig_dtype = self_input.dtype
    if orig_dtype == torch.float16 or orig_dtype == torch.bfloat16:
        max_out = max_out.to(orig_dtype)
        x_torch = x_torch.to(orig_dtype)

    # Convert indices to expected dtype
    indices_dtype = indices.dtype
    max_indices = max_indices.to(indices_dtype)

    # Copy results into existing tensors in-place
    self_input.copy_(x_torch)
    grad_output.copy_(max_out)
    indices.copy_(max_indices)


def aclnn_adaptive_max_pool3d_backward_golden(
    gradOutput, self, indices, gradInput=None, **kwargs
):
    """
    Aclnn golden for aclnnAdaptiveMaxPool3dBackward.
    Parameters follow @aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    grad = gradOutput
    grad_dtype = grad.dtype

    if grad_dtype == torch.float16 or grad_dtype == torch.bfloat16:
        grad = grad.to(torch.float32)
        self = self.to(torch.float32)
    if indices.dtype == torch.int32:
        indices = indices.to(torch.int64)

    backward_out = torch.ops.aten.adaptive_max_pool3d_backward(grad, self, indices)

    if grad_dtype == torch.float16:
        backward_out = backward_out.to(torch.float16)
    elif grad_dtype == torch.bfloat16:
        backward_out = backward_out.to(torch.bfloat16)
    return backward_out
