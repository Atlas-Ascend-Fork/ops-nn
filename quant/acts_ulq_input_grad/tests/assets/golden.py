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


# 本算子仅交付图模式（GEIR）+ 算子 Kernel 两条通路（无 aclnn/e2e，见 CMakeLists ACLNNTYPE aclnn_exclude），
# golden 按通路注册：kernel（算子 UT）与 geir（GE 图模式）共用同一 golden 实现。
__golden__ = {
    "kernel": {
        "acts_ulq_input_grad": "acts_ulq_input_grad_golden",
    },
    "geir": {
        "acts_ulq_input_grad": "acts_ulq_input_grad_golden",
    },
}


def acts_ulq_input_grad_golden(
    y_grad,
    clamp_min_mask,
    clamp_max_mask,
    **kwargs,
):
    """
    Kernel golden for acts_ulq_input_grad.
    All the parameters follow @acts_ulq_input_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
                        input_formats, output_formats, input_ori_formats,
                        output_ori_formats, input_dtypes, output_dtypes.
    """
    del kwargs

    y_grad_tensor = torch.from_numpy(y_grad)
    clamp_min_mask_tensor = torch.from_numpy(clamp_min_mask)
    clamp_max_mask_tensor = torch.from_numpy(clamp_max_mask)
    # 对齐需求交付包（LongTailInfo/ActsULQInputGrad/Golden文件）golden 口径：
    # fp16 输入升 fp32 计算后降回原 dtype（round-nearest-even）。
    # 掩码取值 {0,1} 的合同域内，与原始 dtype 直算位等价（y*1=y、y*0=0 精确）。
    orig_dtype = y_grad_tensor.dtype
    if orig_dtype == torch.float16:
        y_grad_tensor = y_grad_tensor.to(torch.float32)
        clamp_min_mask_tensor = clamp_min_mask_tensor.to(torch.float32)
        clamp_max_mask_tensor = clamp_max_mask_tensor.to(torch.float32)
    signal = torch.mul(clamp_min_mask_tensor, clamp_max_mask_tensor)
    x_grad = torch.mul(y_grad_tensor, signal)
    return x_grad.to(orig_dtype).numpy()
