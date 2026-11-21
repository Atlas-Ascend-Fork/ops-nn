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

__golden__ = {"kernel": {"linear_index_v2": "linear_index_v2_golden"}}


def linear_index_v2_golden(indices_list, stride, value_size, **kwargs):
    """
    Golden function for linear_index_v2.
    All the parameters (names and order) follow @linear_index_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    indices_dtype = indices_list[0].dtype
    output = np.zeros(indices_list[0].shape, dtype=np.int64)

    for i, indices in enumerate(indices_list):
        stride_val = int(stride[i])
        size_val = int(value_size[i])
        a = indices.astype(np.int64)
        if size_val == 0:
            remain = a
        else:
            remain = a - np.floor_divide(a, size_val) * size_val
        output = output + remain * stride_val

    return output.astype(indices_dtype, copy=False)
