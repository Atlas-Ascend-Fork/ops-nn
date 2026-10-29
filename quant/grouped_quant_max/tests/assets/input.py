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

__input__ = {"kernel": {"grouped_quant_max": "grouped_quant_max_input"}}


def grouped_quant_max_input(x, scale, group_list, **kwargs):
    """
    Input function for grouped_quant_max.
    All the parameters (names and order) follow @grouped_quant_max_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    group_list is a cumulative (cumsum) offset over dim0, so it must be
    monotonically non-decreasing with group_list[-1] == x.shape[0]. Fuzz data
    generated from a random data_range does not satisfy this contract, hence we
    normalize it here (in place) so that both the device side and the golden side
    consume the exact same legal tensor.

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    dim_s = x.shape[0]
    group_list.sort()
    group_list = np.minimum(np.maximum(group_list, 0), dim_s)
    group_list[-1] = dim_s

    return [x, scale, group_list]
