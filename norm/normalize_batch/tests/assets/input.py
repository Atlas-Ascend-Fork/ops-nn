#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK input generator for NormalizeBatch.

TTK 的默认随机生成器独立采样 seq_len，会产出 kernel 语义外的值（0 / 负值 / >d）。
kernel 与 golden 均按 clamp(seq_len, 1, d) 防御，但为保证 ST 输入始终落在合法域
（且 dump / golden / kernel 消费同一份数据），此处就地 clamp 到 [1, d]。
"""

import numpy as np

__input__ = {
    "kernel": {
        "normalize_batch": "normalize_batch_inputs",
    },
}


def normalize_batch_inputs(
    input_x, seq_len, *, normalize_type="per_feature", epsilon=1e-5, **kwargs
):
    """kernel 输入插件：就地钳制 seq_len 到 [1, d]。

    TTK 按输入张量槽回写返回元组（与 dump/golden 共享同一 storage）。
    """
    d = int(input_x.shape[-1])
    np.clip(seq_len, 1, d, out=seq_len)
    return (input_x, seq_len)
