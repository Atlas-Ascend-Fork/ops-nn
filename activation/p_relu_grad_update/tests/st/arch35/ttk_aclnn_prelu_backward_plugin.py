# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TTK aclnn golden for aclnnPreluBackward (issue 8246 verification).

Run:
    python3 -m ttk aclnn -i ttk_aclnn_prelu_backward_st.csv \
        --plugin ttk_aclnn_prelu_backward_plugin.py --devices=0 --pc=1 \
        --run=1 --task-prof=false --warmup=false

TTK invokes the golden *unbound* with the API params in C-header order, so the
signature must start with ``gradOutput`` (do NOT add a leading ``self``):
    golden(gradOutput, self, weight, gradInput, gradWeight)

PReLU backward CPU reference:
  gradInput  = gradOutput * (self >= 0 ? 1 : weight[channel])
  gradWeight[channel] = sum(gradOutput * self * (self < 0)) over batch/spatial
For a single-element weight the sum is over all elements. Empty batch (0, C)
yields an empty gradInput and an all-zero gradWeight.

TTK only compares the trailing output slot (gradWeight) for this API, so the
golden returns gradWeight only; gradInput is validated by the aclnn ST/probe.
"""


class AclnnPreluBackwardTestSpec:
    def golden(gradOutput, self_t, weight, gradInput=None, gradWeight=None, **kwargs):
        go = gradOutput.float()
        x = self_t.float()
        w = weight.float()
        neg = x < 0
        if w.numel() == 1:
            gw = (go * x * neg.to(x.dtype)).sum().reshape(w.shape)
        else:
            shape = [1] * x.dim()
            shape[1] = w.numel()
            prod = go * x * neg.to(x.dtype)
            dims = tuple(d for d in range(x.dim()) if d != 1)
            gw = prod.sum(dim=dims).reshape(w.shape)
        gw_dt = gradWeight.dtype if gradWeight is not None else weight.dtype
        return [gw.to(gw_dt)]


__spec__ = {"aclnnPreluBackward": "AclnnPreluBackwardTestSpec"}
