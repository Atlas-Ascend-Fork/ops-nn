#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MultiScaleDeformableAttentionGrad — ATK executor（Ascend910B / arch22）.

用例集: atk_aclnnMultiScaleDeformableAttentionGrad.json

运行:
    atk node --backend npu --devices 0 node --backend cpu task --task accuracy \\
        -c atk_aclnnMultiScaleDeformableAttentionGrad.json \\
        -p executor_aclnnMultiScaleDeformableAttentionGrad.py

aclnn C API 参数顺序 (aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize):
    inputs : value, spatialShape, levelStartIndex, location, attnWeight, gradOutput
    outputs: gradValue, gradLocation, gradAttnWeight

spatialShape / levelStartIndex 是算子的真实输入（int32 tensor），用例 json 只给出形状占位，
真实取值由本 executor 按「每层特征图 h==w、各层均分 num_keys」派生并就地覆写，
CPU 标杆与 NPU 算子使用同一份派生结果。
"""

import math

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


def _derive_shapes(num_keys, num_levels, dtype=torch.int32):
    """按均分 + 正方形特征图派生 spatialShape / levelStartIndex。

    约定: num_keys 能被 num_levels 整除，且每层特征图边长 side = sqrt(num_keys / num_levels) 为整数，
    levelStartIndex[l] = l * (num_keys / num_levels)。
    """
    num_keys = int(num_keys)
    num_levels = int(num_levels)
    if num_levels <= 0 or num_keys % num_levels != 0:
        raise ValueError(
            "num_keys({}) must be divisible by num_levels({})".format(
                num_keys, num_levels
            )
        )
    per_level = num_keys // num_levels
    side = int(round(math.sqrt(per_level)))
    if side * side != per_level:
        raise ValueError(
            "each level must be square, got per-level num_keys={}".format(per_level)
        )
    spatial_shape = torch.tensor([[side, side]] * num_levels, dtype=dtype)
    level_start_index = torch.tensor(
        [level * per_level for level in range(num_levels)], dtype=dtype
    )
    return spatial_shape, level_start_index


def multi_scale_deformable_attention_grad_golden(
    value, spatial_shape, level_start_index, location, attn_weight, grad_output
):
    """CPU 高精度 golden（fp32 累加）。

    gradValue      : 对双线性采样命中的 4 个邻点做散射累加
    gradLocation   : 双线性采样对归一化坐标的偏导，再乘 (W, H)
    gradAttnWeight : sum_c(gradOutput * bilinear(value))
    """
    dtype = value.dtype
    v = value.to(torch.float32)
    loc = location.to(torch.float32)
    aw = attn_weight.to(torch.float32)
    go = grad_output.to(torch.float32)

    ss = spatial_shape.to(torch.int64).cpu().reshape(-1, 2)
    lsi = level_start_index.to(torch.int64).cpu().reshape(-1)

    batch, num_keys, num_heads, channels = v.shape
    num_queries = loc.shape[1]
    num_levels = loc.shape[3]
    num_points = loc.shape[4]
    go = go.reshape(batch, num_queries, num_heads, channels)

    grad_value = torch.zeros_like(v)
    grad_location = torch.zeros_like(loc)
    grad_attn_weight = torch.zeros_like(aw)

    flat_grad_value = grad_value.reshape(batch * num_keys, num_heads, channels)
    b_idx = torch.arange(batch).reshape(batch, 1, 1, 1)
    h_idx = torch.arange(num_heads).reshape(1, 1, num_heads, 1)

    for level in range(num_levels):
        height, width = int(ss[level, 0]), int(ss[level, 1])
        start = int(lsi[level])
        if height <= 0 or width <= 0:
            continue
        level_loc = loc[:, :, :, level]  # [b, nq, nh, np, 2]
        level_aw = aw[:, :, :, level]  # [b, nq, nh, np]
        x = level_loc[..., 0] * width - 0.5
        y = level_loc[..., 1] * height - 0.5
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        ax = x - x0
        ay = y - y0
        x0i = x0.to(torch.int64)
        y0i = y0.to(torch.int64)

        neighbors = (
            ((0, 0), (1 - ay) * (1 - ax)),
            ((0, 1), (1 - ay) * ax),
            ((1, 0), ay * (1 - ax)),
            ((1, 1), ay * ax),
        )

        vagg = torch.zeros(batch, num_queries, num_heads, num_points, channels)
        gathered_cache = {}
        for (dy, dx), weight in neighbors:
            xi = x0i + dx
            yi = y0i + dy
            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            key_idx = (start + yi * width + xi).clamp_(min=0, max=num_keys - 1)
            gathered = v[b_idx, key_idx, h_idx, :] * valid.unsqueeze(-1).to(v.dtype)
            gathered_cache[(dy, dx)] = (gathered, valid, key_idx)
            vagg = vagg + weight.unsqueeze(-1) * gathered

        # gradAttnWeight
        grad_attn_weight[:, :, :, level] = (go.unsqueeze(3) * vagg).sum(dim=-1)

        tilde_g = level_aw.unsqueeze(-1) * go.unsqueeze(3)  # [b, nq, nh, np, c]

        # gradValue: 散射累加（同一 key 多次命中自动累加）
        # flat_grad_value 形如 [batch*num_keys, num_heads, channels]，而 index_add_ 只能索引第 0 维，
        # 故按 head 逐个散射（num_heads 很小，额外开销可忽略）。
        for (dy, dx), weight in neighbors:
            gathered, valid, key_idx = gathered_cache[(dy, dx)]
            contrib = weight.unsqueeze(-1) * tilde_g * valid.unsqueeze(-1)
            flat_idx = (b_idx * num_keys + key_idx).reshape(
                batch, num_queries, num_heads, num_points
            )
            for head in range(num_heads):
                flat_grad_value[:, head, :].index_add_(
                    0,
                    flat_idx[:, :, head, :].reshape(-1),
                    contrib[:, :, head, :, :].reshape(-1, channels),
                )

        # gradLocation
        v00 = gathered_cache[(0, 0)][0]
        v01 = gathered_cache[(0, 1)][0]
        v10 = gathered_cache[(1, 0)][0]
        v11 = gathered_cache[(1, 1)][0]
        d_dx = (1 - ay).unsqueeze(-1) * (v01 - v00) + ay.unsqueeze(-1) * (v11 - v10)
        d_dy = (1 - ax).unsqueeze(-1) * (v10 - v00) + ax.unsqueeze(-1) * (v11 - v01)
        grad_location[:, :, :, level, :, 0] = (tilde_g * d_dx).sum(dim=-1) * width
        grad_location[:, :, :, level, :, 1] = (tilde_g * d_dy).sum(dim=-1) * height

    return (
        grad_value.reshape(value.shape).to(dtype),
        grad_location.reshape(location.shape).to(dtype),
        grad_attn_weight.reshape(attn_weight.shape).to(dtype),
    )


def _prepare_inputs(input_data):
    """就地覆写 spatialShape / levelStartIndex，使其与 value.numKeys 自洽。"""
    value = input_data.kwargs["value"]
    spatial_shape = input_data.kwargs["spatialShape"]
    level_start_index = input_data.kwargs["levelStartIndex"]
    num_levels = input_data.kwargs["location"].shape[3]

    derived_ss, derived_lsi = _derive_shapes(
        value.shape[1], num_levels, dtype=spatial_shape.dtype
    )
    input_data.kwargs["spatialShape"] = derived_ss.to(spatial_shape.device).contiguous()
    input_data.kwargs["levelStartIndex"] = derived_lsi.to(
        level_start_index.device
    ).contiguous()


@register("executor_multi_scale_deformable_attention_grad")
class MultiScaleDeformableAttentionGradApi(BaseApi):
    """CPU 高精度 golden。"""

    def __init__(self, task_result: TaskResult):
        super(MultiScaleDeformableAttentionGradApi, self).__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        _prepare_inputs(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return multi_scale_deformable_attention_grad_golden(
            input_data.kwargs["value"].detach().cpu(),
            input_data.kwargs["spatialShape"].detach().cpu(),
            input_data.kwargs["levelStartIndex"].detach().cpu(),
            input_data.kwargs["location"].detach().cpu(),
            input_data.kwargs["attnWeight"].detach().cpu(),
            input_data.kwargs["gradOutput"].detach().cpu(),
        )


@register("aclnn_multi_scale_deformable_attention_grad")
class AclnnMultiScaleDeformableAttentionGradApi(AclnnBaseApi):
    """NPU aclnn 执行：6 个输入全部必传，3 个输出由框架按算子原型创建。"""

    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        _prepare_inputs(input_data)
        return super().init_by_input_data(input_data)

    def after_call(self, output_packages):
        return super().after_call(output_packages)
