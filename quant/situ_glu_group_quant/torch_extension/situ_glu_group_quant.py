# -----------------------------------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

# Situ激活gate/up拆分因子：x最后一维为2H，激活输出最后一维为H（gate=前半，recipes 固定方向）
SPLIT_FACTOR = 2
# MX量化块大小：每32个元素共享一个scale
MX_BLOCK_SIZE = 32
# y_scale最后一维对齐数：每(align_num * block_size)=64个元素共享一组scale
SCALE_ALIGN_NUM = 2


class SituGluGroupQuantOpBuilder(OpBuilder):
    """
    SituGluGroupQuant 算子的构建器

    基于 aclnnSituGluGroupQuant API 实现（独立移植自 cann-recipes-infer
    SituGluGroupQuant，非合并版）。稀疏 SiTU 门控激活 + MXFP8(e4m3fn,
    group_size=32) 尾轴动态量化融合，面向 MoE 分组场景：
    group_index 逐元素累加得到有效行数 validRows，仅处理前 validRows 行，
    padding 行不写出。与 npu_situ_and_mul_sparse -> npu_dynamic_mx_quant
    (dst=e4m3fn) 串联逐位一致（有效行内）。
    """

    def __init__(self):
        super().__init__("situ_glu_group_quant")

    def sources(self):
        return [self.resolve_source("situ_glu_group_quant.cpp")]

    def schema(self):
        return (
            "situ_glu_group_quant("
            "Tensor x, "
            "Tensor? weight=None, Tensor? group_index=None, "
            "*, "
            "float beta=1.0, "
            "float linear_beta=0.0, "
            "bool activate_left=False, "
            "int axis=-1, "
            "int quant_mode=0, "
            "int dst_type=36, "
            "float dst_type_max=0.0, "
            "str round_mode='rint', "
            "int group_list_type=0"
            ") -> (Tensor y, Tensor y_scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def situ_glu_group_quant_meta(
            x: torch.Tensor,
            weight: torch.Tensor = None,
            group_index: torch.Tensor = None,
            *,
            beta: float = 1.0,
            linear_beta: float = 0.0,
            activate_left: bool = False,
            axis: int = -1,
            quant_mode: int = 0,
            dst_type: int = 36,
            dst_type_max: float = 0.0,
            round_mode: str = "rint",
            group_list_type: int = 0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            torch._check(
                x.dim() >= 1,
                lambda: f"x must be at least 1-dimensional, but got {x.dim()}-d",
            )
            torch._check(
                x.shape[-1] > 0 and x.shape[-1] % SPLIT_FACTOR == 0,
                lambda: f"x last dim must be positive and even, but got {x.shape[-1]}",
            )
            epr = x.shape[-1] // SPLIT_FACTOR
            torch._check(
                x.dtype in (torch.float16, torch.bfloat16),
                lambda: f"x dtype must be float16 or bfloat16, but got {x.dtype}",
            )
            if group_index is not None:
                torch._check(
                    group_index.dtype == torch.int64,
                    lambda: f"group_index dtype must be int64, but got {group_index.dtype}",
                )
            if weight is not None:
                torch._check(
                    weight.dtype == torch.float32,
                    lambda: f"weight dtype must be float32, but got {weight.dtype}",
                )
            torch._check(
                beta > 0.0,
                lambda: f"beta must be greater than 0, but got {beta}",
            )
            torch._check(
                axis == -1,
                lambda: f"axis only supports -1 currently, but got {axis}",
            )
            torch._check(
                quant_mode == 0,
                lambda: f"quant_mode only supports 0 (mx quantization) currently, but got {quant_mode}",
            )
            torch._check(
                dst_type == 36,
                lambda: f"dst_type only supports 36 (FP8_E4M3FN) currently, but got {dst_type}",
            )
            torch._check(
                dst_type_max >= 0.0,
                lambda: f"dst_type_max must be >= 0, but got {dst_type_max}",
            )
            torch._check(
                round_mode == "rint",
                lambda: f'round_mode only supports "rint" currently, but got "{round_mode}"',
            )
            torch._check(
                group_list_type == 1,
                lambda: "group_list_type=0 (cumsum mode) is reserved and not supported yet, "
                f"pass 1 (count mode) instead, but got {group_list_type}",
            )

            y_shape = list(x.shape)
            y_shape[-1] = epr
            scale_group_size = SCALE_ALIGN_NUM * MX_BLOCK_SIZE
            scale_num = (epr + scale_group_size - 1) // scale_group_size
            y_scale_shape = y_shape[:-1] + [scale_num, SCALE_ALIGN_NUM]

            y = torch.empty(y_shape, dtype=torch.float8_e4m3fn, device="meta")
            y_scale = torch.empty(
                y_scale_shape, dtype=torch.float8_e8m0fnu, device="meta"
            )
            return y, y_scale


builder = SituGluGroupQuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def situ_glu_group_quant(
    x: torch.Tensor,
    weight: torch.Tensor = None,
    group_index: torch.Tensor = None,
    *,
    beta: float = 1.0,
    linear_beta: float = 0.0,
    activate_left: bool = False,
    axis: int = -1,
    quant_mode: int = 0,
    dst_type: int = 36,
    dst_type_max: float = 0.0,
    round_mode: str = "rint",
    group_list_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NPU 上的 SituGluGroupQuant —— 稀疏 SiTU 激活 + MXFP8 量化融合算子
    （独立移植自 cann-recipes-infer，非合并版）

    对输入 x 先做 SiTU 激活（gate=前半、up=后半；gate 支路
    beta*tanh(g/beta)*sigmoid(g)，up 支路 linear_beta*tanh(u/linear_beta)，
    linear_beta<=0 时 up 直通不做 tanh 软限幅），
    再按 MX 算法做 FP8(E4M3FN, group=32) 动态量化，返回量化结果 y 与
    E8M0 格式的 scale 张量 y_scale。仅处理 sum(group_index) 覆盖的
    有效行（超出部分为 padding，不写出）。

    Args:
        x (torch.Tensor): 必选，输入Tensor。数据类型为 float16/bfloat16，
            shape 为 (..., 2H)，最后一维必须为正偶数，EPR=H 不可超过 3072；
            必须是NPU上的Tensor，数据格式ND（空Tensor输出为空）。
        group_index (torch.Tensor): 可选，int64 Tensor，非空，第 i 个元素为
            专家 i 分到的 token 行数；kernel 逐元素累加得到有效行数，并按
            min(sum, totalRows) 钳制；不传则全行模式。
        weight (torch.Tensor): 可选，float32 Tensor，逐 token 缩放因子，
            元素个数与总行数一致；不传则不缩放。
        beta (float, optional): SiTU 激活 gate 支路的 beta 参数，必须大于0。默认1.0。
        linear_beta (float, optional): SiTU 激活 up 支路软限幅参数
            （up = linear_beta*tanh(up/linear_beta)）。<=0 时禁用软限幅（up 直通），
            >0 时启用（up 输出幅度钳位在 ±linear_beta 内）。默认 0.0（禁用）。
        activate_left (bool, optional): gate/up 方向选择，默认 False
            （gate 取后半，situ normal 方向）。
        axis (int, optional): 量化轴，当前仅支持 -1（尾轴）。默认 -1。
        quant_mode (int, optional): 量化模式，当前仅支持 0（MX 量化）。默认 0。
        dst_type (int, optional): 目标量化数据类型 35/36/40/41，
            当前仅支持 36（FLOAT8_E4M3FN）。默认 36。
        dst_type_max (float, optional): 目标数据类型数值上限，取值 >= 0。默认 0。
        round_mode (str, optional): 舍入模式 rint/round/floor，当前仅支持 rint。默认 "rint"。
        group_list_type (int, optional): group_index 语义：0=cumsum 模式（预留未实现，会被拒绝），
            1=count 模式（当前语义）。默认 0，当前必须显式传 1。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - y (torch.Tensor): 量化结果，shape 为 x.shape[:-1]+[H]，dtype 为 FLOAT8_E4M3FN。
            - y_scale (torch.Tensor): MX量化的scale，dtype 为 FLOAT8_E8M0，
              shape 为 x.shape[:-1]+[ceil(ceil(H/32)/2), 2]。
    """
    op_module = builder.load()
    return op_module.situ_glu_group_quant(
        x,
        weight,
        group_index,
        beta,
        linear_beta,
        activate_left,
        axis,
        quant_mode,
        dst_type,
        dst_type_max,
        round_mode,
        group_list_type,
    )
