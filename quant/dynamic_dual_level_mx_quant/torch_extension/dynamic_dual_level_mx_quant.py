# -----------------------------------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

# 第一级/第二级 MX 量化块大小，当前仅支持固定值
LEVEL0_BLOCK_SIZE = 512
LEVEL1_BLOCK_SIZE = 32
# level1_scale 每个 E8M0 对齐对覆盖 level1_block_size * 2 个元素
LEVEL1_SCALE_GROUP_SIZE = LEVEL1_BLOCK_SIZE * 2
MX_SCALE_ALIGN = 2
# FLOAT4_E2M1 每字节打包 2 个元素
FP4_PACK_FACTOR = 2
# 支持的量化舍入模式
SUPPORTED_ROUND_MODES = ("rint", "round", "floor")
# 支持的 level1_scale 计算算法：0-OCP，1-ceil 向上取整，2-DynamicDtypeRange
SUPPORTED_SCALE_ALGS = (0, 1, 2)
SCALE_ALG_DYNAMIC = 2
# dst_type_max 仅在 scale_alg=2 时生效，支持 0.0 或 [6.0, 12.0]
DST_TYPE_MAX_MIN = 6.0
DST_TYPE_MAX_MAX = 12.0


def _ceil_div(value: int, factor: int) -> int:
    return (value + factor - 1) // factor


def _y_shape(x: torch.Tensor) -> list:
    # y 为 FLOAT4_E2M1，按每字节 2 个元素打包后用 uint8 承载
    shape = list(x.shape)
    shape[-1] = x.shape[-1] // FP4_PACK_FACTOR
    return shape


def _level0_scale_shape(x: torch.Tensor) -> list:
    shape = list(x.shape)
    shape[-1] = _ceil_div(x.shape[-1], LEVEL0_BLOCK_SIZE)
    return shape


def _level1_scale_shape(x: torch.Tensor) -> list:
    shape = list(x.shape)
    shape[-1] = _ceil_div(x.shape[-1], LEVEL1_SCALE_GROUP_SIZE)
    shape.append(MX_SCALE_ALIGN)
    return shape


def _validate_inputs(
    x: torch.Tensor,
    smooth_scale: Optional[torch.Tensor],
    round_mode: str,
    level0_block_size: int,
    level1_block_size: int,
    scale_alg: int,
    dst_type_max: float,
) -> None:
    torch._check(
        x.dim() >= 1,
        lambda: f"x must be at least 1-dimensional, but got {x.dim()}-d",
    )
    torch._check(
        x.shape[-1] % FP4_PACK_FACTOR == 0,
        lambda: f"x last dim must be even when y is FLOAT4_E2M1, but got {x.shape[-1]}",
    )
    torch._check(
        x.dtype in (torch.float16, torch.bfloat16),
        lambda: f"x dtype must be float16 or bfloat16, but got {x.dtype}",
    )
    torch._check(
        round_mode in SUPPORTED_ROUND_MODES,
        lambda: f"round_mode must be one of {SUPPORTED_ROUND_MODES}, but got {round_mode}",
    )
    torch._check(
        level0_block_size == LEVEL0_BLOCK_SIZE,
        lambda: f"level0_block_size only supports {LEVEL0_BLOCK_SIZE}, but got {level0_block_size}",
    )
    torch._check(
        level1_block_size == LEVEL1_BLOCK_SIZE,
        lambda: f"level1_block_size only supports {LEVEL1_BLOCK_SIZE}, but got {level1_block_size}",
    )
    torch._check(
        scale_alg in SUPPORTED_SCALE_ALGS,
        lambda: f"scale_alg must be one of {SUPPORTED_SCALE_ALGS}, but got {scale_alg}",
    )
    if scale_alg == SCALE_ALG_DYNAMIC:
        torch._check(
            dst_type_max == 0.0
            or (DST_TYPE_MAX_MIN <= dst_type_max <= DST_TYPE_MAX_MAX),
            lambda: (
                f"dst_type_max must be 0.0 or within [{DST_TYPE_MAX_MIN}, "
                f"{DST_TYPE_MAX_MAX}], but got {dst_type_max}"
            ),
        )
    else:
        torch._check(
            dst_type_max == 0.0,
            lambda: (
                f"dst_type_max must be 0.0 when scale_alg is not "
                f"{SCALE_ALG_DYNAMIC}, but got {dst_type_max}"
            ),
        )
    if smooth_scale is not None:
        torch._check(
            smooth_scale.dtype == x.dtype,
            lambda: (
                f"smooth_scale dtype must be the same as x ({x.dtype}), "
                f"but got {smooth_scale.dtype}"
            ),
        )
        torch._check(
            smooth_scale.dim() == 1 and smooth_scale.shape[0] == x.shape[-1],
            lambda: (
                f"smooth_scale must be 1-D with size {x.shape[-1]}, "
                f"but got shape {list(smooth_scale.shape)}"
            ),
        )


class DynamicDualLevelMxQuantOpBuilder(OpBuilder):
    """
    DynamicDualLevelMxQuant 算子的构建器

    基于 aclnnDynamicDualLevelMxQuant/V2 实现：对输入先按 level0_block_size 做一级 MX 量化，
    再按 level1_block_size 做二级量化，输出 FLOAT4_E2M1 量化结果与两级 scale。
    """

    def __init__(self):
        super().__init__("dynamic_dual_level_mx_quant")

    def sources(self):
        return [self.resolve_source("dynamic_dual_level_mx_quant.cpp")]

    def schema(self):
        return (
            "dynamic_dual_level_mx_quant("
            "Tensor x, "
            "Tensor? smooth_scale=None, "
            "*, "
            "str round_mode='rint', "
            "int level0_block_size=512, "
            "int level1_block_size=32, "
            "int scale_alg=0, "
            "float dst_type_max=0.0"
            ") -> (Tensor y, Tensor level0_scale, Tensor level1_scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def dynamic_dual_level_mx_quant_meta(
            x: torch.Tensor,
            smooth_scale: Optional[torch.Tensor] = None,
            *,
            round_mode: str = "rint",
            level0_block_size: int = LEVEL0_BLOCK_SIZE,
            level1_block_size: int = LEVEL1_BLOCK_SIZE,
            scale_alg: int = 0,
            dst_type_max: float = 0.0,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            _validate_inputs(
                x,
                smooth_scale,
                round_mode,
                level0_block_size,
                level1_block_size,
                scale_alg,
                dst_type_max,
            )
            y = x.new_empty(_y_shape(x), dtype=torch.uint8)
            level0_scale = x.new_empty(_level0_scale_shape(x), dtype=torch.float32)
            level1_scale = x.new_empty(
                _level1_scale_shape(x), dtype=torch.float8_e8m0fnu
            )
            return y, level0_scale, level1_scale


builder = DynamicDualLevelMxQuantOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def _dynamic_dual_level_mx_quant_npu(
    x: torch.Tensor,
    smooth_scale: Optional[torch.Tensor] = None,
    *,
    round_mode: str = "rint",
    level0_block_size: int = LEVEL0_BLOCK_SIZE,
    level1_block_size: int = LEVEL1_BLOCK_SIZE,
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = builder.load().dynamic_dual_level_mx_quant(
        x,
        smooth_scale,
        round_mode,
        level0_block_size,
        level1_block_size,
        scale_alg,
        dst_type_max,
    )
    return tuple(output.detach() for output in outputs)


def dynamic_dual_level_mx_quant(
    x: torch.Tensor,
    smooth_scale: Optional[torch.Tensor] = None,
    *,
    round_mode: str = "rint",
    level0_block_size: int = LEVEL0_BLOCK_SIZE,
    level1_block_size: int = LEVEL1_BLOCK_SIZE,
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    NPU 上的 DynamicDualLevelMxQuant —— 目的数据类型为 FLOAT4_E2M1 的 MX 二级量化

    先对输入按 level0_block_size 分块做一级量化，输出 level0_scale 并把中间结果转换回输入
    数据类型；再按 level1_block_size 分块做二级量化，按 scale_alg 指定的算法计算
    level1_scale，并按 round_mode 把元素转换为 FLOAT4_E2M1 输出 y。

    Args:
        x (torch.Tensor): 必选，输入 Tensor。dtype 为 float16/bfloat16，
            shape 为 (..., N)，最后一维必须为偶数；需位于 NPU 上。
        smooth_scale (torch.Tensor, optional): 可选，融合 smooth scale 广播乘，
            dtype 与 x 一致，shape 为 (N,)。不传时不做 smooth scale 运算。
        round_mode (str, optional): 二级量化元素转换的舍入模式，
            取值为 "rint"/"round"/"floor"。默认 "rint"。
        level0_block_size (int, optional): 一级量化块大小，当前仅支持 512。默认 512。
        level1_block_size (int, optional): 二级量化块大小，当前仅支持 32。默认 32。
        scale_alg (int, optional): level1_scale 计算算法，0-OCP，1-ceil 向上取整，
            2-DynamicDtypeRange。默认 0。
        dst_type_max (float, optional): 目标数据类型最大值 Amax(DType)，仅在
            scale_alg=2 时生效；取 0.0 表示使用 FLOAT4_E2M1 的最大值 6，
            或取 [6.0, 12.0] 内的自定义值。默认 0.0。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - y (torch.Tensor): 量化结果，dtype 为 uint8（每字节打包 2 个 FLOAT4_E2M1），
              shape 为 (..., N // 2)。
            - level0_scale (torch.Tensor): 一级量化 scale，dtype 为 float32，
              shape 为 (..., ceil(N / 512))。
            - level1_scale (torch.Tensor): 二级量化 scale，dtype 为 float8_e8m0fnu，
              shape 为 (..., ceil(N / 64), 2)。
    """
    return torch.ops.cann_ops_nn.dynamic_dual_level_mx_quant.default(
        x,
        smooth_scale,
        round_mode=round_mode,
        level0_block_size=level0_block_size,
        level1_block_size=level1_block_size,
        scale_alg=scale_alg,
        dst_type_max=dst_type_max,
    )
