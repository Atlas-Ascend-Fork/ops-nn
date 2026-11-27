# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Ascend950 eager binding and Meta inference for RotateDualLevelQuant."""

from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

_FP4 = 296
_LEVEL0_BLOCK_SIZE = 512
_LEVEL1_BLOCK_SIZE = 32
_SCALE_PAIR_SIZE = 64
_MAX_DIMENSION = 2**31 - 1
_MX_SCALE_DTYPE = getattr(torch, "float8_e8m0fnu", None)
_OUTPUT_DTYPES = {
    _FP4: torch.uint8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
    291: torch.float8_e5m2,
    292: torch.float8_e4m3fn,
}


def _normalize_options(
    dst_dtype, axis, round_mode, scale_alg, dst_type_max, transpose_y
):
    dtype = {torch.float8_e5m2: 23, torch.float8_e4m3fn: 24}.get(dst_dtype, dst_dtype)
    return (
        _FP4 if dtype is None else dtype,
        -1 if axis is None else axis,
        "rint" if round_mode is None else round_mode,
        0 if scale_alg is None else scale_alg,
        0.0 if dst_type_max is None else dst_type_max,
        False if transpose_y is None else transpose_y,
    )


def _check_params(
    x,
    rotation,
    alpha,
    dst_dtype,
    axis,
    round_mode,
    scale_alg,
    dst_type_max,
    transpose_y,
    group_list,
    group_list_type,
    level0_block_size,
    level1_block_size,
):
    torch._check(1 <= x.dim() <= 7, lambda: "x rank must be in [1,7]")
    torch._check(
        x.dtype in (torch.float16, torch.bfloat16), lambda: "x must be FP16/BF16"
    )
    torch._check(rotation.dtype == x.dtype, lambda: "rotation dtype must match x")
    torch._check(rotation.device == x.device, lambda: "rotation must be on x.device")
    torch._check(
        dst_dtype in _OUTPUT_DTYPES, lambda: "dst_dtype must be MXFP4 or MXFP8"
    )
    torch._check(
        _MX_SCALE_DTYPE is not None, lambda: "torch.float8_e8m0fnu is required"
    )
    torch._check(
        axis in (-1, x.dim() - 1), lambda: "axis must select the last dimension"
    )
    torch._check(not transpose_y, lambda: "transpose_y must be False")
    torch._check(
        level0_block_size == _LEVEL0_BLOCK_SIZE, lambda: "level0_block_size must be 512"
    )
    torch._check(
        level1_block_size == _LEVEL1_BLOCK_SIZE, lambda: "level1_block_size must be 32"
    )
    torch._check(group_list_type in (0, 1), lambda: "group_list_type must be 0/1")
    torch._check(round_mode in ("rint", "round", "floor"), lambda: "invalid round_mode")
    fp4 = dst_dtype == _FP4
    torch._check(
        scale_alg in ((0, 2) if fp4 else (0, 1)),
        lambda: "invalid scale_alg for dst_dtype",
    )
    torch._check(fp4 or round_mode == "rint", lambda: "FP8 only supports rint")
    torch._check(
        6.0 <= dst_type_max <= 12.0 if scale_alg == 2 else dst_type_max == 0.0,
        lambda: "dst_type_max must be finite in [6,12] for scale_alg=2, otherwise 0",
    )
    if alpha is not None:
        torch._check(alpha.device == x.device, lambda: "alpha must be on x.device")
        torch._check(
            alpha.dtype == torch.bfloat16 and alpha.dim() == 1 and alpha.shape[0] == 1,
            lambda: "alpha must be BF16 [1]",
        )
    grouped = group_list is not None
    torch._check(
        rotation.dim() in ((3, 4) if grouped else (2, 3)),
        lambda: "rotation rank must be 2/3 in ordinary mode or 3/4 in grouped mode",
    )
    n, k = x.shape[-1], rotation.shape[-1]
    torch._check(k in (32, 64, 128, 256), lambda: "rotation K must be 32/64/128/256")
    torch._check(rotation.shape[-2] == k, lambda: "rotation must be square")
    torch._check(n % k == 0, lambda: "N must be divisible by K")
    if rotation.dim() == (4 if grouped else 3):
        torch._check(
            rotation.shape[-3] == n // k,
            lambda: "rotation column-block count must equal N/K",
        )
    if grouped:
        torch._check(x.dim() == 2 and n > 0, lambda: "grouped x must be [M,N] with N>0")
        torch._check(
            group_list.device == x.device, lambda: "group_list must be on x.device"
        )
        torch._check(
            group_list.dtype == torch.int64 and group_list.dim() == 1,
            lambda: "group_list must be INT64 [E]",
        )
        torch._check(1 <= group_list.shape[0] <= 1024, lambda: "E must be in [1,1024]")
        torch._check(
            rotation.shape[0] == group_list.shape[0],
            lambda: "rotation E must match group_list",
        )
    rows = 1
    for dim in x.shape:
        torch._check(dim <= _MAX_DIMENSION, lambda: "x dimensions must fit INT32")
    for dim in x.shape[:-1]:
        rows *= dim
    torch._check(rows <= _MAX_DIMENSION, lambda: "flattened M must fit INT32")


class RotateDualLevelQuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("rotate_dual_level_quant")

    def sources(self) -> list:
        return [self.resolve_source("rotate_dual_level_quant.cpp")]

    def schema(self) -> str:
        return (
            "rotate_dual_level_quant(Tensor x, Tensor rotation, *, Tensor? alpha=None, "
            "int? dst_dtype=None, int? axis=-1, str? round_mode='rint', "
            "int? scale_alg=0, float? dst_type_max=0.0, bool? transpose_y=False, "
            "Tensor? group_list=None, int group_list_type=0, "
            "int level0_block_size=512, int level1_block_size=32) "
            "-> (Tensor y, Tensor level0_scale, Tensor level1_scale)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def rotate_dual_level_quant_meta(
            x: torch.Tensor,
            rotation: torch.Tensor,
            *,
            alpha: Optional[torch.Tensor] = None,
            dst_dtype: Optional[int] = None,
            axis: Optional[int] = -1,
            round_mode: Optional[str] = "rint",
            scale_alg: Optional[int] = 0,
            dst_type_max: Optional[float] = 0.0,
            transpose_y: Optional[bool] = False,
            group_list: Optional[torch.Tensor] = None,
            group_list_type: int = 0,
            level0_block_size: int = 512,
            level1_block_size: int = 32,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            options = _normalize_options(
                dst_dtype, axis, round_mode, scale_alg, dst_type_max, transpose_y
            )
            _check_params(
                x,
                rotation,
                alpha,
                *options,
                group_list,
                group_list_type,
                level0_block_size,
                level1_block_size,
            )
            n = x.shape[-1]
            y_shape = list(x.shape)
            if options[0] == _FP4:
                y_shape[-1] //= 2
            return (
                x.new_empty(y_shape, dtype=_OUTPUT_DTYPES[options[0]]),
                x.new_empty(
                    list(x.shape[:-1]) + [(n + 511) // 512], dtype=torch.float32
                ),
                x.new_empty(
                    list(x.shape[:-1]) + [(n + 63) // 64, 2], dtype=_MX_SCALE_DTYPE
                ),
            )


rotate_dual_level_quant_builder = RotateDualLevelQuantOpBuilder()
rotate_dual_level_quant_builder._ensure_initialized()


@impl(get_as_library(), rotate_dual_level_quant_builder.name, "PrivateUse1")
def rotate_dual_level_quant(
    x: torch.Tensor,
    rotation: torch.Tensor,
    *,
    alpha: Optional[torch.Tensor] = None,
    dst_dtype: Optional[int] = None,
    axis: Optional[int] = -1,
    round_mode: Optional[str] = "rint",
    scale_alg: Optional[int] = 0,
    dst_type_max: Optional[float] = 0.0,
    transpose_y: Optional[bool] = False,
    group_list: Optional[torch.Tensor] = None,
    group_list_type: int = 0,
    level0_block_size: int = 512,
    level1_block_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate and quantize on Ascend950; eager execution with Meta inference.

    x/rotation are matching FP16/BF16 tensors. Ordinary x has rank 1..7;
    grouped x is [M,N]. Rotation is [K,K]/[N/K,K,K], or grouped
    [E,K,K]/[E,N/K,K,K], with K=32/64/128/256. alpha is optional BF16 [1].
    dst_dtype defaults to FP4 (296), accepts torch_npu.float4_e2m1fn_x2 or
    FP8 scalar-type IDs/dtypes. FP4 supports rint/round/floor and scale_alg 0/2;
    FP8 supports rint and scale_alg 0/1. dst_type_max only affects level1.
    group_list is optional INT64 [E]; type 0=cumsum, 1=count, E in [1,1024].
    Empty groups and prefix coverage are allowed; device values must be valid.
    Both block sizes are fixed at 512/32, axis is last, transpose_y=False.

    Returns (y, level0_scale, level1_scale) on x.device. FP4 y is packed
    uint8 [...,N/2]; FP8 y has x.shape and its native dtype. level0_scale is
    FP32 [...,ceil(N/512)], level1_scale is float8_e8m0fnu [...,ceil(N/64),2].
    Uncovered grouped rows in all three newly allocated outputs are unspecified.
    """
    options = _normalize_options(
        dst_dtype, axis, round_mode, scale_alg, dst_type_max, transpose_y
    )
    _check_params(
        x,
        rotation,
        alpha,
        *options,
        group_list,
        group_list_type,
        level0_block_size,
        level1_block_size,
    )
    return rotate_dual_level_quant_builder.load().rotate_dual_level_quant(
        x,
        rotation,
        alpha,
        *options,
        group_list,
        group_list_type,
        level0_block_size,
        level1_block_size,
    )
