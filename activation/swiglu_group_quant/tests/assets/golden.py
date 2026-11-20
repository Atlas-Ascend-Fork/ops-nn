# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU reference for the Torch-supported quantization modes 0, 1, 2, 3 and 5."""

import numpy as np

from ttk.utilities.dtypes import (
    numpy_to_torch_tensor,
    resolve_numpy_dtype,
    torch_to_numpy_tensor,
)


__golden__ = {
    "aclnn": {
        "aclnnSwigluGroupQuant": "aclnn_swiglu_group_quant_golden",
        "aclnnSwigluGroupQuantV3": "aclnn_swiglu_group_quant_v3_golden",
    },
    "kernel": {
        "swiglu_group_quant": "swiglu_group_quant_golden",
    },
}

__input__ = {
    "aclnn": {
        "aclnnSwigluGroupQuant": "aclnn_swiglu_group_quant_input",
        "aclnnSwigluGroupQuantV3": "aclnn_swiglu_group_quant_v3_input",
    },
    "kernel": {
        "swiglu_group_quant": "swiglu_group_quant_input",
    },
}


DT_FLOAT8_E5M2 = 35
DT_FLOAT8_E4M3FN = 36
DT_FLOAT4_E2M1 = 40
DT_FLOAT4_E1M2 = 41
TORCH_NPU_DTYPE_OFFSET = 256
DEFAULT_CLAMP_LIMIT = -1.0
GROUP_LIST_TYPE_COUNT = 1
GROUP_LIST_TYPE_PAIR = 2

FP8_MAX_VALUE = {
    DT_FLOAT8_E5M2: np.float32(57344.0),
    DT_FLOAT8_E4M3FN: np.float32(448.0),
}
INV_FP8_MAX_VALUE = {
    DT_FLOAT8_E5M2: np.array(0x37924925, dtype=np.uint32).view(np.float32),
    DT_FLOAT8_E4M3FN: np.array(0x3B124925, dtype=np.uint32).view(np.float32),
}
EPS = np.float32(1e-4)
PER_BLOCK_FP16 = 128
PER_MX_FP16 = 32
MX_SCALE_ALIGN = 2
E1M2_STEP = 0.25
E1M2_MAX_CODE = 7


def _to_numpy(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach"):
        return torch_to_numpy_tensor(tensor.detach().cpu())
    if hasattr(tensor, "cpu"):
        return tensor.cpu().numpy()
    return np.asarray(tensor)


def _write_tensor(tensor, value):
    if tensor is None:
        return
    if isinstance(tensor, np.ndarray):
        tensor[...] = value.astype(tensor.dtype, copy=False)
        return
    import torch

    src = torch.as_tensor(value, device=tensor.device)
    if src.dtype != tensor.dtype:
        src = src.to(tensor.dtype)
    tensor.copy_(src.reshape(tensor.shape))


def _dtype_to_int(dst_type):
    if isinstance(dst_type, (int, np.integer)):
        if dst_type >= TORCH_NPU_DTYPE_OFFSET:
            return dst_type - TORCH_NPU_DTYPE_OFFSET
        return {23: DT_FLOAT8_E5M2, 24: DT_FLOAT8_E4M3FN}.get(dst_type, dst_type)
    text = str(dst_type)
    if text.endswith("float8_e5m2"):
        return DT_FLOAT8_E5M2
    if text.endswith("float8_e4m3fn"):
        return DT_FLOAT8_E4M3FN
    if "float4_e2m1" in text:
        return DT_FLOAT4_E2M1
    if "float4_e1m2" in text:
        return DT_FLOAT4_E1M2
    raise ValueError(f"Unsupported dst_type: {dst_type}")


def _fp8_dtype(dst_type):
    if dst_type == DT_FLOAT8_E5M2:
        return resolve_numpy_dtype("float8_e5m2")
    if dst_type == DT_FLOAT8_E4M3FN:
        return resolve_numpy_dtype("float8_e4m3fn")
    raise ValueError(f"Unsupported fp8 dst_type: {dst_type}")


def _fp8_info(dst_type):
    return _fp8_dtype(dst_type), FP8_MAX_VALUE[dst_type], INV_FP8_MAX_VALUE[dst_type]


def _fp4_dtype(dst_type):
    if dst_type == DT_FLOAT4_E2M1:
        return resolve_numpy_dtype("float4_e2m1")
    if dst_type == DT_FLOAT4_E1M2:
        return resolve_numpy_dtype("float4_e1m2")
    raise ValueError(f"Unsupported fp4 dst_type: {dst_type}")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def _ceil_div(x, y):
    return (x + y - 1) // y


def _use_clamp(clamp_limit):
    return clamp_limit is not None and float(clamp_limit) != DEFAULT_CLAMP_LIMIT


def _compute_swiglu(
    x, weight=None, clamp_limit=None, *, mx_v2=False, alpha=1.0, bias=0.0
):
    x = _to_numpy(x)
    input_dtype = x.dtype
    x = x.astype(np.float32)
    orig_shape = x.shape
    x = x.reshape(_token_num(x), orig_shape[-1])
    hidden = orig_shape[-1] // 2
    x0 = x[:, :hidden]
    x1 = x[:, hidden:]

    if _use_clamp(clamp_limit):
        limit = float(clamp_limit)
        x0 = np.minimum(x0, limit)
        x1 = np.minimum(limit, np.maximum(x1, -limit))

    if mx_v2:
        gate = x0 * _sigmoid(np.float32(alpha) * x0)
        y_origin = (gate * (x1 + np.float32(bias))).astype(input_dtype)
    else:
        y_origin = _silu(x0) * x1
    y_weighted = y_origin
    weight = _to_numpy(weight)
    if weight is not None:
        y_weighted = y_origin * weight.reshape(-1, 1).astype(np.float32)
        if mx_v2:
            y_weighted = y_weighted.astype(input_dtype)
    y_origin = y_origin.reshape(*orig_shape[:-1], hidden)
    y_weighted = y_weighted.reshape(*orig_shape[:-1], hidden)
    return y_origin, y_weighted


def _token_num(x):
    x_np = _to_numpy(x)
    return int(np.prod(x_np.shape[:-1]))


def _group_counts(group_index, group_list_type, quant_mode):
    if group_list_type not in (GROUP_LIST_TYPE_COUNT, GROUP_LIST_TYPE_PAIR):
        raise ValueError(f"group_list_type must be 1 or 2, got {group_list_type}")
    if quant_mode == 5 and (
        group_list_type != GROUP_LIST_TYPE_COUNT or group_index is not None
    ):
        raise ValueError("quant_mode=5 requires group_list_type=1 and group_index=None")
    group_index = _to_numpy(group_index)
    if group_index is None:
        if group_list_type == GROUP_LIST_TYPE_PAIR:
            raise ValueError("group_index is required when group_list_type=2")
        return None
    if group_list_type == GROUP_LIST_TYPE_PAIR:
        if group_index.ndim != 2 or group_index.shape[1] != 2:
            raise ValueError(
                "group_index must have shape [E, 2] when group_list_type=2"
            )
        if group_index.shape[0] == 0:
            raise ValueError("group_index must not be empty when group_list_type=2")
        if group_index.dtype != np.int64:
            raise ValueError("group_index must be int64 when group_list_type=2")
        return group_index[:, 1]
    if group_index.ndim != 1:
        raise ValueError("group_index must be 1D when group_list_type=1")
    return group_index


def _merge_output_buffer(value, output_tensor, token_num, real_bs):
    if value is None:
        return None
    if output_tensor is not None:
        merged = _to_numpy(output_tensor).copy()
        if real_bs > 0:
            merged.reshape(token_num, -1)[:real_bs] = value.reshape(token_num, -1)[
                :real_bs
            ]
        return merged
    if real_bs < token_num:
        value = value.copy()
        value.reshape(token_num, -1)[real_bs:] = 0
    return value


def _i32_shl23_as_f32(value):
    value_i64 = np.asarray(value, dtype=np.int64)
    return ((value_i64 << 23) & 0xFFFFFFFF).astype(np.uint32).view(np.float32)


def _round_pow2(scale):
    scale = np.asarray(scale, dtype=np.float32)
    scale_bits = scale.view(np.uint32)
    exp = ((scale_bits >> 23) & 0xFF).astype(np.int32)
    mantissa = scale_bits & ((1 << 23) - 1)
    exp_scale = exp - 127 + (mantissa != 0)
    y_scale_e8m0 = (exp_scale + 127).astype(np.uint8)
    pos_scale = _i32_shl23_as_f32(exp_scale + 127)
    inv_scale = _i32_shl23_as_f32(127 - exp_scale)
    return y_scale_e8m0, inv_scale.astype(np.float32), pos_scale.astype(np.float32)


def _amax_block_fp8(y_block):
    finite = np.isfinite(y_block)
    abs_finite = np.where(finite, np.abs(y_block), np.float32(0.0))
    return np.maximum(abs_finite.max(axis=-1).astype(np.float32), EPS)


def _amax_mx_fp8(y_block):
    return np.maximum(
        np.max(np.abs(y_block.astype(np.float32)), axis=-1).astype(np.float32), EPS
    )


def _quantize_fp8(y_origin, dst_type, quant_mode, round_scale):
    if quant_mode in (1, 5) and not round_scale:
        raise ValueError(f"quant_mode={quant_mode} only supports round_scale=True")

    fp8_type, fp8_max, inv_fp8_max = _fp8_info(dst_type)
    block_size = PER_BLOCK_FP16 if quant_mode == 0 else PER_MX_FP16
    y_shape = y_origin.shape
    hidden = y_shape[-1]
    num_tokens = _token_num(y_origin)
    y_2d = y_origin.reshape(num_tokens, hidden).astype(np.float32)
    num_blocks = _ceil_div(hidden, block_size)
    pad_hidden = num_blocks * block_size
    y_pad = np.pad(y_2d, ((0, 0), (0, pad_hidden - hidden)), mode="constant")
    y_block = y_pad.reshape(num_tokens, num_blocks, block_size)

    if quant_mode == 0:
        amax = _amax_block_fp8(y_block)
    elif quant_mode == 5:
        amax = np.max(np.abs(y_block), axis=-1)
    else:
        amax = _amax_mx_fp8(y_block)
    if round_scale:
        y_scale_e8m0, inv_scale, pos_scale = _round_pow2(amax * inv_fp8_max)
        y_scale = pos_scale if quant_mode == 0 else y_scale_e8m0
        if quant_mode == 5:
            inv_scale[amax == 0] = 0
    else:
        y_scale = (amax * inv_fp8_max).astype(np.float32)
        inv_scale = (fp8_max / amax).astype(np.float32)

    y_scaled = y_block * inv_scale.reshape(num_tokens, num_blocks, 1)
    if quant_mode == 0:
        y_scaled = np.where(np.isfinite(y_scaled), y_scaled, y_block)
    y_fp8 = y_scaled.reshape(num_tokens, pad_hidden)[:, :hidden].astype(fp8_type)

    if quant_mode == 0:
        scale = y_scale.astype(np.float32).reshape(*y_shape[:-1], num_blocks)
    else:
        if num_blocks % 2 != 0:
            y_scale = np.pad(y_scale, ((0, 0), (0, 1)), mode="constant")
        scale = y_scale.view(resolve_numpy_dtype("float8_e8m0")).reshape(
            *y_shape[:-1], _ceil_div(num_blocks, MX_SCALE_ALIGN), 2
        )
    return y_fp8.reshape(y_shape), scale


def _float32_to_bf16_bits(value):
    return value.astype(resolve_numpy_dtype("bfloat16")).view(np.uint16)


def _quantize_e1m2_nibble(y_scaled):
    mag = np.rint(np.abs(y_scaled) / E1M2_STEP).astype(np.int32)
    mag = np.clip(mag, 0, E1M2_MAX_CODE).astype(np.uint8)
    return np.where(y_scaled < 0, mag | 0x8, mag).astype(np.uint8)


def _quantize_fp4(y_origin, dst_type, pack=True):
    block_size = 32
    y_shape = y_origin.shape
    hidden = y_shape[-1]
    num_tokens = _token_num(y_origin)
    y_2d = y_origin.reshape(num_tokens, hidden).astype(np.float32)
    if hidden % block_size != 0:
        raise ValueError("MxFp4 golden only supports hidden divisible by 32")

    num_blocks = hidden // block_size
    y_block = y_2d.reshape(num_tokens, num_blocks, block_size)

    fp4_emax_exp = 2 if dst_type == DT_FLOAT4_E2M1 else 0
    amax = np.max(np.abs(y_block), axis=-1)
    e_amax = ((_float32_to_bf16_bits(amax) >> 7) & 0xFF).astype(np.int32)
    e8m0 = np.maximum(e_amax - fp4_emax_exp, 0).astype(np.uint8)
    inv_scale = np.exp2(127.0 - e8m0.astype(np.float32)).astype(np.float32)
    y_scaled = y_block * inv_scale.reshape(num_tokens, num_blocks, 1)

    if dst_type == DT_FLOAT4_E2M1:
        fp4_values = (
            y_scaled.reshape(num_tokens, hidden)
            .astype(np.float32)
            .astype(_fp4_dtype(dst_type))
        )
        nibble = fp4_values.view(np.uint8).reshape(num_tokens, hidden)
    elif dst_type == DT_FLOAT4_E1M2:
        nibble = _quantize_e1m2_nibble(y_scaled.astype(np.float32)).reshape(
            num_tokens, hidden
        )
        fp4_values = nibble.view(_fp4_dtype(dst_type))
    else:
        raise RuntimeError(
            f"MxFp4 golden only implements FLOAT4_E2M1/E1M2, got {dst_type}"
        )

    if num_blocks % 2 != 0:
        e8m0 = np.pad(e8m0, ((0, 0), (0, 1)), mode="constant")
    scale = e8m0.view(resolve_numpy_dtype("float8_e8m0")).reshape(
        *y_shape[:-1], _ceil_div(num_blocks, 2), 2
    )
    if not pack:
        return fp4_values.reshape(y_shape), scale

    packed = np.bitwise_or(
        nibble[:, 0::2],
        nibble[:, 1::2] << np.uint8(4),
    ).astype(np.uint8)
    packed = packed.view(_fp4_dtype(dst_type))
    return packed.reshape(*y_shape[:-1], _ceil_div(hidden, 2)), scale


def _cast_y_origin(y_origin, x):
    x_np = _to_numpy(x)
    return y_origin.astype(x_np.dtype, copy=False)


def _maybe_to_torch(value, use_torch):
    if not use_torch:
        return value
    if value is None or not isinstance(value, np.ndarray):
        return value
    if value.dtype.name in ("float4_e2m1", "float4_e1m2", "float8_e8m0", "hifloat8"):
        return value
    return numpy_to_torch_tensor(value)


def _quantize_hifloat8(y_weighted, counts, scale, quant_mode, dst_type_max, scale_out):
    hidden = y_weighted.shape[-1]
    values = y_weighted.reshape(-1, hidden)
    group_sizes = [len(values)] if counts is None else [int(count) for count in counts]
    if any(count < 0 for count in group_sizes) or sum(group_sizes) > len(values):
        raise ValueError("HiFloat8 requires nonnegative counts with sum <= token count")

    if quant_mode == 2:
        inverse_scales = _to_numpy(scale)
        if inverse_scales is None or inverse_scales.shape != (len(group_sizes),):
            raise ValueError("quant_mode=2 requires one inverse scale per group")
        y_scale = (
            np.empty(0, dtype=np.float32)
            if scale_out is None
            else _to_numpy(scale_out).copy()
        )
    else:
        maximum = np.float32(dst_type_max)
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("quant_mode=3 requires a positive finite dst_type_max")
        y_scale = (
            np.zeros(len(group_sizes), dtype=np.float32)
            if scale_out is None
            else _to_numpy(scale_out).copy()
        )
        if y_scale.shape != (len(group_sizes),):
            raise ValueError("quant_mode=3 requires one output scale per group")

    hifloat8 = resolve_numpy_dtype("hifloat8")
    y = np.zeros(values.shape, dtype=hifloat8)
    start = 0
    for group, count in enumerate(group_sizes):
        if not count:
            continue
        stop = start + count
        group_values = values[start:stop]
        if quant_mode == 2:
            inverse_scale = np.float32(inverse_scales[group])
        else:
            amax = np.maximum(np.max(np.abs(group_values)), np.float32(1e-8))
            y_scale[group] = amax / maximum
            inverse_scale = (
                np.float32(1) / y_scale[group] if counts is None else maximum / amax
            )
        y[start:stop] = (group_values * inverse_scale).astype(hifloat8)
        start = stop
    return y.reshape(y_weighted.shape), y_scale


def _swiglu_group_quant_reference(
    x,
    weight,
    group_index,
    scale,
    dst_type,
    quant_mode,
    round_scale,
    clamp_limit,
    dst_type_max,
    output_origin,
    group_list_type,
    *,
    alpha=1.0,
    bias=0.0,
    y_out=None,
    scale_out=None,
    origin_out=None,
    pack_fp4=True,
):
    quant_mode = int(quant_mode)
    if quant_mode not in (0, 1, 2, 3, 5):
        raise ValueError(f"Unsupported quant_mode={quant_mode}")
    counts = _group_counts(group_index, group_list_type, quant_mode)
    token_num = _token_num(x)
    y_origin, y_weighted = _compute_swiglu(
        x, weight, clamp_limit, mx_v2=quant_mode == 5, alpha=alpha, bias=bias
    )
    y_origin = _cast_y_origin(y_origin, x) if output_origin else None

    if quant_mode in (2, 3):
        y, y_scale = _quantize_hifloat8(
            y_weighted, counts, scale, quant_mode, dst_type_max, scale_out
        )
        return y, y_scale, y_origin

    dst_type = _dtype_to_int(dst_type)
    if dst_type in (DT_FLOAT4_E2M1, DT_FLOAT4_E1M2):
        # ACLNN test buffers can use either logical FP4 values or packed bytes.
        if y_out is not None and y_out.shape[-1] == y_weighted.shape[-1]:
            pack_fp4 = False
        y, y_scale = _quantize_fp4(y_weighted, dst_type, pack=pack_fp4)
    else:
        y, y_scale = _quantize_fp8(y_weighted, dst_type, quant_mode, bool(round_scale))

    real_bs = (
        token_num if counts is None else min(max(sum(map(int, counts)), 0), token_num)
    )
    return (
        _merge_output_buffer(y, y_out, token_num, real_bs),
        _merge_output_buffer(y_scale, scale_out, token_num, real_bs),
        _merge_output_buffer(y_origin, origin_out, token_num, real_bs),
    )


def aclnn_swiglu_group_quant_golden(
    x,
    weightOptional,
    groupIndexOptional,
    scaleOptional,
    dstType,
    quantMode,
    blockSize,
    roundScale,
    clampLimit,
    dstTypeMax,
    outputOrigin,
    yOut,
    yScaleOut,
    yOriginOut,
    *,
    groupListType=GROUP_LIST_TYPE_COUNT,
    alpha=1.0,
    bias=0.0,
    **kwargs,
):
    del blockSize
    use_torch = kwargs.get("use_torch", False)
    values = _swiglu_group_quant_reference(
        x,
        weightOptional,
        groupIndexOptional,
        scaleOptional,
        dstType,
        quantMode,
        roundScale,
        clampLimit,
        dstTypeMax,
        outputOrigin,
        groupListType,
        alpha=alpha,
        bias=bias,
        y_out=yOut,
        scale_out=yScaleOut,
        origin_out=yOriginOut,
    )
    return tuple(_maybe_to_torch(value, use_torch) for value in values)


def aclnn_swiglu_group_quant_v3_golden(
    x,
    weightOptional,
    groupIndexOptional,
    scaleOptional,
    dstType,
    quantMode,
    blockSize,
    roundScale,
    clampLimit,
    dstTypeMax,
    outputOrigin,
    alpha,
    bias,
    groupListType,
    yOut,
    yScaleOut,
    yOriginOut,
    **kwargs,
):
    return aclnn_swiglu_group_quant_golden(
        x,
        weightOptional,
        groupIndexOptional,
        scaleOptional,
        dstType,
        quantMode,
        blockSize,
        roundScale,
        clampLimit,
        dstTypeMax,
        outputOrigin,
        yOut,
        yScaleOut,
        yOriginOut,
        groupListType=groupListType,
        alpha=alpha,
        bias=bias,
        **kwargs,
    )


def aclnn_swiglu_group_quant_input(
    x,
    weightOptional,
    groupIndexOptional,
    scaleOptional,
    dstType,
    quantMode,
    blockSize,
    roundScale,
    clampLimit,
    dstTypeMax,
    outputOrigin,
    yOut,
    yScaleOut,
    yOriginOut,
    *,
    groupListType=GROUP_LIST_TYPE_COUNT,
    **kwargs,
):
    del blockSize, roundScale
    del scaleOptional, clampLimit, dstTypeMax, outputOrigin, yOut, yScaleOut, yOriginOut
    dst_type = _dtype_to_int(dstType) if quantMode == 1 else None
    if dst_type in (DT_FLOAT4_E2M1, DT_FLOAT4_E1M2):
        x_np = _to_numpy(x)
        hidden = x_np.shape[-1] // 2
        stable_x = np.ones(x_np.shape, dtype=np.float32)
        sign_pattern = np.where(np.arange(hidden) % 2 == 0, 1.0, -1.0).astype(
            np.float32
        )
        stable_x[..., hidden:] = sign_pattern
        _write_tensor(x, stable_x)
        if weightOptional is not None:
            _write_tensor(
                weightOptional,
                np.ones(_to_numpy(weightOptional).shape, dtype=np.float32),
            )
    group_index = _to_numpy(groupIndexOptional)
    if group_index is not None:
        group_index = group_index.copy()
    counts = _group_counts(group_index, groupListType, quantMode)
    if counts is None or counts.size == 0:
        return
    token_num = _token_num(x)
    counts[...] = 0
    if groupListType == GROUP_LIST_TYPE_PAIR:
        group_index[:, 0] = np.arange(counts.size)
    if counts.size == 1:
        counts[0] = token_num
    else:
        counts[0] = token_num // 2
        counts[1] = token_num - counts[0]
    _write_tensor(groupIndexOptional, group_index)


def aclnn_swiglu_group_quant_v3_input(
    x,
    weightOptional,
    groupIndexOptional,
    scaleOptional,
    dstType,
    quantMode,
    blockSize,
    roundScale,
    clampLimit,
    dstTypeMax,
    outputOrigin,
    alpha,
    bias,
    groupListType,
    yOut,
    yScaleOut,
    yOriginOut,
    **kwargs,
):
    del alpha, bias
    return aclnn_swiglu_group_quant_input(
        x,
        weightOptional,
        groupIndexOptional,
        scaleOptional,
        dstType,
        quantMode,
        blockSize,
        roundScale,
        clampLimit,
        dstTypeMax,
        outputOrigin,
        yOut,
        yScaleOut,
        yOriginOut,
        groupListType=groupListType,
        **kwargs,
    )


def swiglu_group_quant_golden(
    x,
    weight,
    group_index,
    scale=None,
    dst_type=DT_FLOAT8_E4M3FN,
    quant_mode=0,
    block_size=0,
    round_scale=False,
    clamp_limit=DEFAULT_CLAMP_LIMIT,
    output_origin=False,
    group_list_type=GROUP_LIST_TYPE_COUNT,
    dst_type_max=15.0,
    alpha=1.0,
    bias=0.0,
    **kwargs,
):
    del block_size, kwargs
    return _swiglu_group_quant_reference(
        x,
        weight,
        group_index,
        scale,
        dst_type,
        quant_mode,
        round_scale,
        clamp_limit,
        dst_type_max,
        output_origin,
        group_list_type,
        alpha=alpha,
        bias=bias,
        pack_fp4=False,
    )


def swiglu_group_quant_input(
    x,
    weight,
    group_index,
    scale=None,
    dst_type=DT_FLOAT8_E4M3FN,
    quant_mode=0,
    block_size=0,
    round_scale=False,
    clamp_limit=DEFAULT_CLAMP_LIMIT,
    output_origin=False,
    group_list_type=GROUP_LIST_TYPE_COUNT,
    **kwargs,
):
    del kwargs
    aclnn_swiglu_group_quant_input(
        x,
        weight,
        group_index,
        None,
        dst_type,
        quant_mode,
        block_size,
        round_scale,
        clamp_limit,
        None,
        output_origin,
        None,
        None,
        None,
        groupListType=group_list_type,
    )
    return x, weight, group_index, scale
