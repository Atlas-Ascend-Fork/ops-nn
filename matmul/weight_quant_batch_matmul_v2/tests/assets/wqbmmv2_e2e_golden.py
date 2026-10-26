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
"""
weight_quant_batch_matmul_v2 E2E golden implementation.

This module contains V2 E2E TestSpec class.
"""

import os
import sys
from importlib import import_module

import numpy as np

_assets = os.path.dirname(os.path.abspath(__file__))
_common = os.path.join(_assets, "../../../common/tests/st/arch35")

for _d in (_common, _assets):
    if _d not in sys.path:
        sys.path.insert(0, _d)

_aclnn = import_module("wqbmmv2_aclnn_golden")
_util = import_module("matmul_golden_util")


def _get_logical_shape(weight, is_transposed):
    """从 weight tensor 获取逻辑形状 (N, K_packed)。

    非转置: view shape = (K, N_packed), storage shape = (K, N_packed)
      → logical = (N_packed, K) 需要交换
    转置: view shape = (K_packed, N), storage shape = (N, K_packed)
      → logical = (N, K_packed) 直接从 storage shape 读取

    返回: (N, K_packed) 用于生成 unpacked 数据
    """
    if is_transposed:
        # 转置: view=(K_packed, N), storage=(N, K_packed)
        # 从 strides 推导 storage shape
        N = weight.shape[1]
        K_packed = weight.shape[0]
        return (N, K_packed)
    else:
        # 非转置: view=(K, N_packed), storage=(K, N_packed)
        K = weight.shape[0]
        N_packed = weight.shape[1]
        return (N_packed, K)


def _write_nz_to_storage(weight, nz_data):
    """将 NZ 数据按 storage_offset 写入 weight 底层 storage（in-place）。

    torch 2.x 的 UntypedStorage 不支持 numpy()/frombuffer，且
    ``storage[slice] = Tensor`` 会报 "can only set storage content with a int
    types"；统一走 util 的 set_ uint8 全 buffer 视图写入。
    """
    _util.overwrite_storage(weight, nz_data)


def _pack_int4_weight(weight, need_nz, is_transposed):
    """int4 打包：生成 unpacked int4 数据 → CPU 打包 → ND→NZ (可选) → in-place 替换。

    纯 CPU 实现，匹配 C++ convert_weight_to_int4_pack 的流程：
      Step 1: 生成 unpacked int4 数据 (logical shape)
      Step 2: Pack (ConvertToB4Pack): 每 8 个 int4 → 1 个 int32
      Step 3: ND→NZ (TransNdToNz): 仅 NZ 场景
      Step 4: 写回 weight (转置时需先转置 packed 以匹配 view shape)

    Args:
        weight: torch.Tensor, weight tensor (view shape already set by framework)
        need_nz: bool, 是否需要转换为 NZ 格式
        is_transposed: bool, weight 是否转置 (从 strides 检测)
    """
    import torch

    if not need_nz and not is_transposed:
        # ND 非转置: 容器 (K, N_packed)，int4 沿 N 打包（与 aclnn golden unpack axis=-1 一致）
        K, N_packed = weight.shape
        unpacked = torch.randint(-8, 8, (K, N_packed * 8), dtype=torch.int32)
        packed = _util.pack_int32_from_int4(unpacked.numpy(), axis=-1)
        weight.copy_(torch.from_numpy(packed))
        return

    if need_nz and not is_transposed:
        # NZ 非转置（packed-quant 载体约定）: 容器 view (K, N_packed)，
        # 逻辑 (K, N) 沿 N 打包后做 ND→NZ；n0 按载体 dtype（int32→2）与
        # aclnn golden 解码侧 get_nz_n0 一致。
        K, N_packed = weight.shape
        unpacked = torch.randint(-8, 8, (K, N_packed * 8), dtype=torch.int32)
        packed = _util.pack_int32_from_int4(unpacked.numpy(), axis=-1)
        n0 = _util.get_nz_n0(str(weight.dtype).replace("torch.", ""))
        nz_data = _util.nd_to_nz(packed, n0=n0)
        _write_nz_to_storage(weight, nz_data)
        return

    # 获取逻辑形状 (N, K) 用于生成 unpacked 数据
    N, K_packed = _get_logical_shape(weight, is_transposed)
    K = K_packed * 8

    # 生成 unpacked 数据，逻辑形状 (N, K)
    unpacked = torch.randint(-8, 8, (N, K), dtype=torch.int32)
    # 打包: (N, K) → (N, K_packed)
    packed = _util.pack_int32_from_int4(unpacked.numpy(), axis=-1)

    if need_nz:
        # NZ 场景: 转置 packed 以匹配 view shape，然后做 ND→NZ。
        # n0 按容器 dtype 取 packed-quant 约定（int32/float32 → 2，与 aclnn golden
        # 解码侧 get_nz_n0 一致；勿用缺省，否则 int64 中间量会按 16 分形导致尺寸不符）
        n0 = _util.get_nz_n0(str(weight.dtype).replace("torch.", ""))
        if is_transposed:
            # packed (N, K_packed) → (K_packed, N) 匹配 view shape
            packed_for_nz = packed.T
        else:
            packed_for_nz = packed
        nz_data = _util.nd_to_nz(packed_for_nz, n0=n0)
        _write_nz_to_storage(weight, nz_data)
    else:
        # ND 场景: 转置 packed 以匹配 view shape，然后 copy_
        if is_transposed:
            # packed (N, K_packed) → (K_packed, N) 匹配 view shape
            packed = packed.T
        weight.copy_(torch.from_numpy(packed))


def _pack_fp4_weight(weight, need_nz, is_transposed):
    """fp4 打包：生成 unpacked fp4 索引 → CPU 打包 → ND→NZ (可选) → in-place 替换。

    纯 CPU 实现，匹配 C++ convert_weight_to_int4_pack 的流程：
      Step 1: 生成 unpacked fp4 索引 (logical shape)
      Step 2: Pack (ConvertToB4Pack): 每 8 个 fp4 → 1 个 int32
      Step 3: ND→NZ (TransNdToNz): 仅 NZ 场景
      Step 4: 写回 weight (转置时需先转置 packed 以匹配 view shape)

    Args:
        weight: torch.Tensor, weight tensor (view shape already set by framework)
        need_nz: bool, 是否需要转换为 NZ 格式
        is_transposed: bool, weight 是否转置 (从 strides 检测)
    """
    import torch

    if need_nz and not is_transposed:
        # NZ 非转置（packed-quant 载体约定）: 容器 view (K, N_packed)，逻辑 (K, N)
        # 沿 N 打包后 ND→NZ；n0 按载体 dtype（float32→2）。
        K, N_packed = weight.shape
        fp4_indices = torch.randint(0, 16, (K, N_packed * 8), dtype=torch.int32)
        packed = _util.pack_int32_from_fp4(fp4_indices.numpy(), axis=-1)
        n0 = _util.get_nz_n0(str(weight.dtype).replace("torch.", ""))
        nz_data = _util.nd_to_nz(packed, n0=n0)
        _write_nz_to_storage(weight, nz_data.view(np.float32))
        return

    if not need_nz and not is_transposed:
        # ND 非转置: 容器 (K, N_packed)，fp4 沿 N 打包（与 aclnn golden unpack axis=-1 一致）
        K, N_packed = weight.shape
        fp4_indices = torch.randint(0, 16, (K, N_packed * 8), dtype=torch.int32)
        packed = _util.pack_int32_from_fp4(fp4_indices.numpy(), axis=-1)
        weight.copy_(torch.from_numpy(packed).view(weight.dtype))
        return

    # 获取逻辑形状 (N, K) 用于生成 unpacked 数据
    N, K_packed = _get_logical_shape(weight, is_transposed)
    K = K_packed * 8

    # 生成 unpacked 数据，逻辑形状 (N, K)
    fp4_indices = torch.randint(0, 16, (N, K), dtype=torch.int32)
    # 打包: (N, K) → (N, K_packed)
    packed = _util.pack_int32_from_fp4(fp4_indices.numpy(), axis=-1)

    if need_nz:
        # NZ 场景: 转置 packed 以匹配 view shape，然后做 ND→NZ（n0 同上按容器 dtype）
        n0 = _util.get_nz_n0(str(weight.dtype).replace("torch.", ""))
        if is_transposed:
            packed_for_nz = packed.T
        else:
            packed_for_nz = packed
        nz_data = _util.nd_to_nz(packed_for_nz, n0=n0)
        _write_nz_to_storage(weight, nz_data.view(np.float32))
    else:
        # ND 场景: 转置 packed 以匹配 view shape，然后 copy_
        if is_transposed:
            packed = packed.T
        weight.copy_(torch.from_numpy(packed).view(weight.dtype))


class WeightQuantBatchMatmulV2TorchApiTestSpec:
    @staticmethod
    def golden(
        x,
        weight,
        antiquant_scale,
        antiquant_offset=None,
        quant_scale=None,
        quant_offset=None,
        bias=None,
        antiquant_group_size=0,
        inner_precise=0,
        weight_dtype=None,
        **kwargs,
    ):
        """E2E golden: 委托 ACLNN golden 完成预处理 + 计算。"""
        kwargs.pop("transpose_x", None)
        kwargs.pop("transpose_weight", None)
        kwargs.pop("transposeX", None)
        kwargs.pop("transposeWeight", None)
        return _aclnn.AclnnWeightQuantBatchMatmulV2TestSpec.golden(
            x,
            weight,
            antiquant_scale,
            antiquant_offset,
            quant_scale,
            quant_offset,
            bias,
            antiquantGroupSize=antiquant_group_size,
            y=x,
            inner_precise=inner_precise,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x,
        weight,
        antiquant_scale,
        antiquant_offset=None,
        quant_scale=None,
        quant_offset=None,
        bias=None,
        antiquant_group_size=0,
        inner_precise=0,
        weight_dtype=None,
        **kwargs,
    ):
        """E2E 输入定制：int4/fp4 打包 + 4 项数据替换（in-place）。"""
        if weight is not None:
            tensor_formats = kwargs.get("tensor_formats", None)
            weight_format = (
                tensor_formats[1]
                if tensor_formats and len(tensor_formats) > 1
                else "ND"
            )
            need_nz = weight_format == "FRACTAL_NZ"
            is_transposed = _util.detect_transpose_from_strides(weight)

            import torch

            if weight.dtype == torch.int32:
                _pack_int4_weight(weight, need_nz, is_transposed)
            elif weight.dtype == torch.float32:
                _pack_fp4_weight(weight, need_nz, is_transposed)

        testcase_name = kwargs.get("testcase_name", "unknown")

        if antiquant_scale is not None:
            sanitized = _util.sanitize_e8m0_scale(antiquant_scale, testcase_name)
            if sanitized is not antiquant_scale:
                _util.write_back(antiquant_scale, sanitized)


__spec__ = {
    "torch_npu.npu_weight_quant_batchmatmul": "WeightQuantBatchMatmulV2TorchApiTestSpec",
}
