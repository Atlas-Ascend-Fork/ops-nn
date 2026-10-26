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
"""E2E 层 golden 实现。

委托 ACLNN golden 做核心计算。
对应 torch API: torch_npu.npu_quant_matmul
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import qbmmv3_aclnn_golden as _aclnn


# ============================================================================
# helpers
# ============================================================================

# torch ScalarType enum → dtype 字符串
# npu_quant_matmul 的 output_dtype 参数使用 torch ScalarType int（不是 CANN dtype int）
_SCALAR_TYPE_TO_STR = {
    1: "int8",
    3: "int32",
    5: "float16",
    6: "float32",
    15: "bfloat16",
}


def _output_dtype_to_str(output_dtype):
    """将 output_dtype 参数（int 或 torch.dtype）转换为 dtype 字符串。

    int 值使用 torch ScalarType enum（不是 CANN dtype enum）。
    """
    if output_dtype is None:
        return None
    if isinstance(output_dtype, int):
        return _SCALAR_TYPE_TO_STR.get(output_dtype, str(output_dtype))
    # torch.dtype
    return _util.dtype_to_str(output_dtype)


# ACL_FLOAT4_E2M1：x1_dtype/x2_dtype 属性声明值（1 字节载体，2 个 fp4/字节）
_ACL_FLOAT4_E2M1 = 296
_FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _decode_fp4_nibble(code):
    val = _FP4_LUT[code & 0x7]
    return np.where(code & 0x8, -val, val).astype(np.float32)


def _decode_fp4_carrier(arr, dtype_code, axis, expect_dim):
    """uint8 载体 fp4 解码为 float32（x1_dtype/x2_dtype 属性=296 时生效）。

    两种载体约定（按 scale 逻辑维自动判定）：
      - 压缩 2 元素/字节（奇数打包维用例的 [L] 路线）：axis 维长 x2 == expect_dim，
        低半字节为前一个元素；
      - 1 元素/字节（无法压缩的奇数逻辑维）：直接按字节低半字节解码。
    """
    if arr is None or dtype_code != _ACL_FLOAT4_E2M1 or arr.dtype != np.uint8:
        return arr
    ax = axis % arr.ndim
    if expect_dim is not None and arr.shape[ax] * 2 == expect_dim:
        lo = _decode_fp4_nibble(arr & 0x0F)
        hi = _decode_fp4_nibble((arr >> 4) & 0x0F)
        stacked = np.stack([lo, hi], axis=ax + 1)
        new_shape = arr.shape[:ax] + (arr.shape[ax] * 2,) + arr.shape[ax + 1 :]
        return stacked.reshape(new_shape)
    return _decode_fp4_nibble(arr & 0x0F)


# ============================================================================
# TestSpec class
# ============================================================================


class TorchNpuQuantMatmulTestSpec:
    """torch_npu.npu_quant_matmul E2E golden。

    参数签名对齐 torch_npu.npu_quant_matmul schema:
      (x1, x2, scale, *, offset, pertoken_scale, bias,
       output_dtype, x1_dtype, x2_dtype, pertoken_scale_dtype,
       scale_dtype, group_sizes, y_scale) -> Tensor

    transpose 从 tensor stride 推断（detect_transpose_from_strides）。
    output_dtype 从参数值推断（CANN int → dtype str）。
    x1_dtype/x2_dtype 等从 tensor dtype 获取，不使用单独参数。
    """

    @staticmethod
    def golden(
        x1,
        x2,
        scale,
        *,
        offset=None,
        pertoken_scale=None,
        bias=None,
        output_dtype=None,
        x1_dtype=None,
        x2_dtype=None,
        pertoken_scale_dtype=None,
        scale_dtype=None,
        group_sizes=None,
        y_scale=None,
        **kwargs,
    ):
        # 1. 方向契约：e2e 通路交付的是逻辑视图值（转置语义已体现在视图数值排布中，
        #    strides 仅供框架在设备侧做派发）；而 kernel compute 采用「原始数组 +
        #    transpose 标志」约定，若此处再按 strides 推断标志传入会形成双重转置
        #    （K 维错位/广播失败）。故计算侧固定按非转置处理，M/N 从逻辑视图形状取。
        transpose_x1 = False
        transpose_x2 = False

        # 2. output_dtype → dtype 字符串
        out_dtype = _output_dtype_to_str(output_dtype)

        # 3. group_sizes → group_size int
        group_size = _util.pack_group_sizes(group_sizes)

        # 4. 委托 ACLNN _kernel_qbmm（绕过 golden() 的 _out_dtype fallback）
        # E2E 模式无 out tensor，output dtype 从 output_dtype 参数获取
        x1_np, x2_np, scale_np, offset_np, pertoken_np, bias_np = _util.to_numpy(
            x1, x2, scale, offset, pertoken_scale, bias
        )

        # fp4 uint8 载体解码（x1_dtype/x2_dtype=296）：压缩轴为 x1 的 M 轴 / x2 的 N 轴，
        # 由 pertoken/scale 的逻辑维自动判定压缩（2/字节）或直存（1/字节）约定。
        # pertoken 约定 (M,G,2)：M 在 -3；scale 约定 (G,N,2)：N 在 -2（低维约定取不到时为 None）
        if pertoken_np is None or pertoken_np.ndim < 2:
            m_expect = None
        elif pertoken_np.ndim >= 3:
            m_expect = pertoken_np.shape[-3]
        else:
            m_expect = pertoken_np.shape[-2]
        if scale_np is None or scale_np.ndim < 2:
            n_expect = None
        elif scale_np.ndim >= 3:
            n_expect = scale_np.shape[-2]
        else:
            n_expect = scale_np.shape[-1]
        x1_np = _decode_fp4_carrier(x1_np, x1_dtype, axis=-2, expect_dim=m_expect)
        x2_np = _decode_fp4_carrier(x2_np, x2_dtype, axis=-1, expect_dim=n_expect)

        deq_scale, u64_offset = _util.decode_u64_scale(scale_np)
        if u64_offset is not None and offset_np is None:
            offset_np = u64_offset

        # 从 x1/x2 shape 推断 output shape（用于 determine_quant_mode 的 M/N）
        if transpose_x1:
            m = x1_np.shape[-1]
        else:
            m = x1_np.shape[-2]
        if transpose_x2:
            n = x2_np.shape[-2]
        else:
            n = x2_np.shape[-1]
        out_shapes = ((m, n),) if x1_np.ndim >= 2 else ()

        return _aclnn._kernel_qbmm(
            x1_np,
            x2_np,
            deq_scale,
            offset_np,
            bias_np,
            pertoken_np,
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
            group_size=group_size,
            out_dtype=out_dtype,
            out_shapes=out_shapes,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        scale,
        *,
        offset=None,
        pertoken_scale=None,
        bias=None,
        output_dtype=None,
        x1_dtype=None,
        x2_dtype=None,
        pertoken_scale_dtype=None,
        scale_dtype=None,
        group_sizes=None,
        y_scale=None,
        **kwargs,
    ):
        """E2E 模式的 customize_inputs（in-place 修改）。

        与 ACLNN customize_inputs 逻辑一致：
        1. E8M0 NaN 清洗
        2. UINT64 scale 合理化重新生成

        注：e2e 路径设备侧由 ttk 依 view 逻辑值做 format cast（NZ 排布由框架生成），
        无需改写 storage，golden 直接消费 view 值即可。
        """
        _aclnn._customize_inputs_impl(
            x1, x2, scale, offset, bias, pertoken_scale, **kwargs
        )


__spec__ = {
    "torch_npu.npu_quant_matmul": "TorchNpuQuantMatmulTestSpec",
}
