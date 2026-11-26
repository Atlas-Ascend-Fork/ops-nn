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
# GE Converter for Graph Mode

try:
    from typing import Optional

    import torch
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        _ge_dtype_to_ge_proto_dtype,
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

# 固定块大小与算法枚举，与 Python 前端保持一致
LEVEL0_BLOCK_SIZE = 512
LEVEL1_BLOCK_SIZE = 32
SCALE_ALG_DEFAULT = 0
DST_TYPE_MAX_DEFAULT = 0.0
# FLOAT4_E2M1 每字节打包 2 个元素
FP4_PACK_FACTOR = 2


def _pack_fp4_output_to_uint8(y, rank):
    """把逻辑形状的 FLOAT4_E2M1 输出打包成 uint8（每字节 2 个元素）。"""
    bit_shape = [1] * (rank - 1)
    bit_shape.append(FP4_PACK_FACTOR)
    div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
    y_shape_fp4 = ge.Shape(y)
    y_shape_uint8 = ge.Div(y_shape_fp4, div_x2)
    y_shape_fp4_2bit = ge.ConcatV2(
        [
            y_shape_uint8,
            ge.Cast(ge.Const([FP4_PACK_FACTOR]), dst_type=DataType.DT_INT32),
        ],
        concat_dim=0,
        N=2,
    )
    y = ge.Bitcast(ge.Reshape(y, y_shape_fp4_2bit), type=DataType.DT_UINT8)
    y = ge.Reshape(y, y_shape_uint8)
    y.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_UINT8)
    return y


if _TORCHAIR_AVAILABLE:

    def DynamicDualLevelMxQuant(
        x: Tensor,
        smooth_scale: Optional[Tensor] = None,
        *,
        round_mode: str = "rint",
        level0_block_size: int = LEVEL0_BLOCK_SIZE,
        level1_block_size: int = LEVEL1_BLOCK_SIZE,
        scale_alg: int = SCALE_ALG_DEFAULT,
        dst_type_max: float = DST_TYPE_MAX_DEFAULT,
    ):
        """
        DynamicDualLevelMxQuant 的 GE op 封装，REG_OP 的 IR 定义如下：

        REG_OP(DynamicDualLevelMxQuant)
            .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
            .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT16, DT_BF16}))
            .OUTPUT(y, TensorType({DT_FLOAT4_E2M1}))
            .OUTPUT(level0_scale, TensorType({DT_FLOAT}))
            .OUTPUT(level1_scale, TensorType({DT_FLOAT8_E8M0}))
            .ATTR(round_mode, String, "rint")
            .ATTR(level0_block_size, Int, 512)
            .ATTR(level1_block_size, Int, 32)
            .ATTR(scale_alg, Int, 0)
            .ATTR(dst_type_max, Float, 0.0)
            .OP_END_FACTORY_REG(DynamicDualLevelMxQuant)
        """
        inputs = {"x": x}
        if smooth_scale is not None:
            inputs["smooth_scale"] = smooth_scale

        return ge_op(
            op_type="DynamicDualLevelMxQuant",
            inputs=inputs,
            attrs={
                "round_mode": attr.String(round_mode),
                "level0_block_size": attr.Int(level0_block_size),
                "level1_block_size": attr.Int(level1_block_size),
                "scale_alg": attr.Int(scale_alg),
                "dst_type_max": attr.Float(dst_type_max),
            },
            outputs=["y", "level0_scale", "level1_scale"],
            ir=IrDef("DynamicDualLevelMxQuant")
            .input("x", "DT_FLOAT16, DT_BF16")
            .optional_input("smooth_scale", "DT_FLOAT16, DT_BF16")
            .attr("round_mode", attr.String("rint"))
            .attr("level0_block_size", attr.Int(LEVEL0_BLOCK_SIZE))
            .attr("level1_block_size", attr.Int(LEVEL1_BLOCK_SIZE))
            .attr("scale_alg", attr.Int(SCALE_ALG_DEFAULT))
            .attr("dst_type_max", attr.Float(DST_TYPE_MAX_DEFAULT))
            .output("y", "DT_FLOAT4_E2M1")
            .output("level0_scale", "DT_FLOAT")
            .output("level1_scale", "DT_FLOAT8_E8M0"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_nn.dynamic_dual_level_mx_quant.default
    )
    def convert_dynamic_dual_level_mx_quant(
        x: Tensor,
        smooth_scale: Optional[Tensor] = None,
        *,
        round_mode: str = "rint",
        level0_block_size: int = LEVEL0_BLOCK_SIZE,
        level1_block_size: int = LEVEL1_BLOCK_SIZE,
        scale_alg: int = SCALE_ALG_DEFAULT,
        dst_type_max: float = DST_TYPE_MAX_DEFAULT,
        meta_outputs: TensorSpec = None,
    ):
        """dynamic_dual_level_mx_quant 图模式 Converter，参数顺序与 schema 一致。"""
        y, level0_scale, level1_scale = DynamicDualLevelMxQuant(
            x,
            smooth_scale,
            round_mode=round_mode,
            level0_block_size=level0_block_size,
            level1_block_size=level1_block_size,
            scale_alg=scale_alg,
            dst_type_max=dst_type_max,
        )
        y.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT4_E2M1)
        level0_scale.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT)
        level1_scale.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT8_E8M0)
        y = _pack_fp4_output_to_uint8(y, x.rank)
        return y, level0_scale, level1_scale

else:

    def convert_dynamic_dual_level_mx_quant(*args, **kwargs):
        raise RuntimeError(
            "DynamicDualLevelMxQuant graph converter: torchair is not available."
        )
