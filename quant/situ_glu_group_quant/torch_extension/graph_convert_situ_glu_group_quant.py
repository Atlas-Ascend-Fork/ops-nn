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


try:
    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

# 整行chunk上限（EPR），与 csrc/meta 的防呆约束一致
MAX_EPR = 3072

if _TORCHAIR_AVAILABLE:
    # standalone torchair exposes attr.String while the torch_npu-bundled
    # variant names it attr.Str — support both.
    _StrAttr = getattr(attr, "String", None) or attr.Str

    def SituGluGroupQuant(
        x: Tensor,
        weight: Tensor = None,
        group_index: Tensor = None,
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
    ):
        """
        SituGluGroupQuant 的 GE op 封装，REG_OP 的 IR 定义如下（独立移植自
        cann-recipes-infer SituGluGroupQuant；非合并版）：

        REG_OP(SituGluGroupQuant)
            .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
            .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
            .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
            .OUTPUT(y, TensorType({DT_FLOAT8_E4M3FN}))
            .OUTPUT(y_scale, TensorType({DT_FLOAT8_E8M0}))
            .ATTR(beta, Float, 1.0f)
            .ATTR(linear_beta, Float, 0.0f)
            .ATTR(activate_left, Bool, false)
            .ATTR(axis, Int, -1)
            .ATTR(quant_mode, Int, 0)
            .ATTR(dst_type, Int, 36)
            .ATTR(dst_type_max, Float, 0.0f)
            .ATTR(round_mode, String, "rint")
            .ATTR(group_list_type, Int, 0)
            .OP_END_FACTORY_REG(SituGluGroupQuant)
        """
        inputs = {"x": x}
        if weight is not None:
            inputs["weight"] = weight
        if group_index is not None:
            inputs["group_index"] = group_index
        return ge_op(
            op_type="SituGluGroupQuant",
            inputs=inputs,
            attrs={
                "beta": attr.Float(beta),
                "linear_beta": attr.Float(linear_beta),
                "activate_left": attr.Bool(activate_left),
                "axis": attr.Int(axis),
                "quant_mode": attr.Int(quant_mode),
                "dst_type": attr.Int(dst_type),
                "dst_type_max": attr.Float(dst_type_max),
                "round_mode": _StrAttr(round_mode),
                "group_list_type": attr.Int(group_list_type),
            },
            outputs=["y", "y_scale"],
            ir=IrDef("SituGluGroupQuant")
            .input("x", "DT_FLOAT16, DT_BF16")
            .input("weight", "DT_FLOAT")
            .input("group_index", "DT_INT64")
            .attr("beta", attr.Float(1.0))
            .attr("linear_beta", attr.Float(0.0))
            .attr("activate_left", attr.Bool(False))
            .attr("axis", attr.Int(-1))
            .attr("quant_mode", attr.Int(0))
            .attr("dst_type", attr.Int(36))
            .attr("dst_type_max", attr.Float(0.0))
            .attr("round_mode", _StrAttr("rint"))
            .attr("group_list_type", attr.Int(0))
            .output("y", "DT_FLOAT8_E4M3FN")
            .output("y_scale", "DT_FLOAT8_E8M0"),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.situ_glu_group_quant.default)
    def convert_situ_glu_group_quant(
        x: Tensor,
        weight: Tensor = None,
        group_index: Tensor = None,
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
        meta_outputs: TensorSpec = None,
    ):
        """situ_glu_group_quant 图模式 Converter，参数顺序与 schema 一致。"""
        return SituGluGroupQuant(
            x,
            weight,
            group_index,
            beta=beta,
            linear_beta=linear_beta,
            activate_left=activate_left,
            axis=axis,
            quant_mode=quant_mode,
            dst_type=dst_type,
            dst_type_max=dst_type_max,
            round_mode=round_mode,
            group_list_type=group_list_type,
        )

else:

    def convert_situ_glu_group_quant(*args, **kwargs):
        raise RuntimeError(
            "SituGluGroupQuant graph converter: torchair is not available."
        )
