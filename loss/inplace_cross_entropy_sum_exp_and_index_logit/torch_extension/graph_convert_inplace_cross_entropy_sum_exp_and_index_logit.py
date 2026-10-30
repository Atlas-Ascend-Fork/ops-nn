# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode (InplaceCrossEntropySumExpAndIndexLogit)

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

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_nn.inplace_cross_entropy_sum_exp_and_index_logit.default
    )
    def convert_inplace_cross_entropy_sum_exp_and_index_logit(
        vocab_parallel_logits: Tensor,
        target: Tensor,
        global_logits_max: Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        meta_outputs: TensorSpec = None,
    ):
        # torch schema returns only 4 non-inplace outputs (predicted, sum_exp, offset, mask);
        # the in-place write-back of vocab_parallel_logits is expressed via the Tensor(a!)
        # alias on the input and restored by auto_functionalized_v2: torchair matches the
        # GE op's same-named input/output "vocab_parallel_logits" pair (GE output index 2)
        # to rebuild the mutated base tensor. The converter MUST return exactly
        # len(schema.returns) == 4 tensors (drop the alias output here), otherwise torchair
        # raises "The length of ge_outputs must be equal to meta_outputs."
        ge_out = ge_op(
            op_type="InplaceCrossEntropySumExpAndIndexLogit",
            inputs={
                "vocab_parallel_logits": vocab_parallel_logits,
                "target": target,
                "global_logits_max": global_logits_max,
            },
            attrs={
                "vocab_start_index": attr.Int(vocab_start_index),
                "vocab_end_index": attr.Int(vocab_end_index),
            },
            outputs=[
                "predicted_logits",
                "sum_exp_logits",
                "vocab_parallel_logits",
                "target_offset",
                "target_mask",
            ],
            ir=IrDef("InplaceCrossEntropySumExpAndIndexLogit")
            .input("vocab_parallel_logits", "DT_FLOAT, DT_BF16")
            .input("target", "DT_INT32")
            .input("global_logits_max", "DT_FLOAT, DT_BF16")
            .required_attr("vocab_start_index", attr.Int)
            .required_attr("vocab_end_index", attr.Int)
            .output("predicted_logits", "DT_FLOAT")
            .output("sum_exp_logits", "DT_FLOAT")
            .output("vocab_parallel_logits", "DT_FLOAT, DT_BF16")
            .output("target_offset", "DT_INT32")
            .output("target_mask", "DT_INT32"),
        )
        # GE 原型 5 输出，index 2 的 vocab_parallel_logits 是原地别名；
        # torch schema 只声明 4 个返回值，别名由 auto_functionalized_v2 通过
        # 同名 input/output 对回写，converter 不能重复返回它。
        (
            predicted_logits,
            sum_exp_logits,
            _vocab_parallel_logits_alias,
            target_offset,
            target_mask,
        ) = ge_out
        return predicted_logits, sum_exp_logits, target_offset, target_mask

else:

    def convert_inplace_cross_entropy_sum_exp_and_index_logit(*args, **kwargs):
        raise RuntimeError(
            "InplaceCrossEntropySumExpAndIndexLogit graph converter: torchair is not available."
        )
