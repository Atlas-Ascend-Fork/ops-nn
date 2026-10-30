# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class InplaceCrossEntropySumExpAndIndexLogitOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("inplace_cross_entropy_sum_exp_and_index_logit")

    def sources(self):
        return [
            self.resolve_source("inplace_cross_entropy_sum_exp_and_index_logit.cpp")
        ]

    def schema(self):
        # vocab_parallel_logits is Tensor(a!) — in-place, not returned
        # Returns only 4 non-inplace outputs
        return (
            "inplace_cross_entropy_sum_exp_and_index_logit(Tensor(a!) vocab_parallel_logits, "
            "Tensor target, Tensor global_logits_max, int vocab_start_index, "
            "int vocab_end_index) -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def inplace_cross_entropy_sum_exp_and_index_logit_meta(
            vocab_parallel_logits,
            target,
            global_logits_max,
            vocab_start_index,
            vocab_end_index,
        ):
            target_shape = list(target.shape)
            predicted_logits = target.new_empty(target_shape, dtype=torch.float32)
            sum_exp_logits = target.new_empty(target_shape, dtype=torch.float32)
            target_offset = target.new_empty(target_shape, dtype=torch.int32)
            target_mask = target.new_empty(target_shape, dtype=torch.int32)
            return (
                predicted_logits,
                sum_exp_logits,
                target_offset,
                target_mask,
            )


builder = InplaceCrossEntropySumExpAndIndexLogitOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), builder.name, "PrivateUse1")
def inplace_cross_entropy_sum_exp_and_index_logit(
    vocab_parallel_logits,
    target,
    global_logits_max,
    vocab_start_index,
    vocab_end_index,
):
    op_module = builder.load()
    return op_module.inplace_cross_entropy_sum_exp_and_index_logit(
        vocab_parallel_logits,
        target,
        global_logits_max,
        vocab_start_index,
        vocab_end_index,
    )
