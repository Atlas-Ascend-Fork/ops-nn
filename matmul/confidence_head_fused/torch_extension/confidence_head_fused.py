# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library

HIDDEN_RANK_WITHOUT_BATCH = 2
HIDDEN_RANK_WITH_BATCH = 3
MAX_DIM = 0xFFFFFFFF


def _check_tensor(
    tensor: torch.Tensor, dtype: torch.dtype, rank: int, device: torch.device, name: str
) -> None:
    torch._check(tensor.dim() == rank, lambda: f"{name} must have rank {rank}")
    torch._check(tensor.dtype == dtype, lambda: f"{name} must have dtype {dtype}")
    torch._check(tensor.layout == torch.strided, lambda: f"{name} must be strided")
    torch._check(
        tensor.device == device, lambda: f"{name} must be on the same device as hidden"
    )
    for dim in tensor.shape:
        torch._check(dim <= MAX_DIM, lambda: f"{name} dimensions must fit UINT32")


class ConfidenceHeadFusedOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("confidence_head_fused")

    def sources(self) -> list:
        return [self.resolve_source("confidence_head_fused.cpp")]

    def schema(self) -> str:
        return (
            "confidence_head_fused(Tensor hidden, Tensor weight, *, "
            "Tensor? markov_emb=None, Tensor? temperature=None, Tensor? bias=None, "
            "Scalar? clamp_low=None, Scalar? clamp_high=None) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def confidence_head_fused_meta(
            hidden: torch.Tensor,
            weight: torch.Tensor,
            *,
            markov_emb: Optional[torch.Tensor] = None,
            temperature: Optional[torch.Tensor] = None,
            bias: Optional[torch.Tensor] = None,
            clamp_low: Optional[float] = None,
            clamp_high: Optional[float] = None,
        ) -> torch.Tensor:
            rank = hidden.dim()
            torch._check(
                rank in (HIDDEN_RANK_WITHOUT_BATCH, HIDDEN_RANK_WITH_BATCH),
                lambda: "hidden must have rank 2 or 3",
            )
            torch._check(
                hidden.dtype in (torch.float16, torch.bfloat16),
                lambda: "hidden must be FP16 or BF16",
            )
            _check_tensor(hidden, hidden.dtype, rank, hidden.device, "hidden")
            _check_tensor(weight, hidden.dtype, 1, hidden.device, "weight")
            k_markov = 0
            if markov_emb is not None:
                _check_tensor(
                    markov_emb, hidden.dtype, rank, hidden.device, "markov_emb"
                )
                for i in range(rank - 1):
                    torch._check(
                        markov_emb.shape[i] == hidden.shape[i],
                        lambda: "hidden and markov_emb leading dimensions must match",
                    )
                k_markov = markov_emb.shape[-1]
            rows = 1
            for dim in hidden.shape[:-1]:
                rows *= dim
            torch._check(rows <= MAX_DIM, lambda: "Flattened rows must fit UINT32")
            torch._check(
                hidden.shape[-1] + k_markov <= MAX_DIM,
                lambda: "K1+K2 must fit UINT32",
            )
            torch._check(
                weight.shape[0] == hidden.shape[-1] + k_markov,
                lambda: "weight length must equal K1+K2",
            )
            if temperature is not None:
                _check_tensor(
                    temperature, torch.float32, 1, hidden.device, "temperature"
                )
                torch._check(
                    temperature.shape[0] == 1, lambda: "temperature must have shape [1]"
                )
            if bias is not None:
                _check_tensor(bias, torch.float32, 1, hidden.device, "bias")
                torch._check(
                    bias.shape[0] == hidden.shape[-2],
                    lambda: "bias must have shape [N], shared across batches",
                )
            # Meta checks metadata only; the kernel handles temperature values and the zero fallback.
            return hidden.new_empty(hidden.shape[:-1], dtype=torch.float32)


confidence_head_fused_builder = ConfidenceHeadFusedOpBuilder()
confidence_head_fused_builder._ensure_initialized()


@impl(get_as_library(), confidence_head_fused_builder.name, "PrivateUse1")
def _confidence_head_fused(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    markov_emb: Optional[torch.Tensor] = None,
    temperature: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    clamp_low: Optional[float] = None,
    clamp_high: Optional[float] = None,
) -> torch.Tensor:
    return confidence_head_fused_builder.load().confidence_head_fused(
        hidden, weight, markov_emb, temperature, bias, clamp_low, clamp_high
    )


def confidence_head_fused(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    markov_emb: Optional[torch.Tensor] = None,
    temperature: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    clamp_low: Optional[float] = None,
    clamp_high: Optional[float] = None,
) -> torch.Tensor:
    """Compute confidence probabilities from FP16/BF16 features and optional FP32 bias[N]."""
    return torch.ops.cann_ops_nn.confidence_head_fused(
        hidden,
        weight,
        markov_emb=markov_emb,
        temperature=temperature,
        bias=bias,
        clamp_low=clamp_low,
        clamp_high=clamp_high,
    )
