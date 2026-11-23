# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class GemmSymmOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("gemm_symm")

    def sources(self) -> list:
        # A source checkout keeps the adapter beside this file. The installed wheel
        # also copies it to the package csrc directory.
        filename = "gemm_symm.cpp"
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [os.path.join(here, "csrc", filename)]
        parent = here
        for _ in range(6):
            parent = os.path.dirname(parent)
            candidates.append(os.path.join(parent, "csrc", filename))
        for path in candidates:
            if os.path.isfile(path):
                return [path]
        return [self.resolve_source(filename)]

    def schema(self) -> str:
        return (
            "gemm_symm(Tensor a, Tensor b, Tensor? c=None, *, float alpha=1.0, float beta=1.0, "
            'bool transpose_a=False, bool transpose_b=False, str symm_type="full", int cube_math_type=0) -> Tensor'
        )

    def register_meta(self):
        @impl(get_as_library(), "gemm_symm", "Meta")
        def gemm_symm_meta(
            a,
            b,
            c=None,
            *,
            alpha=1.0,
            beta=1.0,
            transpose_a=False,
            transpose_b=False,
            symm_type="full",
            cube_math_type=0,
        ):
            m = a.size(-1) if transpose_a else a.size(-2)
            out_shape = (a.size(0), m, m) if a.dim() == 3 else (m, m)
            return torch.empty(out_shape, dtype=a.dtype, device="meta")


gemm_symm_builder = GemmSymmOpBuilder()
gemm_symm_builder._ensure_initialized()


@impl(get_as_library(), "gemm_symm", "PrivateUse1")
def gemm_symm(
    a,
    b,
    c=None,
    *,
    alpha=1.0,
    beta=1.0,
    transpose_a=False,
    transpose_b=False,
    symm_type="full",
    cube_math_type=0,
):
    op_module = gemm_symm_builder.load()
    return op_module.gemm_symm(
        a, b, c, alpha, beta, transpose_a, transpose_b, symm_type, cube_math_type
    )
