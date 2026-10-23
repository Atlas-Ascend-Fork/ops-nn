# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only regression tests for the TTK golden contract (no device access)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "assets" / "golden.py"
_SPEC = importlib.util.spec_from_file_location(
    "golden_contract_" + _PATH.parents[2].name, _PATH
)
_G = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_G)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_promote_does_not_round_to_declared_output(dtype):
    x = np.array([2048], dtype=dtype).reshape(1, 1, 1, 1)
    a = np.array([1], dtype=dtype)
    b = np.array([2], dtype=dtype)
    out = _G.BNInferenceKernelSpec.golden(
        x,
        a,
        b,
        a,
        mode=0,
        input_formats=("NCHW",),
        golden_mode="Promote",
        output_dtypes=("float16",),
    )[0]
    assert out.dtype == dtype
    assert out.item() == 4098


def test_two_party_keeps_typed_add_boundary():
    x = np.array([2048], dtype=np.float16).reshape(1, 1, 1, 1)
    a = np.array([1], dtype=np.float16)
    b = np.array([2], dtype=np.float16)
    out = _G.BNInferenceKernelSpec.golden(
        x, a, b, a, mode=0, input_formats=("NCHW",), golden_mode="Enable"
    )[0]
    assert out.dtype == np.float16
    assert out.item() == 4096


def test_promote_mixed_dtypes_follow_native_tensor_arithmetic():
    x = np.ones((1, 1, 1, 1), np.float32)
    a = np.array([2**-40], np.float64)
    b = np.array([1], np.float64)
    out = _G.BNInferenceKernelSpec.golden(
        x, a, b, b, mode=0, input_formats=("NCHW",), golden_mode="Promote"
    )[0]
    assert out.dtype == np.float64
    assert out.item() == 1 + 2**-40
