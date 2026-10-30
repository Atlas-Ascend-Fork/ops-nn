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
@pytest.mark.parametrize(
    "fmt,shape,axes",
    [
        ("NCDHW", (2, 3, 4), (0, 2)),
        ("NDHWC", (2, 2, 2, 2, 3), (0, 1, 2, 3)),
        ("NDC1HWC0", (2, 2, 1, 2, 2, 16), (0, 1, 3, 4)),
    ],
)
def test_promote_keeps_supplied_precision_and_layout(dtype, fmt, shape, axes):
    x = np.arange(np.prod(shape), dtype=dtype).reshape(shape) / 16
    out = _G.BN3DTrainingReduceKernelSpec.golden(
        x,
        golden_mode="Promote",
        input_formats=(fmt,),
        output_dtypes=("float64", "float64"),
    )
    for actual, expected in zip(
        out,
        [
            np.sum(x, axis=axes, keepdims=fmt == "NDC1HWC0"),
            np.sum(x * x, axis=axes, keepdims=fmt == "NDC1HWC0"),
        ],
    ):
        assert actual.dtype == dtype
        np.testing.assert_array_equal(actual, expected)


def test_two_party_widens_half_to_float32():
    x = np.full((2, 1), 300, dtype=np.float16)
    a, b = _G.BN3DTrainingReduceKernelSpec.golden(x, golden_mode="Enable")
    assert a.dtype == b.dtype == np.float32
    assert b[0] == 180000


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_promote_square_sum_preserves_small_integer_terms(dtype):
    # Every square and the final sum are exactly representable in both dtypes.
    # CPU torch.cumsum accumulates float32 inputs in double; storing a rounded
    # prefix does not round its internal accumulator or change the output dtype.
    x = np.array([2047, 1] * 5, dtype=dtype).reshape(10, 1)
    a, b = _G.BN3DTrainingReduceKernelSpec.golden(
        x, golden_mode="Promote", input_formats=("NCDHW",)
    )
    assert a.dtype == b.dtype == dtype
    assert a[0] == 10240
    assert b[0] == 20951050


def test_special_values_remain_channel_local():
    x = np.array([[1, np.nan, np.inf], [2, 3, 4]], np.float64)
    a, b = _G.BN3DTrainingReduceKernelSpec.golden(x, golden_mode="Promote")
    assert a[0] == 3 and b[0] == 5
    assert np.isnan(a[1]) and np.isnan(b[1])
    assert np.isposinf(a[2]) and np.isposinf(b[2])
