#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Host regressions for the RotateQuant comparator (numpy and ml_dtypes)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))

import unittest
from types import SimpleNamespace

import ml_dtypes
import numpy as np

from rotate_quant_kernel_golden import (
    _finite_values,
    compare_outputs,
    compare_quantized,
    compare_scale,
)

try:
    import en_dtypes
except ImportError:
    en_dtypes = None

FP4 = en_dtypes.float4_e2m1 if en_dtypes else ml_dtypes.float4_e2m1fn
SCALE = en_dtypes.float8_e8m0 if en_dtypes else ml_dtypes.float8_e8m0fnu


class RotateQuantCompareTest(unittest.TestCase):
    def test_scale_delta_one_fails(self):
        actual = np.array([0, 128], dtype=np.uint8).view(SCALE)
        golden = np.array([1, 128], dtype=np.uint8).view(SCALE)
        self.assertFalse(compare_scale(actual, golden)["pass"])

    def test_scale_padding_must_match(self):
        actual = np.array([127, 1], dtype=np.uint8).view(SCALE)
        golden = np.array([127, 0], dtype=np.uint8).view(SCALE)
        self.assertFalse(compare_scale(actual, golden)["pass"])
        self.assertTrue(compare_scale(golden.copy(), golden)["pass"])

    def test_fp4_true_ulp(self):
        golden = np.array([0, -0.5, 3], dtype=FP4)
        adjacent = np.array([0.5, 0, 4], dtype=FP4)
        self.assertTrue(compare_quantized(adjacent, golden)["pass"])
        self.assertFalse(
            compare_quantized(np.array([1], dtype=FP4), golden[:1])["pass"]
        )
        self.assertFalse(
            compare_quantized(np.array([6], dtype=FP4), golden[2:])["pass"]
        )

    def test_fp8_grid_matches_all_finite_codes(self):
        for dtype in (ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2):
            values = np.arange(256, dtype=np.uint8).view(dtype).astype(np.float32)
            expected = np.unique(values[np.isfinite(values)])
            np.testing.assert_array_equal(
                _finite_values(np.dtype(dtype).name), expected
            )

    def test_fp8_small_absolute_error_can_be_many_ulps(self):
        for dtype in (ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2):
            self.assertFalse(
                compare_quantized(
                    np.array([0.25], dtype=dtype), np.array([0], dtype=dtype)
                )["pass"]
            )

    def test_special_values(self):
        dtype = ml_dtypes.float8_e5m2
        golden = np.array([np.nan, np.inf, -np.inf], dtype=dtype)
        self.assertTrue(compare_quantized(golden.copy(), golden)["pass"])
        self.assertFalse(
            compare_quantized(np.array([0, np.inf, -np.inf], dtype=dtype), golden)[
                "pass"
            ]
        )
        self.assertFalse(
            compare_quantized(
                np.array([np.nan, -np.inf, -np.inf], dtype=dtype), golden
            )["pass"]
        )

    def test_shape_and_dtype_mismatch(self):
        values = np.array([1, 2], dtype=FP4)
        self.assertFalse(compare_quantized(values.reshape(1, 2), values)["pass"])
        self.assertFalse(compare_quantized(values.astype(np.float32), values)["pass"])

    def test_context_only_relaxes_y_failure_fraction(self):
        context = SimpleNamespace(
            csv_fields={"precision_tolerances": "((0.001, 0.25),)"}
        )
        y = np.array([0, 0, 0, 1], dtype=FP4)
        golden_y = np.zeros(4, dtype=FP4)
        scale = np.array([128], dtype=np.uint8).view(SCALE)
        golden_scale = np.array([127], dtype=np.uint8).view(SCALE)
        results = compare_outputs(
            y, scale, golden_y, golden_scale, compare_context=context
        )
        self.assertTrue(results[0]["pass"])
        self.assertFalse(results[1]["pass"])

    def test_empty_output(self):
        self.assertTrue(
            compare_quantized(np.array([], dtype=FP4), np.array([], dtype=FP4))["pass"]
        )

    def test_missing_output_fails(self):
        with self.assertRaises(ValueError):
            compare_outputs(np.array([1]))


@unittest.skipUnless(en_dtypes is not None, "Full golden tests require en_dtypes")
class RotateQuantGoldenTest(unittest.TestCase):
    def test_fp4_subnormal_scale_identity_rotation(self):
        from rotate_quant_kernel_golden import compute_rotate_mx

        t = np.float32(2.0**-126)
        x = np.zeros((1, 32), dtype=ml_dtypes.bfloat16)
        x[0, :2] = [6 * t, 3 * t]
        y, scale = compute_rotate_mx(
            x, np.eye(32, dtype=ml_dtypes.bfloat16), scale_alg=2, dst_type_max=8
        )
        np.testing.assert_array_equal(y.astype(np.float32)[0, :2], [6, 6])
        np.testing.assert_array_equal(scale.view(np.uint8).ravel(), [0, 0])

    def test_aclnn_prefers_output_dtype_and_rejects_missing_dtype(self):
        from rotate_quant_aclnn_golden import AclnnRotateQuantTestSpec

        x = np.ones((1, 32), dtype=ml_dtypes.bfloat16)
        rotation = np.eye(32, dtype=ml_dtypes.bfloat16)
        y, _ = AclnnRotateQuantTestSpec.golden(
            x,
            rotation,
            yOut=np.empty((1, 32), dtype=ml_dtypes.float8_e5m2),
            tensor_dtypes=["float4_e2m1"],
        )
        self.assertEqual(y.dtype, np.dtype(ml_dtypes.float8_e5m2))
        with self.assertRaises(ValueError):
            AclnnRotateQuantTestSpec.golden(x, rotation)


if __name__ == "__main__":
    unittest.main()
