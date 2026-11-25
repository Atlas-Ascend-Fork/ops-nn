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

"""CPU regression tests for the Centralization golden."""

import runpy
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch


PLUGIN = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "assets" / "golden.py")
)
golden = PLUGIN["CentralizationSpec"].golden


class TestCentralizationGolden(unittest.TestCase):
    def test_axes_contract(self):
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        for axes in ([-3], [-4], [2], [0, -3]):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                golden(x, axes=axes)
        for axes in ([True], [0.5], "", np.empty((0, 2), dtype=int)):
            with self.subTest(axes=axes), self.assertRaises(TypeError):
                golden(x, axes=axes)
        expected = np.array([[-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(golden(x)[0], expected)
        np.testing.assert_array_equal(golden(x, axes=[-1, np.int32(1)])[0], expected)
        np.testing.assert_array_equal(golden(x, axes=[-2])[0], [[-1.5] * 3, [1.5] * 3])

    def test_input_domain_and_promote(self):
        for shape in ((), (1,) * 9):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                golden(np.zeros(shape, dtype=np.float32))
        for dtype in (np.int32, np.bool_, np.float64):
            with self.subTest(dtype=dtype), self.assertRaises(TypeError):
                golden(np.ones(3, dtype=dtype))
        x = np.array([1, 2, 3], dtype=np.float64)
        result = golden(x, golden_mode="Promote", input_dtypes=("float32",))[0]
        self.assertEqual(result.dtype, x.dtype)
        np.testing.assert_array_equal(result, [-1, 0, 1])
        np.testing.assert_array_equal(
            golden(x, golden_mode="Promote", input_dtypes=(("float64",),))[0], result
        )
        with self.assertRaises(TypeError):
            golden(
                x.astype(np.float32), golden_mode="Promote", input_dtypes=("bfloat16",)
            )

    def test_cancellation_in_sum_and_centering(self):
        large = 2**100
        # Exact means are 1/3 and large + 1/3, respectively.
        rows = np.array(
            [[large, 1, -large], [large, 2 * large, 1], [1, 2, 3]], dtype=np.float32
        )
        expected = np.array(
            [[large, 2 / 3, -large], [-1 / 3, large, -large], [-1, 0, 1]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(golden(rows)[0], expected)
        np.testing.assert_array_equal(golden(rows[:, ::-1])[0], expected[:, ::-1])
        for x, y in zip(rows, expected):
            np.testing.assert_array_equal(golden(x)[0], y)

    def test_promoted_values_match_exact_rational_reference(self):
        rows = np.array(
            [[2**100, 2**46], [1.0, -(2**-54)], [3.0, -7.0]], dtype=np.float64
        )
        expected = []
        for row in rows:
            values = [Fraction.from_float(float(value)) for value in row]
            mean = sum(values) / len(values)
            expected.append([float(value - mean) for value in values])
        actual = golden(rows, golden_mode="Promote", input_dtypes=("float64",))[0]
        np.testing.assert_array_equal(actual, expected)

    def test_nonadjacent_axes_and_nonfinite_values(self):
        x = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
        expected = (
            x.astype(np.float64) - x.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
        ).astype(x.dtype)
        y = golden(x, attributes={"axes": [-1, 0]})[0]
        self.assertEqual(y.dtype, x.dtype)
        np.testing.assert_array_equal(y, expected)
        x = np.array([[np.inf, 1], [-np.inf, 1], [np.inf, -np.inf], [1, np.nan]])
        result = golden(x.astype(np.float32))[0]
        np.testing.assert_array_equal(
            result,
            [[np.nan, -np.inf], [np.nan, np.inf], [np.nan, np.nan], [np.nan, np.nan]],
        )
        np.testing.assert_array_equal(
            golden(x.astype(np.float32), axes=[])[0],
            np.where(np.isfinite(x), 0, np.nan),
        )

    def test_third_party_centers_in_fp32(self):
        compose = PLUGIN["CentralizationSpec"].third_party["torch"](axes=[-1])
        for dtype, values, expected in (
            (torch.float16, [2048, 2050], [-1, 1]),
            (torch.float32, [2**24, 2**24 + 2], [0, 2]),
        ):
            with self.subTest(dtype=dtype):
                x = torch.tensor(values, dtype=dtype)
                (actual,) = compose(x)
                torch.testing.assert_close(
                    actual, torch.tensor(expected, dtype=dtype), rtol=0, atol=0
                )

    def test_empty_dimensions(self):
        x = np.empty((2, 0, 3), dtype=np.float16)
        for axes in ([0], [1], []):
            with self.subTest(axes=axes):
                y = golden(x, axes=axes)[0]
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y.dtype, x.dtype)
        with self.assertRaises(ValueError):
            golden(x, axes=[-4])
