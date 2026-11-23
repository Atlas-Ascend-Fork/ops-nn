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

"""On-device regression tests for compensated means and independent reduction groups."""

import importlib
import unittest

import numpy as np


class TestCentralizationNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import npu_device
        import tensorflow as tf

        cls.device = npu_device.open().as_default()
        op = importlib.import_module("npu_device.npu_device").gen_npu_ops.Centralization

        @tf.function(autograph=False)
        def centralize(x, axis):
            return op(x=x, axes=[axis])

        cls.centralize = staticmethod(centralize)

    def assert_centered(self, x, axis):
        wide = x.astype(np.float64)
        with np.errstate(invalid="ignore", over="ignore"):
            expected = (wide - wide.mean(axis=axis, keepdims=True)).astype(x.dtype)
        np.testing.assert_array_equal(self.centralize(x, axis).numpy(), expected)

    def test_finite_mean_at_float_limit(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                value = sign * float.fromhex("0x1.fffffep+126")
                self.assert_centered(np.full((1, 2), value, dtype=np.float32), 1)

    def test_short_rows_with_overflowing_sums(self):
        limit = np.finfo(np.float32).max
        for width in (2, 3, 4, 5, 6, 64):
            with self.subTest(width=width):
                x = np.tile(np.arange(width, dtype=np.float32), (129, 1))
                x[0] = limit
                x[63] = -limit
                x[64, 0] = np.inf
                x[65, 0] = np.nan
                x[-1] = limit
                if width >= 4:
                    x[1] = 0
                    x[1, :4] = [limit, limit, -limit, -limit]
                    if width > 4:
                        x[1, -1] = 1.0
                        x[3] = x[1]
                        x[3, -1] = 2.0**-120
                    x[2, :3] = [limit, limit, -np.inf]
                self.assert_centered(x, 1)

    def test_nonfinite_mean_in_fast_vector_path(self):
        for value in (np.inf, -np.inf):
            with self.subTest(value=value):
                x = np.zeros((1, 129), dtype=np.float16)
                x[0, 0] = value
                self.assert_centered(x, 1)

    def test_columns_reduce_independently(self):
        for width in (2, 3, 4):
            for value in (np.inf, np.nan):
                with self.subTest(width=width, value=value):
                    x = np.tile(np.arange(1, width + 1, dtype=np.float32), (32768, 1))
                    x[0, 0] = value
                    self.assert_centered(x, 0)


if __name__ == "__main__":
    unittest.main()
