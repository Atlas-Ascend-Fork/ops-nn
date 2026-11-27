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
            return op(x=x, axes=list(axis) if isinstance(axis, tuple) else [axis])

        cls.centralize = staticmethod(centralize)

    def assert_centered(self, x, axis):
        wide = x.astype(np.float64)
        with np.errstate(invalid="ignore", over="ignore"):
            expected = (wide - wide.mean(axis=axis, keepdims=True)).astype(x.dtype)
        np.testing.assert_array_equal(self.centralize(x, axis).numpy(), expected)

    def test_packed_rows_with_partial_tail(self):
        for dtype in (np.float16, np.float32):
            for width in (1, 2, 3, 5, 15, 20, 64):
                with self.subTest(dtype=dtype, width=width):
                    values = np.arange(2053 * width, dtype=np.int32) % 173 - 86
                    x = values.astype(dtype).reshape(-1, width)
                    x[[0, 63, -1], 0] = [np.inf, np.nan, -np.inf]
                    if dtype == np.float32:
                        x[1] = 2.0**-120
                    self.assert_centered(x, 1)

    def test_short_reduction_between_kept_axes(self):
        layouts = ((2, 4), (3, 3), (4, 5), (6, 12))
        for dtype in (np.float16, np.float32):
            extra = ((5, 8), (7, 4), (8, 5)) if dtype == np.float16 else ()
            for width, inner in layouts + extra:
                with self.subTest(dtype=dtype, width=width, inner=inner):
                    x = (
                        np.arange(2053 * width * inner, dtype=np.int32) % 173 - 86
                    ).astype(dtype)
                    x = x.reshape(-1, width, inner)
                    x[0, 0, 0] = np.inf
                    x[-1, :, -1] = np.finfo(dtype).max
                    self.assert_centered(x, 1)

    def test_partial_tiles_and_split_reductions(self):
        layouts = (
            ((97, 1, 70), (2,)),
            ((512, 618), (1,)),
            ((19, 23, 32), (0, 1)),
            ((10001, 108), (0,)),
            ((513, 6, 343), (0, 2)),
            ((2, 4, 128, 120), (0, 3)),
            ((1, 1310721), (1,)),
        )
        for dtype in (np.float16, np.float32):
            for shape, axes in layouts:
                with self.subTest(dtype=dtype, shape=shape, axes=axes):
                    values = np.arange(np.prod(shape), dtype=np.int32) % 173 - 86
                    self.assert_centered(values.astype(dtype).reshape(shape), axes)

    def test_plane_overflow_within_and_between_partials(self):
        x = np.zeros((10001, 108), dtype=np.float32)
        peak = np.float32(2.0**127)
        # A 1001-row split puts the two positive values in the same partial.
        x[1000:1004, 0] = [-peak, peak, peak, -peak]
        # Here each partial is finite and overflow occurs only during the merge.
        x[[0, 1001, 2002, 3003], 1] = [peak, peak, -peak, -peak]
        x[:, 2] = -x[:, 1]
        x[:, 3] = np.float32(2.0**-120)
        self.assert_centered(x, (0,))

    def test_segmented_suffix_cancels_before_overflow(self):
        x = np.zeros((7, 6, 2, 7, 4, 3, 8, 2), dtype=np.float32)
        peak = np.float32(np.finfo(np.float32).max * 0.75)
        # The second segment sums to zero. Flattening both segments would add peak to peak first.
        x[0, 0, 0, 0, 0, 0, 0, 0] = peak
        x[0, 0, 0, 0, 1, 0, 0] = [peak, -peak]
        self.assert_centered(x, (1, 4, 7))

    def test_parallel_suffix_segments(self):
        for dtype in (np.float16, np.float32):
            for outer, suffix in ((97, 160), (151, 159)):
                with self.subTest(dtype=dtype, outer=outer, suffix=suffix):
                    # Each group spans several cores. Exercise both offset-table and computed offsets.
                    rows = np.arange(outer, dtype=dtype)[:, None] - outer // 2
                    x = np.empty((outer, 2, suffix), dtype=dtype)
                    x[:, 0, :] = rows + 32
                    x[:, 1, :] = -rows - 16
                    self.assert_centered(x, (0, 2))
                    x[-1, 0, -1] = np.inf
                    self.assert_centered(x, (0, 2))

    def test_strided_suffix_partial_rows_and_kept_coordinates(self):
        layouts = (
            ((513, 3, 17), (0, 2)),
            ((2, 1201, 2, 3, 2, 3), (1, 3, 5)),
            ((3, 2, 3001, 3, 3), (0, 2, 4)),
            ((3, 3, 8201), (0, 2)),
            ((2, 149, 2, 71, 2), (1, 3)),
            ((8201, 3), (0,)),
            ((2, 8201, 3, 2), (1, 3)),
            ((2, 149, 2, 71, 33), (1, 3)),
        )
        for dtype in (np.float16, np.float32):
            for shape, axes in layouts:
                with self.subTest(dtype=dtype, shape=shape):
                    values = np.arange(np.prod(shape), dtype=np.int32)
                    x = ((values * 13 % 181) - 90).astype(dtype).reshape(shape)
                    self.assert_centered(x, axes)
                    x.flat[0] = np.inf
                    x.flat[-1] = np.nan
                    self.assert_centered(x, axes)

    def test_strided_suffix_sum_at_float_limit(self):
        shape = (513, 3, 17)
        x = np.zeros(shape, dtype=np.float32)
        peak = np.finfo(x.dtype).max
        x[:, 0, :] = peak
        x[:, 1, :] = -peak
        x[0, 2, :4] = [peak, peak, -peak, -peak]
        self.assert_centered(x, (0, 2))

        # A later chunk overflows before cancellation with the first chunk.
        peak = np.float32(peak * 0.75)
        for columns in (2, 33):
            with self.subTest(columns=columns):
                x = np.zeros((2, 149, 2, 71, columns), dtype=np.float32)
                x[0, 0, 0, 0, 0] = -peak
                x[0, 10, 0, :3, 0] = [peak, peak, -peak]
                x[0, :, 0, :, 1] = np.float32(2.0**-105)
                self.assert_centered(x, (1, 3))

    def test_repeating_physical_groups(self):
        for dtype in (np.float16, np.float32):
            for shape in ((16388, 2, 2, 4), (16386, 2, 4)):
                with self.subTest(dtype=dtype, shape=shape):
                    x = np.broadcast_to(
                        np.arange(np.prod(shape[1:])).reshape(shape[1:]), shape
                    ).astype(dtype)
                    x[1::2] += 8
                    self.assert_centered(x, (0, 2))
                    x.reshape(-1)[-1] = np.inf
                    self.assert_centered(x, (0, 2))

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
                        if width == 64:
                            x[4:6] = x[1]
                            x[4:6, -1] = [2.0**-121, 2.0**-126]
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

    def test_wide_packed_rows(self):
        shape = (2305, 195)
        x = (np.arange(np.prod(shape), dtype=np.int32) % 173 - 86).astype(np.float32)
        x = x.reshape(shape)
        x[0] = np.finfo(x.dtype).max
        x[63, 0] = np.inf
        x[-1, -1] = np.nan
        self.assert_centered(x, 1)


if __name__ == "__main__":
    unittest.main()
