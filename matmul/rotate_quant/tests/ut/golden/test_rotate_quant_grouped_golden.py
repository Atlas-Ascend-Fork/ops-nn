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
"""Grouped rotation references: per-N matrices, empty groups and untouched tails."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "st/grouped/assets"))

import unittest

import ml_dtypes
import numpy as np

try:
    from rotate_quant_grouped_golden import AclnnRotateQuantV2TestSpec
    from rotate_quant_kernel_golden import mx_quantize, _apply_clamp, np_mx_scale
except ModuleNotFoundError as error:
    if error.name == "en_dtypes":
        raise unittest.SkipTest("Requires CANN en_dtypes golden dependency") from error
    raise


class RotateQuantGroupedGoldenTest(unittest.TestCase):
    def inputs(self, k=32, q=3, e=5):
        rng = np.random.default_rng(20261021)
        x = rng.uniform(-1, 1, (11, q * k)).astype(ml_dtypes.bfloat16)
        # Both group and N-slice indices change the rotation.
        rot = rng.uniform(-0.25, 0.25, (e, q, k, k)).astype(ml_dtypes.bfloat16)
        y = np.full(x.shape, 3, dtype=ml_dtypes.float8_e4m3fn)
        scale = np.full((11, (q * k + 63) // 64, 2), 129, dtype=np.uint8).view(
            np_mx_scale
        )
        return x, rot, y, scale

    def test_per_n_rotation_matches_independent_group_and_slice_matmuls(self):
        counts = np.array([0, 1, 4, 0, 3], dtype=np.int64)
        for k in (32, 64, 128, 256):
            for mode in (0, 1):
                x, rot, y, scale = self.inputs(k=k)
                alpha = np.array([0.75], dtype=ml_dtypes.bfloat16)
                groups = counts if mode else np.cumsum(counts)
                expected_y, expected_scale = y.copy(), scale.copy()
                start = 0
                for g, size in enumerate(counts):
                    end = start + int(size)
                    if size:
                        z = np.empty((size, x.shape[1]), dtype=ml_dtypes.bfloat16)
                        for b in range(rot.shape[1]):
                            cols = slice(b * k, (b + 1) * k)
                            z[:, cols] = x[start:end, cols].astype(np.float32) @ rot[
                                g, b
                            ].astype(np.float32)
                        z = _apply_clamp(z, alpha)
                        gy, gs = mx_quantize(z, mx_ele_dtype="float8_e4m3fn")
                        expected_y[start:end], expected_scale[start:end] = gy, gs
                    start = end
                actual = AclnnRotateQuantV2TestSpec.golden(
                    x, rot, alpha, groups, mode, yOut=y, scaleOut=scale
                )
                np.testing.assert_array_equal(
                    actual[0].view(np.uint8), expected_y.view(np.uint8)
                )
                np.testing.assert_array_equal(
                    actual[1].view(np.uint8), expected_scale.view(np.uint8)
                )
                np.testing.assert_array_equal(
                    y.view(np.uint8), np.full(y.shape, 3, dtype=y.dtype).view(np.uint8)
                )

    def test_shared_rotation_equals_repeated_per_n_rotation(self):
        counts = np.array([1, 0, 7, 0, 0], dtype=np.int64)
        for q in (1, 3):
            x, rot, y, scale = self.inputs(q=q)
            shared = rot[:, 0].copy()
            repeated = np.repeat(shared[:, None], q, axis=1)
            a = AclnnRotateQuantV2TestSpec.golden(
                x, shared, groupList=counts, groupListType=1, yOut=y, scaleOut=scale
            )
            b = AclnnRotateQuantV2TestSpec.golden(
                x, repeated, groupList=counts, groupListType=1, yOut=y, scaleOut=scale
            )
            for left, right in zip(a, b):
                np.testing.assert_array_equal(left.view(np.uint8), right.view(np.uint8))

    def test_empty_groups_preserve_all_output_codes(self):
        x, rot, y, scale = self.inputs()
        for mode in (0, 1):
            actual = AclnnRotateQuantV2TestSpec.golden(
                x,
                rot,
                groupList=np.zeros(5, dtype=np.int64),
                groupListType=mode,
                yOut=y,
                scaleOut=scale,
            )
            np.testing.assert_array_equal(actual[0].view(np.uint8), y.view(np.uint8))
            np.testing.assert_array_equal(
                actual[1].view(np.uint8), scale.view(np.uint8)
            )

    def test_rejects_mismatched_per_n_rotation_count(self):
        x, rot, y, scale = self.inputs()
        with self.assertRaises(ValueError):
            AclnnRotateQuantV2TestSpec.golden(
                x,
                rot[:, :2],
                groupList=np.ones(5, dtype=np.int64),
                groupListType=1,
                yOut=y,
                scaleOut=scale,
            )


if __name__ == "__main__":
    unittest.main()
