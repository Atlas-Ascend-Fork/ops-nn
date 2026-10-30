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
"""CPU regressions for the ordinary Torch contract and NPU vector-stage boundaries."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets"))

import unittest
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import torch

from rotate_quant_aclnn_golden import AclnnRotateQuantTestSpec
from rotate_quant_e2e_golden import (
    CannOpsNnRotateQuantTestSpec,
    TorchNpuRotateQuantTestSpec,
    _pack_fp4,
    _unpack_fp4,
)
from rotate_quant_kernel_golden import (
    RotateQuantTestSpec,
    _apply_clamp,
    compute_rotate_mx,
    mx_quantize,
    np_bfloat16,
    np_fp4_e2m1,
    np_mx_scale,
)


class RotateQuantE2eGoldenTest(unittest.TestCase):
    def test_original_and_migrated_interfaces_share_quantized_values(self):
        x = torch.arange(192, dtype=torch.bfloat16).reshape(2, 96) / 32
        rotation = torch.eye(32, dtype=x.dtype).repeat(3, 1, 1)
        for options in (
            {"dst_dtype": 296},
            {"dst_dtype": 296, "scale_alg": 2, "dst_type_max": 7.0},
            {"dst_dtype": 291, "scale_alg": 1},
            {"dst_dtype": 292},
        ):
            with self.subTest(options=options):
                original = TorchNpuRotateQuantTestSpec.golden(x, rotation, **options)
                migrated = CannOpsNnRotateQuantTestSpec.golden(x, rotation, **options)
                np.testing.assert_array_equal(
                    original[0].view(np.uint8), migrated[0].view(np.uint8)
                )
                self.assertEqual(original[1].dtype, np.dtype(np_mx_scale))
                self.assertEqual(migrated[1].dtype, np.dtype(np.uint8))
                np.testing.assert_array_equal(original[1].view(np.uint8), migrated[1])

    def test_original_scale_carriers_and_migrated_raw_contract(self):
        y = np.zeros((1, 16), dtype=np.uint8)
        raw = np.array([[[127, 0]]], dtype=np.uint8)
        encoded = raw.view(np_mx_scale)
        for actual in (raw, encoded):
            result = TorchNpuRotateQuantTestSpec.compare(y, actual, y, encoded)
            self.assertTrue(all(item["pass"] for item in result))
        changed = raw.copy()
        changed[0, 0, 1] = 1
        result = TorchNpuRotateQuantTestSpec.compare(y, changed, y, encoded)
        self.assertFalse(result[1]["pass"])
        result = CannOpsNnRotateQuantTestSpec.compare(y, encoded, y, raw)
        self.assertFalse(result[1]["pass"])

    def test_fp4_nibble_order_and_signed_zero(self):
        codes = np.arange(16, dtype=np.uint8).reshape(1, 16)
        packed = _pack_fp4(codes.view(np_fp4_e2m1))
        np.testing.assert_array_equal(
            packed, [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]]
        )
        np.testing.assert_array_equal(_unpack_fp4(packed).view(np.uint8), codes)

    def test_torch_bf16_fp4_identity_has_packed_y_and_raw_scale(self):
        values = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        x = torch.tensor([values * 2], dtype=torch.bfloat16)
        y, scale = CannOpsNnRotateQuantTestSpec.golden(
            x, torch.eye(32, dtype=x.dtype), dst_dtype=296
        )
        self.assertEqual(y.dtype, np.dtype(np.uint8))
        self.assertEqual(y.shape, (1, 16))
        # Matmul need not preserve the sign of zero, so inspect finite nonzero codes.
        np.testing.assert_array_equal(y[0, :4], [0x10, 0x32, 0x54, 0x76])
        np.testing.assert_array_equal(y[0, 5:8], [0xBA, 0xDC, 0xFE])
        np.testing.assert_array_equal(scale, [[[127, 0]]])

    def test_fp8_enums_aliases_and_torch_dtype(self):
        x = torch.ones((2, 96), dtype=torch.bfloat16)
        rotation = torch.eye(32, dtype=x.dtype).repeat(3, 1, 1)
        for dst, dtype, code, value in (
            (23, ml_dtypes.float8_e5m2, 112, 32768),
            (291, ml_dtypes.float8_e5m2, 112, 32768),
            (torch.float8_e5m2, ml_dtypes.float8_e5m2, 112, 32768),
            (24, ml_dtypes.float8_e4m3fn, 119, 256),
            (292, ml_dtypes.float8_e4m3fn, 119, 256),
        ):
            with self.subTest(dst=dst):
                y, scale = CannOpsNnRotateQuantTestSpec.golden(
                    x, rotation, dst_dtype=dst
                )
                self.assertEqual(y.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(
                    y.astype(np.float32), np.full((2, 96), value)
                )
                np.testing.assert_array_equal(
                    scale.reshape(2, 4), [[code, code, code, 0]] * 2
                )

    def test_noncontiguous_inputs_and_none_attributes(self):
        # Build the view after dtype conversion to retain the stride.
        x = torch.arange(384, dtype=torch.bfloat16).reshape(2, 192)[:, 1::2]
        rotation = torch.zeros((3, 32, 64), dtype=x.dtype)[..., 1::2]
        rotation.copy_(torch.eye(32, dtype=x.dtype).repeat(3, 1, 1))
        self.assertFalse(x.is_contiguous())
        self.assertFalse(rotation.is_contiguous())
        actual = CannOpsNnRotateQuantTestSpec.golden(
            x,
            rotation,
            dst_dtype=296,
            axis=None,
            round_mode=None,
            scale_alg=None,
            dst_type_max=None,
            transpose_y=None,
        )
        expected = CannOpsNnRotateQuantTestSpec.golden(
            x.contiguous(), rotation.contiguous(), dst_dtype=296
        )
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_empty_and_high_rank_shapes(self):
        for shape in ((96,), (1, 1, 1, 1, 1, 2, 96), (0, 96), (2, 0)):
            n = shape[-1]
            x = torch.zeros(shape, dtype=torch.bfloat16)
            rotation = torch.eye(32, dtype=x.dtype).repeat(n // 32, 1, 1)
            y, scale = CannOpsNnRotateQuantTestSpec.golden(x, rotation, dst_dtype=296)
            self.assertEqual(y.shape, shape[:-1] + (n // 2,))
            self.assertEqual(scale.shape, shape[:-1] + ((n + 63) // 64, 2))

    def test_rejects_grouped_and_non_mx_contracts(self):
        x = torch.ones((1, 32), dtype=torch.bfloat16)
        rotation = torch.eye(32, dtype=x.dtype)
        for options in (
            {},
            {"dst_dtype": 1},
            {"dst_dtype": 40},
            {"dst_dtype": 296, "group_list": [1]},
            {"dst_dtype": 296, "transpose_y": True},
            {"dst_dtype": 296, "axis": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CannOpsNnRotateQuantTestSpec.golden(x, rotation, **options)

    def test_compare_decodes_fp4_and_checks_scale_padding(self):
        golden_y = np.zeros((1, 2), dtype=np.uint8)
        actual_y = golden_y.copy()
        actual_y[0, 0] = (
            2  # 0 -> 1 differs by two FP4 ULPs in one of four logical values.
        )
        scale = np.array([[[127, 0]]], dtype=np.uint8)
        context = SimpleNamespace(
            csv_fields={"precision_tolerances": "((0.001,0.25),)"}
        )
        result = CannOpsNnRotateQuantTestSpec.compare(
            actual_y, scale, golden_y, scale, compare_context=context
        )
        self.assertTrue(all(item["pass"] for item in result))
        changed = scale.copy()
        changed[0, 0, 1] = 1
        result = CannOpsNnRotateQuantTestSpec.compare(
            golden_y, changed, golden_y, scale, compare_context=context
        )
        self.assertFalse(result[1]["pass"])
        self.assertFalse(
            CannOpsNnRotateQuantTestSpec.compare(actual_y, scale, golden_y, scale)[0][
                "pass"
            ]
        )

    def test_kernel_and_aclnn_compare_receive_context(self):
        y = np.array([0, 0, 0, 1], dtype=np_fp4_e2m1)
        gy = np.zeros(4, dtype=np_fp4_e2m1)
        scale = np.array([127, 0], dtype=np.uint8).view(np_mx_scale)
        context = SimpleNamespace(
            csv_fields={"precision_tolerances": "((0.001,0.25),)"}
        )
        for spec in (RotateQuantTestSpec, AclnnRotateQuantTestSpec):
            self.assertTrue(
                spec.compare(y, scale, gy, scale, compare_context=context)[0]["pass"]
            )


class RotateQuantNpuStageTest(unittest.TestCase):
    def test_fixpipe_bf16_rounding_precedes_fp8_conversion(self):
        x = np.zeros((1, 32), dtype=np.float16)
        x[0, 0] = 1
        rotation = np.zeros((32, 32), dtype=np.float16)
        rotation[0, 0] = 1.0634765625
        y, scale = compute_rotate_mx(x, rotation, mx_ele_dtype="float8_e4m3fn")
        # 1.0634765625 -> BF16 1.0625 -> scaled 272 -> FP8 RINT 256 (not 288).
        self.assertEqual(float(y[0, 0]), 256)
        np.testing.assert_array_equal(scale.view(np.uint8), [[[119, 0]]])

    def test_per_n_rotation_and_scale_pair_padding(self):
        x = np.ones((2, 3, 96), dtype=np_bfloat16)
        rotation = np.stack([np.eye(32) * factor for factor in (1, 2, 4)]).astype(
            np_bfloat16
        )
        y, scale = compute_rotate_mx(x, rotation)
        np.testing.assert_array_equal(y.astype(np.float32), np.full(x.shape, 4))
        np.testing.assert_array_equal(
            scale.view(np.uint8).reshape(6, 4), [[125, 126, 127, 0]] * 6
        )

    def test_alpha_open_interval_and_bf16_limit(self):
        x = np.full((1, 32), 1.0078125, dtype=np_bfloat16)
        for alpha in (0, 1, -0.5, 1.5):
            np.testing.assert_array_equal(
                _apply_clamp(x, np.array([alpha], dtype=np_bfloat16)), x
            )
        result = _apply_clamp(x, np.array([0.50390625], dtype=np_bfloat16))
        np.testing.assert_array_equal(
            result.astype(np.float32), np.full(x.shape, 0.5078125)
        )

    def test_fp4_scale_alg2_mantissa_thresholds(self):
        for max_norm, below, above in ((6, 6.0, 6.03125), (7, 7.0, 7.03125)):
            for maximum, expected_code in ((below, 127), (above, 128)):
                x = np.full((1, 32), maximum, dtype=np_bfloat16)
                _, scale = mx_quantize(x, scale_alg=2, dst_type_max=max_norm)
                np.testing.assert_array_equal(
                    scale.view(np.uint8), [[[expected_code, 0]]]
                )

    def test_zero_scale_code_has_different_ocp_and_cublas_reciprocals(self):
        x = np.full((1, 32), 2.0**-126, dtype=np_bfloat16)
        y, scale = mx_quantize(x, mx_ele_dtype="float8_e4m3fn", scale_alg=1)
        np.testing.assert_array_equal(scale.view(np.uint8), [[[0, 0]]])
        # cuBLAS keeps BF16 0x7f00 (2**127), whereas OCP clears the reciprocal.
        np.testing.assert_array_equal(y.astype(np.float32), np.full(x.shape, 2))
        ocp_y, ocp_scale = mx_quantize(x, mx_ele_dtype="float8_e4m3fn", scale_alg=0)
        np.testing.assert_array_equal(ocp_y.astype(np.float32), np.zeros(x.shape))
        np.testing.assert_array_equal(ocp_scale.view(np.uint8), [[[0, 0]]])

    def test_cublas_fp32_reciprocal_rounding_boundary(self):
        x = np.full((1, 32), 6.5, dtype=np_bfloat16)
        dst_type_max = 6.499999523162842
        y, scale = mx_quantize(x, scale_alg=2, dst_type_max=dst_type_max)
        # Kernel: FP32(6.5 * FP32(1/dstTypeMax)) == 1, so scale code is 127.
        # FP32(6.5/dstTypeMax) is 1.000000119..., which would incorrectly give 128.
        np.testing.assert_array_equal(scale.view(np.uint8), [[[127, 0]]])
        np.testing.assert_array_equal(y.astype(np.float32), np.full(x.shape, 6))

    def test_fp8_saturating_cast_clears_nan_and_keeps_scale_code(self):
        x = np.zeros((1, 32), dtype=np_bfloat16)
        x[0, 0] = np.nan
        y, scale = mx_quantize(x, mx_ele_dtype="float8_e4m3fn")
        np.testing.assert_array_equal(y.astype(np.float32), np.zeros(x.shape))
        np.testing.assert_array_equal(scale.view(np.uint8), [[[255, 0]]])

    def test_fp4_rounding_modes(self):
        x = np.zeros((1, 32), dtype=np_bfloat16)
        x[0, :5] = [0.25, -0.25, 2.5, -2.5, 6]
        for mode, expected in (
            ("rint", [0, 0, 2, -2, 6]),
            ("round", [0.5, -0.5, 3, -3, 6]),
            ("floor", [0, -0.5, 2, -3, 6]),
        ):
            y, _ = mx_quantize(x, round_mode=mode)
            np.testing.assert_array_equal(y.astype(np.float32)[0, :5], expected)


if __name__ == "__main__":
    unittest.main()
