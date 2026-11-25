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

"""CPU golden regressions: python3 -m unittest discover -s <this directory>."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from ml_dtypes import bfloat16


SPEC = importlib.util.spec_from_file_location(
    "dynamic_mx_quant_golden", Path(__file__).parents[1] / "assets" / "golden.py"
)
GOLDEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLDEN)


class DynamicMxQuantGoldenTest(unittest.TestCase):
    def test_zero_blocks_keep_zero_scale_with_positive_lower_bound(self):
        for dtype in (np.float16, bfloat16, np.float32):
            for dst in ("float8_e4m3fn", "float8_e5m2"):
                for axis in (0, -1):
                    for block_size in (32, 64):
                        with self.subTest(
                            dtype=dtype, dst=dst, axis=axis, block=block_size
                        ):
                            # A full zero block, a clamped nonzero block, and a zero tail.
                            x = np.zeros((2, block_size * 2 + 1), dtype=dtype)
                            x[:, 1::2] = -0.0
                            x[:, block_size : 2 * block_size] = 1
                            if axis == 0:
                                x = x.T.copy()
                            scale, y = GOLDEN._mx_quantize(
                                x,
                                mx_ele_dtype=dst,
                                axis=axis,
                                block_size=block_size,
                                scale_alg=1,
                                max_low_bound=57344.0,
                            )
                            expected = np.zeros(scale.shape, np.uint8)
                            # E4M3 uses scale=128, E5M2 uses scale=1.
                            expected[:, 0, 1] = 134 if dst == "float8_e4m3fn" else 127
                            if axis == 0:
                                expected = np.zeros(scale.shape, np.uint8)
                                expected[0, :, 1] = (
                                    134 if dst == "float8_e4m3fn" else 127
                                )
                            np.testing.assert_array_equal(
                                scale.view(np.uint8), expected
                            )
                            zero = x == 0
                            np.testing.assert_array_equal(
                                np.asarray(y, np.float32)[zero], 0
                            )

    def test_zero_blocks_for_all_existing_algorithms(self):
        for dtype in (np.float16, bfloat16, np.float32):
            for alg, dst in (
                (0, "float4_e2m1"),
                (1, "float8_e4m3fn"),
                (2, "float4_e2m1"),
            ):
                with self.subTest(dtype=dtype, alg=alg):
                    x = np.zeros((3, 66), dtype=dtype)
                    x[:, 1::2] = -0.0
                    scale, y = GOLDEN._mx_quantize(x, mx_ele_dtype=dst, scale_alg=alg)
                    np.testing.assert_array_equal(scale.view(np.uint8), 0)
                    np.testing.assert_array_equal(np.asarray(y, np.float32), 0)

    def test_lower_bound_still_clamps_nonzero_blocks(self):
        x = np.ones((2, 32), np.float16)
        scale, _ = GOLDEN._mx_quantize(
            x, mx_ele_dtype="float8_e4m3fn", scale_alg=1, max_low_bound=448.0
        )
        np.testing.assert_array_equal(scale.view(np.uint8), [[[127, 0]], [[127, 0]]])

    def test_algorithm_two_uses_original_fp16_amax(self):
        x = np.zeros((2, 32), np.float16)
        x[:, :2] = 1
        x[:, 2] = [3.0, 3.001953125]
        scale, y = GOLDEN._mx_quantize(x, scale_alg=2)
        np.testing.assert_array_equal(scale.view(np.uint8), [[[126, 0]], [[127, 0]]])
        np.testing.assert_array_equal(
            np.asarray(y, np.float32)[:, :2], [[2, 2], [1, 1]]
        )

    def test_algorithm_two_custom_destination_maximum(self):
        x = np.full((1, 32), 3, np.float16)
        scale, _ = GOLDEN._mx_quantize(x, scale_alg=2, dst_type_max=3.0)
        np.testing.assert_array_equal(scale.view(np.uint8), [[[127, 0]]])

    def test_nvfp4_zero_and_padding_with_lower_bound(self):
        for dtype in (np.float16, bfloat16):
            for k in (2, 16, 18, 32, 48, 66, 65538):
                with self.subTest(dtype=dtype, k=k):
                    y, scale, _ = GOLDEN._nvfp4_reference(
                        np.zeros((2, k), dtype=dtype), max_low_bound=448.0
                    )
                    np.testing.assert_array_equal(y, 0)
                    np.testing.assert_array_equal(scale, 0)

    def test_nvfp4_nan_inf_and_signed_zero(self):
        for dtype in (np.float16, bfloat16):
            with self.subTest(dtype=dtype):
                x = np.zeros((4, 16), dtype=dtype)
                x[0] = np.nan
                x[1, :2] = [np.inf, -np.inf]
                x[2, :2] = [0.0, -0.0]
                x[3, :2] = [1.0, np.nan]
                y, scale, _ = GOLDEN._nvfp4_reference(x)
                np.testing.assert_array_equal(
                    scale.reshape(4, 2)[:, 0], [127, 126, 0, 35]
                )
                self.assertTrue(np.all(y.reshape(4, 8)[0] == 0x77))
                self.assertEqual(int(y.reshape(4, 8)[1, 0]), 0xF7)
                self.assertEqual(int(y.reshape(4, 8)[2, 0]), 0x80)

    def test_nvfp4_aclnn_v3_and_e2e_match_reference_bytes(self):
        for dtype in (np.float16, bfloat16):
            for mode in ("rint", "round", "floor"):
                with self.subTest(dtype=dtype, mode=mode):
                    x = np.zeros((2, 34), dtype=dtype)
                    x[0, :7] = [1, -1, 3, -3, 0.5, -0.5, -0.0]
                    x[1, :3] = [np.nan, np.inf, -np.inf]
                    packed, scale_bytes, _ = GOLDEN._nvfp4_reference(
                        x, round_mode=mode, max_low_bound=4.0
                    )
                    acl_y, acl_scale = GOLDEN.aclnn_dynamic_mx_quant_v3_golden(
                        x, -1, mode, 40, 16, 3, 0.0, 4.0, None, None
                    )
                    e2e_y, e2e_scale = GOLDEN.npu_dynamic_mx_quant_golden(
                        x,
                        axis=-1,
                        round_mode=mode,
                        dst_type=296,
                        block_size=16,
                        scale_alg=3,
                        max_low_bound=4.0,
                    )
                    self.assertEqual(acl_y.shape, x.shape)
                    self.assertEqual(acl_scale.shape, (2, 2, 2))
                    np.testing.assert_array_equal(
                        np.ascontiguousarray(acl_scale).view(np.uint8).ravel(),
                        scale_bytes,
                    )
                    self.assertEqual(e2e_y.shape, x.shape)
                    np.testing.assert_array_equal(e2e_y, acl_y)
                    np.testing.assert_array_equal(e2e_scale.ravel(), scale_bytes)
                    # Repack the logical FP4 codes to verify the device bytes.
                    codes = np.ascontiguousarray(e2e_y).view(np.uint8) & 15
                    packed_from_e2e = codes[..., ::2] | (codes[..., 1::2] << 4)
                    np.testing.assert_array_equal(packed_from_e2e.ravel(), packed)

    def test_nvfp4_torch_bfloat16_input_and_promoted_carrier(self):
        import torch

        x = torch.tensor([[1, -0.0, 3, -3] + [0] * 14], dtype=torch.bfloat16)
        expected_y, expected_scale, _ = GOLDEN._nvfp4_reference(
            x.float().numpy().astype(bfloat16)
        )
        acl_y, acl_scale = GOLDEN.aclnn_dynamic_mx_quant_v3_golden(
            x, -1, None, 40, 16, 3, 0.0, 0.0, None, None
        )
        e2e_y, e2e_scale = GOLDEN.npu_dynamic_mx_quant_golden(
            x, dst_type=296, block_size=16, scale_alg=3
        )
        promoted_y, promoted_scale = GOLDEN.npu_dynamic_mx_quant_golden(
            x.float(),
            dst_type=296,
            block_size=16,
            scale_alg=3,
            golden_mode="Promote",
        )
        e2e_codes = np.ascontiguousarray(e2e_y).view(np.uint8) & 15
        np.testing.assert_array_equal(
            (e2e_codes[..., ::2] | (e2e_codes[..., 1::2] << 4)).ravel(),
            expected_y,
        )
        np.testing.assert_array_equal(e2e_scale.ravel(), expected_scale)
        np.testing.assert_array_equal(promoted_y, e2e_y)
        np.testing.assert_array_equal(promoted_scale, e2e_scale)
        np.testing.assert_array_equal(acl_scale.view(np.uint8).ravel(), expected_scale)
        self.assertEqual(acl_y.shape, (1, 18))
        # The ACLNN runner promotes low precision inputs and replaces
        # tensor_dtypes with float32 before calling a custom golden.
        promoted_acl_y, promoted_acl_scale = GOLDEN.aclnn_dynamic_mx_quant_v3_golden(
            x.float(),
            -1,
            "rint",
            40,
            16,
            3,
            0.0,
            0.0,
            None,
            None,
            tensor_dtypes=("float32", "float4_e2m1", "float8_e4m3fn"),
        )
        np.testing.assert_array_equal(promoted_acl_y, acl_y)
        np.testing.assert_array_equal(promoted_acl_scale, acl_scale)
        with self.assertRaisesRegex(ValueError, "source metadata"):
            GOLDEN.npu_dynamic_mx_quant_golden(
                x.float(), dst_type=296, block_size=16, scale_alg=3
            )

    def test_legacy_aclnn_versions_and_e2e_fp4_packing(self):
        x = np.array([[1, -1] * 32], dtype=np.float16)
        for alg, dst, torch_dst in ((0, 40, 296), (1, 36, 292), (2, 40, 296)):
            with self.subTest(alg=alg):
                expected_y, expected_scale = GOLDEN.dynamic_mx_quant_golden(
                    x,
                    dst_type=dst,
                    scale_alg=alg,
                    output_dtypes=(GOLDEN.DATA_TYPE_INT_TO_STR[dst], "float8_e8m0"),
                )
                acl_v1 = GOLDEN.aclnn_dynamic_mx_quant_golden(
                    x, -1, "rint", dst, 32, alg, None, None
                )
                acl_v2 = GOLDEN.aclnn_dynamic_mx_quant_v2_golden(
                    x, -1, "rint", dst, 32, alg, 0.0, None, None
                )
                acl_v3 = GOLDEN.aclnn_dynamic_mx_quant_v3_golden(
                    x, -1, "rint", dst, 32, alg, 0.0, 0.0, None, None
                )
                for y, scale in (acl_v1, acl_v2, acl_v3):
                    np.testing.assert_array_equal(y, expected_y)
                    np.testing.assert_array_equal(scale, expected_scale)
                e2e_y, e2e_scale = GOLDEN.npu_dynamic_mx_quant_golden(
                    x, dst_type=torch_dst, scale_alg=alg
                )
                np.testing.assert_array_equal(e2e_y, expected_y)
                np.testing.assert_array_equal(
                    e2e_scale, np.ascontiguousarray(expected_scale).view(np.uint8)
                )


if __name__ == "__main__":
    unittest.main()
