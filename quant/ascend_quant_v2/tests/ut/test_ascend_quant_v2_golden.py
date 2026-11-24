# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "quant_golden", Path(__file__).resolve().parents[1] / "assets" / "golden.py"
)
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)


class AscendQuantGoldenTest(unittest.TestCase):
    def test_nonlast_axis_kernel_and_aclnn(self):
        x = torch.ones(2, 3)
        scale = torch.tensor([2.0, 4.0])
        offset = torch.tensor([1.0, -1.0])
        expected = torch.full((2, 3), 3, dtype=torch.int8)
        for axis in (0, -2):
            with self.subTest(axis=axis):
                actual = golden.aclnn_ascend_quant_v3_golden(
                    x,
                    scale,
                    offset,
                    False,
                    "round",
                    2,
                    axis,
                    torch.empty_like(expected),
                )[0]
                self.assertTrue(torch.equal(actual, expected))
                array = golden.ascend_quant_v2_golden(
                    x.numpy(),
                    scale.numpy(),
                    offset.numpy(),
                    axis=axis,
                    output_dtypes=["int8"],
                )
                np.testing.assert_array_equal(array, expected.numpy())

    def test_square_shape_does_not_silently_scale_columns(self):
        actual = golden.aclnn_ascend_quant_v3_golden(
            torch.ones(2, 2),
            torch.tensor([2.0, 4.0]),
            None,
            False,
            "round",
            2,
            -2,
            torch.empty(2, 2, dtype=torch.int8),
        )[0]
        self.assertTrue(
            torch.equal(actual, torch.tensor([[2, 2], [4, 4]], dtype=torch.int8))
        )

    def test_sqrt_mode_and_broadcast_offset(self):
        actual = golden.aclnn_ascend_quant_v3_golden(
            torch.ones(2, 3),
            torch.tensor([2.0, 3.0]),
            torch.tensor([1.0, 2.0]),
            True,
            "round",
            2,
            0,
            torch.empty(2, 3, dtype=torch.int8),
        )[0]
        self.assertTrue(
            torch.equal(
                actual, torch.tensor([[5, 5, 5], [11, 11, 11]], dtype=torch.int8)
            )
        )

    def test_trunc_uses_toward_zero_for_both_signs(self):
        x = torch.tensor([[-1.9, -0.9, 0.9, 1.9]])
        actual = golden.aclnn_ascend_quant_v3_golden(
            x,
            torch.ones(1),
            None,
            False,
            b"trunc",
            2,
            -1,
            torch.empty(1, 4, dtype=torch.int8),
        )[0]
        self.assertTrue(
            torch.equal(actual, torch.tensor([[-1, 0, 0, 1]], dtype=torch.int8))
        )

    def test_last_axis_and_full_rank_scale_keep_semantics(self):
        x = torch.ones(2, 3)
        for scale in (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([[1.0, 2.0, 3.0]])):
            actual = golden.aclnn_ascend_quant_v3_golden(
                x,
                scale,
                None,
                False,
                "round",
                2,
                -1,
                torch.empty(2, 3, dtype=torch.int8),
            )[0]
            self.assertTrue(
                torch.equal(
                    actual, torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.int8)
                )
            )


if __name__ == "__main__":
    unittest.main()
