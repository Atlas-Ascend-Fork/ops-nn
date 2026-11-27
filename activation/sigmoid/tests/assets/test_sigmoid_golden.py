# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


def load_golden(path):
    spec = importlib.util.spec_from_file_location("sigmoid_golden_reference", path)
    module = importlib.util.module_from_spec(spec)
    # Only the ACLNN references need torch; the kernel reference uses NumPy.
    with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
        spec.loader.exec_module(module)
    return module.sigmoid_golden


class SigmoidGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[4]
        cls.references = [
            (path, load_golden(repo / path))
            for path in (
                "activation/sigmoid/tests/assets/golden.py",
                "experimental/activation/sigmoid/tests/assets/golden.py",
            )
        ]

    def test_float16_negative_tail_stays_nonzero(self):
        x = np.array([-17, -16, -15, -14, -13, -12, -11], dtype=np.float16)
        expected = np.array(
            [1 / (1 + math.exp(-float(v))) for v in x], dtype=np.float16
        )
        for path, golden in self.references:
            with self.subTest(path=path), np.errstate(over="raise", invalid="raise"):
                result = golden(x)
                self.assertEqual(result.dtype, x.dtype)
                self.assertTrue(np.all(result > 0))
                np.testing.assert_array_equal(result, expected)

    def test_float16_matrix_and_special_values(self):
        x = np.array([[-12, -2, 0], [2, np.inf, -np.inf]], dtype=np.float16)
        expected = np.array(
            [
                [1 / (1 + math.exp(12)), 1 / (1 + math.exp(2)), 0.5],
                [1 / (1 + math.exp(-2)), 1, 0],
            ],
            dtype=np.float16,
        )
        for path, golden in self.references:
            with self.subTest(path=path):
                result = golden(x)
                self.assertEqual(result.shape, x.shape)
                self.assertEqual(result.dtype, x.dtype)
                np.testing.assert_array_equal(result, expected)

    def test_float32_and_scalar_inputs(self):
        for path, golden in self.references:
            for dtype in (np.float16, np.float32):
                with self.subTest(path=path, dtype=dtype):
                    x = np.array(-12, dtype=dtype)
                    result = golden(x)
                    self.assertIsInstance(result, np.ndarray)
                    self.assertEqual(result.shape, ())
                    self.assertEqual(result.dtype, x.dtype)
                    self.assertAlmostEqual(
                        float(result), 1 / (1 + math.exp(12)), delta=1e-8
                    )

    @unittest.skipIf(ml_dtypes is None, "ml_dtypes is not installed")
    def test_bfloat16_output_dtype(self):
        x = np.array([-20, -12, 0, 12, 20], dtype=ml_dtypes.bfloat16)
        expected = np.array([1 / (1 + math.exp(-float(v))) for v in x], dtype=x.dtype)
        for path, golden in self.references:
            with self.subTest(path=path):
                result = golden(x)
                self.assertEqual(result.dtype, x.dtype)
                np.testing.assert_array_equal(result, expected)

    def test_float16_extreme_values(self):
        x = np.array([-20, -12, 12, 20], dtype=np.float16)
        expected = np.array([1 / (1 + math.exp(-float(v))) for v in x], dtype=x.dtype)
        for path, golden in self.references:
            with self.subTest(path=path):
                result = golden(x)
                self.assertEqual(result.dtype, x.dtype)
                np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
