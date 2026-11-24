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

import torch

spec = importlib.util.spec_from_file_location(
    "quant_golden", Path(__file__).resolve().parents[1] / "assets" / "golden.py"
)
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)


class QuantizeGoldenTest(unittest.TestCase):
    def run_golden(self, values, dtype):
        x = torch.tensor([values], dtype=torch.float32)
        out = torch.empty_like(x, dtype=dtype)
        return golden.aclnn_quantize_golden(
            x,
            torch.ones(1),
            None,
            {torch.int8: 2, torch.uint8: 4, torch.int32: 3}[dtype],
            -1,
            out,
        )[0]

    def test_uint8_uses_unsigned_range(self):
        actual = self.run_golden([-2, 0, 127, 128, 254.6, 300], torch.uint8)
        expected = torch.tensor([[0, 0, 127, 128, 255, 255]], dtype=torch.uint8)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_int32_preserves_values_outside_int8(self):
        actual = self.run_golden([-300, 128, 1024], torch.int32)
        expected = torch.tensor([[-300, 128, 1024]], dtype=torch.int32)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_int32_saturates_without_float32_endpoint_overflow(self):
        values = [-3000000000.0, -2147483648.0, 2147483648.0, 3000000000.0]
        actual = self.run_golden(values, torch.int32)
        expected = torch.tensor(
            [[-2147483648, -2147483648, 2147483647, 2147483647]], dtype=torch.int32
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_int8_keeps_round_to_even_and_saturation(self):
        actual = self.run_golden([-200, -1.5, -0.5, 0.5, 1.5, 200], torch.int8)
        expected = torch.tensor([[-128, -2, 0, 0, 2, 127]], dtype=torch.int8)
        self.assertTrue(torch.equal(actual, expected))

    def test_uint8_per_channel_axis(self):
        x = torch.tensor([[20.0, 40.0, 60.0], [100.0, 200.0, 300.0]])
        out = torch.empty_like(x, dtype=torch.uint8)
        actual = golden.aclnn_quantize_golden(
            x, torch.tensor([2.0, 10.0]), torch.tensor([1.0, 2.0]), 4, 0, out
        )[0]
        self.assertTrue(
            torch.equal(
                actual, torch.tensor([[11, 21, 31], [12, 22, 32]], dtype=torch.uint8)
            )
        )


if __name__ == "__main__":
    unittest.main()
