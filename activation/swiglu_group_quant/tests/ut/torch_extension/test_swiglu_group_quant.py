# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import unittest

import torch


class TestSwigluGroupQuant(unittest.TestCase):
    def test_dynamic_hifloat8_pair_scale_has_one_value_per_group(self):
        from cann_ops_nn import swiglu_group_quant

        x = torch.empty((8, 256), device="meta", dtype=torch.float16)
        pairs = torch.empty((3, 2), device="meta", dtype=torch.int64)
        _, scale, _ = swiglu_group_quant(
            x, group_index=pairs, quant_mode=3, group_list_type=2
        )

        self.assertEqual(scale.shape, (3,))
        self.assertEqual(scale.dtype, torch.float32)

    def test_mx_accepts_noncontiguous_pairs(self):
        from cann_ops_nn import swiglu_group_quant

        if not torch.npu.is_available():
            self.skipTest("NPU is unavailable")

        x = torch.linspace(-1, 1, 8 * 256, dtype=torch.float16).reshape(8, 256).npu()
        pairs = torch.tensor([[19, 3, 7], [1, 2, 5]], dtype=torch.int64).npu().t()
        self.assertFalse(pairs.is_contiguous())
        options = dict(
            quant_mode=1, round_scale=True, output_origin=True, group_list_type=2
        )
        expected = swiglu_group_quant(x, group_index=pairs.contiguous(), **options)
        actual = swiglu_group_quant(x, group_index=pairs, **options)

        for result, reference in zip(actual, expected):
            torch.testing.assert_close(
                result.cpu().view(torch.uint8),
                reference.cpu().view(torch.uint8),
                rtol=0,
                atol=0,
            )

    def test_pairs_backward_matches_counts_after_sum(self):
        from cann_ops_nn import swiglu_group_quant

        if not torch.npu.is_available():
            self.skipTest("NPU is unavailable")

        x = torch.ones((8, 256), dtype=torch.float16).npu().requires_grad_()
        weight = torch.ones((8, 1), dtype=torch.float32).npu().requires_grad_()
        counts = torch.tensor([1, 0, 7], dtype=torch.int64).npu()
        pairs = torch.tensor([[19, 1], [3, 0], [7, 7]], dtype=torch.int64).npu()
        _, _, count_origin = swiglu_group_quant(x, weight, counts, output_origin=True)
        _, _, pair_origin = swiglu_group_quant(
            x, weight, pairs, output_origin=True, group_list_type=2
        )

        count_grads = torch.autograd.grad(count_origin.sum(), (x, weight))
        pair_grads = torch.autograd.grad(pair_origin.sum(), (x, weight))
        for pair_grad, count_grad in zip(pair_grads, count_grads):
            torch.testing.assert_close(
                pair_grad.cpu(), count_grad.cpu(), rtol=0, atol=0
            )
