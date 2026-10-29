# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run with unittest discovery; no Torch or NPU runtime is required."""

import unittest
from scalenz_test_utils import (
    AclnnInputRejected,
    TORCH_WEIGHT_NZ_GUARD,
    is_input_rejection,
)


class RejectionClassificationTest(unittest.TestCase):
    def test_query_parameter_error(self):
        self.assertTrue(
            is_input_rejection(AclnnInputRejected("aclnnQuantMatmulWeightNz", 161002))
        )

    def test_query_runtime_error(self):
        self.assertFalse(
            is_input_rejection(AclnnInputRejected("aclnnQuantMatmulWeightNz", 561000))
        )

    def test_torch_query_diagnostic(self):
        error = RuntimeError(
            "call aclnnQuantMatmulWeightNz failed, error code is 161002; "
            "aclnnQuantMatmulWeightNzGetWorkspaceSize: invalid scale shape"
        )
        self.assertTrue(is_input_rejection(error))

    def test_torch_execution_error(self):
        self.assertFalse(
            is_input_rejection(
                RuntimeError(
                    "call aclnnQuantMatmulWeightNz failed, error code is 161002"
                )
            )
        )

    def test_missing_binary(self):
        self.assertFalse(
            is_input_rejection(
                RuntimeError(
                    "call aclnnQuantMatmulWeightNz failed, error code is 561000; Cannot find binary"
                )
            )
        )

    def test_capture_or_replay_error(self):
        self.assertFalse(
            is_input_rejection(RuntimeError("ACLNN Graph runtime failed: 161002"))
        )

    def test_graph_precision_assertion(self):
        self.assertFalse(
            is_input_rejection(AssertionError("Tensor-likes are not close!"))
        )

    def test_assertion_cannot_impersonate_query(self):
        self.assertFalse(
            is_input_rejection(
                AssertionError(
                    "call aclnnQuantMatmulWeightNz failed, error code is 161002; "
                    "aclnnQuantMatmulWeightNzGetWorkspaceSize"
                )
            )
        )

    def test_weight_guard_is_case_specific(self):
        error = RuntimeError(TORCH_WEIGHT_NZ_GUARD)
        self.assertFalse(is_input_rejection(error))
        self.assertTrue(is_input_rejection(error, allow_weight_guard=True))

    def test_unrelated_failure(self):
        self.assertFalse(is_input_rejection(RuntimeError("out of memory")))


if __name__ == "__main__":
    unittest.main()
