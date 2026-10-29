# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Parameter-rejection classification for the ScaleNZ system tests."""

import re


ACLNN_ERR_PARAM_INVALID = 161002
TORCH_WEIGHT_NZ_GUARD = (
    "ScaleNZ requires x2 in WeightNZ format; ND x2 is not supported."
)


class AclnnInputRejected(RuntimeError):
    """GetWorkspaceSize rejected an input, before allocation/capture/execution."""

    def __init__(self, api_name, status):
        self.api_name = api_name
        self.status = status
        super().__init__(f"{api_name}GetWorkspaceSize rejected input: {status}")


def is_input_rejection(error, allow_weight_guard=False):
    """Accept known parameter checks; execution and precision errors must fail."""
    if isinstance(error, AclnnInputRejected):
        return error.status == ACLNN_ERR_PARAM_INVALID
    if not isinstance(error, RuntimeError):
        return False
    message = str(error)
    if allow_weight_guard and TORCH_WEIGHT_NZ_GUARD in message:
        return True
    # Torch wraps both query and execution failures in RuntimeError. Require
    # the invalid-parameter code and a diagnostic naming the workspace query.
    for api_name in ("aclnnQuantMatmulWeightNz", "aclnnQuantMatmulV5"):
        if f"{api_name}GetWorkspaceSize" in message and re.search(
            rf"call {api_name} failed, error code is {ACLNN_ERR_PARAM_INVALID}\b",
            message,
        ):
            return True
    return False
