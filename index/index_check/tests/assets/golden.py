# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy

__spec__ = {"index_check": "IndexCheckTestSpec"}


class IndexCheckTestSpec:
    """IndexCheck operator test spec (kernel flow).

    IndexCheck is a CANN native aclnn L0 auxiliary operator with NO output.
    It checks if indices are within valid bounds and asserts (aborts) if out of range.
    For each index tensor i with bound bounds[i]:
      - If idx < 0: idx += bounds[i] (negative-index wrapping)
      - Assert idx >= 0  (not too negative)
      - Assert idx < bounds[i]  (not exceeding upper bound)

    Golden returns [None] (single suppressed output). IndexCheck has no real
    output, so a dummy output ((1,) int32) is declared in the CSV and the
    golden returns [None] for it. In TTK's comparison module, when golden
    is None the per-output result is SUPPRESSED -> PASS, while keeping the
    outputs list non-empty so the comparison loop is entered instead of
    short-circuiting to UNKNOWN/FAIL (which happens when outputs is empty).

    customize_inputs ensures bounds are positive integers and clips indices
    to the valid range [-bound, bound-1] so that valid-case test cases never
    trigger the kernel's Trap().
    """

    def golden(bounds, indices_list, **kwargs):
        # IndexCheck has no real output. Return [None] to suppress the
        # per-output comparison (golden is None -> SUPPRESSED -> PASS),
        # while keeping the outputs list non-empty so the comparison loop
        # is entered instead of short-circuiting to UNKNOWN/FAIL.
        return [None]

    # -- third_party — 竞品参考实现（torch 语义：同样的越界断言无输出行为） --
    class ThirdPartyImpl:
        def __init__(self, **kwargs):
            pass

        def __call__(self, bounds, indices_list, **kwargs):
            # 与 golden 一致：无输出算子，返回 [None] 以抑制逐输出比对
            return [None]

    third_party = {"torch": ThirdPartyImpl}

    def customize_inputs(bounds, indices_list, **kwargs):
        bounds_arr = numpy.asarray(bounds, dtype=numpy.int64)
        n = len(indices_list)
        if bounds_arr.size != n:
            bounds_arr = numpy.full((n,), 100, dtype=numpy.int64)
        # Violating inputs pass through unchanged so the kernel Trap path stays
        # reachable for exception cases (non-positive bounds or out-of-range
        # indices must abort, not be silently repaired).
        if numpy.any(bounds_arr <= 0):
            return (bounds_arr, [numpy.asarray(t) for t in indices_list])
        for i, t in enumerate(indices_list):
            if i >= bounds_arr.size:
                break
            arr = numpy.asarray(t)
            b = int(bounds_arr[i])
            if numpy.any((arr < -b) | (arr >= b)):
                return (bounds_arr, [numpy.asarray(t) for t in indices_list])
        # Valid inputs: defensive clip (no-op for the archived case ranges).
        clipped = []
        for i, t in enumerate(indices_list):
            t = numpy.asarray(t)
            b = int(bounds_arr[i]) if i < bounds_arr.size else 100
            t = numpy.clip(t, -b, b - 1)
            clipped.append(t.astype(t.dtype))
        return (bounds_arr, clipped)

    tolerance = {
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }
