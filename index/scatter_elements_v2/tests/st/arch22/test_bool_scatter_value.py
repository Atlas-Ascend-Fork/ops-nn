# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A2/A3 device ST. Source CANN, then run from the repository root:

ASCEND_CUSTOM_OPP_PATH=/path/to/vendor SCATTER_OPAPI_LIBRARY=/path/to/libcust_opapi.so python -m unittest \
    index.scatter_elements_v2.tests.st.arch22.test_bool_scatter_value
"""

import csv
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from index.scatter_elements_v2.tests.st.arch35.test_scatter_elements_v2_scalar import (
    ScatterApi,
)


# Expected scatter task on stock CANN 9.1.0 (910B2). Shapes identify tasks in msprof.
# name, self shape, index shape, scatter axis, AICore
ROUTE_CASES = [
    ("47_rows", (47, 257), (47, 9), -1, False),
    ("48_rows", (48, 257), (48, 9), -1, True),
    ("scalar_out", (128, 67), (128, 23), -1, True),
    ("tensor_out", (128, 69), (128, 23), -1, True),
    ("tensor_inplace", (128, 71), (128, 23), -1, True),
    ("strided_index", (128, 73), (128, 27), -1, True),
    ("partial_rows", (128, 75), (64, 23), -1, True),
    ("deterministic", (128, 79), (128, 23), -1, True),
    ("bool_add", (128, 81), (128, 23), -1, False),
    ("first", (4096, 4096), (2048, 4096), 0, True),
    ("middle", (256, 512, 128), (256, 256, 128), 1, True),
    ("first_single_index", (64, 100000), (1, 100000), 0, True),
    ("first_multirow_wide", (64, 262144), (32, 262144), 0, True),
    ("first_wide", (512, 40000), (129, 40000), 0, True),
    ("middle_wide", (64, 8, 40000), (64, 4, 40000), 1, True),
]


class BoolScatterValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scatter = ScatterApi(
            os.environ.get("SCATTER_OPAPI_LIBRARY", "libopapi_nn.so")
        )

    def test_row_widths(self):
        for width, value in ((39999, False), (40000, True), (40001, False)):
            with self.subTest(width=width):
                source = torch.full((128, width), not value, dtype=torch.bool)
                # Different targets per row; exercise both endpoints and repeated writes.
                rows = torch.arange(128)
                index = torch.stack(
                    (rows, rows, torch.zeros_like(rows), width - 1 - rows), dim=1
                )
                expected = source.clone().scatter_(-1, index, value)
                actual = self.scatter(
                    source.npu(), -1, index.npu(), value, inplace=True
                )
                torch.testing.assert_close(actual.cpu(), expected)

    def test_first_axis_with_wide_suffix(self):
        for shape in ((8, 45000), (8, 200, 201)):
            with self.subTest(shape=shape):
                source = torch.zeros(shape, dtype=torch.bool)
                index = torch.full((4, *shape[1:]), 7, dtype=torch.int64)
                expected = source.clone().scatter_(0, index, True)
                actual = self.scatter(source.npu(), 0, index.npu(), True, inplace=True)
                torch.testing.assert_close(actual.cpu(), expected)

    def test_out_of_place_preserves_self(self):
        original = torch.zeros((128, 67), dtype=torch.bool)
        index = torch.tensor([0, 1, 66]).repeat(128, 1)
        for update in (True, torch.ones_like(index, dtype=torch.bool)):
            with self.subTest(tensor_update=isinstance(update, torch.Tensor)):
                source = original.npu()
                expected = original.clone().scatter_(1, index, update)
                device_update = (
                    update.npu() if isinstance(update, torch.Tensor) else update
                )
                out = self.scatter(source, 1, index.npu(), device_update)
                torch.testing.assert_close(out.cpu(), expected)
                torch.testing.assert_close(source.cpu(), original)

    def test_separate_views_in_one_allocation(self):
        storage = torch.zeros((2, 128, 67), dtype=torch.bool).npu()
        source, out = storage.unbind()
        index = torch.tensor([0, 66]).repeat(128, 1)
        expected = torch.zeros((128, 67), dtype=torch.bool).scatter_(1, index, True)
        device_index = index.npu()
        # Supply the sibling view where the ACLNN test helper allocates its output.
        with patch.object(torch, "empty_like", return_value=out):
            result = self.scatter(source, 1, device_index, True)
        self.assertIs(result, out)
        torch.testing.assert_close(out.cpu(), expected)
        self.assertFalse(source.cpu().any().item())

    def test_zero_dim_tensor_update(self):
        index = torch.arange(23).repeat(128, 1)
        device_index = index.npu()
        original = torch.zeros((128, 67), dtype=torch.bool)
        tensor_value = torch.tensor(True).npu()
        expected = original.clone().scatter_(1, index, True)
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                source = original.npu()
                api = "aclnnInplaceScatter" if inplace else "aclnnScatter"
                with self.assertRaisesRegex(
                    RuntimeError, rf"^{api}GetWorkspaceSize: 161002$"
                ):
                    self.scatter(source, 1, device_index, tensor_value, inplace=inplace)
                torch.testing.assert_close(source.cpu(), original)
                out = self.scatter(source, 1, device_index, True, inplace=inplace)
                torch.testing.assert_close(out.cpu(), expected)

    def test_scalar_broadcast_indices(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(index_dtype=dtype):
                for count in (16, 8192):
                    source = torch.ones((128, 259), dtype=torch.bool)
                    index = (
                        torch.arange(count, dtype=dtype).remainder(259).repeat(128, 1)
                    )
                    result = self.scatter(
                        source.npu(),
                        -1,
                        index.npu(),
                        False,
                        inplace=True,
                    )
                    torch.testing.assert_close(
                        result.cpu(), source.clone().scatter_(-1, index.long(), False)
                    )

    def test_dispatch_inputs(self):
        for name, shape, index_shape, axis, _ in ROUTE_CASES:
            with self.subTest(case=name):
                source = torch.zeros(shape, dtype=torch.bool)
                index = torch.randint(
                    shape[axis],
                    index_shape,
                    generator=torch.Generator().manual_seed(42),
                )
                device_index = index.npu()
                value, device_value = True, True
                if name.startswith("tensor_"):
                    # Unique destinations allow mixed BOOL updates to be checked element by element.
                    index = torch.arange(index_shape[-1]).repeat(index_shape[0], 1)
                    device_index = index.npu()
                    value = index.remainder(2).bool()
                    device_value = value.npu()
                elif name == "strided_index":
                    device_index = torch.stack((device_index, device_index), dim=-1)[
                        ..., 1
                    ]
                if name == "bool_add":
                    expected = source.clone().scatter_add_(
                        axis, index, torch.ones_like(index, dtype=torch.bool)
                    )
                else:
                    expected = source.clone().scatter_(axis, index, value)
                device_source = source.npu()
                inplace = not name.endswith("_out")
                reduction = "add" if name == "bool_add" else "none"
                previous = torch.are_deterministic_algorithms_enabled()
                try:
                    torch.use_deterministic_algorithms(name == "deterministic")
                    result = self.scatter(
                        device_source,
                        axis,
                        device_index,
                        device_value,
                        reduction,
                        inplace,
                    )
                    torch.testing.assert_close(result.cpu(), expected)
                    if not inplace:
                        torch.testing.assert_close(device_source.cpu(), source)
                finally:
                    torch.use_deterministic_algorithms(previous)
                    self.scatter.set_deterministic(previous)

    def test_dispatch_profile(self):
        # Numerical results alone also pass when every shape runs on AI_CPU.
        suite = "index.scatter_elements_v2.tests.st.arch22.test_bool_scatter_value.BoolScatterValueTest"
        methods = (
            "test_row_widths",
            "test_first_axis_with_wide_suffix",
            "test_dispatch_inputs",
            "test_scalar_broadcast_indices",
        )
        application = shlex.join(
            [
                sys.executable,
                "-m",
                "unittest",
                *(f"{suite}.{method}" for method in methods),
            ]
        )
        core, cpu = ("ScatterElementsV2", "MIX_AIV"), ("ScatterElements", "AI_CPU")
        expected = {
            "128,39999": core,
            "128,40000": cpu,
            "128,40001": cpu,
            "8,45000": cpu,
            "8,200,201": cpu,
            "128,259": core,
            **{
                ",".join(map(str, shape)): core if aicore else cpu
                for _, shape, _, _, aicore in ROUTE_CASES
            },
        }
        observed = set()
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                ["msprof", f"--output={directory}", f"--application={application}"],
                check=True,
            )
            for summary in Path(directory).rglob("op_summary_*.csv"):
                with summary.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        if row["OP Type"] not in (
                            "ScatterElements",
                            "ScatterElementsV2",
                        ):
                            continue
                        shape, index_shape, update_shape = (
                            row["Input Shapes"].strip('"').split(";")
                        )
                        self.assertIn(shape, expected)
                        self.assertEqual(
                            (row["OP Type"], row["Task Type"]), expected[shape]
                        )
                        if expected[shape] == core:
                            self.assertEqual(update_shape, index_shape, shape)
                        observed.add(shape)
        self.assertEqual(observed, set(expected))
