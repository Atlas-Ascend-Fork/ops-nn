# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU-only Torch schema/Meta tests; substitute the JIT loader, not the Torch dispatcher."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch


class TestRotateQuantTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = torch.library.Library("rotate_quant_torch_test", "DEF")
        library = cls.library

        class TestOpBuilder:
            def __init__(self, name):
                self.name = name

            def resolve_source(self, filename):
                return "csrc/" + filename

            def _ensure_initialized(self):
                library.define(self.schema())
                self.register_meta()

            def load(self):
                raise AssertionError("Meta dispatch must not load the NPU extension")

        builder_module = types.ModuleType("cann_ops_nn.op_builder")
        builder_module.OpBuilder = TestOpBuilder
        builder_module.get_as_library = lambda: library
        source = (
            Path(__file__).resolve().parents[3] / "torch_extension" / "rotate_quant.py"
        )
        spec = importlib.util.spec_from_file_location("rotate_quant_under_test", source)
        cls.binding = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "cann_ops_nn": types.ModuleType("cann_ops_nn"),
                "cann_ops_nn.op_builder": builder_module,
            },
        ):
            spec.loader.exec_module(cls.binding)
        cls.op = torch.ops.rotate_quant_torch_test.rotate_quant

    def tensor(self, shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="meta")

    def test_ordinary_mx_shapes_and_packing(self):
        for k in (32, 64, 128, 256):
            for rank in range(1, 8):
                shape = (2,) * (rank - 1) + (k * 3,)
                for segmented in (False, True):
                    rotation = self.tensor((3, k, k) if segmented else (k, k))
                    for dtype, expected_dtype, pack in (
                        (296, torch.uint8, 2),
                        (291, torch.float8_e5m2, 1),
                        (292, torch.float8_e4m3fn, 1),
                    ):
                        with self.subTest(
                            k=k, rank=rank, segmented=segmented, dtype=dtype
                        ):
                            y, scale = self.op(
                                self.tensor(shape),
                                rotation,
                                dst_dtype=dtype,
                                axis=rank - 1,
                            )
                            self.assertEqual(
                                tuple(y.shape), shape[:-1] + (shape[-1] // pack,)
                            )
                            self.assertEqual(y.dtype, expected_dtype)
                            self.assertEqual(
                                tuple(scale.shape),
                                shape[:-1] + ((shape[-1] + 63) // 64, 2),
                            )
                            self.assertEqual(scale.dtype, torch.uint8)
                            self.assertEqual(y.device.type, "meta")

    def test_grouped_shapes_count_cumsum_and_group_limit(self):
        # Meta tensors have no readable values: list metadata must suffice, including partial coverage.
        for e in (1, 4, 1024):
            for m in (0, 640):
                for group_type in (0, 1):
                    for rot_shape in ((e, 128, 128), (e, 32, 128, 128)):
                        for dtype in (296, 291, 292):
                            with self.subTest(
                                e=e,
                                m=m,
                                group_type=group_type,
                                rot_shape=rot_shape,
                                dtype=dtype,
                            ):
                                y, scale = self.op(
                                    self.tensor((m, 4096)),
                                    self.tensor(rot_shape),
                                    dst_dtype=dtype,
                                    group_list=self.tensor((e,), torch.int64),
                                    group_list_type=group_type,
                                )
                                self.assertEqual(
                                    tuple(y.shape), (m, 2048 if dtype == 296 else 4096)
                                )
                                self.assertEqual(tuple(scale.shape), (m, 64, 2))

    def test_int_outputs_defaults_and_dtype_objects(self):
        x, rotation = (
            self.tensor((8, 512), torch.float16),
            self.tensor((64, 64), torch.float16),
        )
        for kwargs, shape, dtype in (
            ({}, (8, 512), torch.int8),
            ({"dst_dtype": torch.int8}, (8, 512), torch.int8),
            ({"dst_dtype": torch.quint4x2}, (8, 64), torch.int32),
        ):
            y, scale = self.op(x, rotation, **kwargs)
            self.assertEqual(tuple(y.shape), shape)
            self.assertEqual(y.dtype, dtype)
            self.assertEqual(tuple(scale.shape), (8,))
            self.assertEqual(scale.dtype, torch.float32)
        y, _ = self.op(
            x,
            rotation,
            dst_dtype=None,
            axis=None,
            round_mode=None,
            scale_alg=None,
            dst_type_max=None,
            transpose_y=None,
        )
        self.assertEqual(y.dtype, torch.int8)
        for dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            y, _ = self.op(x, rotation, dst_dtype=dtype)
            self.assertEqual(y.dtype, dtype)

    def test_empty_and_noncontiguous_inputs(self):
        for shape in ((0, 128), (8, 0)):
            for dtype in (296, 291, 1, 16):
                y, _ = self.op(
                    self.tensor(shape), self.tensor((64, 64)), dst_dtype=dtype
                )
                self.assertEqual(y.numel(), 0)
        x = self.tensor((128, 4)).transpose(0, 1)
        rotation = self.tensor((64, 64)).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        y, scale = self.op(x, rotation, dst_dtype=296)
        self.assertEqual(tuple(y.shape), (4, 64))
        self.assertEqual(tuple(scale.shape), (4, 2, 2))

    def test_quantization_attributes(self):
        x, rotation = self.tensor((2, 128)), self.tensor((64, 64))
        for mode in ("rint", "round", "floor"):
            for max_value in (6.0, 12.0):
                self.op(
                    x,
                    rotation,
                    dst_dtype=296,
                    scale_alg=2,
                    dst_type_max=max_value,
                    round_mode=mode,
                    alpha=self.tensor((1,)),
                )
        for dtype in (291, 292):
            self.op(x, rotation, dst_dtype=dtype, scale_alg=1)
        for kwargs in (
            {"axis": 0},
            {"transpose_y": True},
            {"round_mode": "trunc"},
            {"dst_dtype": 296, "scale_alg": 1},
            {"dst_dtype": 296, "scale_alg": 2},
            {"dst_dtype": 291, "round_mode": "floor"},
            {"dst_dtype": 292, "scale_alg": 2},
            {"dst_type_max": 6.0},
            {"dst_dtype": 0},
            {"group_list_type": 2},
            {"alpha": self.tensor((1,))},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.op(x, rotation, **kwargs)

    def test_invalid_group_and_rotation_metadata(self):
        x = self.tensor((640, 4096))
        group = self.tensor((4,), torch.int64)
        valid_rotation = self.tensor((4, 128, 128))
        cases = (
            (x, valid_rotation, self.tensor((0,), torch.int64)),
            (x, self.tensor((1025, 128, 128)), self.tensor((1025,), torch.int64)),
            (x, valid_rotation, self.tensor((4,), torch.int32)),
            (x, valid_rotation, self.tensor((4, 1), torch.int64)),
            (x, self.tensor((3, 128, 128)), group),
            (x, self.tensor((4, 31, 128, 128)), group),
            (x, self.tensor((4, 64, 128)), group),
            (self.tensor((2, 320, 4096)), valid_rotation, group),
            (self.tensor((640, 0)), valid_rotation, group),
            (self.tensor((640, 4097)), valid_rotation, group),
        )
        for input_x, rotation, groups in cases:
            with (
                self.subTest(x=input_x.shape, rot=rotation.shape, groups=groups.shape),
                self.assertRaises(RuntimeError),
            ):
                self.op(input_x, rotation, dst_dtype=296, group_list=groups)
        with self.assertRaises(RuntimeError):
            self.op(x, valid_rotation, group_list=group)
        with self.assertRaises(RuntimeError):
            self.op(x, self.tensor((31, 128, 128)), dst_dtype=296)

    def test_privateuse1_forwards_optional_inputs_without_reading_values(self):
        x = self.tensor((640, 4096))
        rotation = self.tensor((4, 32, 128, 128))
        alpha, groups = self.tensor((1,)), self.tensor((4,), torch.int64)
        native = Mock()
        native.rotate_quant.return_value = ("y", "scale")
        with patch.object(
            self.binding.rotate_quant_builder, "load", return_value=native
        ):
            result = self.binding.rotate_quant(
                x,
                rotation,
                alpha=alpha,
                dst_dtype=296,
                axis=1,
                round_mode="round",
                scale_alg=2,
                dst_type_max=8.0,
                group_list=groups,
                group_list_type=1,
            )
        self.assertEqual(result, ("y", "scale"))
        native.rotate_quant.assert_called_once_with(
            x, rotation, alpha, 296, 1, "round", 2, 8.0, False, groups, 1
        )


if __name__ == "__main__":
    unittest.main()
