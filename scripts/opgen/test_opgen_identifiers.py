#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("opgen_standalone.py")


class OpgenIdentifierTest(unittest.TestCase):
    def test_invalid_names_do_not_create_projects(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("123foo", "_2foo", "___", "class", "float"):
                with self.subTest(name=name):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "-t",
                            "math",
                            "-n",
                            name,
                            "-p",
                            directory,
                        ],
                        capture_output=True,
                        text=True,
                        errors="replace",
                    )
                    self.assertNotEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertFalse((Path(directory) / "math" / name).exists())

    def test_valid_names_generate_both_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, variant in (("my_op_2", "default"), ("_my_op", "aicpu")):
                with self.subTest(name=name, variant=variant):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT),
                            "-t",
                            "math",
                            "-n",
                            name,
                            "-p",
                            directory,
                            "-v",
                            variant,
                        ],
                        capture_output=True,
                        text=True,
                        errors="replace",
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    generated = Path(directory) / "math" / name
                    if variant == "default":
                        kernel = generated / "op_kernel" / f"{name}.cpp"
                        self.assertIn(
                            f"void {name}(", kernel.read_text(encoding="utf-8")
                        )
                    else:
                        kernel = generated / "op_kernel_aicpu" / f"{name}_aicpu.cpp"
                        self.assertIn(
                            "MyOpCpuKernel::Compute", kernel.read_text(encoding="utf-8")
                        )


if __name__ == "__main__":
    unittest.main()
