# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("cmake"), "requires CMake")
class OpInfoCMakeTest(unittest.TestCase):
    def test_parser_result_and_fallback(self):
        source = (ROOT / "cmake/ut.cmake").read_text(encoding="utf8")
        start = source.index(
            "                    execute_process(\n",
            source.index("if (EXISTS ${opsInfo})"),
        )
        end = source.index(
            '                    message(STATUS "get_opfile_from_opsinfo', start
        )
        section = source[start:end]
        for name, content, expected, success in (
            (
                "valid",
                json.dumps({"MyOp": {"opFile": {"value": "special"}}}),
                "special",
                True,
            ),
            ("empty", "{}", "MyOp", True),
            ("invalid", "{", "", False),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                info = root / "info.json"
                info.write_text(content, encoding="utf8")
                text = (
                    f'set(ASCEND_PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
                    f'set(CMAKE_SOURCE_DIR "{ROOT.as_posix()}")\n'
                    f'set(opsInfo "{info.as_posix()}")\nset(opType MyOp)\n'
                    + section
                    + '\nif("${kernel_cpp_name}" STREQUAL "")\n set(kernel_cpp_name MyOp)\nendif()\n'
                    + 'message(STATUS "NAME=${kernel_cpp_name}")\n'
                )
                script = root / "test.cmake"
                script.write_text(text, encoding="utf8")
                result = subprocess.run(
                    ["cmake", "-P", str(script)], capture_output=True, text=True
                )
                self.assertEqual(result.returncode == 0, success, result.stderr)
                if success:
                    self.assertIn("NAME=" + expected, result.stdout)
                else:
                    self.assertIn("Failed to read operator info", result.stderr)


if __name__ == "__main__":
    unittest.main()
