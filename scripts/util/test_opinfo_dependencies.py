#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    shutil.which("cmake") and shutil.which("ninja"), "CMake and Ninja required"
)
class OpInfoDependenciesTest(unittest.TestCase):
    def test_incremental_updates_and_optional_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            util = root / "scripts" / "util"
            binary = root / "scripts" / "kernel" / "binary_script"
            util.mkdir(parents=True)
            binary.mkdir(parents=True)
            shutil.copyfile(
                ROOT / "scripts/util/parse_ini_to_json.py",
                util / "parse_ini_to_json.py",
            )
            shutil.copyfile(
                ROOT / "scripts/kernel/binary_script/merge_ini_files.py",
                binary / "merge_ini_files.py",
            )
            source = (ROOT / "cmake/gen_ops_info.cmake").read_text(encoding="utf-8")
            if os.name == "nt":
                # The production commands assume a Unix shell; adapt only those commands for this host.
                source = source.replace(
                    "COMMAND mkdir -p ${opinfo_file_path}",
                    "COMMAND ${CMAKE_COMMAND} -E make_directory ${opinfo_file_path}",
                ).replace(
                    "COMMAND touch ${MGINI_OPS_INFO_DIR}/aic-merged-${MGINI_COMPUTE_UNIT}-ops-info.ini",
                    "COMMAND ${CMAKE_COMMAND} -E touch ${MGINI_OPS_INFO_DIR}/aic-merged-${MGINI_COMPUTE_UNIT}-ops-info.ini",
                )
            (root / "gen_ops_info.cmake").write_text(source, encoding="utf-8")
            autogen = root / "autogen"
            (autogen / "inner").mkdir(parents=True)
            (autogen / "exc").mkdir()
            ini = autogen / "aic-test-ops-info.ini"
            ini.write_text(
                "[Example]\noutput0.name=one\noutput0.paramType=required\n",
                encoding="utf-8",
            )
            (root / "CMakeLists.txt").write_text(
                f'''cmake_minimum_required(VERSION 3.16)
project(opinfo NONE)
set(ASCEND_PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}")
set(OPS_KERNEL_BINARY_SCRIPT "{binary.as_posix()}")
set(ASCEND_KERNEL_CONF_DST "${{CMAKE_BINARY_DIR}}/conf")
file(MAKE_DIRECTORY "${{ASCEND_KERNEL_CONF_DST}}")
include("{(root / "gen_ops_info.cmake").as_posix()}")
add_ops_info_target_v1(TARGET opinfo OPS_INFO_DIR "{autogen.as_posix()}" COMPUTE_UNIT test OUTPUT "${{CMAKE_BINARY_DIR}}/out/ops.json")
merge_ini_files(TARGET mergeinfo OPS_INFO_DIR "{autogen.as_posix()}" COMPUTE_UNIT test)
''',
                encoding="utf-8",
            )
            build = root / "build"

            def run(*args):
                result = subprocess.run(
                    args, capture_output=True, text=True, errors="replace"
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout + result.stderr

            run("cmake", "-S", str(root), "-B", str(build), "-G", "Ninja")
            run("cmake", "--build", str(build), "--target", "opinfo", "mergeinfo")
            output = build / "out/ops.json"
            merged = build / "conf/aic-test-ops-info.ini"
            self.assertEqual(
                json.loads(output.read_text())["Example"]["output0"]["name"], "one"
            )
            self.assertNotIn(
                "Generating",
                run("cmake", "--build", str(build), "--target", "opinfo", "mergeinfo"),
            )
            time.sleep(1.1)
            ini.write_text(
                "[Example]\noutput0.name=two\noutput0.paramType=required\n",
                encoding="utf-8",
            )
            run("cmake", "--build", str(build), "--target", "opinfo", "mergeinfo")
            with self.subTest(target="json", event="modify"):
                self.assertEqual(
                    json.loads(output.read_text())["Example"]["output0"]["name"], "two"
                )
            with self.subTest(target="ini", event="modify"):
                self.assertIn("output0.name=two", merged.read_text())
            optional = autogen / "inner/aic-test-ops-info.ini"
            optional.write_text(
                "[Other]\noutput0.name=z\noutput0.paramType=required\n",
                encoding="utf-8",
            )
            run("cmake", "--build", str(build), "--target", "opinfo", "mergeinfo")
            with self.subTest(target="json", event="add"):
                self.assertIn("Other", json.loads(output.read_text()))
            with self.subTest(target="ini", event="add"):
                self.assertIn("[Other]", merged.read_text())
            optional.unlink()
            log = run("cmake", "--build", str(build), "--target", "opinfo", "mergeinfo")
            with self.subTest(target="json", event="remove"):
                self.assertIn("Generating out/ops.json", log)
            with self.subTest(target="ini", event="remove"):
                self.assertNotIn("[Other]", merged.read_text())
            # A separate known issue may leave old JSON trailing bytes after a shorter rewrite.


if __name__ == "__main__":
    unittest.main()
