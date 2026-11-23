# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Command failures must reach the installer and kernel UT callers unchanged."""

from pathlib import Path
import re
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")


def write_script(path, text):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


@pytest.mark.parametrize("exit_code", [0, 1, 2, 42, 127])
def test_installer_preserves_command_exit_status(tmp_path, exit_code):
    source = (REPO / "install_deps.sh").read_text(encoding="utf-8")
    function = re.search(r"(?ms)^run_command\(\) \{\n.*?^\}", source)
    assert function is not None
    script = tmp_path / "installer-command.sh"
    write_script(
        script,
        "set -euo pipefail\n"
        + function.group()
        + '\nrun_command "$@"\nprintf "INSTALL_CONTINUED\\n"\n',
    )
    result = subprocess.run(
        [
            BASH,
            script.as_posix(),
            BASH,
            "-c",
            'printf "stdout: %s\\n" "$1"; printf "stderr: %s\\n" "$1" >&2; exit "$2"',
            "_",
            "argument with spaces",
            str(exit_code),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert ("INSTALL_CONTINUED" in result.stdout) == (exit_code == 0)
    if exit_code:
        assert f"Exit code: {exit_code}" in result.stdout
        assert "stdout: argument with spaces" in result.stdout
        assert "stderr: argument with spaces" in result.stdout


@pytest.mark.parametrize("exit_code", [0, 1, 42, 200])
def test_kernel_ut_preserves_status_through_tee(tmp_path, exit_code):
    source = (REPO / ".gitcode/scripts/ut.sh").read_text(encoding="utf-8")
    branch = re.search(r"(?ms)^\s*opkernel\)\n(.*?)^\s*;;", source)
    assert branch is not None
    checker = tmp_path / "scripts/ci/check_kernel_ut.sh"
    checker.parent.mkdir(parents=True)
    write_script(
        checker,
        f'printf "kernel UT output\\n"\nexit {exit_code}\n',
    )
    script = tmp_path / "kernel-ut.sh"
    write_script(
        script,
        "set +e\nrepo_name=ops-nn\n"
        + branch.group(1)
        + '\nprintf "UT_STATUS=%s\\n" "$ret"\n',
    )
    result = subprocess.run(
        [BASH, script.as_posix()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"UT_STATUS={exit_code}" in result.stdout
    assert "kernel UT output" in result.stdout
    assert (tmp_path / "output.txt").read_text() == "kernel UT output\n"
