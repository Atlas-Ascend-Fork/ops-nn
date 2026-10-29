# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU integration test: bad-scale compute errors must fail the main suite.

Use the same environment as test_scalenz_input_contract.py. The invalid scale
is intercepted before submission, simulating execution/precision failures;
all other cases use real ACLNN capture/replay and the compatibility kernels.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import test_scalenz_input_contract as contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    results = []
    original_call = contract.DirectAclnn.call
    for kind in ("precision", "execution"):
        injected = []

        def failing_call(
            self,
            args,
            expected,
            fp4,
            ta,
            tb,
            weight_nz,
            capture=False,
            fault=None,
            scale_nz=True,
            check_descriptors=False,
        ):
            if (
                capture
                and scale_nz
                and weight_nz
                and args[2].shape[-2] != expected.shape[-1]
            ):
                injected.append(kind)
                if kind == "precision":
                    raise AssertionError(
                        "INJECTED_GRAPH_PRECISION_FAILURE: Tensor-likes are not close!"
                    )
                raise RuntimeError(
                    "INJECTED_GRAPH_EXECUTION_FAILURE: ACLNN Graph runtime failed: 161002"
                )
            return original_call(
                self,
                args,
                expected,
                fp4,
                ta,
                tb,
                weight_nz,
                capture=capture,
                fault=fault,
                scale_nz=scale_nz,
                check_descriptors=check_descriptors,
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "records.json"
            argv = [
                "test_scalenz_input_contract.py",
                "--mode",
                "aclnn_graph",
                "--smoke",
                "--output",
                str(output),
            ]
            exit_code = 0
            with (
                patch.object(sys, "argv", argv),
                patch.object(contract.DirectAclnn, "call", failing_call),
            ):
                try:
                    contract.main()
                except SystemExit as error:
                    exit_code = error.code
            rows = json.loads(output.read_text())["records"]
        failures = [row for row in rows if "INJECTED_GRAPH_" in row.get("error", "")]
        assert len(injected) == len(failures) == 2, (kind, injected, failures)
        assert exit_code == 1, (kind, exit_code)
        assert all(row["status"] == "error" for row in failures), failures
        assert not any(
            row["status"] not in ("pass", "expected_reject", "skipped_existing_limit")
            for row in rows
            if row not in failures
        ), rows
        results.append(
            dict(
                kind=kind,
                suite_exit=exit_code,
                injected_errors=len(failures),
                counts=dict(Counter(row["status"] for row in rows)),
                failures=failures,
            )
        )
    if options.output:
        options.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"failure_propagation": results}), flush=True)


if __name__ == "__main__":
    main()
