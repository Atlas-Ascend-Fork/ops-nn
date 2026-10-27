#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Generate deterministic TTK inputs covering AR/RA reduction problem shapes.

Run from the task directory, with --output pointing into its MyTestCase checkout.
The CSV contains absolute paths to the generated inputs; no golden is generated
here. TTK loads the operator's golden.py when executing the CSV.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def generate(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    routes = (
        ("ar", "NCDHW", (2, 1, 1, 1, 32768)),
        ("split_r", "NDHWC", (1, 1, 1, 65536, 16)),
        ("ra", "NDHWC", (1, 1, 1, 8192, 513)),
    )
    for route, fmt, shape in routes:
        channels = shape[1] if fmt == "NCDHW" else shape[-1]
        for dtype in ("float32", "float16"):
            for mode in ("posinf", "neginf", "mixedinf", "oneinf", "nan", "finite"):
                name = f"welford_{route}_{dtype}_{mode}"
                data_dir = output / name
                data_dir.mkdir(exist_ok=True)
                x = np.ones(shape, dtype=dtype)
                channel_rows = x.reshape(-1, channels)
                if mode in ("posinf", "neginf", "nan"):
                    x.fill({"posinf": np.inf, "neginf": -np.inf, "nan": np.nan}[mode])
                elif mode == "mixedinf":
                    channel_rows[::2] = np.inf
                    channel_rows[1::2] = -np.inf
                elif mode == "oneinf":
                    channel_rows[0] = np.inf
                else:
                    channel_rows[1::2] = 2
                inputs = (
                    x,
                    np.ones(channels, dtype="float32"),
                    np.zeros(channels, dtype="float32"),
                )
                paths = []
                for index, value in enumerate(inputs):
                    path = data_dir / f"input_{index}.npy"
                    np.save(path, value)
                    paths.append(str(path))
                cases.append(
                    {
                        "testcase_name": name,
                        "op_name": "batch_norm3_d",
                        "input_shapes": repr(
                            (shape, (channels,), (channels,), None, None)
                        ),
                        "input_dtypes": repr((dtype, "float32", "float32", None, None)),
                        "input_formats": repr((fmt, "ND", "ND", None, None)),
                        "output_shapes": repr((shape,) + ((channels,),) * 4),
                        "output_dtypes": repr((dtype,) + ("float32",) * 4),
                        "output_formats": repr((fmt,) + ("ND",) * 4),
                        "attributes": repr(
                            {"epsilon": 0.0001, "data_format": fmt, "is_training": True}
                        ),
                        "manual_input_binaries": repr(tuple(paths) + (None, None)),
                        "is_enabled": "True",
                    }
                )
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cases[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    return output / "cases.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.output))
