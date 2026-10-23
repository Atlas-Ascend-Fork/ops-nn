#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate frozen TTK inputs; use tests/assets/golden.py for all references."""

import argparse
import csv
from pathlib import Path

import ml_dtypes
import numpy as np


def generate(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def append(name, array, fmt="NCDHW"):
        path = output_dir / f"{name}.npy"
        np.save(path, array)
        output_shape = (
            (1, 1, array.shape[2], 1, 1, array.shape[5])
            if fmt == "NDC1HWC0"
            else (array.shape[1],)
        )
        rows.append(
            {
                "testcase_name": name,
                "op_name": "bn3d_training_reduce",
                "input_shapes": repr((array.shape,)),
                "input_dtypes": repr((str(array.dtype),)),
                "input_formats": repr((fmt,)),
                "input_ori_shapes": repr((array.shape,)),
                "input_ori_formats": repr((fmt,)),
                "output_shapes": repr((output_shape, output_shape)),
                "output_dtypes": repr(("float32", "float32")),
                "output_formats": repr((fmt, fmt)),
                "output_ori_shapes": repr((output_shape, output_shape)),
                "output_ori_formats": repr((fmt, fmt)),
                "attributes": "{}",
                "manual_input_binaries": repr((str(path.resolve()),)),
                "manual_golden_binaries": "()",
                "remark": "Frozen overflow/cancellation and finite controls; repository golden and default comparison.",
            }
        )

    shapes = {
        "r0_one": (4, 64, 1),
        "fused": (2, 64, 2),
        "short": (1, 1, 256),
        "full_r": (1025, 8, 64),
        "sub_r": (1, 8, 65537),
        "split": (1, 1, 1048577),
    }
    for dtype in (np.float32, ml_dtypes.bfloat16):
        maximum = (
            float(np.finfo(np.float32).max)
            if dtype == np.float32
            else float(ml_dtypes.finfo(dtype).max)
        )
        for route, (n, c, r) in shapes.items():
            flat = np.zeros((c, n * r), dtype=dtype)
            positions = (0, 2, 3) if n * r == 4 else (0, 64, n * r - 1)
            flat[:, positions] = [maximum, maximum, -maximum]
            array = flat.reshape(c, n, r).transpose(1, 0, 2).copy()
            append(f"bn3_recover_{np.dtype(dtype).name}_{route}", array)

        # Packed channels include opposing signs so lane/channel mapping matters.
        packed = np.zeros((2, 1, 4, 1, 2, 16), dtype=dtype)
        packed[0, 0, :, 0, 0, :] = maximum
        packed[1, 0, :, 0, 0, :] = maximum
        packed[1, 0, :, 0, 1, :] = -maximum
        packed[..., 1::2] *= -1
        append(f"bn3_recover_{np.dtype(dtype).name}_packed", packed, "NDC1HWC0")

    maximum = np.finfo(np.float32).max
    cases = {
        "cancellation": [maximum, maximum, -maximum, -maximum],
        "negative": [-maximum, 0, -maximum, maximum],
        "true_overflow": [maximum, 0, maximum, 0],
        "positive_inf": [np.inf, 1, 2, 3],
        "negative_inf": [-np.inf, 1, 2, 3],
        "both_inf": [np.inf, -np.inf, 1, 2],
        "nan": [np.nan, 1, 2, 3],
        "finite_control": [256, 0, 256, -256],
    }
    for name, values in cases.items():
        array = np.repeat(
            np.asarray(values, dtype=np.float32).reshape(2, 1, 2), 64, axis=1
        )
        append(f"bn3_recover_{name}", array)
    append("bn3_recover_fp16_control", np.full((2, 64, 2), 65504, dtype=np.float16))
    append("bn3_recover_empty_channels", np.empty((2, 0, 2), dtype=np.float32))
    with (output_dir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Generated {generate(args.output_dir.resolve())} cases")
