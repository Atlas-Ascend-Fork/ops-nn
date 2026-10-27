#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate fixed TTK inputs for training and inference precision.

The output directory should be inside the task's MyTestCase checkout. Input
construction targets cancellation, nonfinite affine parameters, extreme finite
values, packed channel tails and multiple reduction cores. TTK independently
computes all expected outputs using golden.py.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def generate(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261024)
    cases = []

    def add(
        name,
        values,
        gamma,
        beta,
        fmt,
        epsilon=1e-5,
        running=None,
        input_dtype="float32",
    ):
        # values is [R,C]; layouts retain the same reduction and channel axes.
        if values.ndim == 5:
            x = values
            channels = x.shape[1 if fmt == "NCDHW" else -1]
        else:
            count, channels = values.shape
            if fmt == "NDHWC":
                x = values.reshape(1, 1, 1, count, channels)
            else:
                x = values.T.copy().reshape(1, channels, 1, 1, count)
        data_dir = output / name
        data_dir.mkdir(exist_ok=True)
        paths = []
        inputs = (x, gamma, beta) + (() if running is None else running)
        for index, value in enumerate(inputs):
            path = data_dir / f"input_{index}.npy"
            np.save(
                path, np.asarray(value, dtype=input_dtype if index == 0 else "float32")
            )
            paths.append(str(path))
        optional_shape = (None, None) if running is None else ((channels,),) * 2
        optional_dtype = (None, None) if running is None else ("float32",) * 2
        optional_format = (None, None) if running is None else ("ND",) * 2
        cases.append(
            {
                "testcase_name": name,
                "op_name": "batch_norm3_d",
                "input_shapes": repr(
                    (x.shape, (channels,), (channels,)) + optional_shape
                ),
                "input_dtypes": repr(
                    (input_dtype,) + ("float32",) * 2 + optional_dtype
                ),
                "input_formats": repr((fmt, "ND", "ND") + optional_format),
                "output_shapes": repr((x.shape,) + ((channels,),) * 4),
                "output_dtypes": repr((input_dtype,) + ("float32",) * 4),
                "output_formats": repr((fmt,) + ("ND",) * 4),
                "attributes": repr(
                    {
                        "epsilon": epsilon,
                        "data_format": fmt,
                        "is_training": running is None,
                    }
                ),
                "manual_input_binaries": repr(
                    tuple(paths) + ((None, None) if running is None else ())
                ),
                "is_enabled": "True",
            }
        )

    for fmt in ("NCDHW", "NDHWC"):
        for channels in (3, 24, 65):
            values = rng.uniform(0.0, 0.001, (8193, channels)).astype(np.float32)
            gamma = rng.uniform(1, 1000, channels).astype(np.float32)
            precise = values.astype(np.float64)
            normalized = (precise[0] - precise.mean(0)) / np.sqrt(
                precise.var(0) + float(np.float32(1e-5))
            )
            # Cancellation at the first row leaves the rounding residual of
            # beta, so losing low bits of mean/rstd is observable in y.
            beta = -(normalized * gamma).astype(np.float32)
            add(f"precision_cancel_{fmt}_{channels}", values, gamma, beta, fmt)

        values = np.ones((1025, 4), dtype=np.float32)
        values[-1] = np.nextafter(np.float32(1), np.float32(2))
        add(
            f"precision_mean_low_inf_{fmt}",
            values,
            np.array([np.inf, -np.inf, 1, -1], dtype=np.float32),
            np.zeros(4),
            fmt,
        )

        limit = float(np.finfo(np.float32).max)
        values = (rng.uniform(-1, 1, (16385, 3)) * limit).astype(np.float32)
        add(
            f"precision_extreme_{fmt}",
            values,
            np.array([1, limit, -limit]),
            np.array([0, -limit, limit]),
            fmt,
        )
        for exponent in (10, 25, 35):
            values = (rng.uniform(-1, 1, (8193, 3)) * (10.0**exponent)).astype(
                np.float32
            )
            add(
                f"precision_scaled_{fmt}_{exponent}",
                values,
                np.ones(3),
                np.zeros(3),
                fmt,
            )
        values = np.full((8193, 3), limit, dtype=np.float32)
        add(f"precision_constant_{fmt}", values, np.ones(3), np.zeros(3), fmt)

        values = rng.uniform(-1, 1, (129, 3)).astype(np.float32)
        add(
            f"precision_epsilon_large_{fmt}",
            values,
            np.full(3, limit),
            np.zeros(3),
            fmt,
            epsilon=limit,
        )
        values.fill(0.25)
        add(
            f"precision_epsilon_zero_{fmt}",
            values,
            np.ones(3),
            np.zeros(3),
            fmt,
            epsilon=0.0,
        )
        add(
            f"precision_epsilon_negative_{fmt}",
            values,
            np.ones(3),
            np.zeros(3),
            fmt,
            epsilon=-1.0,
        )

    # Exercise each inference tiling route with cancellation and a finite
    # variance + epsilon sum that exceeds the FP32 range.
    routes = (
        ("ar", "NCDHW", (2, 3, 2, 2, 17)),
        ("small_ab1", "NCDHW", (65537, 2, 1, 1, 2)),
        ("ra", "NDHWC", (2, 2, 2, 17, 65)),
        ("small_a", "NDHWC", (1, 1, 1, 65537, 3)),
        ("continuous_a", "NDHWC", (1, 1, 1, 65537, 129)),
    )
    epsilon = float(np.float32(1e-4))
    tiny = np.finfo(np.float32).tiny
    center = np.float32(np.sqrt(1.0 + epsilon) / float(tiny))
    center_bits = int(center.view(np.uint32))
    limit = float(np.finfo(np.float32).max)
    for route, fmt, shape in routes:
        channels = shape[1 if fmt == "NCDHW" else -1]
        bits = rng.integers(
            center_bits - 1024, center_bits + 1025, shape, dtype=np.uint32
        )
        values = bits.view(np.float32)
        add(
            f"precision_infer_cancel_{route}",
            values,
            np.full(channels, -tiny),
            np.ones(channels),
            fmt,
            epsilon=epsilon,
            running=(np.full(channels, -1), np.ones(channels)),
        )
        values = np.full(shape, limit, dtype=np.float32)
        add(
            f"precision_infer_large_sum_{route}",
            values,
            np.ones(channels),
            np.zeros(channels),
            fmt,
            epsilon=limit,
            running=(np.zeros(channels), np.full(channels, limit)),
        )

    for channels in (3, 24, 65):
        values = rng.uniform(0.001, 0.01, (8193, channels)).astype(np.float16)
        gamma = rng.uniform(10, 1000, channels).astype(np.float32)
        precise = values.astype(np.float64)
        normalized = (precise[0] - precise.mean(0)) / np.sqrt(
            precise.var(0) + float(np.float32(1e-5))
        )
        beta = -(normalized * gamma).astype(np.float32)
        add(
            f"precision_half_cancel_{channels}",
            values,
            gamma,
            beta,
            "NDHWC",
            input_dtype="float16",
        )
        values = rng.uniform(-65504, 65504, (16384, channels)).astype(np.float16)
        values[1::2] = -values[::2]
        # Exact cancellation of each adjacent pair gives an exactly zero mean.
        add(
            f"precision_half_zero_mean_{channels}",
            values,
            np.ones(channels),
            np.zeros(channels),
            "NDHWC",
            input_dtype="float16",
        )

    for route, fmt, shape in routes:
        channels = shape[1 if fmt == "NCDHW" else -1]
        signs = np.where(np.arange(channels) % 2 == 0, 1.0, -1.0)
        variance = np.resize(np.array([0.01, 0.02, -0.01, np.inf]), channels)
        for dtype in ("float16", "float32"):
            # Finite products can exceed FP32 before an infinite bias is added.
            # Negative variance must still preserve NaN instead of taking bias.
            add(
                f"precision_infer_inf_bias_{route}_{dtype}",
                np.zeros(shape),
                -signs,
                signs * np.inf,
                fmt,
                epsilon=1e-4,
                running=(np.full(channels, -limit), variance),
                input_dtype=dtype,
            )

    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cases[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)
    return output / "cases.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(generate(parser.parse_args().output))
