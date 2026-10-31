#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import glob
import os

try:
    import ml_dtypes

    BFLOAT16 = ml_dtypes.bfloat16
except ImportError:
    BFLOAT16 = None

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file_lists, output_file_lists, d_type):
    if not golden_file_lists or not output_file_lists:
        print("FAILED! No golden or output bin files found.")
        return False
    if len(golden_file_lists) != len(output_file_lists):
        print(
            f"FAILED! golden file count {len(golden_file_lists)} != output file count {len(output_file_lists)}"
        )
        return False
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
    }
    if BFLOAT16 is not None:
        dtype_map["bfloat16_t"] = BFLOAT16
    if d_type not in dtype_map:
        raise ValueError(f"unsupported dtype for comparison: {d_type!r}")
    np_dtype = dtype_map[d_type]

    if d_type == "float32":
        precision = 1 / 10000
    else:
        precision = 1 / 1000

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        if os.path.getsize(gold) == 0 or os.path.getsize(out) == 0:
            print(f"FAILED! empty bin file: golden={gold}, output={out}")
            data_same = False
            continue
        tmp_out = np.fromfile(out, np_dtype).astype(np.float32)
        tmp_gold = np.fromfile(gold, np_dtype).astype(np.float32)
        diff_res = np.isclose(tmp_out, tmp_gold, precision, 0, True)
        diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    return compare_data(golden_file_lists, output_file_lists, d_type)


if __name__ == "__main__":
    if not process(sys.argv[1]):
        sys.exit(1)
