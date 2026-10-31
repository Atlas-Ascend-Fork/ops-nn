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
import os
import numpy as np
import re
import tensorflow as tf
import torch


def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 ,]+)\}", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",")]
        shape_list.append(single_shape)
    return shape_list


def _numpy_to_torch(array, d_type):
    """Bridge a numpy array to a torch tensor for golden computation.

    torch.from_numpy does not accept the numpy bfloat16 dtype produced by
    tf.bfloat16.as_numpy_dtype, so reinterpret bf16 through its 16-bit view.
    """
    if d_type == "bfloat16_t":
        return torch.from_numpy(np.ascontiguousarray(array).view(np.int16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(np.ascontiguousarray(array))


def _torch_to_numpy(tensor, d_type, np_type):
    """Bridge the torch golden back to numpy for bin writing (no math in numpy)."""
    if d_type == "bfloat16_t":
        return tensor.view(torch.int16).numpy().view(np_type)
    return tensor.numpy().astype(np_type)


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype,
    }
    np_type = d_type_dict[d_type]
    shape_list = parse_str_to_shape_list(shape_str)

    tensor1_list = [(np.ones(shape) * 3).astype(np_type) for shape in shape_list]
    tensor2_list = [(np.ones(shape) * 10).astype(np_type) for shape in shape_list]
    weight_list = [(np.ones(shape) * 10).astype(np_type) for shape in shape_list]

    # Golden is produced by the direct competitor torch._foreach_lerp (GOLDEN.05);
    # numpy only bridges the arrays and writes the bin files.
    golden_list = torch._foreach_lerp(
        [_numpy_to_torch(tensor, d_type) for tensor in tensor1_list],
        [_numpy_to_torch(tensor, d_type) for tensor in tensor2_list],
        [_numpy_to_torch(tensor, d_type) for tensor in weight_list],
    )

    for index in range(len(shape_list)):
        tensor1_list[index].astype(np_type).tofile(
            f"{d_type}_input_t_foreach_lerp_list{index}_1.bin"
        )
        tensor2_list[index].astype(np_type).tofile(
            f"{d_type}_input_t_foreach_lerp_list{index}_2.bin"
        )
        weight_list[index].astype(np_type).tofile(
            f"{d_type}_input_t_foreach_lerp_list{index}_3.bin"
        )
        _torch_to_numpy(golden_list[index], d_type, np_type).astype(np_type).tofile(
            f"{d_type}_golden_t_foreach_lerp_list{index}.bin"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
