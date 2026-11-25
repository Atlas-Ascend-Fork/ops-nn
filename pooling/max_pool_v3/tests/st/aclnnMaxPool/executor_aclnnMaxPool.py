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

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


def _resolve_api(api_name: str):
    if not isinstance(api_name, str) or not api_name:
        raise ValueError(f"Invalid api_name: {api_name!r}")
    parts = api_name.split(".")
    if any(not part.isidentifier() for part in parts):
        raise ValueError(f"Unsafe api_name: {api_name!r}")
    if parts[0] != "torch":
        raise ValueError(f"Unsupported api_name root: {parts[0]!r}")
    obj = torch
    for attr in parts[1:]:
        if attr.startswith("__"):
            raise ValueError(f"Disallowed dunder attribute in api_name: {api_name!r}")
        obj = getattr(obj, attr)
    if not callable(obj):
        raise ValueError(f"Resolved api_name is not callable: {api_name!r}")
    return obj


@register("aclnn_max_pool")
class AclnnMaxPoolApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_dict = dict()
        input_dict["kernel_size"] = input_data.kwargs["kernel_size"]
        input_dict["stride"] = input_data.kwargs["stride"]
        input_dict["padding"] = input_data.kwargs["padding"]
        input_dict["dilation"] = input_data.kwargs["dilation"]
        input_dict["ceil_mode"] = False if input_data.kwargs["ceil_mode"] == 0 else True
        output = None
        if self.device == "npu":
            torch.npu.set_compile_mode(jit_compile=False)
            device = f"npu:{self.device_id}"
            model = _resolve_api(self.api_name)(**input_dict).to(device)
            output = model(input_data.kwargs["input"].to(device))

        if self.device == "cpu":
            model = _resolve_api(self.api_name)(**input_dict)

            ori_type = input_data.kwargs["input"].dtype
            output = model(input_data.kwargs["input"]).to(torch.float32)
            output = output.to(ori_type)

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
            model = _resolve_api(self.api_name)(**input_dict).to(device)
            output = model(input_data.kwargs["input"].to(device))

        return output.cpu()

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if len(input_data.kwargs["input"].shape) == 3:
            return AclFormat.ACL_FORMAT_ND
        return AclFormat.ACL_FORMAT_NCHW
