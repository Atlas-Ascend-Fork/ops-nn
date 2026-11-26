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
#

# exec_offset 与 exec_no_offset 的合并执行器：
#   - offset 通路（不对称量化，带 offset 输出）：gdd_exec_aclnnDynamicQuantV2 /
#     aclnn_DynamicQuantV2 / aclnn_DynamicQuantV2_abnormal
#   - no_offset 通路（对称量化，无 offset）：gdd_exec_aclnnDynamicQuantV2_noffset /
#     gdd_exec_aclnnDynamicQuantV2_iscontinue_noffset / aclnn_DynamicQuantV2_noffset
# 公共标杆实现 test_sim / test_asym_sim / tensor_int32_to_int4s 合并为一份。
import copy

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


def test_sim(x, smooth_scales=None, group=None, dstType=2):
    x_fp32 = x.to(torch.float)

    if dstType == 2:
        min_value = -128
        max_value = 127
    elif dstType == 3:
        min_value = -8
        max_value = 7
    elif dstType == 29:
        min_value = -8
        max_value = 7

    if smooth_scales is not None:
        smooth_scales = smooth_scales.to(torch.float)
        x_fp32 = x_fp32.view(-1, x_fp32.shape[-1])
        if group is not None:
            x_calc = torch.zeros_like(x_fp32)
            group_start_idx = 0
            for group_idx in range(group.shape[0]):
                group_end_idx = group[group_idx].item()
                x_calc[group_start_idx:group_end_idx] = (
                    x_fp32[group_start_idx:group_end_idx] * smooth_scales[group_idx]
                )
                group_start_idx = group_end_idx
        else:
            x_calc = x_fp32 * smooth_scales
        x_fp32 = x_calc.view(x.shape)

    row_max = torch.abs(x_fp32).max(dim=-1, keepdim=True)[0]
    scale = row_max / max_value
    out = x_fp32 * (1 / scale)

    out = torch.round(out)
    out = torch.clamp(out, min_value, max_value).to(torch.int8)
    quant_scale = scale.view(*scale.shape[:-1])
    return out, quant_scale


def test_asym_sim(x, smooth_scales=None, group=None, dstType=2):
    x_fp32 = x.to(torch.float)
    if dstType == 2:
        offset_base = 127
        quant_scale_value = 255
        min_value = -128
        max_value = 127
    elif dstType == 3:
        offset_base = 7
        quant_scale_value = 15
        min_value = -8
        max_value = 7
    elif dstType == 29:
        offset_base = 7
        quant_scale_value = 15
        min_value = -8
        max_value = 7
    if smooth_scales is not None:
        smooth_scales = smooth_scales.to(torch.float)
        x_fp32 = x_fp32.view(-1, x_fp32.shape[-1])
        if group is not None:
            x_calc = torch.zeros_like(x_fp32)
            group_start_idx = 0
            for group_idx in range(group.shape[0]):
                group_end_idx = group[group_idx].item()
                x_calc[group_start_idx:group_end_idx] = (
                    x_fp32[group_start_idx:group_end_idx] * smooth_scales[group_idx]
                )
                group_start_idx = group_end_idx
        else:
            x_calc = x_fp32 * smooth_scales
        x_fp32 = x_calc.view(x.shape)

    row_max = x_fp32.max(dim=-1, keepdim=True)[0]
    row_min = x_fp32.min(dim=-1, keepdim=True)[0]
    scale = torch.clamp((row_max - row_min) / quant_scale_value, 1e-12)

    offset = offset_base - row_max / scale

    out = x_fp32 * (1 / scale) + offset

    out = torch.round(out)
    out = torch.clamp(out, min_value, max_value).to(torch.int8)
    quant_scale = scale.view(*scale.shape[:-1])
    offset.squeeze_(-1)
    return out, quant_scale, offset


def tensor_int32_to_int4s(tensor_int32):
    tensor_array = tensor_int32.reshape(1, tensor_int32.numel())[0]
    int32_array = tensor_array.view(torch.int32).cpu().numpy()
    masks = np.array([0xF << (i * 4) for i in range(8)], dtype=np.uint32)
    shifted = (int32_array[:, None] & masks) >> np.arange(0, 32, 4)
    sign_extended = np.where(shifted & 0x8, shifted - 16, shifted)
    out = torch.tensor(sign_extended.astype(np.int32)).to(torch.int8)
    return out


# ---------------- offset 通路（不对称量化，带 offset 输出） ----------------


@register("gdd_exec_aclnnDynamicQuantV2")
class FunctionDynamicQuantOffsetApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        torch.manual_seed(self.task_result.case_config.id)
        x = input_data.kwargs["x"]
        if len(x.shape) > 0 and 0 not in x.shape:
            group_index = input_data.kwargs["groupIndexOptional"]
            smooth_scale = input_data.kwargs["smoothScalesOptional"]
            if len(smooth_scale.shape) == 2:
                max_group_index = x.numel() // x.shape[-1]
                group_index = torch.randint(
                    0, max_group_index, (smooth_scale.shape[0],), dtype=torch.int32
                )
                group_index = torch.sort(group_index)[0]
                group_index[-1] = max_group_index

            if self.device == "pyaclnn":
                input_data.kwargs["groupIndexOptional"] = group_index.npu()
            elif self.device in ["cpu", "gpu"]:
                input_data.kwargs["groupIndexOptional"] = group_index

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            x = input_data.kwargs["x"]
            if len(x.shape) == 0 or 0 in x.shape:
                return x, x, x
            smooth = input_data.kwargs["smoothScalesOptional"]
            group = input_data.kwargs["groupIndexOptional"]
            dstType = input_data.kwargs["dstType"]
            if len(smooth.shape) == 0:
                smooth = None
            if len(group.shape) == 0:
                group = None
            x = x if x.dtype == torch.float32 else x.type(torch.float32)
            if smooth is not None:
                smooth = (
                    smooth
                    if smooth.dtype == torch.float32
                    else smooth.type(torch.float32)
                )

            output = test_asym_sim(x, smooth, group, dstType)
            # output = test_sim(x ,smooth ,group,dstType)
        return output[0], output[1], output[2]


@register("aclnn_DynamicQuantV2")
class AclnnDynamicQuantOffsetApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

        groupIndexOptional = input_data.kwargs["groupIndexOptional"]
        smooth = input_data.kwargs["smoothScalesOptional"]

        dst_type = input_args[3].value
        # 当dst_type为3时，创建int32的tensor作为输出
        if dst_type == 3:
            out_data = self.acl_tensor_to_torch(input_args[4])
            out_shape = out_data.shape
            out_data = torch.zeros(
                (*out_shape[:-1], out_shape[-1] // 8), dtype=torch.int32
            ).npu()
            out_acl_tensor = self.torch_tensor_to_acl(out_data)
            output_packages[0] = input_args[4] = out_acl_tensor

        if groupIndexOptional.shape == torch.Size([]):
            input_args[2] = TensorPtr()
        if smooth.shape == torch.Size([]):
            input_args[1] = TensorPtr()
        # input_args.append(TensorPtr())
        return input_args, output_packages

    def after_call(self, output_packages):
        output = super().after_call(output_packages)
        if output[0].dtype == torch.int32:
            # 将int32的输出解包为int8的输出
            output[0] = tensor_int32_to_int4s(output[0])
        return output


@register("aclnn_DynamicQuantV2_abnormal")
class AclnnDynamicQuantOffsetAbnormalApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

        groupIndexOptional = input_data.kwargs["groupIndexOptional"]
        smooth = input_data.kwargs["smoothScalesOptional"]

        dst_type = input_args[3].value
        # 当dst_type为3时，创建int32的tensor作为输出
        if dst_type == 3:
            out_data = self.acl_tensor_to_torch(input_args[4])
            out_shape = out_data.shape
            out_data = torch.zeros(
                (*out_shape[:-1], out_shape[-1] // 8), dtype=torch.int32
            ).npu()
            out_acl_tensor = self.torch_tensor_to_acl(out_data)
            output_packages[0] = input_args[4] = out_acl_tensor

        if groupIndexOptional.shape == torch.Size([]):
            input_args[2] = TensorPtr()
        if smooth.shape == torch.Size([]):
            input_args[1] = TensorPtr()
        # input_args.append(TensorPtr())

        id = self.task_result.case_config.id

        print("input_args参数个数：", len(input_args))
        if id == 0:
            input_args[0] = TensorPtr()
        elif id == 1:
            new_tensor = torch.randn(input_args[0].pytensor.shape, dtype=torch.double)
            acl_tensor = self.backend.torch_tensor_to_acl(
                new_tensor, AclFormat.ACL_FORMAT_ND
            )
            input_args[0] = acl_tensor
        elif id == 2:
            new_tensor = torch.tensor(
                input_args[0].pytensor.shape, dtype=input_args[0].pytensor.dtype
            )
            acl_tensor = self.backend.torch_tensor_to_acl(
                new_tensor, AclFormat.ACL_FORMAT_NCHW
            )
            input_args[0] = acl_tensor
        return input_args, output_packages

    def after_call(self, output_packages):
        output = super().after_call(output_packages)
        if output[0].dtype == torch.int32:
            # 将int32的输出解包为int8的输出
            output[0] = tensor_int32_to_int4s(output[0])
        return output


# ---------------- no_offset 通路（对称量化，无 offset） ----------------


@register("gdd_exec_aclnnDynamicQuantV2_noffset")
class FunctionDynamicQuantNoOffsetApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        torch.manual_seed(self.task_result.case_config.id)
        x = input_data.kwargs["x"]
        group_index = input_data.kwargs["groupIndexOptional"]
        smooth_scale = input_data.kwargs["smoothScalesOptional"]
        if len(smooth_scale.shape) == 2:
            max_group_index = x.numel() // x.shape[-1]
            group_index = torch.randint(
                0, max_group_index, (smooth_scale.shape[0],), dtype=torch.int32
            )
            group_index = torch.sort(group_index)[0]
            group_index[-1] = max_group_index

        if self.device == "pyaclnn":
            input_data.kwargs["groupIndexOptional"] = group_index.npu()
        elif self.device in ["cpu", "gpu"]:
            input_data.kwargs["groupIndexOptional"] = group_index

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            x = input_data.kwargs["x"]
            smooth = input_data.kwargs["smoothScalesOptional"]
            group = input_data.kwargs["groupIndexOptional"]
            dstType = input_data.kwargs["dstType"]
            if len(smooth.shape) == 0:
                smooth = None
            if len(group.shape) == 0:
                group = None
            x = x if x.dtype == torch.float32 else x.type(torch.float32)
            if smooth is not None:
                smooth = (
                    smooth
                    if smooth.dtype == torch.float32
                    else smooth.type(torch.float32)
                )

            # output = test_asym_sim(x ,smooth ,group,dstType)
            output = test_sim(x, smooth, group, dstType)
        return output[0], output[1]


@register("gdd_exec_aclnnDynamicQuantV2_iscontinue_noffset")
class FunctionDynamicQuantIsContinueNoOffsetApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self.Input_Data = copy.deepcopy(input_data)
        torch.manual_seed(self.task_result.case_config.id)
        input_data.kwargs["x"] = input_data.kwargs["x"][..., ::2]
        # input_data.kwargs["groupIndexOptional"] = input_data.kwargs["groupIndexOptional"][..., ::2]
        # input_data.kwargs["smoothScalesOptional"] = input_data.kwargs["smoothScalesOptional"][..., ::2]
        x = input_data.kwargs["x"]
        group_index = input_data.kwargs["groupIndexOptional"]
        smooth_scale = input_data.kwargs["smoothScalesOptional"]
        if len(smooth_scale.shape) == 2:
            max_group_index = x.numel() // x.shape[-1]
            group_index = torch.randint(
                0, max_group_index, (smooth_scale.shape[0],), dtype=torch.int32
            )
            group_index = torch.sort(group_index)[0]
            group_index[-1] = max_group_index

        if self.device == "pyaclnn":
            input_data.kwargs["groupIndexOptional"] = group_index.npu()
        elif self.device in ["cpu", "gpu"]:
            input_data.kwargs["groupIndexOptional"] = group_index

    def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
        if name is not None:
            return self.Input_Data.kwargs[
                name
            ].shape  # 把原始参数里的shape作为storage shape给aclnn用
        else:
            return None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            x = input_data.kwargs["x"]
            smooth = input_data.kwargs["smoothScalesOptional"]
            group = input_data.kwargs["groupIndexOptional"]
            dstType = input_data.kwargs["dstType"]
            if len(smooth.shape) == 0:
                smooth = None
            if len(group.shape) == 0:
                group = None
            x = x if x.dtype == torch.float32 else x.type(torch.float32)
            if smooth is not None:
                smooth = (
                    smooth
                    if smooth.dtype == torch.float32
                    else smooth.type(torch.float32)
                )

            # output = test_asym_sim(x ,smooth ,group,dstType)
            output = test_sim(x, smooth, group, dstType)
        return output[0], output[1]


@register("aclnn_DynamicQuantV2_noffset")
class AclnnDynamicQuantNoOffsetApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

        groupIndexOptional = input_data.kwargs["groupIndexOptional"]
        smooth = input_data.kwargs["smoothScalesOptional"]

        dst_type = input_args[3].value
        # 当dst_type为3时，创建int32的tensor作为输出
        if dst_type == 3:
            out_data = self.acl_tensor_to_torch(input_args[4])
            out_shape = out_data.shape
            out_data = torch.zeros(
                (*out_shape[:-1], out_shape[-1] // 8), dtype=torch.int32
            ).npu()
            out_acl_tensor = self.torch_tensor_to_acl(out_data)
            output_packages[0] = input_args[4] = out_acl_tensor

        if groupIndexOptional.shape == torch.Size([]):
            input_args[2] = TensorPtr()
        if smooth.shape == torch.Size([]):
            input_args[1] = TensorPtr()
        input_args.append(TensorPtr())
        return input_args, output_packages

    def after_call(self, output_packages):
        output = super().after_call(output_packages)
        if output[0].dtype == torch.int32:
            # 将int32的输出解包为int8的输出
            output[0] = tensor_int32_to_int4s(output[0])
        return output
