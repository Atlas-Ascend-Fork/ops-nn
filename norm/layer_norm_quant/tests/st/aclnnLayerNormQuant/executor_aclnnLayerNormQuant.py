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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
import torch.nn.functional as F
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
import copy


@register("aclnn_layer_norm_quant_continue")
class LayerNormQuantNonContiguous(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        self.Input_Data = copy.deepcopy(input_data)  # 备份一个做切分之前的原始参数
        input_data.kwargs["x"] = input_data.kwargs["x"][..., ::2]
        input_data.kwargs["gamma"] = input_data.kwargs["gamma"][..., ::2]
        input_data.kwargs["beta"] = input_data.kwargs["beta"][..., ::2]
        input_data.kwargs["scale"] = input_data.kwargs["scale"][..., ::2]
        input_data.kwargs["zeroPointsOptional"] = input_data.kwargs[
            "zeroPointsOptional"
        ][..., ::2]

    def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
        if name is not None:
            return self.Input_Data.kwargs[
                name
            ].shape  # 把原始参数里的shape作为storage shape给aclnn用
        else:
            return None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs["x"]
        input_gamma = input_data.kwargs["gamma"]
        input_beta = input_data.kwargs["beta"]
        input_scale = input_data.kwargs["scale"]
        input_zeroPointsOptional = input_data.kwargs["zeroPointsOptional"]
        epsilon = input_data.kwargs["eps"]
        # print("golden==========")
        if self.device == "cpu":
            if len(input_x.shape) == 0:
                return torch.zeros(input_scale.shape).to(torch.float32), torch.zeros(
                    input_scale.shape
                ).to(torch.float32)
            gamma = input_gamma.reshape(-1)
            beta = input_beta.reshape(-1)
            res = F.layer_norm(
                input_x.float(),
                gamma.shape,
                weight=gamma.float(),
                bias=beta.float(),
                eps=epsilon,
            ).half()
            # print("res====================")
            # print(res)
            output = res / input_scale + input_zeroPointsOptional
            output = torch.clamp(output, -128, 127)
            output = output.to(torch.int8)
            # print("output====================")
            # print(output)
        # 不比较这个输出
        # scaleOut = torch.zeros(input_scale.shape).to(torch.float32)
        scaleOut = torch.empty(input_x.shape[:-1]).to(torch.float32)
        return output, scaleOut


@register("aclnn_layer_norm_quant")
class LayerNormQuant(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs["x"]
        input_gamma = input_data.kwargs["gamma"]
        input_beta = input_data.kwargs["beta"]
        input_scale = input_data.kwargs["scale"]
        input_zeroPointsOptional = input_data.kwargs["zeroPointsOptional"]
        epsilon = input_data.kwargs["eps"]
        # print("golden==========")
        if self.device == "cpu":
            if len(input_x.shape) == 0:
                return torch.zeros(input_scale.shape).to(torch.float32), torch.zeros(
                    input_scale.shape
                ).to(torch.float32)
            gamma = input_gamma.reshape(-1)
            beta = input_beta.reshape(-1)
            res = F.layer_norm(
                input_x.float(),
                gamma.shape,
                weight=gamma.float(),
                bias=beta.float(),
                eps=epsilon,
            ).half()
            # print("res====================")
            # print(res)
            output = res / input_scale + input_zeroPointsOptional
            output = torch.clamp(output, -128, 127)
            output = output.to(torch.int8)
            # print("output====================")
            # print(output)
        # 不比较这个输出
        # scaleOut = torch.zeros(input_scale.shape).to(torch.float32)
        scaleOut = torch.empty(input_x.shape[:-1]).to(torch.float32)
        return output, scaleOut


@register("pyaclnn_layer_norm_quant")
class SampleAclnnApi(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        # print(">>>>>>>>>>>>>>>>>>>>>>x=", input_args[0].pytensor.shape)
        # print(">>>>>>>>>>>>>>>>>>>>>>scaleOut=", input_args[8].pytensor.shape)

        return input_args, output_packages

    # pyaclnn的自定义api
    def after_call(self, output_packages):
        output = []
        output1 = self.acl_tensor_to_torch(output_packages[0])
        if output1 is None or output1.numel() == 0:
            output.append(output1)
            tensorA = torch.zeros(1)
            output2 = torch.zeros(tensorA.shape).to(torch.float32)
            output.append(output2)
        else:
            output.append(output1)
        return output

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnLayerNormQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* gammma, const aclTensor* beta, const aclTensor* scalesOptional, \
                    const aclTensor* zeroPointsOptional, int quantMode, double epsilon, aclTensor* res, aclTensor* scaleOut, uint64_t* workspaceSize, aclOpExecutor** executor)"


@register("pyaclnn_layer_norm_quant_abnormal")
class SampleAclnnApiExp(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)

        id = self.task_result.case_config.id

        if id == 0:
            input_args[0] = TensorPtr()
        elif id == 1:
            input_args[7] = TensorPtr()
        elif id == 2:
            new_tensor = torch.randn(input_args[0].pytensor.shape, dtype=torch.double)
            acl_tensor = self.backend.torch_tensor_to_acl(
                new_tensor, AclFormat.ACL_FORMAT_ND
            )
            input_args[0] = acl_tensor
        elif id == 3:
            new_tensor = torch.randn(1, 182, dtype=torch.double)
            acl_tensor = self.backend.torch_tensor_to_acl(
                new_tensor, AclFormat.ACL_FORMAT_ND
            )
            input_args[1] = acl_tensor
        elif id == 4:
            new_tensor = torch.randn(1, 182, dtype=torch.double)
            acl_tensor = self.backend.torch_tensor_to_acl(
                new_tensor, AclFormat.ACL_FORMAT_ND
            )
            input_args[7] = acl_tensor
        elif id == 5:
            new_tensor = torch.randn(1, 182, dtype=torch.double)
            acl_tensor = self.backend.torch_tensor_to_acl(
                new_tensor, AclFormat.ACL_FORMAT_ND
            )
            input_args[2] = acl_tensor
        elif id == 6:
            new_tensor = torch.randn(182, dtype=torch.double)
            acl_tensor = self.backend.torch_tensor_to_acl(
                new_tensor, AclFormat.ACL_FORMAT_ND
            )
            input_args[4] = acl_tensor

        return input_args, output_packages

    # pyaclnn的自定义api
    def after_call(self, output_packages):
        output = []
        output1 = self.acl_tensor_to_torch(output_packages[0])
        output.append(output1)
        tensorA = torch.ones(1)
        output2 = torch.zeros(tensorA.shape).to(torch.float32)
        output.append(output2)
        return output
