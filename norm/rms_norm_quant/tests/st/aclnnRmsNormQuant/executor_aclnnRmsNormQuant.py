#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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


@register("function_rms_norm_quant")
class RmsNormQuant(BaseApi):
    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnRmsNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, const aclTensor *scale, const aclTensor *offset, double epsilon, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)"

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["x"]
        beta = input_data.kwargs["beta"]
        gamma = input_data.kwargs["gamma"]
        scale = input_data.kwargs["scale"]
        offset = input_data.kwargs["offset"]
        epsilon = input_data.kwargs["epsilon"]
        return npu_add_rms_norm_quant_golden(
            self, x, beta, gamma, scale, offset, epsilon=epsilon
        )


def npu_add_rms_norm_quant_golden(self, x, beta, gamma, scale, offset, epsilon=1e-06):
    def rms_norm(inputX, inputGamma, inputBeta, eps=1e-6):
        square_sum = torch.sum(torch.square(inputX), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / inputX.shape[-1] + eps)
        return inputX * factor * inputGamma + inputBeta

    inputX = x.float()
    inputGamma = gamma.float()
    inputBeta = beta.float()
    inputScale = scale.float()
    inputOffset = offset.float()
    quant_in = rms_norm(inputX, inputGamma=inputGamma, inputBeta=inputBeta, eps=epsilon)
    outputQ = torch.quantize_per_tensor(quant_in, inputScale, inputOffset, torch.qint8)
    y1_np = outputQ.int_repr().detach().clone().cpu().numpy()
    return torch.tensor(y1_np).type(torch.int8).reshape(x.shape)


@register("aclnn_function")
class AclnnRmsNormQuant(AclnnBaseApi):
    def after_call(self, output_packages):
        output = []
        y1 = self.acl_tensor_to_torch(output_packages[0])
        output.append(y1)
        return output
