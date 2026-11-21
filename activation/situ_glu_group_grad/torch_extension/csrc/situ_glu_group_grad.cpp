/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace activation {
namespace {
constexpr int64_t kSplitFactor = 2;
// group_list_type: 0 = group_index is a cumsum list (its last element is the total row count),
// 1 = group_index is a count list (the sum of its elements is the total row count).
constexpr int64_t kGroupListTypeCumsum = 0;
constexpr int64_t kGroupListTypeCount = 1;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

void CheckOptionalNpuTensor(const c10::optional<at::Tensor>& tensor, const char* name)
{
    if (tensor.has_value() && tensor.value().defined()) {
        CheckNpuTensor(tensor.value(), name);
    }
}

void ValidateDim(const at::Tensor& x, const int64_t dim)
{
    TORCH_CHECK(x.dim() > 0, "x rank should be greater than 0");
    int64_t actualDim = dim;
    if (actualDim < 0) {
        actualDim += x.dim();
    }
    TORCH_CHECK(actualDim >= 0 && actualDim < x.dim(), "dim should be in [", -x.dim(), ", ", x.dim() - 1, "], but got ",
                dim);
    const int64_t dimSize = x.size(actualDim);
    TORCH_CHECK(dimSize % kSplitFactor == 0, "x size at dim ", actualDim, " should be even, but got ", dimSize);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> situ_glu_group_grad(const at::Tensor& grad_y, const at::Tensor& x,
                                                       const c10::optional<at::Tensor>& weight,
                                                       const c10::optional<at::Tensor>& group_index, const int64_t dim,
                                                       const double beta, const double linear_beta,
                                                       const bool activate_left, const int64_t group_list_type)
{
    CheckNpuTensor(grad_y, "grad_y");
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(weight, "weight");
    CheckOptionalNpuTensor(group_index, "group_index");
    ValidateDim(x, dim);
    TORCH_CHECK(group_list_type == kGroupListTypeCumsum || group_list_type == kGroupListTypeCount,
                "group_list_type must be 0 (cumsum) or 1 (count), but got ", group_list_type);

    int64_t totalRows = 1;
    int64_t actualDim = dim < 0 ? dim + x.dim() : dim;
    for (int64_t i = 0; i < actualDim; ++i) {
        totalRows *= x.size(i);
    }

    const bool hasWeight = weight.has_value() && weight.value().defined();
    if (hasWeight) {
        TORCH_CHECK(weight.value().scalar_type() == at::kFloat, "weight dtype must be FLOAT");
        TORCH_CHECK(weight.value().numel() == totalRows,
                    "weight element num must equal the product of x dims "
                    "before the dim axis (",
                    totalRows, "), but got ", weight.value().numel());
    }
    if (group_index.has_value() && group_index.value().defined()) {
        TORCH_CHECK(group_index.value().dim() == 1, "group_index must be 1D when present");
        TORCH_CHECK(group_index.value().numel() > 0, "group_index must not be empty when present");
        TORCH_CHECK(group_index.value().scalar_type() == at::kLong, "group_index dtype must be INT64");
    }

    at::Tensor grad_x = at::empty(x.sizes(), x.options());
    // grad_weight is a required output: weight shape when weight is given, empty [0] otherwise
    // (no weight in forward means no weight gradient chain).
    at::Tensor grad_weight = at::empty({0}, x.options().dtype(at::kFloat));
    if (hasWeight) {
        grad_weight = at::empty(weight.value().sizes(), weight.value().options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnSituGluGroupGrad, grad_y, x, weight, group_index, dim, beta, linear_beta, activate_left,
              group_list_type, grad_x, grad_weight);
    return std::make_tuple(grad_x, grad_weight);
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("situ_glu_group_grad", &cann_ops_nn::activation::situ_glu_group_grad, "SituGluGroupGrad on NPU");
}
