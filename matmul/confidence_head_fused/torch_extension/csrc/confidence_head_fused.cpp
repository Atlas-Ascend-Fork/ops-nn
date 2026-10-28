/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include "aclnnop/aclnn_confidence_head_fused.h"
#include "aclnn_common.h"

namespace cann_ops_nn::matmul {
namespace {
constexpr int64_t HIDDEN_RANK_WITHOUT_BATCH = 2;
constexpr int64_t HIDDEN_RANK_WITH_BATCH = 3;
constexpr int64_t MAX_DIM = std::numeric_limits<uint32_t>::max();

void CheckTensor(const at::Tensor& tensor, at::ScalarType dtype, int64_t rank, const c10::Device& device,
                 const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device() == device, name, " must be on the same NPU as hidden");
    TORCH_CHECK(tensor.layout() == at::kStrided, name, " must be strided");
    TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == rank, name, " has an invalid dtype or rank");
    for (auto dim : tensor.sizes()) {
        TORCH_CHECK(dim <= MAX_DIM, name, " dimensions must fit UINT32");
    }
}

bool IsPresent(const c10::optional<at::Tensor>& tensor) { return tensor.has_value() && tensor->defined(); }

void CheckScalarTensor(const c10::optional<at::Tensor>& tensor, const c10::Device& device, const char* name)
{
    if (IsPresent(tensor)) {
        CheckTensor(*tensor, at::kFloat, 1, device, name);
        TORCH_CHECK(tensor->size(0) == 1, name, " must have shape [1]");
    }
}

} // namespace

at::Tensor confidence_head_fused(const at::Tensor& hidden, const at::Tensor& weight,
                                 const c10::optional<at::Tensor>& markov_emb,
                                 const c10::optional<at::Tensor>& temperature, const c10::optional<at::Tensor>& bias,
                                 const c10::optional<double>& clamp_low, const c10::optional<double>& clamp_high)
{
    TORCH_CHECK(hidden.defined() && hidden.device().type() == at::kPrivateUse1, "hidden must be on NPU");
    TORCH_CHECK(hidden.dim() == HIDDEN_RANK_WITHOUT_BATCH || hidden.dim() == HIDDEN_RANK_WITH_BATCH,
                "hidden must have rank 2 or 3");
    TORCH_CHECK(hidden.scalar_type() == at::kHalf || hidden.scalar_type() == at::kBFloat16,
                "hidden must be FP16 or BF16");
    auto inputDtype = hidden.scalar_type();
    int64_t rank = hidden.dim();
    CheckTensor(hidden, inputDtype, rank, hidden.device(), "hidden");
    CheckTensor(weight, inputDtype, 1, hidden.device(), "weight");
    int64_t kMarkov = 0;
    if (IsPresent(markov_emb)) {
        CheckTensor(*markov_emb, inputDtype, rank, hidden.device(), "markov_emb");
        for (int64_t i = 0; i + 1 < rank; ++i) {
            TORCH_CHECK(markov_emb->size(i) == hidden.size(i), "hidden and markov_emb leading dimensions must match");
        }
        kMarkov = markov_emb->size(rank - 1);
    }
    int64_t rows = 1;
    for (int64_t i = 0; i + 1 < rank; ++i) {
        int64_t dim = hidden.size(i);
        TORCH_CHECK(dim == 0 || rows <= MAX_DIM / dim, "Flattened rows must fit UINT32");
        rows *= dim;
    }
    TORCH_CHECK(hidden.size(rank - 1) <= MAX_DIM - kMarkov, "K1+K2 must fit UINT32");
    TORCH_CHECK(weight.size(0) == hidden.size(rank - 1) + kMarkov, "weight length must equal K1+K2");
    CheckScalarTensor(temperature, hidden.device(), "temperature");
    if (IsPresent(bias)) {
        CheckTensor(*bias, at::kFloat, 1, hidden.device(), "bias");
        TORCH_CHECK(bias->size(0) == hidden.size(rank - 2), "bias must have shape [N], shared across batches");
    }
    const c10::OptionalDeviceGuard guard(hidden.device());
    auto outShape = hidden.sizes().vec();
    outShape.pop_back();
    auto out = at::empty(outShape, hidden.options().dtype(at::kFloat));
    c10::optional<at::Scalar> clampLowScalar;
    c10::optional<at::Scalar> clampHighScalar;
    if (clamp_low.has_value()) {
        clampLowScalar = at::Scalar(static_cast<float>(*clamp_low));
    }
    if (clamp_high.has_value()) {
        clampHighScalar = at::Scalar(static_cast<float>(*clamp_high));
    }
    ACLNN_CMD(aclnnConfidenceHeadFused, hidden, weight, markov_emb, temperature, bias, clampLowScalar, clampHighScalar,
              out);
    return out;
}

} // namespace cann_ops_nn::matmul

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("confidence_head_fused", &cann_ops_nn::matmul::confidence_head_fused, "ConfidenceHeadFused on NPU");
}
