/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {
namespace {
// Situ激活gate/up拆分因子：x最后一维为2H，激活输出最后一维为H
constexpr int64_t kSplitFactor = 2;
// MX量化块大小：每32个元素共享一个scale；scale最后一维对齐数=2
constexpr int64_t kMxBlockSize = 32;
constexpr int64_t kScaleAlignNum = 2;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

bool TensorAbsent(const c10::optional<at::Tensor>& tensor) { return !tensor.has_value() || !tensor->defined(); }
} // namespace

std::tuple<at::Tensor, at::Tensor> situ_glu_group_quant(const at::Tensor& x, const c10::optional<at::Tensor>& weight,
                                                        const c10::optional<at::Tensor>& group_index, double beta,
                                                        double linearBeta, bool activate_left, int64_t axis,
                                                        int64_t quant_mode, int64_t dst_type, double dst_type_max,
                                                        std::string round_mode, int64_t group_list_type)
{
    // 入参校验：设备/维度/数据类型/取值范围；更全的约束由 aclnn 壳防呆拦截。
    // group_index 可选（owner 决策）：不传 = 全行模式（所有行参与计算，由
    // kernel fullRowsMode 处理）；传 = recipes 分组稀疏语义（sum 钳制 + padding 跳行）。
    const bool isGrouped = !TensorAbsent(group_index);
    const bool hasWeight = !TensorAbsent(weight);
    CheckNpuTensor(x, "x");
    TORCH_CHECK(x.dim() >= 1, "x must be at least 1-dimensional, but got ", x.dim());
    const int64_t lastDim = x.size(x.dim() - 1);
    const int64_t epr = lastDim / kSplitFactor;
    TORCH_CHECK(lastDim > 0 && lastDim % kSplitFactor == 0, "x last dim must be positive and even, but got ", lastDim);
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x dtype must be float16 or bfloat16, but got ", x.scalar_type());
    if (isGrouped) {
        TORCH_CHECK(group_index->scalar_type() == at::kLong, "group_index must be int64, but got ",
                    group_index->scalar_type());
    }
    if (hasWeight) {
        TORCH_CHECK(weight->scalar_type() == at::kFloat, "weight must be float32, but got ", weight->scalar_type());
    }
    TORCH_CHECK(beta > 0.0, "beta must be greater than 0, but got ", beta);
    // linearBeta: <= 0 disables the up-branch tanh soft-clamp (up passthrough);
    // > 0 enables it — any value is legal, no rejection.

    // 扩展属性（situ_mx_quant caliber）：当前均为单一支持值，与 host tiling 校验口径一致
    TORCH_CHECK(axis == -1, "axis only supports -1 currently, but got ", axis);
    TORCH_CHECK(quant_mode == 0, "quant_mode only supports 0 (mx quantization) currently, but got ", quant_mode);
    TORCH_CHECK(dst_type == 36, "dst_type only supports 36 (FP8_E4M3FN) currently, but got ", dst_type);
    TORCH_CHECK(dst_type_max >= 0.0, "dst_type_max must be >= 0, but got ", dst_type_max);
    TORCH_CHECK(round_mode == "rint", "round_mode only supports \"rint\" currently, but got \"", round_mode, "\"");
    TORCH_CHECK(group_list_type == 1,
                "group_list_type=0 (cumsum mode) is reserved and not supported yet, pass 1 (count mode) instead, "
                "but got ",
                group_list_type);

    // 输出y：shape与x一致，最后一维减半（gate/up激活输出）
    auto yShape = op_infer::array_to_small_vector(x.sizes());
    yShape[x.dim() - 1] = epr;

    // 输出y_scale：最后一维替换为ceil(EPR / 64)，再追加长度为kScaleAlignNum的维度
    constexpr int64_t scaleGroupSize = kScaleAlignNum * kMxBlockSize;
    auto y_scaleShape = yShape;
    y_scaleShape[x.dim() - 1] = (epr + scaleGroupSize - 1) / scaleGroupSize;
    y_scaleShape.push_back(kScaleAlignNum);

    // y固定e4m3fn、y_scale固定e8m0。均用at::empty按目标dtype直接申请：
    // NPU zero_算子不支持FP8/E8M0 dtype，禁止用at::zeros初始化FP8类输出。
    at::Tensor y = at::empty(yShape, x.options().dtype(at::ScalarType::Float8_e4m3fn));
    at::Tensor y_scale = at::empty(y_scaleShape, x.options().dtype(at::ScalarType::Float8_e8m0fnu));

    // 拉起aclnnSituGluGroupQuant kernel，入参顺序与aclnn接口定义一致。
    // 注意：aclnn C接口的浮点标量原型为double（勿按文档float截断，否则按ABI错位读参）。
    char* roundModePtr = const_cast<char*>(round_mode.data());
    ACLNN_CMD(aclnnSituGluGroupQuant, x, weight, group_index, beta, linearBeta, activate_left, axis, quant_mode,
              dst_type, dst_type_max, roundModePtr, group_list_type, y, y_scale);

    return std::make_tuple(y, y_scale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("situ_glu_group_quant", &cann_ops_nn::quant::situ_glu_group_quant, "SituGluGroupQuant on NPU");
}
