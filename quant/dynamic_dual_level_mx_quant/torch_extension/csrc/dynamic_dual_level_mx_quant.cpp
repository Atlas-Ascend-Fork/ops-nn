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
namespace quant {
namespace {
// 第一级/第二级 MX 量化块大小，当前仅支持固定值
constexpr int64_t kLevel0BlockSize = 512;
constexpr int64_t kLevel1BlockSize = 32;
// level1_scale 每个 E8M0 对齐对覆盖 level1_block_size * 2 个元素
constexpr int64_t kLevel1ScaleGroupSize = kLevel1BlockSize * 2;
constexpr int64_t kMxScaleAlign = 2;
// FLOAT4_E2M1 每字节打包 2 个元素
constexpr int64_t kFp4PackFactor = 2;
// level1_scale 计算算法：0-OCP，1-ceil 向上取整，2-DynamicDtypeRange
constexpr int64_t kScaleAlgOcp = 0;
constexpr int64_t kScaleAlgDynamic = 2;
// dst_type_max 仅在 scale_alg=2 时生效，支持 0.0 或 [6.0, 12.0]
constexpr double kDstTypeMaxMin = 6.0;
constexpr double kDstTypeMaxMax = 12.0;

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

int64_t CeilDiv(int64_t value, int64_t factor) { return (value + factor - 1) / factor; }
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> dynamic_dual_level_mx_quant(
    const at::Tensor& x, const c10::optional<at::Tensor>& smooth_scale, const std::string& round_mode,
    int64_t level0_block_size, int64_t level1_block_size, int64_t scale_alg, double dst_type_max)
{
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(smooth_scale, "smooth_scale");

    TORCH_CHECK(x.dim() >= 1, "x must be at least 1-dimensional, but got ", x.dim());
    const int64_t last_dim = x.size(x.dim() - 1);
    TORCH_CHECK(last_dim % kFp4PackFactor == 0, "x last dim must be even when y is FLOAT4_E2M1, but got ", last_dim);
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x dtype must be float16 or bfloat16, but got ", x.scalar_type());
    TORCH_CHECK(round_mode == "rint" || round_mode == "round" || round_mode == "floor",
                "round_mode must be 'rint', 'round' or 'floor', but got ", round_mode);
    TORCH_CHECK(level0_block_size == kLevel0BlockSize, "level0_block_size only supports ", kLevel0BlockSize,
                ", but got ", level0_block_size);
    TORCH_CHECK(level1_block_size == kLevel1BlockSize, "level1_block_size only supports ", kLevel1BlockSize,
                ", but got ", level1_block_size);
    TORCH_CHECK(scale_alg >= kScaleAlgOcp && scale_alg <= kScaleAlgDynamic, "scale_alg must be 0, 1 or 2, but got ",
                scale_alg);
    if (scale_alg == kScaleAlgDynamic) {
        TORCH_CHECK(dst_type_max == 0.0 || (dst_type_max >= kDstTypeMaxMin && dst_type_max <= kDstTypeMaxMax),
                    "dst_type_max must be 0.0 or within [", kDstTypeMaxMin, ", ", kDstTypeMaxMax, "], but got ",
                    dst_type_max);
    } else {
        TORCH_CHECK(dst_type_max == 0.0, "dst_type_max must be 0.0 when scale_alg is not 2, but got ", dst_type_max);
    }
    if (smooth_scale.has_value() && smooth_scale.value().defined()) {
        const at::Tensor& smooth = smooth_scale.value();
        TORCH_CHECK(smooth.scalar_type() == x.scalar_type(), "smooth_scale dtype must be the same as x, but got ",
                    smooth.scalar_type(), " and ", x.scalar_type());
        TORCH_CHECK(smooth.dim() == 1 && smooth.size(0) == last_dim,
                    "smooth_scale must be 1-D and its size must equal x last dim ", last_dim, ", but got shape ",
                    smooth.sizes());
    }

    // y 为 FLOAT4_E2M1，按每字节 2 个元素打包后用 uint8 承载
    auto yShape = op_infer::array_to_small_vector(x.sizes());
    yShape[x.dim() - 1] = last_dim / kFp4PackFactor;
    // level0_scale 为 float32，最后一维为 ceil(N / level0_block_size)
    auto level0Shape = op_infer::array_to_small_vector(x.sizes());
    level0Shape[x.dim() - 1] = CeilDiv(last_dim, level0_block_size);
    // level1_scale 为 FLOAT8_E8M0，最后一维为 ceil(N / 64)，再追加长度为 2 的对齐维
    auto level1Shape = op_infer::array_to_small_vector(x.sizes());
    level1Shape[x.dim() - 1] = CeilDiv(last_dim, kLevel1ScaleGroupSize);
    level1Shape.push_back(kMxScaleAlign);

    at::Tensor y = at::empty(yShape, x.options().dtype(at::kByte));
    at::Tensor level0Scale = at::empty(level0Shape, x.options().dtype(at::kFloat));
    at::Tensor level1Scale = at::empty(level1Shape, x.options().dtype(at::ScalarType::Float8_e8m0fnu));

    // 打包后的 uint8 通过 TensorWrapper 还原为 FLOAT4_E2M1 逻辑 shape 传给 aclnn
    TensorWrapper yWrapper{y, ACL_FLOAT4_E2M1};
    // scale_alg=0 且 dst_type_max=0 时行为与 V1 一致，优先走 V1 以兼容旧版 CANN
    const bool useV2 = scale_alg != kScaleAlgOcp || dst_type_max != 0.0;
    if (useV2) {
        ACLNN_CMD(aclnnDynamicDualLevelMxQuantV2, x, smooth_scale, round_mode, level0_block_size, level1_block_size,
                  scale_alg, dst_type_max, yWrapper, level0Scale, level1Scale);
    } else {
        ACLNN_CMD(aclnnDynamicDualLevelMxQuant, x, smooth_scale, round_mode, level0_block_size, level1_block_size,
                  yWrapper, level0Scale, level1Scale);
    }
    return std::make_tuple(y, level0Scale, level1Scale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dynamic_dual_level_mx_quant", &cann_ops_nn::quant::dynamic_dual_level_mx_quant,
          "DynamicDualLevelMxQuant on NPU");
}
