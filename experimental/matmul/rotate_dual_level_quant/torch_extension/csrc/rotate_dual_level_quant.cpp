/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <vector>
#include <torch/extension.h>
#include "aclnnop/aclnn_rotate_dual_level_quant.h"

// Declare before aclnn_common.h so its ConvertTypes template sees this overload.
// The local wrapper enforces ND without changing other operators' converters.
struct RotateDualLevelQuantNdTensor {
    const at::Tensor& tensor;
    aclDataType dtype;
    int64_t packedDim = -1;
};
inline aclTensor* ConvertType(const RotateDualLevelQuantNdTensor& wrapper);
#include "aclnn_common.h"

template <>
struct is_tensor_wrapper<RotateDualLevelQuantNdTensor> : std::true_type {};

inline aclTensor* ConvertType(const RotateDualLevelQuantNdTensor& wrapper)
{
    const auto& tensor = wrapper.tensor;
    if (!tensor.defined()) {
        return nullptr;
    }
    static const auto createTensor = GET_OP_API_FUNC(CreateTensor);
    TORCH_CHECK(createTensor != nullptr, "aclCreateTensor is unavailable");
    auto meta = PrepareTensorMeta(tensor, wrapper.dtype, wrapper.packedDim);
    // Base-format tensors retain logical sizes, strides and offsets. Rank must
    // not turn these ND descriptors into NCL/NCHW/NCDHW descriptors.
    return createTensor(meta.wrapperShape.data(), tensor.dim(), wrapper.dtype, meta.wrapperStride.data(),
                        ConvertToAclStorageOffset(tensor, wrapper.dtype), ACL_FORMAT_ND, meta.storageDims.data(),
                        meta.storageDims.size(), const_cast<void*>(tensor.storage().data()));
}

namespace cann_ops_nn {
namespace matmul {
namespace {
constexpr int64_t LEVEL0_BLOCK_SIZE = 512;
constexpr int64_t LEVEL1_BLOCK_SIZE = 32;
constexpr int64_t SCALE_PAIR_SIZE = 64;
constexpr int64_t FP4_VALUES_PER_BYTE = 2;

void CheckDevice(const at::Tensor& tensor, const at::Tensor& x, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device() == x.device(), name, " must be on ", x.device());
    TORCH_CHECK(IsOpInputBaseFormat(tensor), name, " must use a base storage format for ND conversion");
}

void CheckShapes(const at::Tensor& x, const at::Tensor& rotation, const c10::optional<at::Tensor>& alpha,
                 const c10::optional<at::Tensor>& groups)
{
    TORCH_CHECK(x.dim() >= 1 && x.dim() <= 7, "x rank must be in [1,7]");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "x must be FP16/BF16");
    TORCH_CHECK(rotation.scalar_type() == x.scalar_type(), "rotation dtype must match x");
    const bool grouped = groups.has_value();
    TORCH_CHECK(grouped ? (rotation.dim() == 3 || rotation.dim() == 4) : (rotation.dim() == 2 || rotation.dim() == 3),
                "rotation rank must be 2/3 in ordinary mode or 3/4 in grouped mode");
    const int64_t n = x.size(-1);
    const int64_t k = rotation.size(-1);
    TORCH_CHECK(k == 32 || k == 64 || k == 128 || k == 256, "rotation K must be 32/64/128/256");
    TORCH_CHECK(rotation.size(-2) == k && n % k == 0, "rotation must be square and K must divide N");
    if (rotation.dim() == (grouped ? 4 : 3)) {
        TORCH_CHECK(rotation.size(-3) == n / k, "rotation column-block count must equal N/K");
    }
    bool zeroLeading = false;
    for (int64_t i = 0; i < x.dim(); ++i) {
        TORCH_CHECK(x.size(i) <= std::numeric_limits<int32_t>::max(), "x dimensions must fit INT32");
        zeroLeading = zeroLeading || (i + 1 < x.dim() && x.size(i) == 0);
    }
    if (!zeroLeading) {
        int64_t rows = 1;
        for (int64_t i = 0; i + 1 < x.dim(); ++i) {
            TORCH_CHECK(rows <= std::numeric_limits<int32_t>::max() / x.size(i), "flattened M must fit INT32");
            rows *= x.size(i);
        }
    }
    if (alpha.has_value()) {
        const auto& a = alpha.value();
        TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.dim() == 1 && a.size(0) == 1, "alpha must be BF16 [1]");
    }
    if (grouped) {
        const auto& g = groups.value();
        TORCH_CHECK(x.dim() == 2 && n > 0, "grouped x must be [M,N] with N>0");
        TORCH_CHECK(g.scalar_type() == at::kLong && g.dim() == 1, "group_list must be INT64 [E]");
        TORCH_CHECK(g.size(0) >= 1 && g.size(0) <= 1024 && rotation.size(0) == g.size(0),
                    "group count must match rotation and be in [1,1024]");
    }
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> rotate_dual_level_quant(
    const at::Tensor& x, const at::Tensor& rotation, const c10::optional<at::Tensor>& alpha,
    c10::optional<int64_t> dst_dtype, c10::optional<int64_t> axis, const c10::optional<std::string>& round_mode,
    c10::optional<int64_t> scale_alg, c10::optional<double> dst_type_max, c10::optional<bool> transpose_y,
    const c10::optional<at::Tensor>& group_list, int64_t group_list_type, int64_t level0_block_size,
    int64_t level1_block_size)
{
    TORCH_CHECK(x.defined() && x.device().type() == at::kPrivateUse1, "x must be on NPU");
    CheckDevice(x, x, "x");
    CheckDevice(rotation, x, "rotation");
    if (alpha.has_value()) {
        CheckDevice(alpha.value(), x, "alpha");
    }
    if (group_list.has_value()) {
        CheckDevice(group_list.value(), x, "group_list");
    }
    CheckShapes(x, rotation, alpha, group_list);
    TORCH_CHECK(level0_block_size == LEVEL0_BLOCK_SIZE && level1_block_size == LEVEL1_BLOCK_SIZE,
                "level0_block_size must be 512 and level1_block_size must be 32");
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1, "group_list_type must be 0/1");
    const int64_t dtype_value = dst_dtype.value_or(static_cast<int64_t>(ACL_FLOAT4_E2M1) + g_toAclOffset);
    const aclDataType out_acl_type = GetAclDataType(dtype_value);
    at::ScalarType out_type;
    switch (out_acl_type) {
        case ACL_FLOAT4_E2M1:
            TORCH_CHECK(dtype_value == static_cast<int64_t>(ACL_FLOAT4_E2M1) + g_toAclOffset,
                        "FP4 dst_dtype must be torch_npu.float4_e2m1fn_x2 (296)");
            out_type = at::kByte;
            break;
        case ACL_FLOAT8_E4M3FN:
            out_type = at::kFloat8_e4m3fn;
            break;
        case ACL_FLOAT8_E5M2:
            out_type = at::kFloat8_e5m2;
            break;
        default:
            TORCH_CHECK(false, "dst_dtype must be MXFP4 or MXFP8");
    }
    const bool fp4 = out_acl_type == ACL_FLOAT4_E2M1;
    const int64_t axis_value = axis.value_or(-1);
    const int64_t alg = scale_alg.value_or(0);
    const double max_value = dst_type_max.value_or(0.0);
    const bool trans = transpose_y.value_or(false);
    std::string rounding = round_mode.value_or("rint");
    TORCH_CHECK(axis_value == -1 || axis_value == x.dim() - 1, "axis must select the last dimension");
    TORCH_CHECK(!trans, "transpose_y must be False");
    TORCH_CHECK(rounding == "rint" || rounding == "round" || rounding == "floor", "invalid round_mode");
    TORCH_CHECK(fp4 || rounding == "rint", "FP8 only supports rint");
    TORCH_CHECK(fp4 ? (alg == 0 || alg == 2) : (alg == 0 || alg == 1), "invalid scale_alg for dst_dtype");
    TORCH_CHECK(std::isfinite(max_value) && (alg == 2 ? (max_value >= 6.0 && max_value <= 12.0) : max_value == 0.0),
                "dst_type_max must be finite in [6,12] for scale_alg=2, otherwise 0");

    const int64_t n = x.size(-1);
    std::vector<int64_t> y_shape(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> level0_shape = y_shape;
    std::vector<int64_t> level1_shape = y_shape;
    if (fp4) {
        y_shape.back() /= FP4_VALUES_PER_BYTE;
    }
    level0_shape.back() = (n + LEVEL0_BLOCK_SIZE - 1) / LEVEL0_BLOCK_SIZE;
    level1_shape.back() = (n + SCALE_PAIR_SIZE - 1) / SCALE_PAIR_SIZE;
    level1_shape.push_back(2);
    auto y = at::empty(y_shape, x.options().dtype(out_type));
    auto level0 = at::empty(level0_shape, x.options().dtype(at::kFloat));
    auto level1 = at::empty(level1_shape, x.options().dtype(at::kByte)).view(at::ScalarType::Float8_e8m0fnu);
    at::Tensor alpha_tensor = alpha.value_or(at::Tensor());
    at::Tensor group_tensor = group_list.value_or(at::Tensor());
    const auto input_type = ConvertToAclDataType(x.scalar_type());
    RotateDualLevelQuantNdTensor x_wrapper{x, input_type};
    RotateDualLevelQuantNdTensor rotation_wrapper{rotation, input_type};
    RotateDualLevelQuantNdTensor alpha_wrapper{alpha_tensor, ACL_BF16};
    RotateDualLevelQuantNdTensor group_wrapper{group_tensor, ACL_INT64};
    RotateDualLevelQuantNdTensor y_wrapper{y, out_acl_type, fp4 ? y.dim() - 1 : -1};
    RotateDualLevelQuantNdTensor level0_wrapper{level0, ACL_FLOAT};
    RotateDualLevelQuantNdTensor level1_wrapper{level1, ACL_FLOAT8_E8M0};
    char* round_ptr = rounding.data();
    ACLNN_CMD(aclnnRotateDualLevelQuant, x_wrapper, rotation_wrapper, alpha_wrapper, group_wrapper, group_list_type,
              axis_value, round_ptr, alg, max_value, trans, level0_block_size, level1_block_size, y_wrapper,
              level0_wrapper, level1_wrapper);
    return std::make_tuple(y, level0, level1);
}
} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rotate_dual_level_quant", &cann_ops_nn::matmul::rotate_dual_level_quant, "RotateDualLevelQuant on NPU");
}
