/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <tuple>
#include <vector>
#include <torch/extension.h>
#include "aclnnop/aclnn_rotate_quant_v2.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace matmul {
namespace {
constexpr int64_t INT4_VALUES_PER_INT32 = 8;
constexpr int64_t FP4_VALUES_PER_BYTE = 2;
constexpr int64_t MX_SCALE_PAIR_SIZE = 64;
constexpr int64_t MX_SCALES_PER_PAIR = 2;
constexpr int64_t MIN_INPUT_RANK = 1;
constexpr int64_t MAX_INPUT_RANK = 7;
constexpr int64_t MATRIX_RANK = 2;
constexpr int64_t GROUP_LIST_TYPE_CUMSUM = 0;
constexpr int64_t GROUP_LIST_TYPE_COUNT = 1;

void CheckInputDevice(const at::Tensor& input, const at::Tensor& x, const char* name)
{
    TORCH_CHECK(input.defined(), name, " must be defined");
    TORCH_CHECK(input.device() == x.device(), name, " must be on ", x.device(), ", but got ", input.device());
}
} // namespace

std::tuple<at::Tensor, at::Tensor> rotate_quant(const at::Tensor& x, const at::Tensor& rotation,
                                                const c10::optional<at::Tensor>& alpha, c10::optional<int64_t> dstDtype,
                                                c10::optional<int64_t> axis,
                                                const c10::optional<std::string>& roundMode,
                                                c10::optional<int64_t> scaleAlg, c10::optional<double> dstTypeMax,
                                                c10::optional<bool> transposeY,
                                                const c10::optional<at::Tensor>& groupList, int64_t groupListType)
{
    TORCH_CHECK(x.defined() && x.device().type() == at::kPrivateUse1, "x must be on NPU device");
    CheckInputDevice(rotation, x, "rotation");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x dtype must be float16 or bfloat16, but got ", x.scalar_type());
    TORCH_CHECK(rotation.scalar_type() == x.scalar_type(), "x and rotation must have the same dtype, but got ",
                x.scalar_type(), " and ", rotation.scalar_type());
    TORCH_CHECK(x.dim() >= MIN_INPUT_RANK && x.dim() <= MAX_INPUT_RANK,
                "x must have between 1 and 7 dimensions, but got ", x.dim());
    TORCH_CHECK(groupListType == GROUP_LIST_TYPE_CUMSUM || groupListType == GROUP_LIST_TYPE_COUNT,
                "group_list_type must be 0 or 1, but got ", groupListType);
    if (alpha.has_value()) {
        CheckInputDevice(alpha.value(), x, "alpha");
    }
    const bool grouped = groupList.has_value();
    if (grouped) {
        CheckInputDevice(groupList.value(), x, "group_list");
    }

    const int64_t dstDtypeValue = dstDtype.value_or(static_cast<int64_t>(at::kChar));
    const bool isInt4 = dstDtypeValue == static_cast<int64_t>(at::ScalarType::QUInt4x2);
    const aclDataType outAclType = isInt4 ? ACL_INT32 : GetAclDataType(dstDtypeValue);
    at::ScalarType outScalarType;
    switch (outAclType) {
        case ACL_INT8:
            outScalarType = at::kChar;
            break;
        case ACL_INT32:
            TORCH_CHECK(isInt4, "INT4 output must be requested with dst_dtype=torch.quint4x2");
            outScalarType = at::kInt;
            break;
        case ACL_FLOAT4_E2M1:
            TORCH_CHECK(dstDtypeValue == static_cast<int64_t>(ACL_FLOAT4_E2M1) + g_toAclOffset,
                        "FP4 output must be requested with dst_dtype=torch_npu.float4_e2m1fn_x2");
            outScalarType = at::kByte;
            break;
        case ACL_FLOAT8_E5M2:
            outScalarType = at::kFloat8_e5m2;
            break;
        case ACL_FLOAT8_E4M3FN:
            outScalarType = at::kFloat8_e4m3fn;
            break;
        default:
            TORCH_CHECK(false, "unsupported dst_dtype for rotate_quant: ", dstDtypeValue);
    }
    const bool isMx = outAclType == ACL_FLOAT4_E2M1 || outAclType == ACL_FLOAT8_E5M2 || outAclType == ACL_FLOAT8_E4M3FN;
    TORCH_CHECK(!grouped || isMx, "grouped rotate_quant only supports MXFP4/MXFP8 output");
    TORCH_CHECK(isMx || x.dim() == MATRIX_RANK, "INT8/INT4 x must be 2-dimensional [M,N]");
    TORCH_CHECK(!grouped || x.dim() == MATRIX_RANK, "grouped x must be 2-dimensional [M,N]");

    std::vector<int64_t> outShape(x.sizes().begin(), x.sizes().end());
    const int64_t n = outShape.back();
    if (isInt4) {
        TORCH_CHECK(n % INT4_VALUES_PER_INT32 == 0, "INT4 N must be divisible by 8");
        outShape.back() /= INT4_VALUES_PER_INT32;
    } else if (outAclType == ACL_FLOAT4_E2M1) {
        TORCH_CHECK(n % FP4_VALUES_PER_BYTE == 0, "FP4 N must be divisible by 2");
        outShape.back() /= FP4_VALUES_PER_BYTE;
    }
    std::vector<int64_t> scaleShape;
    if (isMx) {
        scaleShape.assign(x.sizes().begin(), x.sizes().end());
        scaleShape.back() = n / MX_SCALE_PAIR_SIZE + (n % MX_SCALE_PAIR_SIZE != 0);
        scaleShape.push_back(MX_SCALES_PER_PAIR);
    } else {
        scaleShape = {x.size(0)};
    }
    auto y = at::empty(outShape, x.options().dtype(outScalarType));
    auto scale = at::empty(scaleShape, x.options().dtype(isMx ? at::kByte : at::kFloat));
    // Keep FP4 output packed in uint8 for Torch versions without native FP4 support.
    // Expose E8M0 scales with their native dtype for subsequent operators.
    if (isMx) {
        scale = scale.view(at::ScalarType::Float8_e8m0fnu);
    }
    // Torch counts packed FP4 bytes; ACLNN uses logical FP4 elements.
    TensorWrapper yWrapper{y, outAclType, outAclType == ACL_FLOAT4_E2M1 ? y.dim() - 1 : -1};
    TensorWrapper scaleWrapper{scale, isMx ? ACL_FLOAT8_E8M0 : ACL_FLOAT};

    const int64_t axisValue = axis.value_or(-1);
    const int64_t scaleAlgValue = scaleAlg.value_or(0);
    const double dstTypeMaxValue = dstTypeMax.value_or(0.0);
    const bool transposeYValue = transposeY.value_or(false);
    std::string roundModeValue = roundMode.value_or("rint");
    char* roundModePtr = roundModeValue.data();
    // Preserve input strides: ACLNN performs the same Contiguous conversion as ordinary RotateQuant.
    ACLNN_CMD(aclnnRotateQuantV2, x, rotation, alpha, groupList, groupListType, axisValue, roundModePtr, scaleAlgValue,
              dstTypeMaxValue, transposeYValue, yWrapper, scaleWrapper);
    return std::make_tuple(y, scale);
}

} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rotate_quant", &cann_ops_nn::matmul::rotate_quant, "RotateQuant on NPU");
}
