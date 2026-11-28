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

namespace {
constexpr int64_t OFFSET_32_BITS = 32;
constexpr int64_t OFFSET_16_BITS = 16;
constexpr uint64_t GROUP_MAX = 65535UL;
constexpr size_t GROUP_DIM = 3;

void CheckPermList(const c10::optional<std::vector<int64_t>>& perm, const char* name)
{
    if (!perm.has_value()) {
        return;
    }
    TORCH_CHECK(perm->size() == GROUP_DIM, name, " must contain 3 elements, but got ", perm->size());
    for (const auto value : perm.value()) {
        TORCH_CHECK(value >= 0 && value < static_cast<int64_t>(GROUP_DIM), name,
                    " elements must be in [0, 2], but got ", value);
    }
}

int64_t CheckAndGetGroupSize(at::IntArrayRef groupSizeList)
{
    int64_t groups = 0;
    if (groupSizeList.empty()) {
        return groups;
    }
    size_t groupDim = groupSizeList.size();
    TORCH_CHECK(groupDim == GROUP_DIM, "group_sizes only support input with three elements, but got ", groupDim);
    int64_t groupM = static_cast<int64_t>(groupSizeList[0]);
    int64_t groupN = static_cast<int64_t>(groupSizeList[1]);
    int64_t groupK = static_cast<int64_t>(groupSizeList[2]);
    bool invalidGroupParam = ((groupM <= GROUP_MAX && groupM >= 0) && (groupN <= GROUP_MAX && groupN >= 0) &&
                              (groupK <= GROUP_MAX && groupK >= 0));
    TORCH_CHECK(invalidGroupParam, "group param value must conform to range [0, 65535]");
    groups = static_cast<int64_t>((static_cast<uint64_t>(groupM) << OFFSET_32_BITS) +
                                  (static_cast<uint64_t>(groupN) << OFFSET_16_BITS) + static_cast<uint64_t>(groupK));
    return groups;
}
} // namespace

namespace cann_ops_nn {
namespace matmul {

at::Tensor transpose_quant_batch_mat_mul(
    const at::Tensor& x1, const at::Tensor& x2, int64_t dtype, const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& x1Scale, const c10::optional<at::Tensor>& x2Scale,
    c10::optional<std::vector<int64_t>>& groupSizes, c10::optional<std::vector<int64_t>>& permX1,
    c10::optional<std::vector<int64_t>>& permX2, c10::optional<std::vector<int64_t>>& permY,
    c10::optional<int64_t> batchSplitFactor, c10::optional<int64_t> x1Dtype, c10::optional<int64_t> x2Dtype,
    c10::optional<int64_t> x1ScaleDtype, c10::optional<int64_t> x2ScaleDtype)
{
    const at::Tensor& biasReal = bias.value_or(at::Tensor());
    const at::Tensor& x1ScaleReal = x1Scale.value_or(at::Tensor());
    const at::Tensor& x2ScaleReal = x2Scale.value_or(at::Tensor());

    // The torch interface supports MX fp4/fp8 (float4_e2m1/float8_e4m3fn with
    // e8m0 scale), FP8 (float8_e4m3fn/float8_e5m2 with float scale), HIFP8
    // (hifloat8 with uint64 scale), HIFP4 (hifloat4 with hifloat4_scale), and
    // A2/A3 INT8 W8A8. The aclnn op_api layer handles dtype dispatch.
    const aclDataType x1Acl = x1Dtype.has_value() ? GetAclDataType(x1Dtype.value()) :
                                                    ConvertToAclDataType(x1.scalarType());
    const aclDataType x2Acl = x2Dtype.has_value() ? GetAclDataType(x2Dtype.value()) :
                                                    ConvertToAclDataType(x2.scalarType());
    const bool x1Supported = (x1Acl == ACL_FLOAT4_E2M1) || (x1Acl == ACL_FLOAT8_E4M3FN) || (x1Acl == ACL_FLOAT8_E5M2) ||
                             (x1Acl == ACL_HIFLOAT8) || (x1Acl == ACL_INT8) || (x1Acl == ACL_HIFLOAT4_NUM);
    const bool x2Supported = (x2Acl == ACL_FLOAT4_E2M1) || (x2Acl == ACL_FLOAT8_E4M3FN) || (x2Acl == ACL_FLOAT8_E5M2) ||
                             (x2Acl == ACL_HIFLOAT8) || (x2Acl == ACL_INT8) || (x2Acl == ACL_HIFLOAT4_NUM);
    TORCH_CHECK(x1Supported,
                "x1 must be float4_e2m1, float8_e4m3fn, float8_e5m2, hifloat8, hifloat4, or int8 in the torch "
                "interface, got aclDataType ",
                x1Acl);
    TORCH_CHECK(x2Supported,
                "x2 must be float4_e2m1, float8_e4m3fn, float8_e5m2, hifloat8, hifloat4, or int8 in the torch "
                "interface, got aclDataType ",
                x2Acl);
    const aclDataType x1ScaleAcl = x1ScaleDtype.has_value() ?
                                       GetAclDataType(x1ScaleDtype.value()) :
                                       (x1ScaleReal.defined() ? ConvertToAclDataType(x1ScaleReal.scalarType()) :
                                                                ACL_DT_UNDEFINED);
    const aclDataType x2ScaleAcl = x2ScaleDtype.has_value() ?
                                       GetAclDataType(x2ScaleDtype.value()) :
                                       (x2ScaleReal.defined() ? ConvertToAclDataType(x2ScaleReal.scalarType()) :
                                                                ACL_DT_UNDEFINED);
    // Scale dtype validation: e8m0 for MX, float for FP8, uint64 for HIFP8,
    // hifloat4_scale for HIFP4.
    const bool isHifp4 = (x1Acl == ACL_HIFLOAT4_NUM) || (x2Acl == ACL_HIFLOAT4_NUM);
    const bool isHifp8 = !isHifp4 && ((x1Acl == ACL_HIFLOAT8) || (x2Acl == ACL_HIFLOAT8));
    const bool isMx = !isHifp4 && !isHifp8 && (x1ScaleAcl == ACL_FLOAT8_E8M0 || x2ScaleAcl == ACL_FLOAT8_E8M0);
    if (isHifp4) {
        TORCH_CHECK(x1Acl == ACL_HIFLOAT4_NUM && x2Acl == ACL_HIFLOAT4_NUM,
                    "x1 and x2 must both be hifloat4 for HIFP4 input");
        TORCH_CHECK(x1ScaleReal.defined() && x2ScaleReal.defined(), "x1_scale/x2_scale are required for HIFP4 input");
        constexpr int64_t kHifp4ScaleElemBytes = 4;
        TORCH_CHECK(
            x1ScaleReal.element_size() == kHifp4ScaleElemBytes && x2ScaleReal.element_size() == kHifp4ScaleElemBytes,
            "x1_scale/x2_scale must use 4-byte storage (e.g. float32) for HIFP4 input, got ",
            x1ScaleReal.element_size(), " and ", x2ScaleReal.element_size(), " bytes");
        TORCH_CHECK(x1ScaleAcl == ACL_HIFLOAT4_SCALE_NUM && x2ScaleAcl == ACL_HIFLOAT4_SCALE_NUM,
                    "x1_scale/x2_scale must be hifloat4_scale for HIFP4 input, got aclDataType ", x1ScaleAcl, " and ",
                    x2ScaleAcl);
    } else if (isMx) {
        TORCH_CHECK(x1ScaleAcl == ACL_DT_UNDEFINED || x1ScaleAcl == ACL_FLOAT8_E8M0,
                    "x1_scale must be float8_e8m0 (MX scale) for MX input, got aclDataType ", x1ScaleAcl);
        TORCH_CHECK(x2ScaleAcl == ACL_DT_UNDEFINED || x2ScaleAcl == ACL_FLOAT8_E8M0,
                    "x2_scale must be float8_e8m0 (MX scale) for MX input, got aclDataType ", x2ScaleAcl);
    } else if (isHifp8) {
        // HIFP8: x2Scale is a per-channel 64-bit scale. torch uint64 tensors are not
        // available on every backend, so int64 is accepted too and reinterpreted as
        // uint64 when building the aclnn argument (see x2ScaleAcltype below).
        TORCH_CHECK(x2ScaleAcl == ACL_UINT64 || x2ScaleAcl == ACL_INT64 || x2ScaleAcl == ACL_DT_UNDEFINED,
                    "x2_scale must carry the uint64 bit pattern (int64 is reinterpreted bitwise) for HIFP8 ",
                    "input, got aclDataType ", x2ScaleAcl);
    } else {
        // FP8: x2Scale must be float (per-channel) or undefined; x1Scale optional (float per-token)
        TORCH_CHECK(x2ScaleAcl == ACL_DT_UNDEFINED || x2ScaleAcl == ACL_FLOAT,
                    "x2_scale must be float (FP8 per-channel scale) for FP8 input, got aclDataType ", x2ScaleAcl);
    }

    const int64_t bIdx = 0;
    const int64_t mIdx = 1;
    const int64_t kaIdx = 2;
    const int64_t kbIdx = 1;
    const int64_t nIdx = 2;
    const std::vector<int64_t> defaultPermX1 = {mIdx, bIdx, kaIdx};
    const std::vector<int64_t> defaultPermX2 = {bIdx, kbIdx, nIdx};
    const std::vector<int64_t> defaultPermY = {mIdx, bIdx, nIdx};
    const std::vector<int64_t> defaultGroupSizes = {0, 0, 0};

    CheckPermList(permX1, "perm_x1");
    CheckPermList(permX2, "perm_x2");
    CheckPermList(permY, "perm_y");
    const auto permX1Real = permX1.has_value() ? at::IntArrayRef(permX1.value()) : at::IntArrayRef(defaultPermX1);
    const auto permX2Real = permX2.has_value() ? at::IntArrayRef(permX2.value()) : at::IntArrayRef(defaultPermX2);
    const auto permYReal = permY.has_value() ? at::IntArrayRef(permY.value()) : at::IntArrayRef(defaultPermY);
    int64_t groupSizeValue = CheckAndGetGroupSize(groupSizes.has_value() ? at::IntArrayRef(groupSizes.value()) :
                                                                           at::IntArrayRef(defaultGroupSizes));
    int32_t batchSplitFactorValue = static_cast<int32_t>(batchSplitFactor.value_or(1));

    // Is4BitDtype() covers ACL_HIFLOAT4_NUM (see aclnn_common.h), so HIFP4
    // uses the same packed-axis logical-length doubling as MXFP4.
    const bool x1IsFp4 = Is4BitDtype(x1Acl);
    const bool x2IsFp4 = Is4BitDtype(x2Acl);
    const int64_t x1LastDim = static_cast<int64_t>(x1.sizes().size()) - 1;
    const int64_t x2LastDim = static_cast<int64_t>(x2.sizes().size()) - 1;

    auto mDim = x1.size(permX1Real[1]);
    if (x1IsFp4 && permX1Real[1] == x1LastDim) {
        mDim *= FP4_IN_INT8;
    }
    auto batchDim = x1.size(permX1Real[0]);
    auto nDim = x2.size(permX2Real[2]);
    if (x2IsFp4 && permX2Real[2] == x2LastDim) {
        nDim *= FP4_IN_INT8;
    }

    c10::SmallVector<int64_t, op_infer::SIZE> outputSize = {mDim, batchDim, nDim};

    if (batchSplitFactorValue > 1) {
        outputSize = {batchSplitFactorValue, mDim, batchDim * nDim / batchSplitFactorValue};
    }

    aclDataType dtypeValue = static_cast<aclDataType>(dtype);
    TORCH_CHECK(!(isHifp4 && dtypeValue == ACL_HIFLOAT8), "hifp4 output supports only float16 (1) and bfloat16 (27)");
    at::ScalarType scalarType;
    if (dtypeValue == ACL_FLOAT16) {
        scalarType = at::ScalarType::Half;
    } else if (dtypeValue == ACL_BF16) {
        scalarType = at::ScalarType::BFloat16;
    } else if (dtypeValue == ACL_HIFLOAT8) {
        scalarType = at::ScalarType::Byte;
    } else {
        TORCH_CHECK(false, "unsupported output dtype, only support float16(1), bfloat16(27) and hifloat8(34), got ",
                    dtype);
    }

    at::Tensor result = at::empty(outputSize, at::TensorOptions().dtype(scalarType).device(at::kPrivateUse1));

    bool isNz = at_npu::native::get_npu_format(x2) == ACL_FORMAT_FRACTAL_NZ;
    TORCH_CHECK(!(isHifp4 && isNz), "HIFP4 does not support x2 FRACTAL_NZ in the torch interface");
    bool isNdNzFormat = isNz && at_npu::native::get_npu_format(x1) != ACL_FORMAT_FRACTAL_NZ;

    TensorWrapper x1Wrapper = {
        x1, x1Dtype.has_value() ? GetAclDataType(x1Dtype.value()) : ConvertToAclDataType(x1.scalarType())};
    TensorWrapper x2Wrapper = {
        x2, x2Dtype.has_value() ? GetAclDataType(x2Dtype.value()) : ConvertToAclDataType(x2.scalarType())};
    TensorWrapper resultWrapper = {result, dtypeValue};

    aclDataType x1ScaleAcltype = ACL_DT_UNDEFINED;
    if (x1ScaleDtype.has_value()) {
        x1ScaleAcltype = GetAclDataType(x1ScaleDtype.value());
    } else if (x1ScaleReal.defined()) {
        x1ScaleAcltype = ConvertToAclDataType(x1ScaleReal.scalarType());
    }
    TensorWrapper x1ScaleWrapper = {x1ScaleReal, x1ScaleAcltype};

    aclDataType x2ScaleAcltype = ACL_DT_UNDEFINED;
    if (x2ScaleDtype.has_value()) {
        x2ScaleAcltype = GetAclDataType(x2ScaleDtype.value());
    } else if (x2ScaleReal.defined()) {
        x2ScaleAcltype = ConvertToAclDataType(x2ScaleReal.scalarType());
    }
    if (isHifp8 && x2ScaleAcltype == ACL_INT64) {
        x2ScaleAcltype = ACL_UINT64;
    }
    TensorWrapper x2ScaleWrapper = {x2ScaleReal, x2ScaleAcltype};
    int32_t dtypeValueI32 = static_cast<int32_t>(dtypeValue);

    if (isNdNzFormat) {
        auto weightNzFunc = GetOpApiFuncAddr("aclnnTransposeQuantBatchMatMulWeightNzGetWorkspaceSize");
        TORCH_CHECK(weightNzFunc != nullptr,
                    "In the current CANN version, aclnnTransposeQuantBatchMatMulWeightNz does not support "
                    "x2 as WeightNz input. Please upgrade the CANN package to version 9.1 or higher, "
                    "or set the x2 to ND mode.");
        ACLNN_CMD(aclnnTransposeQuantBatchMatMulWeightNz, x1Wrapper, x2Wrapper, biasReal, x1ScaleWrapper,
                  x2ScaleWrapper, dtypeValueI32, groupSizeValue, permX1Real, permX2Real, permYReal,
                  batchSplitFactorValue, resultWrapper);
    } else {
        ACLNN_CMD(aclnnTransposeQuantBatchMatMul, x1Wrapper, x2Wrapper, biasReal, x1ScaleWrapper, x2ScaleWrapper,
                  dtypeValueI32, groupSizeValue, permX1Real, permX2Real, permYReal, batchSplitFactorValue,
                  resultWrapper);
    }

    return result;
}

} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("transpose_quant_batch_mat_mul", &cann_ops_nn::matmul::transpose_quant_batch_mat_mul,
          "TransposeQuantBatchMatMul on NPU");
}
