/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnnop/aclnn_flat_quant_v3.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {

constexpr int64_t QUINT4X2_DTYPE = 16;
constexpr int64_t FLOAT4_E2M1FN_X2_DTYPE = 296;
constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;
constexpr int64_t FP4_NUMS_PER_BYTE = 2;
constexpr int64_t MXFP4_SCALE_BLOCK_SIZE = 64;
constexpr int64_t MXFP4_SCALE_LAST_DIM = 2;
constexpr double DST_TYPE_MAX_MIN = 6.0;
constexpr double DST_TYPE_MAX_MAX = 12.0;

std::tuple<at::Tensor, at::Tensor> flat_quant(const at::Tensor& x, const at::Tensor& kroneckerP1,
                                              const at::Tensor& kroneckerP2, double clipRatio, int64_t dstDtype,
                                              double dstTypeMax, const c10::optional<at::Tensor>& groupList,
                                              int64_t groupListType)
{
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on NPU device");
    TORCH_CHECK(kroneckerP1.device().type() == at::kPrivateUse1, "kronecker_p1 must be on NPU device");
    TORCH_CHECK(kroneckerP2.device().type() == at::kPrivateUse1, "kronecker_p2 must be on NPU device");

    TORCH_CHECK(x.dim() == 3, "x must be 3-dimensional [M, N1, N2], but got ", x.dim());
    int64_t m = x.size(0);
    int64_t n1 = x.size(1);
    int64_t n2 = x.size(2);

    TORCH_CHECK(kroneckerP1.dim() == 2, "kronecker_p1 must be 2-dimensional [N1, N1], but got ", kroneckerP1.dim());
    TORCH_CHECK(kroneckerP1.size(0) == n1 && kroneckerP1.size(1) == n1, "kronecker_p1 shape must be [", n1, ", ", n1,
                "], but got [", kroneckerP1.size(0), ", ", kroneckerP1.size(1), "]");

    TORCH_CHECK(kroneckerP2.dim() == 2, "kronecker_p2 must be 2-dimensional [N2, N2], but got ", kroneckerP2.dim());

    auto inputDtype = x.scalar_type();
    TORCH_CHECK(inputDtype == at::kHalf || inputDtype == at::kBFloat16, "x dtype must be float16 or bfloat16, but got ",
                inputDtype);
    TORCH_CHECK(kroneckerP1.scalar_type() == inputDtype, "kronecker_p1 dtype must match x dtype: ", inputDtype, " vs ",
                kroneckerP1.scalar_type());
    TORCH_CHECK(kroneckerP2.scalar_type() == inputDtype, "kronecker_p2 dtype must match x dtype: ", inputDtype, " vs ",
                kroneckerP2.scalar_type());

    if (groupList.has_value() && groupList.value().defined()) {
        TORCH_CHECK(groupList.value().device().type() == at::kPrivateUse1, "group_list must be on NPU device");
        TORCH_CHECK(groupList.value().scalar_type() == at::kLong, "group_list dtype must be INT64, but got ",
                    groupList.value().scalar_type());
        auto groupShape = groupList.value().sizes();
        TORCH_CHECK(groupShape.size() == 1 || groupShape.size() == 2, "group_list must be 1D or 2D, but got ",
                    groupShape.size(), "D");
    }

    TORCH_CHECK(clipRatio > 0.0 && clipRatio <= 1.0, "clip_ratio must be in range (0, 1], but got ", clipRatio);
    TORCH_CHECK(dstTypeMax == 0.0 || (dstTypeMax >= DST_TYPE_MAX_MIN && dstTypeMax <= DST_TYPE_MAX_MAX),
                "dst_type_max must be 0 or in range [6, 12], but got ", dstTypeMax);
    TORCH_CHECK(groupListType >= 0 && groupListType <= 2, "group_list_type must be in range [0, 2], but got ",
                groupListType);

    aclDataType outAclType = ACL_INT32;
    aclDataType scaleAclType = ACL_FLOAT;
    if (dstDtype == QUINT4X2_DTYPE) {
        // torch.quint4x2: the output tensor is INT32, 8 INT4 elements packed into 1 INT32.
        TORCH_CHECK(n2 % INT4_NUMS_IN_INT32_SPACE == 0, "N2 must be divisible by 8 for INT4 output, but got N2=", n2);
    } else if (dstDtype == FLOAT4_E2M1FN_X2_DTYPE) {
        outAclType = ACL_FLOAT4_E2M1;
        scaleAclType = ACL_FLOAT8_E8M0;
        TORCH_CHECK(n2 % 2 == 0, "N2 must be even for FP4 output, but got N2=", n2);
    } else {
        TORCH_CHECK(false, "dst_dtype must be INT4 or FP4, got dst_dtype=", dstDtype);
    }

    bool isP2ZeroShape = (kroneckerP2.size(0) == 0 && kroneckerP2.size(1) == 0);
    if (outAclType == ACL_FLOAT4_E2M1) {
        TORCH_CHECK(isP2ZeroShape || (kroneckerP2.size(0) == n2 && kroneckerP2.size(1) == n2),
                    "kronecker_p2 shape must be [", n2, ", ", n2, "] or [0, 0] for FP4 output, but got [",
                    kroneckerP2.size(0), ", ", kroneckerP2.size(1), "]");
    } else {
        TORCH_CHECK(kroneckerP2.size(0) == n2 && kroneckerP2.size(1) == n2, "kronecker_p2 shape must be [", n2, ", ",
                    n2, "], but got [", kroneckerP2.size(0), ", ", kroneckerP2.size(1), "]");
    }

    std::vector<int64_t> outShape;
    std::vector<int64_t> quantScaleShape;
    if (outAclType == ACL_INT32) {
        // INT4: [M, N1, N2/8].
        outShape = {m, n1, n2 / INT4_NUMS_IN_INT32_SPACE};
        quantScaleShape = {m};
    } else {
        // FP4: [M, N1*N2/2] with scale [M, ceil(N1*N2/64), 2].
        outShape = {m, n1 * n2 / FP4_NUMS_PER_BYTE};
        quantScaleShape = {m, (n1 * n2 + MXFP4_SCALE_BLOCK_SIZE - 1) / MXFP4_SCALE_BLOCK_SIZE, MXFP4_SCALE_LAST_DIM};
    }

    at::ScalarType outScalarType = outAclType == ACL_INT32 ? at::kInt : at::kByte;
    at::Tensor out = at::empty(outShape, at::TensorOptions().dtype(outScalarType).device(at::kPrivateUse1));
    at::ScalarType scaleScalarType = outAclType == ACL_INT32 ? at::kFloat : at::kByte;
    at::Tensor quantScale = at::empty(quantScaleShape,
                                      at::TensorOptions().dtype(scaleScalarType).device(at::kPrivateUse1));

    TensorWrapper outWrapper{out, outAclType};
    TensorWrapper scaleWrapper{quantScale, scaleAclType};
    ACLNN_CMD(aclnnFlatQuantV3, x, kroneckerP1, kroneckerP2, groupList, clipRatio, dstTypeMax, groupListType,
              outWrapper, scaleWrapper);

    return std::make_tuple(out, quantScale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flat_quant", &cann_ops_nn::quant::flat_quant, "FlatQuant quantization on NPU");
}
