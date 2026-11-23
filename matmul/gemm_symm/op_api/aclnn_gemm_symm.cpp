/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_gemm_symm.h"

#include <string>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "gemm_symm.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace {
constexpr float kDefaultScale = 1.0F;

struct GemmSymmParams {
    const aclTensor* a{nullptr};
    const aclTensor* b{nullptr};
    const aclTensor* c{nullptr};
    aclTensor* y{nullptr};
    const aclScalar* alphaOptional{nullptr};
    const aclScalar* betaOptional{nullptr};
    bool transposeA{false};
    bool transposeB{false};
    const char* fillMode{nullptr};
    int8_t cubeMathType{0};
};

float ReadScale(const aclScalar* scalar, float fallback) { return scalar == nullptr ? fallback : scalar->ToFloat(); }

int64_t LogicalM(const aclTensor* tensor, bool transposed)
{
    const auto& shape = tensor->GetViewShape();
    const int64_t rank = shape.GetDimNum();
    return transposed ? shape.GetDim(rank - 1) : shape.GetDim(rank - 2);
}

int64_t LogicalK(const aclTensor* tensor, bool transposed)
{
    const auto& shape = tensor->GetViewShape();
    const int64_t rank = shape.GetDimNum();
    return transposed ? shape.GetDim(rank - 2) : shape.GetDim(rank - 1);
}

aclnnStatus CheckParams(const GemmSymmParams& params)
{
    OP_CHECK_NULL(params.a, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.b, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.y, return ACLNN_ERR_PARAM_NULLPTR);
    const auto dtype = params.a->GetDataType();
    if ((dtype != DataType::DT_FLOAT16 && dtype != DataType::DT_BF16) || params.b->GetDataType() != dtype ||
        params.y->GetDataType() != dtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GemmSymm supports float16 and bfloat16, and a, b, y must match.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t rank = params.a->GetViewShape().GetDimNum();
    if ((rank != 2 && rank != 3) || params.b->GetViewShape().GetDimNum() != rank ||
        params.y->GetViewShape().GetDimNum() != rank) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "a, b and y must be rank 2 or 3.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const float beta = ReadScale(params.betaOptional, kDefaultScale);
    if (beta != 0.0F && params.c == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "c is required when beta is not 0.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (params.c != nullptr && (params.c->GetDataType() != dtype || params.c->GetViewShape().GetDimNum() != rank)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "c must match a in dtype and rank.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.fillMode != nullptr && std::string(params.fillMode) != "full") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fillMode only supports full.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cubeMathType < 0 || params.cubeMathType > 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cubeMathType supports 0, 1, 2, 3 and 4.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t m = LogicalM(params.a, params.transposeA);
    const int64_t k = LogicalK(params.a, params.transposeA);
    const int64_t batch = rank == 3 ? params.a->GetViewShape().GetDim(0) : 1;
    if (m <= 0 || k < 0 || LogicalK(params.b, !params.transposeB) != k || LogicalM(params.b, !params.transposeB) != m ||
        LogicalM(params.y, false) != m || LogicalK(params.y, false) != m ||
        (rank == 3 && (params.b->GetViewShape().GetDim(0) != batch || params.y->GetViewShape().GetDim(0) != batch))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "b or y shape does not match a.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.c != nullptr && (LogicalM(params.c, false) != m || LogicalK(params.c, false) != m ||
                                (rank == 3 && params.c->GetViewShape().GetDim(0) != batch))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "c must be [..., m, m] and match the batch of a.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND950) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "GemmSymm only supports Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus BuildGraph(const GemmSymmParams& params, aclOpExecutor* executor)
{
    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto aContig = l0op::Contiguous(params.a, executor);
    auto bContig = l0op::Contiguous(params.b, executor);
    CHECK_RET(aContig != nullptr && bContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* cContig = nullptr;
    if (params.c != nullptr) {
        cContig = l0op::Contiguous(params.c, executor);
        CHECK_RET(cContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto yContig = l0op::Contiguous(params.y, executor);
    CHECK_RET(yContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const float alpha = ReadScale(params.alphaOptional, kDefaultScale);
    const float beta = ReadScale(params.betaOptional, kDefaultScale);
    auto result = l0op::GemmSymm(aContig, bContig, cContig, yContig, alpha, beta, params.transposeA, params.transposeB,
                                 params.fillMode, params.cubeMathType, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result, params.y, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGemmSymmGetWorkspaceSize(const aclTensor* a, const aclTensor* b, const aclTensor* c, aclTensor* y,
                                          const aclScalar* alphaOptional, const aclScalar* betaOptional,
                                          bool transposeA, bool transposeB, const char* fillMode, int8_t cubeMathType,
                                          uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGemmSymm, DFX_IN(a, b, c, alphaOptional, betaOptional, cubeMathType), DFX_OUT(y));
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = BuildGraph(
        GemmSymmParams{a, b, c, y, alphaOptional, betaOptional, transposeA, transposeB, fillMode, cubeMathType},
        uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGemmSymm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGemmSymm);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
