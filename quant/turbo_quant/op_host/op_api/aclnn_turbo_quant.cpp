/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_turbo_quant.cpp
 * \brief aclnn interface of TurboQuant
 */
#include "aclnn_turbo_quant.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "turbo_quant.h"
#include "op_api/aclnn_util.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t TQ_LATENT_DIM_NUM = 2;
constexpr size_t TQ_DIM_TOKEN = 0;
constexpr size_t TQ_DIM_HEAD = 1;
constexpr int64_t TQ_N_CENT = 16;
constexpr int64_t TQ_SUPPORTED_HEAD_DIM = 512;
constexpr int64_t TQ_ELEMENTS_PER_BYTE = 2;

static bool CheckNotNull(const aclTensor* latent, const aclTensor* centroids, const aclTensor* y,
                         const aclTensor* scale)
{
    OP_CHECK_NULL(latent, return false);
    OP_CHECK_NULL(centroids, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(scale, return false);
    return true;
}

static bool CheckDtype(const aclTensor* latent, const aclTensor* centroids, const aclTensor* y, const aclTensor* scale)
{
    OP_CHECK_DTYPE_NOT_MATCH(latent, op::DataType::DT_FLOAT, return false);
    OP_CHECK_DTYPE_NOT_MATCH(centroids, op::DataType::DT_FLOAT, return false);
    OP_CHECK_DTYPE_NOT_MATCH(y, op::DataType::DT_UINT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(scale, op::DataType::DT_FLOAT16, return false);
    return true;
}

static bool CheckShape(const aclTensor* latent, const aclTensor* centroids, const aclTensor* y, const aclTensor* scale)
{
    const auto& latentView = latent->GetViewShape();
    if (latentView.GetDimNum() != TQ_LATENT_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The rank[%lu] of latent must be 2.", latentView.GetDimNum());
        return false;
    }
    const int64_t headDim = latentView.GetDim(TQ_DIM_HEAD);
    if (headDim != TQ_SUPPORTED_HEAD_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The headDim[%ld] of latent must be %ld.", headDim, TQ_SUPPORTED_HEAD_DIM);
        return false;
    }
    const int64_t numTokens = latentView.GetDim(TQ_DIM_TOKEN);
    if (numTokens < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The numTokens[%ld] of latent must be non-negative.", numTokens);
        return false;
    }

    op::Shape expectedCentroidsShape;
    expectedCentroidsShape.AppendDim(TQ_N_CENT);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(centroids, expectedCentroidsShape, return false);

    op::Shape expectedYShape;
    expectedYShape.AppendDim(numTokens);
    expectedYShape.AppendDim(headDim / TQ_ELEMENTS_PER_BYTE);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(y, expectedYShape, return false);

    op::Shape expectedScaleShape;
    expectedScaleShape.AppendDim(numTokens);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(scale, expectedScaleShape, return false);
    return true;
}
} // namespace

aclnnStatus aclnnTurboQuantGetWorkspaceSize(const aclTensor* latent, const aclTensor* centroids, const aclTensor* y,
                                            const aclTensor* scale, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnTurboQuant, DFX_IN(latent, centroids), DFX_OUT(y, scale));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(CheckNotNull(latent, centroids, y, scale), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(latent, centroids, y, scale), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(latent, centroids, y, scale), ACLNN_ERR_PARAM_INVALID);

    // Empty tensors carry no workload: with numTokens 0, y is [0, headDim/2] and scale is [0].
    if (latent->IsEmpty() || y->IsEmpty() || scale->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto latentContiguous = l0op::Contiguous(latent, uniqueExecutor.get());
    CHECK_RET(latentContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto centroidsContiguous = l0op::Contiguous(centroids, uniqueExecutor.get());
    CHECK_RET(centroidsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto turboQuantOut = l0op::TurboQuant(latentContiguous, centroidsContiguous, uniqueExecutor.get());
    auto yOut = std::get<0>(turboQuantOut);
    auto scaleOut = std::get<1>(turboQuantOut);
    CHECK_RET(yOut != nullptr && scaleOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto yViewCopyResult = l0op::ViewCopy(yOut, y, uniqueExecutor.get());
    CHECK_RET(yViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto scaleViewCopyResult = l0op::ViewCopy(scaleOut, scale, uniqueExecutor.get());
    CHECK_RET(scaleViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTurboQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTurboQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
