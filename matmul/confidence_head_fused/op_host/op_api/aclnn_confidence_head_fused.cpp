/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_confidence_head_fused.h"
#include <initializer_list>
#include <limits>
#include "confidence_head_fused.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace {
constexpr size_t MATRIX_DIM_NUM = 2;
constexpr size_t BATCH_MATRIX_DIM_NUM = 3;
constexpr int64_t MAX_DIM = std::numeric_limits<uint32_t>::max();
constexpr float CLAMP_LIMIT = static_cast<float>(std::numeric_limits<int32_t>::max());
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* hidden, const aclTensor* weight, const aclTensor* out)
{
    OP_CHECK_NULL(hidden, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckSocVersion()
{
    auto soc = GetCurrentPlatformInfo().GetSocVersion();
    if (soc != SocVersion::ASCEND950 && soc != SocVersion::ASCEND350) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ConfidenceHeadFused only supports Ascend950 and Ascend350.");
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* hidden, const aclTensor* weight, const aclTensor* markovEmbOptional,
                            const aclTensor* temperatureOptional, const aclTensor* biasOptional, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(hidden, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(weight, hidden->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, DataType::DT_FLOAT, return false);
    if (markovEmbOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(markovEmbOptional, hidden->GetDataType(), return false);
    }
    if (temperatureOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(temperatureOptional, DataType::DT_FLOAT, return false);
    }
    if (biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(biasOptional, DataType::DT_FLOAT, return false);
    }
    return true;
}

static bool CheckDimRange(const aclTensor* tensor, const char* name)
{
    if (tensor == nullptr) {
        return true;
    }
    for (size_t i = 0; i < tensor->GetViewShape().GetDimNum(); ++i) {
        int64_t dim = tensor->GetViewShape().GetDim(i);
        if (dim < 0 || dim > MAX_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dimensions must be in [0, UINT32_MAX].", name);
            return false;
        }
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* hidden, const aclTensor* weight, const aclTensor* markovEmbOptional,
                            const aclTensor* temperatureOptional, const aclTensor* biasOptional, const aclTensor* out)
{
    size_t rank = hidden->GetViewShape().GetDimNum();
    if (rank != MATRIX_DIM_NUM && rank != BATCH_MATRIX_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hidden must be 2D or 3D.");
        return false;
    }
    OP_CHECK_WRONG_DIMENSION(weight, 1, return false);
    OP_CHECK_WRONG_DIMENSION(out, rank - 1, return false);
    if (markovEmbOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(markovEmbOptional, rank, return false);
    }
    if (temperatureOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(temperatureOptional, 1, return false);
    }
    if (biasOptional != nullptr) {
        OP_CHECK_WRONG_DIMENSION(biasOptional, 1, return false);
    }
    if (!CheckDimRange(hidden, "hidden") || !CheckDimRange(weight, "weight") || !CheckDimRange(out, "out") ||
        !CheckDimRange(markovEmbOptional, "markovEmbOptional") ||
        !CheckDimRange(temperatureOptional, "temperatureOptional") || !CheckDimRange(biasOptional, "biasOptional")) {
        return false;
    }
    if (temperatureOptional != nullptr && temperatureOptional->GetViewShape().GetDim(0) != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "temperatureOptional must have shape [1].");
        return false;
    }
    if (biasOptional != nullptr && biasOptional->GetViewShape().GetDim(0) != hidden->GetViewShape().GetDim(rank - 2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "biasOptional must have shape [N], shared across batches.");
        return false;
    }
    int64_t rows = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        int64_t dim = hidden->GetViewShape().GetDim(i);
        if (out->GetViewShape().GetDim(i) != dim ||
            (markovEmbOptional != nullptr && markovEmbOptional->GetViewShape().GetDim(i) != dim) ||
            (dim != 0 && rows > MAX_DIM / dim)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Leading dimensions must match and flattened rows must fit UINT32.");
            return false;
        }
        rows *= dim;
    }
    int64_t kEmbed = hidden->GetViewShape().GetDim(rank - 1);
    int64_t kMarkov = markovEmbOptional == nullptr ? 0 : markovEmbOptional->GetViewShape().GetDim(rank - 1);
    if (kEmbed > MAX_DIM - kMarkov || weight->GetViewShape().GetDim(0) != kEmbed + kMarkov) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight length must equal K1+K2 and fit UINT32.");
        return false;
    }
    return true;
}

static bool CheckTensorFormat(const aclTensor* tensor, const char* name)
{
    if (tensor == nullptr) {
        return true;
    }
    if (tensor->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s does not support FRACTAL_NZ format.", name);
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* hidden, const aclTensor* weight, const aclTensor* markovEmbOptional,
                        const aclTensor* temperatureOptional, const aclTensor* biasOptional, const aclTensor* out)
{
    return CheckTensorFormat(hidden, "hidden") && CheckTensorFormat(weight, "weight") &&
           CheckTensorFormat(out, "out") && CheckTensorFormat(markovEmbOptional, "markovEmbOptional") &&
           CheckTensorFormat(temperatureOptional, "temperatureOptional") &&
           CheckTensorFormat(biasOptional, "biasOptional");
}

static bool CheckClampScalar(const aclScalar* scalar, const char* name)
{
    if (scalar != nullptr && !CanCast(scalar->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be convertible to FP32.", name);
        return false;
    }
    return true;
}

aclnnStatus CheckParams(const aclTensor* hidden, const aclTensor* weight, const aclTensor* markovEmbOptional,
                        const aclTensor* temperatureOptional, const aclTensor* biasOptional,
                        const aclScalar* clampLowOptional, const aclScalar* clampHighOptional, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(hidden, weight, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckSocVersion(), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(hidden, weight, markovEmbOptional, temperatureOptional, biasOptional, out),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(hidden, weight, markovEmbOptional, temperatureOptional, biasOptional, out),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(hidden, weight, markovEmbOptional, temperatureOptional, biasOptional, out),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckClampScalar(clampLowOptional, "clampLowOptional"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckClampScalar(clampHighOptional, "clampHighOptional"), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclTensor* PrepareInput(const aclTensor* input, aclOpExecutor* executor)
{
    if (input == nullptr) {
        return nullptr;
    }
    // Normalize metadata for zero-element views.
    auto contiguousInput = l0op::Contiguous(input, executor);
    if (contiguousInput == nullptr || contiguousInput->GetStorageFormat() == Format::FORMAT_ND) {
        return contiguousInput;
    }
    return l0op::ReFormat(contiguousInput, Format::FORMAT_ND, executor);
}
} // namespace

aclnnStatus aclnnConfidenceHeadFusedGetWorkspaceSize(const aclTensor* hidden, const aclTensor* weight,
                                                     const aclTensor* markovEmbOptional,
                                                     const aclTensor* temperatureOptional,
                                                     const aclTensor* biasOptional, const aclScalar* clampLowOptional,
                                                     const aclScalar* clampHighOptional, const aclTensor* out,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnConfidenceHeadFused,
                   DFX_IN(hidden, weight, markovEmbOptional, temperatureOptional, biasOptional, clampLowOptional,
                          clampHighOptional),
                   DFX_OUT(out));
    if (workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto ret = CheckParams(hidden, weight, markovEmbOptional, temperatureOptional, biasOptional, clampLowOptional,
                           clampHighOptional, out);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto hiddenContiguous = PrepareInput(hidden, uniqueExecutor.get());
    auto weightContiguous = PrepareInput(weight, uniqueExecutor.get());
    auto markovContiguous = PrepareInput(markovEmbOptional, uniqueExecutor.get());
    auto temperatureContiguous = PrepareInput(temperatureOptional, uniqueExecutor.get());
    auto biasContiguous = PrepareInput(biasOptional, uniqueExecutor.get());
    if (hiddenContiguous == nullptr || weightContiguous == nullptr ||
        (markovEmbOptional != nullptr && markovContiguous == nullptr) ||
        (temperatureOptional != nullptr && temperatureContiguous == nullptr) ||
        (biasOptional != nullptr && biasContiguous == nullptr)) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    float low = clampLowOptional == nullptr ? -CLAMP_LIMIT : clampLowOptional->ToFloat();
    float high = clampHighOptional == nullptr ? CLAMP_LIMIT : clampHighOptional->ToFloat();
    auto result = l0op::ConfidenceHeadFused(hiddenContiguous, weightContiguous, markovContiguous, temperatureContiguous,
                                            biasContiguous, low, high, uniqueExecutor.get());
    if (result == nullptr || (!out->IsEmpty() && l0op::ViewCopy(result, out, uniqueExecutor.get()) == nullptr)) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConfidenceHeadFused(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConfidenceHeadFused);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
