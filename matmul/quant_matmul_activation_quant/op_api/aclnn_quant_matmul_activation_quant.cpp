/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_matmul_activation_quant.h"

#include <string>

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "quant_matmul_activation_quant_checker.h"
#include "quant_matmul_activation_quant_util.h"

using namespace op;
using namespace QBMMActivationQuant;
using Ops::NN::FormatString;

namespace {

constexpr const char* API_NAME = "aclnnQuantMatmulActivationQuantGetWorkspaceSize";

static aclnnStatus CheckInputRanks(const QuantMatmulActivationQuantParams& params)
{
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto x2DimNum = params.x2->GetViewShape().GetDimNum();
    if (x1DimNum < MIN_MATRIX_RANK || x1DimNum > MAX_MATRIX_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "x1", FormatString("%zuD", x1DimNum).c_str(),
                                                 FormatString("the rank of x1 must be in the range [2, 6]"));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2DimNum < MIN_MATRIX_RANK || x2DimNum > MAX_MATRIX_RANK) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "x2", FormatString("%zuD", x2DimNum).c_str(),
                                                 FormatString("the rank of x2 must be in the range [2, 6]"));
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckWeightNdInputs(const aclTensor* x1, const aclTensor* x2)
{
    const auto status = CheckPlatformAndInputs(x1, x2, API_NAME);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    OP_LOGD("WeightND input validation succeeded.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const QuantMatmulActivationQuantParams& params)
{
    const auto shapeInfo = GetMatmulShapeInfo(params);
    CHECK_RET(CheckKDimensionsMatch(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckMKN(shapeInfo.mDim, shapeInfo.kDim, shapeInfo.nDim, API_NAME), ACLNN_ERR_PARAM_INVALID);
    if (IsMxFp4Input(params.x1, params.x2, params.y, params.yScale)) {
        CHECK_RET(CheckFp4Shape(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckMxScaleLastDim(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckBiasShape(params, shapeInfo.nDim, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckInputScaleShapes(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutputShape(params, shapeInfo, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const QuantMatmulActivationQuantParams& params)
{
    OP_LOGD("Parameter validation started.");
    CHECK_RET(CheckDtype(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormats(params, Format::FORMAT_ND, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttributes(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("Parameter validation succeeded.");

    return ACLNN_SUCCESS;
}

static aclnnStatus BuildExecutor(QuantMatmulActivationQuantParams& params, aclOpExecutor* executor)
{
    const bool transposeX1Before = params.transposeX1;
    CHECK_RET(PrepareCommonInputs(params, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    auto reformattedX2 = QuantMatmulActivationQuantAclnnCheck::SetTensorToNDFormat(params.x2);
    CHECK_RET(reformattedX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    params.x2 = reformattedX2;
    const bool transposeX2Before = params.transposeX2;
    CHECK_RET(QuantMatmulActivationQuantAclnnCheck::TensorContiguousProcess(params.x2, params.transposeX2, executor),
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MxScaleContiguousProcess(params.x1Scale, params.transposeX1 != transposeX1Before, executor),
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MxScaleContiguousProcess(params.x2Scale, params.transposeX2 != transposeX2Before, executor),
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(CheckGroupSize(params, API_NAME), ACLNN_ERR_PARAM_INVALID);

    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return AddQuantMatmulActivationQuantTasks(params, executor);
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnQuantMatmulActivationQuantGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2,
                                                            const aclTensor* x1ScaleOptional, const aclTensor* x2Scale,
                                                            const aclTensor* biasOptional, bool transposeX1,
                                                            bool transposeX2, int64_t groupSize, char* activationType,
                                                            char* quantMode, char* roundMode, int64_t scaleAlg,
                                                            double dstTypeMax, aclTensor* yOut, aclTensor* yScaleOut,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulActivationQuant,
                   DFX_IN(x1, x2, x1ScaleOptional, x2Scale, biasOptional, transposeX1, transposeX2, groupSize,
                          activationType, quantMode, roundMode, scaleAlg, dstTypeMax),
                   DFX_OUT(yOut, yScaleOut));

    auto ret = CheckWeightNdInputs(x1, x2);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto yDtype = x1->GetDataType();
    QuantMatmulActivationQuantParams params{
        x1,        x2,        x1ScaleOptional, x2Scale, biasOptional, yOut,       yScaleOut,   activationType,
        quantMode, roundMode, groupSize,       yDtype,  scaleAlg,     dstTypeMax, transposeX1, transposeX2};

    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckNotEmpty(params, API_NAME) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckInputRanks(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(executorPtr != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(PrepareInputViews(params, executorPtr), ACLNN_ERR_INNER_NULLPTR);

    ret = BuildExecutor(params, executorPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulActivationQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulActivationQuant);
    OP_CHECK(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
             OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER, "This is an error in QuantMatmulActivationQuant launch aicore."),
             return ACLNN_ERR_INNER);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
