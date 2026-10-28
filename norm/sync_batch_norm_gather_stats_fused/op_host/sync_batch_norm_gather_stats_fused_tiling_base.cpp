/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_fused_tiling_base.cpp
 * \brief
 */

#include "sync_batch_norm_gather_stats_fused_tiling.h"

using namespace Ops::Base;

namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "SyncBatchNormGatherStatsFused",
                                               reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

static const size_t INPUT_IDX_ZERO = 0;
static const size_t INPUT_IDX_ONE = 1;
static const size_t INPUT_IDX_TWO = 2;
static const size_t INPUT_IDX_THREE = 3;
static const size_t INPUT_IDX_FOUR = 4;
static const size_t OUTPUT_IDX_ZERO = 0;
static const size_t OUTPUT_IDX_ONE = 1;
static const size_t OUTPUT_IDX_TWO = 2;
static const size_t OUTPUT_IDX_THREE = 3;
static const size_t BASE_WSP_SIZE = 0;

static const std::vector<std::string> inputIdx = {"mean", "invstd", "counts", "running_mean", "running_var"};
static const std::vector<std::string> outputIdx = {"mean_all_out", "invstd_all_out", "running_mean_out",
                                                   "running_var_out"};

ge::graphStatus SyncBatchNormGatherStatsFusedTilingBase::ValidateInputDtypes()
{
    auto meanDesc = context_->GetInputDesc(INPUT_IDX_ZERO);
    OP_CHECK_IF(
        meanDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "mean",
                                                 "the input desc of required input mean is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    commonParams.dtype = meanDesc->GetDataType();

    auto invstdDesc = context_->GetInputDesc(INPUT_IDX_ONE);
    OP_CHECK_IF(
        invstdDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "invstd",
                                                 "the input desc of required input invstd is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    ge::DataType invstdDtype = invstdDesc->GetDataType();

    auto countsDesc = context_->GetInputDesc(INPUT_IDX_TWO);
    OP_CHECK_IF(
        countsDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "counts",
                                                 "the input desc of required input counts is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    ge::DataType countsDtype = countsDesc->GetDataType();

    auto runningMeanDesc = context_->GetInputDesc(INPUT_IDX_THREE);
    OP_CHECK_IF(runningMeanDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "running_mean",
                    "the input desc of required input running_mean is missing in the tiling context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    ge::DataType runningMeanDtype = runningMeanDesc->GetDataType();

    auto runningVarDesc = context_->GetInputDesc(INPUT_IDX_FOUR);
    OP_CHECK_IF(runningVarDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context_->GetNodeName(), "running_var",
                    "the input desc of required input running_var is missing in the tiling context; check that the "
                    "graph node provides all required inputs"),
                return ge::GRAPH_FAILED);
    ge::DataType runningVarDtype = runningVarDesc->GetDataType();

    OP_CHECK_IF((commonParams.dtype != ge::DataType::DT_FLOAT) && (commonParams.dtype != ge::DataType::DT_FLOAT16) &&
                    (commonParams.dtype != ge::DataType::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "mean", ToString(commonParams.dtype).c_str(),
                                          "FLOAT, FLOAT16 or BF16"),
                return ge::GRAPH_FAILED);

    if (invstdDtype != commonParams.dtype) {
        std::string dtypeMsg = ToString(invstdDtype) + " and " + ToString(commonParams.dtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "invstd and mean", dtypeMsg.c_str(),
                                               "The dtypes of input invstd and input mean should be the same");
        return ge::GRAPH_FAILED;
    }

    if (runningMeanDtype != commonParams.dtype) {
        std::string dtypeMsg = ToString(runningMeanDtype) + " and " + ToString(commonParams.dtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "running_mean and mean", dtypeMsg.c_str(),
                                               "The dtypes of input running_mean and input mean should be the same");
        return ge::GRAPH_FAILED;
    }

    if (runningVarDtype != commonParams.dtype) {
        std::string dtypeMsg = ToString(runningVarDtype) + " and " + ToString(commonParams.dtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "running_var and mean", dtypeMsg.c_str(),
                                               "The dtypes of input running_var and input mean should be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedTilingBase::ValidateInputShapes()
{
    auto mean = context_->GetInputShape(INPUT_IDX_ZERO);
    OP_CHECK_IF(
        mean == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "mean",
                                                 "the input shape of required input mean is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto meanShape = mean->GetStorageShape();
    int64_t meanDimNum = meanShape.GetDimNum();

    OP_CHECK_IF(
        meanDimNum != 2,
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "mean", std::to_string(meanDimNum).c_str(),
                                                  "The dim num of input mean should be 2 (N*C)"),
        return ge::GRAPH_FAILED);

    commonParams.nLength = meanShape.GetDim(0);
    commonParams.cLength = meanShape.GetDim(1);

    OP_CHECK_IF((commonParams.nLength <= 0) || (commonParams.cLength <= 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->GetNodeName(), "mean", ToString(meanShape).c_str(),
                    "The shape of input mean can not be an empty tensor or an invalid tensor with a negative dim"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedTilingBase::GetAndValidateAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        ReportTilingStageError(context_->GetNodeName(),
                               "GetShapeAttrsInfo: GetAttrs() returned null; check the TilingContext construction"),
        return ge::GRAPH_FAILED);

    const float* momentumPtr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_IF(
        momentumPtr == nullptr,
        ReportTilingStageError(
            context_->GetNodeName(),
            "the required attr momentum is missing in the tiling context; check the attr registration on the node"),
        return ge::GRAPH_FAILED);
    commonParams.momentum = *momentumPtr;

    const float* epsPtr = attrs->GetAttrPointer<float>(1);
    OP_CHECK_IF(epsPtr == nullptr,
                ReportTilingStageError(
                    context_->GetNodeName(),
                    "the required attr eps is missing in the tiling context; check the attr registration on the node"),
                return ge::GRAPH_FAILED);
    commonParams.eps = *epsPtr;

    OP_CHECK_IF((commonParams.momentum < 0.0f) || (commonParams.momentum > 1.0f),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "momentum",
                                                      std::to_string(commonParams.momentum).c_str(),
                                                      "the value of attr momentum must be in range [0, 1]"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        commonParams.eps <= 0.0f,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "eps", std::to_string(commonParams.eps).c_str(),
                                              "the value of attr eps must be greater than 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void SyncBatchNormGatherStatsFusedTilingBase::SetDtypeKey()
{
    if (commonParams.dtype == ge::DataType::DT_FLOAT) {
        commonParams.dtypeKey = DtypeKey::FLOAT32;
    } else if (commonParams.dtype == ge::DataType::DT_FLOAT16) {
        commonParams.dtypeKey = DtypeKey::FLOAT16;
    } else if (commonParams.dtype == ge::DataType::DT_BF16) {
        commonParams.dtypeKey = DtypeKey::BFLOAT16;
    }
}

ge::graphStatus SyncBatchNormGatherStatsFusedTilingBase::GetShapeAttrsInfo()
{
    if (ValidateInputDtypes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (ValidateInputShapes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (GetAndValidateAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    SetDtypeKey();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedTilingBase::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<SyncBatchNormGatherStatsFusedCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                ReportTilingStageError(
                    context_->GetNodeName(),
                    "GetPlatformInfo: compile info is null; check the SoC-specific OPP package and its loading logs"),
                return ge::GRAPH_FAILED);

    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    commonParams.blockSize = compileInfo->blockSize;
    commonParams.vlFp32 = compileInfo->vlFp32;

    OP_CHECK_IF(commonParams.coreNum <= 0,
                ReportTilingStageError(context_->GetNodeName(), "GetCoreNumAiv returned " +
                                                                    std::to_string(commonParams.coreNum) +
                                                                    "; check the platform info and SoC"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(commonParams.ubSizePlatForm <= 0,
                ReportTilingStageError(context_->GetNodeName(), "GetCoreMemSize(UB) returned " +
                                                                    std::to_string(commonParams.ubSizePlatForm) +
                                                                    "; check the platform info and SoC"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool SyncBatchNormGatherStatsFusedTilingBase::IsCapable() { return true; }

ge::graphStatus SyncBatchNormGatherStatsFusedTilingBase::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SyncBatchNormGatherStatsFusedTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SyncBatchNormGatherStatsFusedTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr,
                ReportTilingStageError(context_->GetNodeName(),
                                       "GetWorkspaceSizes returned null; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SyncBatchNormGatherStatsFusedTilingBase::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t SyncBatchNormGatherStatsFusedTilingBase::GetTilingKey() const { return 0; }

} // namespace optiling
