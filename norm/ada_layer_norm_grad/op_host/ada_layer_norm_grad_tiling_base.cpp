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
 * \file ada_layer_norm_grad_tiling_base.cc
 * \brief
 */

#include "ada_layer_norm_grad_tiling.h"

namespace optiling {
// Tiling阶段框架契约类失败(框架对象为空/平台信息异常/策略失败)统一上报EZ1008, 不再生成EZ9999内部码
static void ReportTilingStageError(const char* opName, const std::string& reason)
{
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opName != nullptr ? opName : "AdaLayerNormGrad", reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
}

static const size_t INPUT_IDX_ZERO = 0;
static const size_t INPUT_IDX_ONE = 1;
static const size_t INPUT_IDX_TWO = 2;
static const size_t INPUT_IDX_THREE = 3;
static const size_t INPUT_IDX_FOUR = 4;
static const size_t INPUT_IDX_FIVE = 5;
static const size_t INPUT_IDX_SIX = 6;
static const size_t OUTPUT_IDX_ZERO = 0;
static const size_t OUTPUT_IDX_ONE = 1;
static const size_t OUTPUT_IDX_TWO = 2;
static const size_t OUTPUT_IDX_THREE = 3;
static const size_t OUTPUT_IDX_FOUR = 4;
static const size_t BASE_WSP_SIZE = 0;
static const size_t GRAD_OUT_NUM = 5;

static bool CheckShapeSame(const gert::TilingContext* context_, const size_t leftIndex, const size_t rightIndex,
                           const bool isLeftInput, const bool isRightInput)
{
    const gert::StorageShape* leftShape = nullptr;
    const gert::StorageShape* rightShape = nullptr;
    if (isLeftInput) {
        leftShape = context_->GetInputShape(leftIndex);
        OP_CHECK_IF(leftShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(),
                                                             ("input[" + std::to_string(leftIndex) + "]").c_str(),
                                                             "the input shape is missing in the tiling context; check "
                                                             "that the graph node provides all required inputs"),
                    return ge::GRAPH_FAILED);
    } else {
        leftShape = context_->GetOutputShape(leftIndex);
        OP_CHECK_IF(leftShape == nullptr,
                    ReportTilingStageError(context_->GetNodeName(),
                                           "the output shape is missing in the tiling context; check that InferShape "
                                           "executed and the framework propagated the output shapes"),
                    return ge::GRAPH_FAILED);
    }
    if (isRightInput) {
        rightShape = context_->GetInputShape(rightIndex);
        OP_CHECK_IF(rightShape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(),
                                                             ("input[" + std::to_string(rightIndex) + "]").c_str(),
                                                             "the input shape is missing in the tiling context; check "
                                                             "that the graph node provides all required inputs"),
                    return ge::GRAPH_FAILED);
    } else {
        rightShape = context_->GetOutputShape(rightIndex);
        OP_CHECK_IF(rightShape == nullptr,
                    ReportTilingStageError(context_->GetNodeName(),
                                           "the output shape is missing in the tiling context; check that InferShape "
                                           "executed and the framework propagated the output shapes"),
                    return ge::GRAPH_FAILED);
    }

    // get storage shape
    gert::Shape leftShapeVal = leftShape->GetStorageShape();
    gert::Shape rightShapeVal = rightShape->GetStorageShape();

    // check the leftIndex shape and rightIndex shape are the same
    OP_CHECK_IF((leftShapeVal != rightShapeVal),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "left, right",
                    (Ops::Base::ToString(leftShapeVal) + ", " + Ops::Base::ToString(rightShapeVal)).c_str(),
                    ("the shape at leftIndex " + std::to_string(leftIndex) +
                     " must be the same as the shape at rightIndex " + std::to_string(rightIndex))
                        .c_str()),
                return ge::GRAPH_FAILED);
    return true;
}

static ge::graphStatus InputDtypeCheck(gert::TilingContext* context_, ge::DataType dyDtype, ge::DataType xDtype,
                                       ge::DataType rstdDtype, ge::DataType meanDtype, ge::DataType scaleDtype,
                                       ge::DataType gammaDtype, ge::DataType betaDtype)
{
    OP_LOGI(context_->GetNodeName(),
            "Input dtype check start: dy=%d, x=%d, rstd=%d, mean=%d, gamma=%d, beta=%d, scale=%d", dyDtype, xDtype,
            rstdDtype, meanDtype, gammaDtype, betaDtype, scaleDtype);

    // input check
    OP_CHECK_IF(
        (dyDtype != ge::DataType::DT_FLOAT) && (dyDtype != ge::DataType::DT_FLOAT16) &&
            (dyDtype != ge::DataType::DT_BF16),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "dy", Ops::Base::ToString(dyDtype).c_str(),
                                              "The dtype of dy must be FLOAT, FLOAT16 or BF16."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDtype != dyDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "x, dy",
                    (Ops::Base::ToString(xDtype) + ", " + Ops::Base::ToString(dyDtype)).c_str(),
                    "The dtypes of x and dy must be the same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((rstdDtype != ge::DataType::DT_FLOAT) || (meanDtype != ge::DataType::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "rstd, mean",
                    (Ops::Base::ToString(rstdDtype) + ", " + Ops::Base::ToString(meanDtype)).c_str(),
                    "The dtypes of rstd and mean must be FLOAT."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (gammaDtype != dyDtype) && (gammaDtype != ge::DataType::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "gamma", Ops::Base::ToString(gammaDtype).c_str(),
                                              "The dtype of gamma must be the same as dy or FLOAT."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (betaDtype != dyDtype) && (betaDtype != ge::DataType::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "beta", Ops::Base::ToString(betaDtype).c_str(),
                                              "The dtype of beta must be the same as dy or FLOAT."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (scaleDtype != dyDtype) && (scaleDtype != ge::DataType::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "scale", Ops::Base::ToString(scaleDtype).c_str(),
                                              "The dtype of scale must be the same as dy or FLOAT."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void CheckInputShapes(gert::TilingContext* context_)
{
    CheckShapeSame(context_, INPUT_IDX_ZERO, INPUT_IDX_ONE, true, true);
    CheckShapeSame(context_, INPUT_IDX_ZERO, OUTPUT_IDX_ZERO, true, false);
    CheckShapeSame(context_, INPUT_IDX_TWO, INPUT_IDX_THREE, true, true);
    CheckShapeSame(context_, INPUT_IDX_FOUR, OUTPUT_IDX_ONE, true, false);
}

static bool GetAllInputDtypes(gert::TilingContext* context_, ParamsAdaLayerNormGrad& commonParams)
{
    auto dyDesc = context_->GetInputDesc(INPUT_IDX_ZERO);
    OP_CHECK_IF(
        dyDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "dy",
                                                 "the input desc of required input dy is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    commonParams.dyDtype = dyDesc->GetDataType();

    auto xDesc = context_->GetInputDesc(INPUT_IDX_ONE);
    OP_CHECK_IF(
        xDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "x",
                                                 "the input desc of required input x is missing in the tiling context; "
                                                 "check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    commonParams.xDtype = xDesc->GetDataType();

    auto rstdDesc = context_->GetInputDesc(INPUT_IDX_TWO);
    OP_CHECK_IF(
        rstdDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "rstd",
                                                 "the input desc of required input rstd is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    commonParams.rstdDtype = rstdDesc->GetDataType();

    auto meanDesc = context_->GetInputDesc(INPUT_IDX_THREE);
    OP_CHECK_IF(
        meanDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "mean",
                                                 "the input desc of required input mean is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    commonParams.meanDtype = meanDesc->GetDataType();

    auto scaleDesc = context_->GetInputDesc(INPUT_IDX_FOUR);
    OP_CHECK_IF(
        scaleDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "scale",
                                                 "the input desc of required input scale is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    commonParams.scaleDtype = scaleDesc->GetDataType();

    auto gammaDesc = context_->GetInputDesc(INPUT_IDX_FIVE);
    OP_CHECK_IF(
        gammaDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "gamma",
                                                 "the input desc of required input gamma is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    commonParams.gammaDtype = gammaDesc->GetDataType();

    auto betaDesc = context_->GetInputDesc(INPUT_IDX_SIX);
    OP_CHECK_IF(
        betaDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "beta",
                                                 "the input desc of required input beta is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    commonParams.betaDtype = betaDesc->GetDataType();
    return true;
}

static ge::graphStatus ExtraShape(const gert::TilingContext* context_, ParamsAdaLayerNormGrad& commonParams)
{
    // get shape
    auto dy = context_->GetInputShape(INPUT_IDX_ZERO);
    OP_CHECK_IF(
        dy == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "dy",
                                                 "the input shape of required input dy is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto dyShape = dy->GetStorageShape();
    int64_t dyDimNum = dyShape.GetDimNum();

    auto gamma = context_->GetInputShape(INPUT_IDX_FIVE);
    OP_CHECK_IF(
        gamma == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(), "gamma",
                                                 "the input shape of required input gamma is missing in the tiling "
                                                 "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    auto gammaShape = gamma->GetStorageShape();
    int64_t gammaDimNum = gammaShape.GetDimNum();

    OP_CHECK_IF((dyDimNum < gammaDimNum),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context_->GetNodeName(), "dy, gamma",
                    (std::to_string(dyDimNum) + ", " + std::to_string(gammaDimNum)).c_str(),
                    "the dim num of dy must not be less than the dim num of gamma"),
                return ge::GRAPH_FAILED);
    // fuse dims
    int64_t row = 1;
    int64_t col = 1;
    int64_t seq = dyShape.GetDim(dyDimNum - gammaDimNum - 1);
    int64_t batch = 1;
    for (int64_t i = 0; i < dyDimNum; i++) {
        OP_CHECK_IF(
            (dyShape.GetDim(i) <= 0),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "dy", Ops::Base::ToString(dyShape).c_str(),
                                                  "All axes of dy must be positive numbers"),
            return ge::GRAPH_FAILED);

        if (i < dyDimNum - gammaDimNum - 1) {
            batch *= dyShape.GetDim(i);
        }
        if (i < dyDimNum - gammaDimNum) {
            row *= dyShape.GetDim(i);
        } else {
            OP_CHECK_IF((dyShape.GetDim(i) != gammaShape.GetDim(i - dyDimNum + gammaDimNum)),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context_->GetNodeName(), "dy, gamma",
                            (Ops::Base::ToString(dyShape) + ", " + Ops::Base::ToString(gammaShape)).c_str(),
                            "the last dims of dy must be the same as the dims of gamma"),
                        return ge::GRAPH_FAILED);
            col *= dyShape.GetDim(i);
        }
    }
    commonParams.batchSize = batch;
    commonParams.seqSize = seq;
    commonParams.colSize = col;
    commonParams.rowSize = row;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetOutputDtype(const gert::TilingContext* context_, int idx, ge::DataType& dtype)
{
    auto desc = context_->GetOutputDesc(idx);
    OP_CHECK_IF(desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->GetNodeName(),
                                                         ("output[" + std::to_string(idx) + "]").c_str(),
                                                         "the output desc is missing in the tiling context; check that "
                                                         "the graph node provides all required outputs"),
                return ge::GRAPH_FAILED);
    dtype = desc->GetDataType();
    return desc ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus CheckOutputDtype(const gert::TilingContext* context_, ParamsAdaLayerNormGrad& params)
{
    const char* nodeName = context_->GetNodeName();

    // 检查dx/dscale/dshift（与dy类型一致）
    if (params.pdxIsRequire && GetOutputDtype(context_, OUTPUT_IDX_ZERO, params.dxDtype) == ge::GRAPH_SUCCESS) {
        OP_CHECK_IF(params.dxDtype != params.dyDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        nodeName, "pd_x, dy",
                        (Ops::Base::ToString(params.dxDtype) + ", " + Ops::Base::ToString(params.dyDtype)).c_str(),
                        "The dtypes of pd_x and dy must be the same when pd_x is required."),
                    return ge::GRAPH_FAILED);
    }
    if (params.pdscaleIsRequire && GetOutputDtype(context_, OUTPUT_IDX_ONE, params.dscaleDtype) == ge::GRAPH_SUCCESS) {
        OP_CHECK_IF(params.dscaleDtype != params.dyDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        nodeName, "pd_scale, dy",
                        (Ops::Base::ToString(params.dscaleDtype) + ", " + Ops::Base::ToString(params.dyDtype)).c_str(),
                        "The dtypes of pd_scale and dy must be the same when pd_scale is required."),
                    return ge::GRAPH_FAILED);
    }
    if (params.pdshiftIsRequire && GetOutputDtype(context_, OUTPUT_IDX_TWO, params.dshiftDtype) == ge::GRAPH_SUCCESS) {
        OP_CHECK_IF(params.dshiftDtype != params.dyDtype,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        nodeName, "pd_shift, dy",
                        (Ops::Base::ToString(params.dshiftDtype) + ", " + Ops::Base::ToString(params.dyDtype)).c_str(),
                        "The dtypes of pd_shift and dy must be the same when pd_shift is required."),
                    return ge::GRAPH_FAILED);
    }

    // 检查dgamma/dbeta（与gamma一致或为float32）
    if (params.pdgammaIsRequire &&
        GetOutputDtype(context_, OUTPUT_IDX_THREE, params.dgammaDtype) == ge::GRAPH_SUCCESS) {
        OP_CHECK_IF((params.dgammaDtype != params.gammaDtype) && (params.dgammaDtype != ge::DataType::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "pd_gamma", Ops::Base::ToString(params.dgammaDtype).c_str(),
                        "The dtype of pd_gamma must be the same as gamma or FLOAT when pd_gamma is required."),
                    return ge::GRAPH_FAILED);
    }
    if (params.pdbetaIsRequire && GetOutputDtype(context_, OUTPUT_IDX_FOUR, params.dbetaDtype) == ge::GRAPH_SUCCESS) {
        OP_CHECK_IF((params.dbetaDtype != params.gammaDtype) && (params.dbetaDtype != ge::DataType::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "pd_beta", Ops::Base::ToString(params.dbetaDtype).c_str(),
                        "The dtype of pd_beta must be the same as gamma or FLOAT when pd_beta is required."),
                    return ge::GRAPH_FAILED);
    }

    // 检查dgamma和dbeta类型一致性
    if (params.pdgammaIsRequire && params.pdbetaIsRequire) {
        OP_CHECK_IF(
            params.dgammaDtype != params.dbetaDtype,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                nodeName, "pd_gamma, pd_beta",
                (Ops::Base::ToString(params.dgammaDtype) + ", " + Ops::Base::ToString(params.dbetaDtype)).c_str(),
                "The dtypes of pd_gamma and pd_beta must be the same when both are required."),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaLayerNormGradTilingBase::GetShapeAttrsInfo()
{
    CheckInputShapes(context_);

    GetAllInputDtypes(context_, commonParams);
    // clac shpape
    OP_CHECK_IF(ExtraShape(context_, commonParams) == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "get shape failed."), return ge::GRAPH_FAILED);

    commonParams.pdxIsRequire = true;
    commonParams.pdgammaIsRequire = true;
    commonParams.pdbetaIsRequire = true;
    commonParams.pdscaleIsRequire = true;
    commonParams.pdshiftIsRequire = true;

    // check input dtype
    OP_CHECK_IF(InputDtypeCheck(context_, commonParams.dyDtype, commonParams.xDtype, commonParams.rstdDtype,
                                commonParams.meanDtype, commonParams.scaleDtype, commonParams.gammaDtype,
                                commonParams.betaDtype) == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "input dtype check failed."), return ge::GRAPH_FAILED);

    // check output dtype
    OP_CHECK_IF(CheckOutputDtype(context_, commonParams) == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "output dtype check failed."), return ge::GRAPH_FAILED);

    commonParams.colAlign = (commonParams.colSize + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM *
                            B16_BLOCK_ALIGN_NUM;
    commonParams.isDeterministicKey = 1;
    if (commonParams.dyDtype == ge::DataType::DT_FLOAT) {
        commonParams.colAlign = (commonParams.colSize + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM *
                                B32_BLOCK_ALIGN_NUM;
        commonParams.dtypeKey = LNGDtypeKey::FLOAT_FLOAT;
    } else if (commonParams.dyDtype == ge::DataType::DT_FLOAT16) {
        if (commonParams.gammaDtype == ge::DataType::DT_FLOAT16) {
            commonParams.dtypeKey = LNGDtypeKey::FLOAT16_FLOAT16;
        } else if (commonParams.gammaDtype == ge::DataType::DT_FLOAT) {
            commonParams.dtypeKey = LNGDtypeKey::FLOAT16_FLOAT;
        }
    } else if (commonParams.dyDtype == ge::DataType::DT_BF16) {
        if (commonParams.gammaDtype == ge::DataType::DT_BF16) {
            commonParams.dtypeKey = LNGDtypeKey::BFLOAT16_BFLOAT16;
        } else if (commonParams.gammaDtype == ge::DataType::DT_FLOAT) {
            commonParams.dtypeKey = LNGDtypeKey::BFLOAT16_FLOAT;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaLayerNormGradTilingBase::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<AdaLayerNormGradCompileInfo>();
    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
    commonParams.isRegBase = compileInfo->isRegBase;
    commonParams.blockSize = compileInfo->blockSize;
    commonParams.vlFp32 = compileInfo->vlFp32;
    return ge::GRAPH_SUCCESS;
}

bool AdaLayerNormGradTilingBase::IsCapable() { return true; }

ge::graphStatus AdaLayerNormGradTilingBase::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaLayerNormGradTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AdaLayerNormGradTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = BASE_WSP_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaLayerNormGradTilingBase::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t AdaLayerNormGradTilingBase::GetTilingKey() const { return 0; }

constexpr static int64_t CONST_ZERO = 0;
constexpr static int64_t CONST_ONE = 1;
constexpr static int64_t CONST_TWO = 2;
constexpr static int64_t CONST_FOUR = 4;
constexpr static int64_t CONST_SIXTY_THREE = 63;

int64_t AdaLayerNormGradTilingBase::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - __builtin_clzl(num);
        return (CONST_ONE << pow);
    }
}

int64_t AdaLayerNormGradTilingBase::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

} // namespace optiling
