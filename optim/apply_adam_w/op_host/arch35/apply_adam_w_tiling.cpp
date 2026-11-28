/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_adam_w_tiling.cpp
 * \brief
 */
#include <iostream>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "../op_kernel/arch35/apply_adam_w_dag.h"
#include "apply_adam_w_tiling.h"

using namespace ge;
using namespace ApplyAdamWOp;
using namespace ApplyAdamWNs;

namespace optiling {
const size_t SYS_WORKSPACE = 16777216; // 16M
constexpr static int32_t OPTIONAL_INPUT_INDEX = 11;
const static std::map<size_t, string> TENSOR_INDEX_LIST = {{1, "m"}, {2, "v"}, {10, "grad"}};
const static std::map<size_t, string> SCALAR_INDEX_LIST = {
    {3, "beta1_power"}, {4, "beta2_power"}, {5, "lr"}, {6, "weight_decay"}, {7, "beta1"}, {8, "beta2"}, {9, "epsilon"}};

ge::graphStatus ApplyAdamWTiling::SetTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Enter SetTilingData");
    if (maximizeAttr_) {
        tiling_->maximizeFactor = static_cast<float>(-1.0);
    } else {
        tiling_->maximizeFactor = static_cast<float>(1.0);
    }
    auto rawTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, rawTilingData);

    size_t* currentWorkspace = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, currentWorkspace);
    currentWorkspace[0] = SYS_WORKSPACE;

    tilingKey = GET_TPL_TILING_KEY(schMode, amsgrad, dType);
    tilingContext_->SetTilingKey(tilingKey);
    tilingContext_->SetBlockDim(tiling_->eleBaseTilingData.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWTiling::CheckIsScalar(size_t inputIdx)
{
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto storageShape = inputShape->GetStorageShape();
    if (storageShape.IsScalar() || storageShape.GetShapeSize() == 1) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus ApplyAdamWTiling::CheckSameShape(size_t inputIdx, const gert::Shape& input0Shape)
{
    auto inputShape = tilingContext_->GetInputShape(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto curStorageShape = inputShape->GetStorageShape();
    if (curStorageShape != input0Shape) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWTiling::CheckSameDtype(size_t inputIdx, const ge::DataType& input0Dtype,
                                                 ge::DataType& curDtype)
{
    auto inputDesc = tilingContext_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    curDtype = inputDesc->GetDataType();
    if (curDtype != input0Dtype) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWTiling::CheckShapeAndType()
{
    auto inputShape = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape);
    auto inputStorageShape = inputShape->GetStorageShape();

    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    auto inputDtype = inputDesc->GetDataType();
    for (const auto& pair : SCALAR_INDEX_LIST) {
        const gert::StorageShape* curShape = tilingContext_->GetInputShape(pair.first);
        OP_CHECK_IF(
            CheckIsScalar(pair.first) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                tilingContext_->GetNodeName(), pair.second.c_str(),
                (curShape != nullptr ? Ops::Base::ToString(curShape->GetStorageShape()) : std::string("null")).c_str(),
                "Input should be a scalar or have shape size 1"),
            return ge::GRAPH_FAILED);
        ge::DataType curDtype = ge::DT_UNDEFINED;
        OP_CHECK_IF(CheckSameDtype(pair.first, inputDtype, curDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        tilingContext_->GetNodeName(), (pair.second + " and var").c_str(),
                        (Ops::Base::ToString(curDtype) + " and " + Ops::Base::ToString(inputDtype)).c_str(),
                        "Their dtypes should be the same"),
                    return ge::GRAPH_FAILED);
    }
    for (const auto& pair : TENSOR_INDEX_LIST) {
        const gert::StorageShape* curShape = tilingContext_->GetInputShape(pair.first);
        OP_CHECK_IF(CheckSameShape(pair.first, inputStorageShape) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        tilingContext_->GetNodeName(), (std::string("var and ") + pair.second).c_str(),
                        (Ops::Base::ToString(inputStorageShape) + " and " +
                         (curShape != nullptr ? Ops::Base::ToString(curShape->GetStorageShape()) : std::string("null")))
                            .c_str(),
                        (std::string("The shapes of input ") + pair.second + " and var should be the same").c_str()),
                    return ge::GRAPH_FAILED);
        ge::DataType curDtype = ge::DT_UNDEFINED;
        OP_CHECK_IF(CheckSameDtype(pair.first, inputDtype, curDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        tilingContext_->GetNodeName(), (pair.second + " and var").c_str(),
                        (Ops::Base::ToString(curDtype) + " and " + Ops::Base::ToString(inputDtype)).c_str(),
                        "Their dtypes should be the same"),
                    return ge::GRAPH_FAILED);
    }

    auto maxGradNormDesc = tilingContext_->GetOptionalInputDesc(OPTIONAL_INPUT_INDEX);
    if (maxGradNormDesc != nullptr) {
        auto maxGradNormDtype = maxGradNormDesc->GetDataType();
        if (maxGradNormDtype != inputDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                tilingContext_->GetNodeName(), "max_grad_norm and var",
                (Ops::Base::ToString(maxGradNormDtype) + " and " + Ops::Base::ToString(inputDtype)).c_str(),
                "The dtypes of max_grad_norm and var should be the same");
            return ge::GRAPH_FAILED;
        }
    }

    auto maxGradNormShape = tilingContext_->GetOptionalInputShape(OPTIONAL_INPUT_INDEX);
    if (maxGradNormShape != nullptr) {
        auto maxGradNormStorageShape = maxGradNormShape->GetStorageShape();
        if (maxGradNormStorageShape != inputStorageShape) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                tilingContext_->GetNodeName(), "max_grad_norm and var",
                (Ops::Base::ToString(maxGradNormStorageShape) + " and " + Ops::Base::ToString(inputStorageShape))
                    .c_str(),
                "The shapes of max_grad_norm and var should be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyAdamWTiling::DoElewiseTiling()
{
    ElewiseBaseTiling eleBaseTiling(tilingContext_);
    auto input0Desc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    ge::graphStatus ret = ge::GRAPH_FAILED;
    if (amsgradAttr_) {
        amsgrad = ATTR_IS_TRUE;
        if (input0DType == ge::DT_FLOAT16) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWAmsGradDAG<half, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_FP16;
        } else if (input0DType == ge::DT_BF16) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWAmsGradDAG<half, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_BF16;
        } else if (input0DType == ge::DT_FLOAT) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWAmsGradDAG<float, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_FP32;
        } else {
            OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "var (amsgrad branch)",
                                      Ops::Base::ToString(input0DType).c_str(), "fp16, bf16 or fp32");
            ret = ge::GRAPH_FAILED;
        }
    } else {
        if (input0DType == ge::DT_FLOAT16) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWDAG<half, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_FP16;
        } else if (input0DType == ge::DT_BF16) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWDAG<half, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_BF16;
        } else if (input0DType == ge::DT_FLOAT) {
            ret = eleBaseTiling.DoTiling<ApplyAdamWDAG<float, float>::OpDag>(tiling_->eleBaseTilingData);
            dType = APPLY_ADAM_W_TPL_FP32;
        } else {
            OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "var (non-amsgrad branch)",
                                      Ops::Base::ToString(input0DType).c_str(), "fp16, bf16 or fp32");
            ret = ge::GRAPH_FAILED;
        }
    }
    return ret;
}

ge::graphStatus ApplyAdamWTiling::RunTiling()
{
    if (tilingContext_ == nullptr) {
        const std::vector<const char*> msgKey = {"opName", "reason"};
        const std::vector<const char*> msgValue = {
            "ApplyAdamW", "RunTiling: TilingContext is nullptr. Check the GE tiling invocation."};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
        OP_LOGE_WITHOUT_REPORT("ApplyAdamW", "RunTiling: TilingContext is nullptr. Check the GE tiling invocation.");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(CheckShapeAndType() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(tilingContext_->GetNodeName(),
                                       "CheckShapeAndType failed; propagate the lower-layer first error."),
                return ge::GRAPH_FAILED);
    ElewiseBaseTiling elewiseBaseTiling(tilingContext_);
    tiling_ = tilingContext_->GetTilingData<ApplyAdamWTilingData>();
    if (tiling_ == nullptr) {
        const std::vector<const char*> msgKey = {"opName", "reason"};
        const std::vector<const char*> msgValue = {
            "ApplyAdamW", "RunTiling: GetTilingData returns nullptr. Check the GE tiling data allocation."};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
        OP_LOGE_WITHOUT_REPORT("ApplyAdamW", "RunTiling: GetTilingData returns nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const bool* amsgradAttr = attrs->GetAttrPointer<bool>(0);
    amsgradAttr_ = amsgradAttr != nullptr ? *amsgradAttr : false;

    // amsgrad=true 时可选输入 max_grad_norm 必选（对齐 aclnn 层校验与 apply_adam_w_v2 的
    // SEQ28 裁决修复）：GE/GEIR 图通路不经过 aclnn，tiling 是唯一拦截点；
    // 缺失时 kernel amsgrad DAG 会对 nullptr GM 地址执行 CopyIn（非法地址访问）。
    OP_CHECK_IF(amsgradAttr_ && tilingContext_->GetOptionalInputDesc(OPTIONAL_INPUT_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ApplyAdamW", "max_grad_norm", "None",
                                                      "max_grad_norm is required when amsgrad is true"),
                return ge::GRAPH_FAILED);

    const bool* maximizeAttr = attrs->GetAttrPointer<bool>(1);
    maximizeAttr_ = maximizeAttr != nullptr ? *maximizeAttr : false;
    OP_CHECK_IF(DoElewiseTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT(tilingContext_->GetNodeName(),
                                       "DoElewiseTiling failed; propagate the lower-layer first error."),
                return ge::GRAPH_FAILED);
    return SetTilingData();
}

static ge::graphStatus TilingForApplyAdamW(gert::TilingContext* context)
{
    if (context == nullptr) {
        const std::vector<const char*> msgKey = {"opName", "reason"};
        const std::vector<const char*> msgValue = {
            "ApplyAdamW", "TilingForApplyAdamW: TilingContext is nullptr. Check the GE tiling invocation."};
        REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgValue);
        OP_LOGE_WITHOUT_REPORT("ApplyAdamW",
                               "TilingForApplyAdamW: TilingContext is nullptr. Check the GE tiling invocation.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD("ApplyAdamWTiling", "Enter TilingForApplyAdamW");

    auto compileInfo = reinterpret_cast<const ApplyAdamWCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD("ApplyAdamWTiling", "Enter new ApplyAdamWTiling");
    ApplyAdamWTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepareForApplyAdamW(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ApplyAdamWCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyAdamW).Tiling(TilingForApplyAdamW).TilingParse<ApplyAdamWCompileInfo>(TilingPrepareForApplyAdamW);
} // namespace optiling
